/*
 * Copyright 2026 François Lagunas
 * Copyright 2020-2022 GoPro Inc.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <nopemd.h>

#include <CoreVideo/CoreVideo.h>
#include <Metal/Metal.h>

#include "config.h"

#include "hwmap.h"
#include "image.h"
#include "internal.h"
#include "log.h"
#include "ngpu/format.h"
#include "ngpu/metal/ctx_mtl.h"
#include "ngpu/metal/format_mtl.h"
#include "ngpu/metal/texture_mtl.h"
#include "nopegl.h"

/*
 * Zero-copy mapping of VideoToolbox frames: each plane of the IOSurface
 * backing the CVPixelBuffer is exposed as a Metal texture through a
 * CVMetalTextureCache, then wrapped as a ngpu texture.
 */

struct format_desc {
    enum image_layout layout;
    size_t nb_planes;
    enum ngpu_format planes[2];
};

static int vt_get_format_desc(OSType format, struct format_desc *desc)
{
    switch (format) {
    case kCVPixelFormatType_32BGRA:
        *desc = (struct format_desc){NGLI_IMAGE_LAYOUT_DEFAULT, 1, {NGPU_FORMAT_B8G8R8A8_UNORM}};
        return 0;
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
        *desc = (struct format_desc){NGLI_IMAGE_LAYOUT_NV12, 2, {NGPU_FORMAT_R8_UNORM, NGPU_FORMAT_R8G8_UNORM}};
        return 0;
    case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
    case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
        *desc = (struct format_desc){NGLI_IMAGE_LAYOUT_NV12, 2, {NGPU_FORMAT_R16_UNORM, NGPU_FORMAT_R16G16_UNORM}};
        return 0;
    default:
        LOG(ERROR, "unsupported pixel format %d", format);
        return NGL_ERROR_UNSUPPORTED;
    }
}

struct hwmap_vt_mtl {
    struct nmd_frame *frame;
    struct ngpu_texture *planes[2];
    CVMetalTextureRef texture_refs[2];
    struct format_desc format_desc;
    CVMetalTextureCacheRef texture_cache;
};

static void release_texture_refs(struct hwmap_vt_mtl *vt)
{
    for (size_t i = 0; i < 2; i++) {
        if (vt->texture_refs[i])
            CFRelease(vt->texture_refs[i]);
        vt->texture_refs[i] = NULL;
    }
}

static int vt_mtl_map_frame(struct hwmap *hwmap, struct nmd_frame *frame)
{
    struct hwmap_vt_mtl *vt = hwmap->hwmap_priv_data;
    const struct hwmap_params *params = &hwmap->params;

    /* The previous frame and its textures are kept alive until now: the GPU
     * may still be sampling them */
    release_texture_refs(vt);
    nmd_frame_releasep(&vt->frame);
    vt->frame = frame;

    CVPixelBufferRef cvpixbuf = (CVPixelBufferRef)frame->datap[0];

    for (size_t i = 0; i < vt->format_desc.nb_planes; i++) {
        const size_t width = CVPixelBufferGetWidthOfPlane(cvpixbuf, i);
        const size_t height = CVPixelBufferGetHeightOfPlane(cvpixbuf, i);
        const enum ngpu_format format = vt->format_desc.planes[i];

        CVReturn status = CVMetalTextureCacheCreateTextureFromImage(NULL, vt->texture_cache, cvpixbuf, NULL,
                                                                    ngpu_format_ngl_to_mtl(format),
                                                                    width, height, i, &vt->texture_refs[i]);
        if (status != kCVReturnSuccess) {
            LOG(ERROR, "could not create texture from image on plane %zu: %d", i, status);
            return NGL_ERROR_GRAPHICS_GENERIC;
        }
        id<MTLTexture> texture = CVMetalTextureGetTexture(vt->texture_refs[i]);

        struct ngpu_texture *plane = vt->planes[i];
        const struct ngpu_texture_mtl *plane_mtl = (const struct ngpu_texture_mtl *)plane;
        if (plane_mtl->wrapped && plane->params.width == width && plane->params.height == height) {
            ngpu_texture_mtl_set_texture(plane, texture);
            continue;
        }

        const struct ngpu_texture_params plane_params = {
            .type       = NGPU_TEXTURE_TYPE_2D,
            .format     = format,
            .width      = (uint32_t)width,
            .height     = (uint32_t)height,
            .min_filter = params->texture_min_filter,
            .mag_filter = params->texture_mag_filter,
            .wrap_s     = params->texture_wrap_s,
            .wrap_t     = params->texture_wrap_t,
            .usage      = params->texture_usage,
        };
        int ret = ngpu_texture_mtl_wrap(plane, &plane_params, texture);
        if (ret < 0)
            return ret;
    }

    return 0;
}

static bool support_direct_rendering(struct hwmap *hwmap)
{
    const struct hwmap_params *params = &hwmap->params;

    bool direct_rendering = NGLI_HAS_ALL_FLAGS(params->image_layouts, NGLI_IMAGE_LAYOUT_NV12_BIT);
    if (direct_rendering && params->texture_mipmap_filter) {
        LOG(WARNING, "IOSurface buffers do not support mipmapping: disabling direct rendering");
        direct_rendering = false;
    }

    return direct_rendering;
}

static int vt_mtl_init(struct hwmap *hwmap, struct nmd_frame *frame)
{
    struct ngl_ctx *ctx = hwmap->ctx;
    struct ngpu_ctx *gpu_ctx = ctx->gpu_ctx;
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)gpu_ctx;
    struct hwmap_vt_mtl *vt = hwmap->hwmap_priv_data;

    CVPixelBufferRef cvpixbuf = (CVPixelBufferRef)frame->datap[0];
    int ret = vt_get_format_desc(CVPixelBufferGetPixelFormatType(cvpixbuf), &vt->format_desc);
    if (ret < 0)
        return ret;

    CVReturn status = CVMetalTextureCacheCreate(NULL, NULL, gpu_ctx_mtl->device, NULL, &vt->texture_cache);
    if (status != kCVReturnSuccess) {
        LOG(ERROR, "could not create Metal texture cache: %d", status);
        return NGL_ERROR_GRAPHICS_GENERIC;
    }

    for (size_t i = 0; i < 2; i++) {
        vt->planes[i] = ngpu_texture_create(gpu_ctx);
        if (!vt->planes[i])
            return NGL_ERROR_MEMORY;
    }

    const struct image_params image_params = {
        .width       = frame->width,
        .height      = frame->height,
        .layout      = vt->format_desc.layout,
        .color_scale = 1.f,
        .color_info  = ngli_color_info_from_nopemd_frame(frame),
    };
    ngli_image_init(&hwmap->mapped_image, &image_params, vt->planes);

    hwmap->require_hwconv = !support_direct_rendering(hwmap);

    return 0;
}

static void vt_mtl_uninit(struct hwmap *hwmap)
{
    struct hwmap_vt_mtl *vt = hwmap->hwmap_priv_data;

    for (size_t i = 0; i < 2; i++)
        ngpu_texture_freep(&vt->planes[i]);
    release_texture_refs(vt);
    if (vt->texture_cache) {
        CFRelease(vt->texture_cache);
        vt->texture_cache = NULL;
    }
    nmd_frame_releasep(&vt->frame);
}

const struct hwmap_class ngli_hwmap_vt_mtl_class = {
    .name      = "videotoolbox (iosurface → nv12)",
    .hwformat  = NMD_PIXFMT_VT,
    .layouts   = (const enum image_layout[]){
        NGLI_IMAGE_LAYOUT_DEFAULT,
        NGLI_IMAGE_LAYOUT_NV12,
        NGLI_IMAGE_LAYOUT_NONE
    },
    .flags     = HWMAP_FLAG_FRAME_OWNER,
    .priv_size = sizeof(struct hwmap_vt_mtl),
    .init      = vt_mtl_init,
    .map_frame = vt_mtl_map_frame,
    .uninit    = vt_mtl_uninit,
};

const struct hwmap_class *ngli_hwmap_mtl_classes[] = {
    &ngli_hwmap_vt_mtl_class,
    NULL
};
