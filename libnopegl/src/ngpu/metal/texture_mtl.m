/*
 * Copyright 2026 François Lagunas
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

#include "config.h"

#include "ctx_mtl.h"
#include "format_mtl.h"
#include "log.h"
#include "nopegl.h"
#include "texture_mtl.h"
#include "utils/bits.h"
#include "utils/memory.h"
#include "utils/utils.h"

static const MTLSamplerMinMagFilter filter_map[NGPU_NB_FILTER] = {
    [NGPU_FILTER_NEAREST] = MTLSamplerMinMagFilterNearest,
    [NGPU_FILTER_LINEAR]  = MTLSamplerMinMagFilterLinear,
};

static const MTLSamplerMipFilter mipmap_filter_map[NGPU_NB_MIPMAP] = {
    [NGPU_MIPMAP_FILTER_NONE]    = MTLSamplerMipFilterNotMipmapped,
    [NGPU_MIPMAP_FILTER_NEAREST] = MTLSamplerMipFilterNearest,
    [NGPU_MIPMAP_FILTER_LINEAR]  = MTLSamplerMipFilterLinear,
};

static const MTLSamplerAddressMode wrap_map[NGPU_NB_WRAP] = {
    [NGPU_WRAP_CLAMP_TO_EDGE]   = MTLSamplerAddressModeClampToEdge,
    [NGPU_WRAP_MIRRORED_REPEAT] = MTLSamplerAddressModeMirrorRepeat,
    [NGPU_WRAP_REPEAT]          = MTLSamplerAddressModeRepeat,
};

static MTLTextureType get_texture_type(const struct ngpu_texture_params *params)
{
    switch (params->type) {
    case NGPU_TEXTURE_TYPE_2D:
        return params->samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
    case NGPU_TEXTURE_TYPE_2D_ARRAY:
        return params->samples > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray;
    case NGPU_TEXTURE_TYPE_3D:
        return MTLTextureType3D;
    case NGPU_TEXTURE_TYPE_CUBE:
        return MTLTextureTypeCube;
    default:
        ngli_assert(0);
    }
}

static MTLTextureUsage get_texture_usage(uint32_t usage)
{
    MTLTextureUsage mtl_usage = 0;
    if (usage & NGPU_TEXTURE_USAGE_SAMPLED_BIT)
        mtl_usage |= MTLTextureUsageShaderRead;
    if (usage & NGPU_TEXTURE_USAGE_STORAGE_BIT)
        mtl_usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    if (usage & (NGPU_TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | NGPU_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
        mtl_usage |= MTLTextureUsageRenderTarget;
    return mtl_usage;
}

static uint32_t get_mipmap_levels(uint32_t width, uint32_t height)
{
    return ngli_log2(width | height | 1);
}

static void init_fields(struct ngpu_texture *s, const struct ngpu_texture_params *params)
{
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)s;

    s->params = *params;
    ngli_assert(params->width && params->height);

    s->params.depth = params->type == NGPU_TEXTURE_TYPE_3D ? params->depth : 1;

    s_priv->array_layers = 1;
    if (params->type == NGPU_TEXTURE_TYPE_CUBE)
        s_priv->array_layers = 6;
    else if (params->type == NGPU_TEXTURE_TYPE_2D_ARRAY)
        s_priv->array_layers = params->depth;

    s_priv->mipmap_levels = 1;
    if (params->mipmap_filter != NGPU_MIPMAP_FILTER_NONE)
        s_priv->mipmap_levels = get_mipmap_levels(params->width, params->height);

    s_priv->bytes_per_pixel = ngpu_format_get_bytes_per_pixel(params->format);
}

static int create_sampler(struct ngpu_texture *s)
{
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)s->gpu_ctx;
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)s;
    const struct ngpu_texture_params *params = &s->params;

    if (!(params->usage & NGPU_TEXTURE_USAGE_SAMPLED_BIT))
        return 0;

    MTLSamplerDescriptor *desc = [[MTLSamplerDescriptor alloc] init];
    desc.minFilter    = filter_map[params->min_filter];
    desc.magFilter    = filter_map[params->mag_filter];
    desc.mipFilter    = mipmap_filter_map[params->mipmap_filter];
    desc.sAddressMode = wrap_map[params->wrap_s];
    desc.tAddressMode = wrap_map[params->wrap_t];
    desc.rAddressMode = wrap_map[params->wrap_r];
    s_priv->sampler = [gpu_ctx_mtl->device newSamplerStateWithDescriptor:desc];
    [desc release];

    return s_priv->sampler ? 0 : NGL_ERROR_GRAPHICS_GENERIC;
}

struct ngpu_texture *ngpu_texture_mtl_create(struct ngpu_ctx *gpu_ctx)
{
    struct ngpu_texture_mtl *s = ngli_calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->parent.gpu_ctx = gpu_ctx;
    return (struct ngpu_texture *)s;
}

int ngpu_texture_mtl_init(struct ngpu_texture *s, const struct ngpu_texture_params *params)
{
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)s->gpu_ctx;
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)s;

    init_fields(s, params);

    const MTLPixelFormat format = ngpu_format_ngl_to_mtl(params->format);
    if (format == MTLPixelFormatInvalid) {
        LOG(ERROR, "unsupported texture format %d", params->format);
        return NGL_ERROR_GRAPHICS_UNSUPPORTED;
    }

    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType      = get_texture_type(&s->params);
    desc.pixelFormat      = format;
    desc.width            = params->width;
    desc.height           = params->height;
    desc.depth            = s->params.depth;
    desc.arrayLength      = params->type == NGPU_TEXTURE_TYPE_2D_ARRAY ? s_priv->array_layers : 1;
    desc.mipmapLevelCount = s_priv->mipmap_levels;
    desc.sampleCount      = NGLI_MAX(params->samples, 1);
    desc.usage            = get_texture_usage(params->usage);
    desc.storageMode      = MTLStorageModePrivate;
    s_priv->texture = [gpu_ctx_mtl->device newTextureWithDescriptor:desc];
    [desc release];

    if (!s_priv->texture) {
        LOG(ERROR, "unable to create %ux%ux%u texture", params->width, params->height, s->params.depth);
        return NGL_ERROR_GRAPHICS_MEMORY;
    }

    return create_sampler(s);
}

int ngpu_texture_mtl_wrap(struct ngpu_texture *s, const struct ngpu_texture_params *params, id<MTLTexture> texture)
{
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)s;

    /* A wrapper may be re-wrapped when the wrapped texture changes size */
    [s_priv->texture release];
    [s_priv->sampler release];
    s_priv->sampler = nil;

    init_fields(s, params);
    s_priv->wrapped = 1;
    s_priv->texture = [texture retain];

    return create_sampler(s);
}

void ngpu_texture_mtl_set_texture(struct ngpu_texture *s, id<MTLTexture> texture)
{
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)s;
    ngli_assert(s_priv->wrapped);
    [texture retain];
    [s_priv->texture release];
    s_priv->texture = texture;
}

int ngpu_texture_mtl_upload(struct ngpu_texture *s, const uint8_t *data, uint32_t linesize)
{
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)s;
    const struct ngpu_texture_params *params = &s->params;
    const struct ngpu_texture_transfer_params transfer_params = {
        .width          = params->width,
        .height         = params->height,
        .depth          = params->depth,
        .base_layer     = 0,
        .layer_count    = s_priv->array_layers,
        .pixels_per_row = linesize ? linesize : params->width,
    };
    return ngpu_texture_mtl_upload_with_params(s, data, &transfer_params);
}

int ngpu_texture_mtl_upload_with_params(struct ngpu_texture *s, const uint8_t *data,
                                        const struct ngpu_texture_transfer_params *transfer_params)
{
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)s->gpu_ctx;
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)s;
    const struct ngpu_texture_params *params = &s->params;

    ngli_assert(!s_priv->wrapped);
    ngli_assert(params->usage & NGPU_TEXTURE_USAGE_TRANSFER_DST_BIT);

    if (!data)
        return 0;

    const size_t bytes_per_row = (size_t)transfer_params->pixels_per_row * s_priv->bytes_per_pixel;
    const size_t bytes_per_image = bytes_per_row * transfer_params->height;
    const size_t layer_size = bytes_per_image * transfer_params->depth;
    const size_t size = layer_size * transfer_params->layer_count;

    @autoreleasepool {
        /* A fresh staging buffer per upload: the command buffer retains it
         * until the copy has executed, so no synchronization is needed */
        id<MTLBuffer> staging = [gpu_ctx_mtl->device newBufferWithBytes:data
                                                                  length:size
                                                                 options:MTLResourceStorageModeShared];
        if (!staging)
            return NGL_ERROR_GRAPHICS_MEMORY;

        id<MTLBlitCommandEncoder> blit = ngpu_ctx_mtl_begin_blit(s->gpu_ctx);
        for (uint32_t i = transfer_params->base_layer; i < transfer_params->layer_count; i++) {
            [blit copyFromBuffer:staging
                    sourceOffset:i * layer_size
               sourceBytesPerRow:bytes_per_row
             sourceBytesPerImage:params->type == NGPU_TEXTURE_TYPE_3D ? bytes_per_image : 0
                      sourceSize:MTLSizeMake(transfer_params->width, transfer_params->height, transfer_params->depth)
                       toTexture:s_priv->texture
                destinationSlice:i
                destinationLevel:0
               destinationOrigin:MTLOriginMake(transfer_params->x, transfer_params->y, transfer_params->z)];
        }
        if (s_priv->mipmap_levels > 1)
            [blit generateMipmapsForTexture:s_priv->texture];
        ngpu_ctx_mtl_end_blit(s->gpu_ctx, blit);

        [staging release];
    }

    return 0;
}

int ngpu_texture_mtl_generate_mipmap(struct ngpu_texture *s)
{
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)s;

    if (s_priv->mipmap_levels <= 1)
        return 0;

    @autoreleasepool {
        id<MTLBlitCommandEncoder> blit = ngpu_ctx_mtl_begin_blit(s->gpu_ctx);
        [blit generateMipmapsForTexture:s_priv->texture];
        ngpu_ctx_mtl_end_blit(s->gpu_ctx, blit);
    }

    return 0;
}

void ngpu_texture_mtl_copy_to_buffer(struct ngpu_texture *s, id<MTLBuffer> buffer)
{
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)s;
    const struct ngpu_texture_params *params = &s->params;

    @autoreleasepool {
        id<MTLBlitCommandEncoder> blit = ngpu_ctx_mtl_begin_blit(s->gpu_ctx);
        [blit copyFromTexture:s_priv->texture
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(params->width, params->height, 1)
                     toBuffer:buffer
            destinationOffset:0
       destinationBytesPerRow:params->width * s_priv->bytes_per_pixel
     destinationBytesPerImage:params->width * params->height * s_priv->bytes_per_pixel];
        ngpu_ctx_mtl_end_blit(s->gpu_ctx, blit);
    }
}

void ngpu_texture_mtl_freep(struct ngpu_texture **sp)
{
    if (!*sp)
        return;
    struct ngpu_texture_mtl *s_priv = (struct ngpu_texture_mtl *)*sp;
    [s_priv->texture release];
    [s_priv->sampler release];
    ngli_freep(sp);
}
