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

#include <string.h>

#include "config.h"

#include "bindgroup_mtl.h"
#include "buffer_mtl.h"
#include "ctx_mtl.h"
#include "format_mtl.h"
#include "log.h"
#include "math_utils.h"
#include "ngpu/vulkan/glslang_utils.h"
#include "ngpu/vulkan/wsi_apple.h"
#include "nopegl.h"
#include "pipeline_mtl.h"
#include "program_mtl.h"
#include "rendertarget_mtl.h"
#include "texture_mtl.h"
#include "utils/memory.h"
#include "utils/utils.h"

#define COLOR_USAGE (NGPU_TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | NGPU_TEXTURE_USAGE_TRANSFER_SRC_BIT)
#define DEPTH_USAGE NGPU_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT

static const enum ngpu_format onscreen_color_format = NGPU_FORMAT_B8G8R8A8_UNORM;
static const enum ngpu_format offscreen_color_format = NGPU_FORMAT_R8G8B8A8_UNORM;
static const enum ngpu_format depth_format = NGPU_FORMAT_D32_SFLOAT;
static const enum ngpu_format depth_stencil_format = NGPU_FORMAT_D32_SFLOAT_S8_UINT;

static void check_cmd_buffer(id<MTLCommandBuffer> cmd_buffer)
{
    if (cmd_buffer.status == MTLCommandBufferStatusError)
        LOG(ERROR, "command buffer failed: %s", cmd_buffer.error.localizedDescription.UTF8String);
}

static void commit(struct ngpu_ctx_mtl *s, id<MTLCommandBuffer> cmd_buffer)
{
    [cmd_buffer commit];
    [cmd_buffer retain];
    [s->last_cmd_buffer release];
    s->last_cmd_buffer = cmd_buffer;
}

static void commit_and_wait(struct ngpu_ctx_mtl *s, id<MTLCommandBuffer> cmd_buffer)
{
    commit(s, cmd_buffer);
    [cmd_buffer waitUntilCompleted];
    check_cmd_buffer(cmd_buffer);
}

static id<MTLCommandBuffer> get_or_create_cmd_buffer(struct ngpu_ctx_mtl *s)
{
    if (s->cmd_buffer)
        return s->cmd_buffer;
    ngli_assert(!s->transient_cmd_buffer);
    s->transient_cmd_buffer = [[s->queue commandBuffer] retain];
    return s->transient_cmd_buffer;
}

static void execute_transient(struct ngpu_ctx_mtl *s)
{
    if (!s->transient_cmd_buffer)
        return;
    commit_and_wait(s, s->transient_cmd_buffer);
    [s->transient_cmd_buffer release];
    s->transient_cmd_buffer = nil;
}

id<MTLBlitCommandEncoder> ngpu_ctx_mtl_begin_blit(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    ngli_assert(!s_priv->render_encoder);
    return [get_or_create_cmd_buffer(s_priv) blitCommandEncoder];
}

void ngpu_ctx_mtl_end_blit(struct ngpu_ctx *s, id<MTLBlitCommandEncoder> encoder)
{
    [encoder endEncoding];
    execute_transient((struct ngpu_ctx_mtl *)s);
}

id<MTLComputeCommandEncoder> ngpu_ctx_mtl_begin_compute(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    ngli_assert(!s_priv->render_encoder);
    return [get_or_create_cmd_buffer(s_priv) computeCommandEncoder];
}

void ngpu_ctx_mtl_end_compute(struct ngpu_ctx *s, id<MTLComputeCommandEncoder> encoder)
{
    [encoder endEncoding];
    execute_transient((struct ngpu_ctx_mtl *)s);
}

void ngpu_ctx_mtl_wait_submitted(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    for (uint32_t i = 0; i < s->nb_in_flight_frames; i++)
        [s_priv->frame_cmd_buffers[i] waitUntilCompleted];
    [s_priv->last_cmd_buffer waitUntilCompleted];
}

static int create_texture(struct ngpu_ctx *s, enum ngpu_format format, uint32_t samples, uint32_t usage,
                          struct ngpu_texture **texturep)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;

    struct ngpu_texture *texture = ngpu_texture_create(s);
    if (!texture)
        return NGL_ERROR_MEMORY;

    const struct ngpu_texture_params params = {
        .type    = NGPU_TEXTURE_TYPE_2D,
        .format  = format,
        .width   = s_priv->width,
        .height  = s_priv->height,
        .samples = samples,
        .usage   = usage,
    };

    int ret = ngpu_texture_init(texture, &params);
    if (ret < 0) {
        ngpu_texture_freep(&texture);
        return ret;
    }

    *texturep = texture;
    return 0;
}

static int create_dummy_texture(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;

    s_priv->dummy_texture = ngpu_texture_create(s);
    if (!s_priv->dummy_texture)
        return NGL_ERROR_MEMORY;

    const struct ngpu_texture_params params = {
        .type   = NGPU_TEXTURE_TYPE_2D,
        .format = NGPU_FORMAT_R8G8B8A8_UNORM,
        .width  = 1,
        .height = 1,
        .usage  = NGPU_TEXTURE_USAGE_SAMPLED_BIT |
                  NGPU_TEXTURE_USAGE_STORAGE_BIT |
                  NGPU_TEXTURE_USAGE_TRANSFER_DST_BIT,
    };

    int ret = ngpu_texture_init(s_priv->dummy_texture, &params);
    if (ret < 0)
        return ret;

    const uint8_t buf[4] = {0};
    return ngpu_texture_upload(s_priv->dummy_texture, buf, 0);
}

static int create_rendertarget(struct ngpu_ctx *s, struct ngpu_texture *color, struct ngpu_texture *resolve_color,
                               struct ngpu_texture *depth_stencil, enum ngpu_load_op load_op,
                               struct ngpu_rendertarget **rendertargetp)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    const struct ngl_config *config = &s->config;

    struct ngpu_rendertarget *rendertarget = ngpu_rendertarget_create(s);
    if (!rendertarget)
        return NGL_ERROR_MEMORY;

    const struct ngpu_rendertarget_params params = {
        .width     = s_priv->width,
        .height    = s_priv->height,
        .nb_colors = 1,
        .colors[0] = {
            .attachment     = color,
            .resolve_target = resolve_color,
            .load_op        = load_op,
            .clear_value[0] = config->clear_color[0],
            .clear_value[1] = config->clear_color[1],
            .clear_value[2] = config->clear_color[2],
            .clear_value[3] = config->clear_color[3],
            .store_op       = NGPU_STORE_OP_STORE,
        },
        .depth_stencil = {
            .attachment = depth_stencil,
            .load_op    = load_op,
            .store_op   = NGPU_STORE_OP_STORE,
        },
    };

    int ret = ngpu_rendertarget_init(rendertarget, &params);
    if (ret < 0) {
        ngpu_rendertarget_freep(&rendertarget);
        return ret;
    }

    *rendertargetp = rendertarget;
    return 0;
}

static int push_texture(struct darray *array, struct ngpu_texture *texture)
{
    if (!ngli_darray_push(array, &texture)) {
        ngpu_texture_freep(&texture);
        return NGL_ERROR_MEMORY;
    }
    return 0;
}

static int push_rendertarget(struct darray *array, struct ngpu_rendertarget *rt)
{
    if (!ngli_darray_push(array, &rt)) {
        ngpu_rendertarget_freep(&rt);
        return NGL_ERROR_MEMORY;
    }
    return 0;
}

static int create_render_resources(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    const struct ngl_config *config = &s->config;

    /* Offscreen: one color target per in-flight frame. Onscreen: a single
     * wrapper whose texture is the drawable of the current frame. */
    const enum ngpu_format color_format = config->offscreen ? offscreen_color_format : onscreen_color_format;
    const uint32_t nb_images = config->offscreen ? s->nb_in_flight_frames : 1;
    for (uint32_t i = 0; i < nb_images; i++) {
        struct ngpu_texture *color = NULL;
        if (config->offscreen) {
            int ret = create_texture(s, color_format, 0, COLOR_USAGE, &color);
            if (ret < 0)
                return ret;
        } else {
            color = ngpu_texture_create(s);
            if (!color)
                return NGL_ERROR_MEMORY;
            const struct ngpu_texture_params params = {
                .type   = NGPU_TEXTURE_TYPE_2D,
                .format = color_format,
                .width  = s_priv->width,
                .height = s_priv->height,
                .usage  = COLOR_USAGE,
            };
            int ret = ngpu_texture_mtl_wrap(color, &params, nil);
            if (ret < 0) {
                ngpu_texture_freep(&color);
                return ret;
            }
        }
        int ret = push_texture(&s_priv->colors, color);
        if (ret < 0)
            return ret;

        struct ngpu_texture *depth_stencil = NULL;
        ret = create_texture(s, depth_stencil_format, config->samples, DEPTH_USAGE, &depth_stencil);
        if (ret < 0)
            return ret;
        ret = push_texture(&s_priv->depth_stencils, depth_stencil);
        if (ret < 0)
            return ret;

        struct ngpu_texture *ms_color = NULL;
        if (config->samples) {
            ret = create_texture(s, color_format, config->samples, COLOR_USAGE, &ms_color);
            if (ret < 0)
                return ret;
            ret = push_texture(&s_priv->ms_colors, ms_color);
            if (ret < 0)
                return ret;
        }

        struct ngpu_texture *target_color = ms_color ? ms_color : color;
        struct ngpu_texture *resolve_color = ms_color ? color : NULL;

        struct ngpu_rendertarget *rt = NULL;
        ret = create_rendertarget(s, target_color, resolve_color, depth_stencil, NGPU_LOAD_OP_CLEAR, &rt);
        if (ret < 0)
            return ret;
        ret = push_rendertarget(&s_priv->rts, rt);
        if (ret < 0)
            return ret;

        struct ngpu_rendertarget *rt_load = NULL;
        ret = create_rendertarget(s, target_color, resolve_color, depth_stencil, NGPU_LOAD_OP_LOAD, &rt_load);
        if (ret < 0)
            return ret;
        ret = push_rendertarget(&s_priv->rts_load, rt_load);
        if (ret < 0)
            return ret;
    }

    if (config->offscreen) {
        s_priv->capture_buffer_size = (size_t)s_priv->width * s_priv->height * ngpu_format_get_bytes_per_pixel(color_format);
        s_priv->capture_buffer = [s_priv->device newBufferWithLength:s_priv->capture_buffer_size
                                                             options:MTLResourceStorageModeShared];
        if (!s_priv->capture_buffer)
            return NGL_ERROR_GRAPHICS_MEMORY;
    }

    s_priv->default_rt_layout = (struct ngpu_rendertarget_layout) {
        .samples   = config->samples,
        .nb_colors = 1,
        .colors[0] = {
            .format  = color_format,
            .resolve = config->samples > 0,
        },
        .depth_stencil = {
            .format = depth_stencil_format,
        },
    };

    return 0;
}

static void destroy_render_resources(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;

    ngli_darray_clear(&s_priv->rts);
    ngli_darray_clear(&s_priv->rts_load);
    ngli_darray_clear(&s_priv->colors);
    ngli_darray_clear(&s_priv->ms_colors);
    ngli_darray_clear(&s_priv->depth_stencils);
    s_priv->default_rt = NULL;
    s_priv->default_rt_load = NULL;

    [s_priv->capture_buffer release];
    s_priv->capture_buffer = nil;
}

static void free_texture(void *user_arg, void *data)
{
    ngpu_texture_freep(data);
}

static void free_rendertarget(void *user_arg, void *data)
{
    ngpu_rendertarget_freep(data);
}

static struct ngpu_ctx *mtl_create(const struct ngl_config *config)
{
    struct ngpu_ctx_mtl *s = ngli_calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    return (struct ngpu_ctx *)s;
}

static uint32_t get_max_samples(id<MTLDevice> device)
{
    static const uint32_t samples[] = {8, 4, 2};
    for (size_t i = 0; i < NGLI_ARRAY_NB(samples); i++)
        if ([device supportsTextureSampleCount:samples[i]])
            return samples[i];
    return 1;
}

static void init_limits(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    id<MTLDevice> device = s_priv->device;
    struct ngpu_limits *limits = &s->limits;

    /* Values from the Metal feature set tables for Apple7+ GPUs */
    const MTLSize max_threads = device.maxThreadsPerThreadgroup;
    limits->max_vertex_attributes              = 31;
    limits->max_texture_image_units            = 31;
    limits->max_image_units                    = 8;
    limits->max_compute_work_group_count[0]    = 65535;
    limits->max_compute_work_group_count[1]    = 65535;
    limits->max_compute_work_group_count[2]    = 65535;
    limits->max_compute_work_group_invocations = (uint32_t)max_threads.width;
    limits->max_compute_work_group_size[0]     = (uint32_t)max_threads.width;
    limits->max_compute_work_group_size[1]     = (uint32_t)max_threads.height;
    limits->max_compute_work_group_size[2]     = (uint32_t)max_threads.depth;
    limits->max_compute_shared_memory_size     = (uint32_t)device.maxThreadgroupMemoryLength;
    limits->max_uniform_block_size             = 65536;
    limits->min_uniform_block_offset_alignment = 256;
    limits->max_storage_block_size             = 1U << 30;
    limits->min_storage_block_offset_alignment = 256;
    limits->max_samples                        = get_max_samples(device);
    limits->max_texture_dimension_1d           = 16384;
    limits->max_texture_dimension_2d           = 16384;
    limits->max_texture_dimension_3d           = 2048;
    limits->max_texture_dimension_cube         = 16384;
    limits->max_texture_array_layers           = 2048;
    limits->max_color_attachments              = NGPU_MAX_COLOR_ATTACHMENTS;
    limits->max_draw_buffers                   = NGPU_MAX_COLOR_ATTACHMENTS;
}

static int mtl_init(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    const struct ngl_config *config = &s->config;

    if (config->offscreen) {
        if (config->width <= 0 || config->height <= 0) {
            LOG(ERROR, "could not create offscreen context with invalid dimensions (%dx%d)",
                config->width, config->height);
            return NGL_ERROR_INVALID_ARG;
        }
        if (config->capture_buffer_type != NGL_CAPTURE_BUFFER_TYPE_CPU) {
            LOG(ERROR, "unsupported capture buffer type");
            return NGL_ERROR_UNSUPPORTED;
        }
    } else if (config->capture_buffer) {
        LOG(ERROR, "capture_buffer is not supported by onscreen context");
        return NGL_ERROR_INVALID_ARG;
    }

    ngli_darray_init(&s_priv->colors, sizeof(struct ngpu_texture *), 0);
    ngli_darray_init(&s_priv->ms_colors, sizeof(struct ngpu_texture *), 0);
    ngli_darray_init(&s_priv->depth_stencils, sizeof(struct ngpu_texture *), 0);
    ngli_darray_init(&s_priv->rts, sizeof(struct ngpu_rendertarget *), 0);
    ngli_darray_init(&s_priv->rts_load, sizeof(struct ngpu_rendertarget *), 0);
    ngli_darray_set_free_func(&s_priv->colors, free_texture, NULL);
    ngli_darray_set_free_func(&s_priv->ms_colors, free_texture, NULL);
    ngli_darray_set_free_func(&s_priv->depth_stencils, free_texture, NULL);
    ngli_darray_set_free_func(&s_priv->rts, free_rendertarget, NULL);
    ngli_darray_set_free_func(&s_priv->rts_load, free_rendertarget, NULL);

    @autoreleasepool {
        s_priv->device = MTLCreateSystemDefaultDevice();
        if (!s_priv->device) {
            LOG(ERROR, "no Metal device available");
            return NGL_ERROR_GRAPHICS_UNSUPPORTED;
        }
        s_priv->queue = [s_priv->device newCommandQueue];
        if (!s_priv->queue)
            return NGL_ERROR_GRAPHICS_GENERIC;

        s->version = 300;
        s->language_version = 450;
        s->features = NGPU_FEATURE_COMPUTE |
                      NGPU_FEATURE_IMAGE_LOAD_STORE |
                      NGPU_FEATURE_STORAGE_BUFFER |
                      NGPU_FEATURE_BUFFER_MAP_PERSISTENT |
                      NGPU_FEATURE_DEPTH_STENCIL_RESOLVE;
        init_limits(s);

        s->nb_in_flight_frames = 2;
        s_priv->frame_cmd_buffers = ngli_calloc(s->nb_in_flight_frames, sizeof(*s_priv->frame_cmd_buffers));
        if (!s_priv->frame_cmd_buffers)
            return NGL_ERROR_MEMORY;

        int ret = ngli_glslang_init();
        if (ret < 0)
            return ret;

        s_priv->width = config->width;
        s_priv->height = config->height;

        if (!config->offscreen) {
            s_priv->layer = [ngpu_window_get_metal_layer((const void *)config->window) retain];
            if (!s_priv->layer)
                return NGL_ERROR_INVALID_ARG;
            if (!s_priv->width || !s_priv->height) {
                const CGSize size = s_priv->layer.drawableSize;
                s_priv->width = (uint32_t)size.width;
                s_priv->height = (uint32_t)size.height;
            }
            s_priv->layer.device = s_priv->device;
            s_priv->layer.pixelFormat = ngpu_format_ngl_to_mtl(onscreen_color_format);
            s_priv->layer.framebufferOnly = NO;
            s_priv->layer.drawableSize = CGSizeMake(s_priv->width, s_priv->height);
        }

        ret = create_dummy_texture(s);
        if (ret < 0)
            return ret;

        ret = create_render_resources(s);
        if (ret < 0)
            return ret;
    }

    return 0;
}

static int mtl_resize(struct ngpu_ctx *s, uint32_t width, uint32_t height)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    const struct ngl_config *config = &s->config;

    if (config->offscreen) {
        LOG(ERROR, "resize operation is not supported by offscreen context");
        return NGL_ERROR_UNSUPPORTED;
    }

    ngpu_ctx_mtl_wait_submitted(s);
    destroy_render_resources(s);
    s_priv->width = width;
    s_priv->height = height;
    s_priv->layer.drawableSize = CGSizeMake(width, height);
    @autoreleasepool {
        return create_render_resources(s);
    }
}

static int mtl_set_capture_buffer(struct ngpu_ctx *s, void *capture_buffer)
{
    struct ngl_config *config = &s->config;

    if (!config->offscreen) {
        LOG(ERROR, "capture_buffer is not supported by onscreen context");
        return NGL_ERROR_UNSUPPORTED;
    }

    config->capture_buffer = capture_buffer;
    return 0;
}

/* Limit the number of frames in flight */
static void wait_frame(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    id<MTLCommandBuffer> cmd_buffer = s_priv->frame_cmd_buffers[s->current_frame_index];
    [cmd_buffer waitUntilCompleted];
    check_cmd_buffer(cmd_buffer);
}

static int mtl_begin_update(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;

    wait_frame(s);
    ngli_assert(!s_priv->cmd_buffer);
    s_priv->cmd_buffer = [[s_priv->queue commandBuffer] retain];
    return 0;
}

static int mtl_end_update(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;

    commit(s_priv, s_priv->cmd_buffer);
    [s_priv->cmd_buffer release];
    s_priv->cmd_buffer = nil;
    return 0;
}

static int mtl_begin_draw(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    const struct ngl_config *config = &s->config;

    wait_frame(s);
    ngli_assert(!s_priv->cmd_buffer);
    s_priv->cmd_buffer = [[s_priv->queue commandBuffer] retain];

    if (config->offscreen) {
        struct ngpu_rendertarget **rts = ngli_darray_data(&s_priv->rts);
        struct ngpu_rendertarget **rts_load = ngli_darray_data(&s_priv->rts_load);
        s_priv->default_rt = rts[s->current_frame_index];
        s_priv->default_rt_load = rts_load[s->current_frame_index];
    } else {
        @autoreleasepool {
            s_priv->drawable = [[s_priv->layer nextDrawable] retain];
        }
        if (!s_priv->drawable) {
            LOG(ERROR, "unable to acquire a drawable");
            return NGL_ERROR_GRAPHICS_GENERIC;
        }
        struct ngpu_texture **colors = ngli_darray_data(&s_priv->colors);
        ngpu_texture_mtl_set_texture(colors[0], s_priv->drawable.texture);

        struct ngpu_rendertarget **rts = ngli_darray_data(&s_priv->rts);
        struct ngpu_rendertarget **rts_load = ngli_darray_data(&s_priv->rts_load);
        s_priv->default_rt = rts[0];
        s_priv->default_rt_load = rts_load[0];
    }

    return 0;
}

static int mtl_query_draw_time(struct ngpu_ctx *s, int64_t *time)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    const struct ngl_config *config = &s->config;

    if (!config->hud)
        return NGL_ERROR_INVALID_USAGE;

    id<MTLCommandBuffer> cmd_buffer = s_priv->cmd_buffer;
    commit_and_wait(s_priv, cmd_buffer);
    *time = (int64_t)((cmd_buffer.GPUEndTime - cmd_buffer.GPUStartTime) * 1e9);
    [cmd_buffer release];
    s_priv->cmd_buffer = [[s_priv->queue commandBuffer] retain];

    return 0;
}

static int mtl_end_draw(struct ngpu_ctx *s, double t)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    const struct ngl_config *config = &s->config;
    id<MTLCommandBuffer> cmd_buffer = s_priv->cmd_buffer;

    if (config->offscreen) {
        if (config->capture_buffer) {
            struct ngpu_texture **colors = ngli_darray_data(&s_priv->colors);
            ngpu_texture_mtl_copy_to_buffer(colors[s->current_frame_index], s_priv->capture_buffer);
            commit_and_wait(s_priv, cmd_buffer);
            memcpy(config->capture_buffer, s_priv->capture_buffer.contents, s_priv->capture_buffer_size);
        } else {
            commit(s_priv, cmd_buffer);
        }
    } else {
        [cmd_buffer presentDrawable:s_priv->drawable];
        commit(s_priv, cmd_buffer);
        struct ngpu_texture **colors = ngli_darray_data(&s_priv->colors);
        ngpu_texture_mtl_set_texture(colors[0], nil);
        [s_priv->drawable release];
        s_priv->drawable = nil;
    }

    [s_priv->frame_cmd_buffers[s->current_frame_index] release];
    s_priv->frame_cmd_buffers[s->current_frame_index] = cmd_buffer;
    s_priv->cmd_buffer = nil;

    return 0;
}

static void mtl_wait_idle(struct ngpu_ctx *s)
{
    ngpu_ctx_mtl_wait_submitted(s);
}

static void mtl_destroy(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;

    if (!s_priv->device)
        return;

    if (s_priv->frame_cmd_buffers) {
        ngpu_ctx_mtl_wait_submitted(s);
        for (uint32_t i = 0; i < s->nb_in_flight_frames; i++)
            [s_priv->frame_cmd_buffers[i] release];
        ngli_freep(&s_priv->frame_cmd_buffers);
    }

    destroy_render_resources(s);
    ngli_darray_reset(&s_priv->rts);
    ngli_darray_reset(&s_priv->rts_load);
    ngli_darray_reset(&s_priv->colors);
    ngli_darray_reset(&s_priv->ms_colors);
    ngli_darray_reset(&s_priv->depth_stencils);
    ngpu_texture_freep(&s_priv->dummy_texture);

    [s_priv->cmd_buffer release];
    [s_priv->last_cmd_buffer release];
    [s_priv->drawable release];
    [s_priv->layer release];
    [s_priv->queue release];
    [s_priv->device release];
    s_priv->device = nil;

    ngli_glslang_uninit();
}

/*
 * The vertex shaders flip their output Y axis (see shader_mtl.c), so the
 * projection follows the Vulkan conventions. Unlike Vulkan, Metal determines
 * the winding after that flip, so the cull mode is left untouched.
 */
static enum ngpu_cull_mode mtl_transform_cull_mode(struct ngpu_ctx *s, enum ngpu_cull_mode cull_mode)
{
    return cull_mode;
}

static void mtl_transform_projection_matrix(struct ngpu_ctx *s, float *dst)
{
    static const NGLI_ALIGNED_MAT(matrix) = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f,-1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 0.0f,
        0.0f, 0.0f, 0.5f, 1.0f,
    };
    ngli_mat4_mul(dst, matrix, dst);
}

static void mtl_get_rendertarget_uvcoord_matrix(struct ngpu_ctx *s, float *dst)
{
    static const NGLI_ALIGNED_MAT(matrix) = NGLI_MAT4_IDENTITY;
    memcpy(dst, matrix, 4 * 4 * sizeof(float));
}

static struct ngpu_rendertarget *mtl_get_default_rendertarget(struct ngpu_ctx *s, enum ngpu_load_op load_op)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    switch (load_op) {
    case NGPU_LOAD_OP_DONT_CARE:
    case NGPU_LOAD_OP_CLEAR:
        return s_priv->default_rt;
    case NGPU_LOAD_OP_LOAD:
        return s_priv->default_rt_load;
    default:
        ngli_assert(0);
    }
}

static const struct ngpu_rendertarget_layout *mtl_get_default_rendertarget_layout(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    return &s_priv->default_rt_layout;
}

static void mtl_get_default_rendertarget_size(struct ngpu_ctx *s, uint32_t *width, uint32_t *height)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    *width = s_priv->width;
    *height = s_priv->height;
}

static void mtl_begin_render_pass(struct ngpu_ctx *s, struct ngpu_rendertarget *rt)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    ngli_assert(!s_priv->render_encoder);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buffer = get_or_create_cmd_buffer(s_priv);
        MTLRenderPassDescriptor *desc = ngpu_rendertarget_mtl_get_descriptor(rt);
        s_priv->render_encoder = [[cmd_buffer renderCommandEncoderWithDescriptor:desc] retain];
    }

    const MTLViewport viewport = {0.0, 0.0, rt->width, rt->height, 0.0, 1.0};
    [s_priv->render_encoder setViewport:viewport];
    const MTLScissorRect scissor = {0, 0, rt->width, rt->height};
    [s_priv->render_encoder setScissorRect:scissor];
}

static void mtl_end_render_pass(struct ngpu_ctx *s)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;

    [s_priv->render_encoder endEncoding];
    [s_priv->render_encoder release];
    s_priv->render_encoder = nil;

    execute_transient(s_priv);
}

static void mtl_set_viewport(struct ngpu_ctx *s, const struct ngpu_viewport *viewport)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    const MTLViewport vp = {viewport->x, viewport->y, viewport->width, viewport->height, 0.0, 1.0};
    [s_priv->render_encoder setViewport:vp];
}

static void mtl_set_scissor(struct ngpu_ctx *s, const struct ngpu_scissor *scissor)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    const struct ngpu_rendertarget *rt = s->rendertarget;

    /* nope.gl scissors have a bottom-left origin; Metal requires the rectangle
     * to lie within the render target */
    const int64_t x0 = NGLI_MIN((int64_t)scissor->x, (int64_t)rt->width);
    const int64_t y0 = NGLI_MAX((int64_t)rt->height - scissor->y - scissor->height, 0);
    const int64_t x1 = NGLI_MIN((int64_t)scissor->x + scissor->width, (int64_t)rt->width);
    const int64_t y1 = NGLI_MIN(NGLI_MAX((int64_t)rt->height - scissor->y, 0), (int64_t)rt->height);
    const MTLScissorRect rect = {
        .x      = (NSUInteger)x0,
        .y      = (NSUInteger)NGLI_MIN(y0, (int64_t)rt->height),
        .width  = (NSUInteger)NGLI_MAX(x1 - x0, 0),
        .height = (NSUInteger)NGLI_MAX(y1 - NGLI_MIN(y0, (int64_t)rt->height), 0),
    };
    [s_priv->render_encoder setScissorRect:rect];
}

static enum ngpu_format mtl_get_preferred_depth_format(struct ngpu_ctx *s)
{
    return depth_format;
}

static enum ngpu_format mtl_get_preferred_depth_stencil_format(struct ngpu_ctx *s)
{
    return depth_stencil_format;
}

static uint32_t mtl_get_format_features(struct ngpu_ctx *s, enum ngpu_format format)
{
    struct ngpu_ctx_mtl *s_priv = (struct ngpu_ctx_mtl *)s;
    return ngpu_format_mtl_get_features(s_priv->device, format);
}

static void mtl_generate_texture_mipmap(struct ngpu_ctx *s, struct ngpu_texture *texture)
{
    ngpu_texture_generate_mipmap(texture);
}

static void mtl_set_bindgroup(struct ngpu_ctx *s, struct ngpu_bindgroup *bindgroup, const uint32_t *offsets, size_t nb_offsets)
{
}

static void mtl_set_pipeline(struct ngpu_ctx *s, struct ngpu_pipeline *pipeline)
{
}

static void mtl_draw(struct ngpu_ctx *s, uint32_t nb_vertices, uint32_t nb_instances, uint32_t first_vertex)
{
    ngpu_pipeline_mtl_draw(s->pipeline, nb_vertices, nb_instances, first_vertex);
}

static void mtl_draw_indexed(struct ngpu_ctx *s, uint32_t nb_indices, uint32_t nb_instances)
{
    ngpu_pipeline_mtl_draw_indexed(s->pipeline, nb_indices, nb_instances);
}

static void mtl_dispatch(struct ngpu_ctx *s, uint32_t nb_group_x, uint32_t nb_group_y, uint32_t nb_group_z)
{
    ngpu_pipeline_mtl_dispatch(s->pipeline, nb_group_x, nb_group_y, nb_group_z);
}

static void mtl_set_vertex_buffer(struct ngpu_ctx *s, uint32_t index, const struct ngpu_buffer *buffer)
{
}

static void mtl_set_index_buffer(struct ngpu_ctx *s, const struct ngpu_buffer *buffer, enum ngpu_format format)
{
}

const struct ngpu_ctx_class ngpu_ctx_mtl = {
    .id                                 = NGL_BACKEND_METAL,
    .create                             = mtl_create,
    .init                               = mtl_init,
    .resize                             = mtl_resize,
    .set_capture_buffer                 = mtl_set_capture_buffer,
    .begin_update                       = mtl_begin_update,
    .end_update                         = mtl_end_update,
    .begin_draw                         = mtl_begin_draw,
    .end_draw                           = mtl_end_draw,
    .query_draw_time                    = mtl_query_draw_time,
    .wait_idle                          = mtl_wait_idle,
    .destroy                            = mtl_destroy,

    .transform_cull_mode                = mtl_transform_cull_mode,
    .transform_projection_matrix        = mtl_transform_projection_matrix,
    .get_rendertarget_uvcoord_matrix    = mtl_get_rendertarget_uvcoord_matrix,

    .get_default_rendertarget           = mtl_get_default_rendertarget,
    .get_default_rendertarget_layout    = mtl_get_default_rendertarget_layout,
    .get_default_rendertarget_size      = mtl_get_default_rendertarget_size,

    .begin_render_pass                  = mtl_begin_render_pass,
    .end_render_pass                    = mtl_end_render_pass,

    .set_viewport                       = mtl_set_viewport,
    .set_scissor                        = mtl_set_scissor,

    .get_preferred_depth_format         = mtl_get_preferred_depth_format,
    .get_preferred_depth_stencil_format = mtl_get_preferred_depth_stencil_format,
    .get_format_features                = mtl_get_format_features,

    .generate_texture_mipmap            = mtl_generate_texture_mipmap,

    .set_bindgroup                      = mtl_set_bindgroup,

    .set_pipeline                       = mtl_set_pipeline,
    .draw                               = mtl_draw,
    .draw_indexed                       = mtl_draw_indexed,
    .dispatch                           = mtl_dispatch,

    .set_vertex_buffer                  = mtl_set_vertex_buffer,
    .set_index_buffer                   = mtl_set_index_buffer,

    .buffer_create                      = ngpu_buffer_mtl_create,
    .buffer_init                        = ngpu_buffer_mtl_init,
    .buffer_wait                        = ngpu_buffer_mtl_wait,
    .buffer_upload                      = ngpu_buffer_mtl_upload,
    .buffer_map                         = ngpu_buffer_mtl_map,
    .buffer_unmap                       = ngpu_buffer_mtl_unmap,
    .buffer_freep                       = ngpu_buffer_mtl_freep,

    .bindgroup_layout_create            = ngpu_bindgroup_layout_mtl_create,
    .bindgroup_layout_init              = ngpu_bindgroup_layout_mtl_init,
    .bindgroup_layout_freep             = ngpu_bindgroup_layout_mtl_freep,

    .bindgroup_create                   = ngpu_bindgroup_mtl_create,
    .bindgroup_init                     = ngpu_bindgroup_mtl_init,
    .bindgroup_update_texture           = ngpu_bindgroup_mtl_update_texture,
    .bindgroup_update_buffer            = ngpu_bindgroup_mtl_update_buffer,
    .bindgroup_freep                    = ngpu_bindgroup_mtl_freep,

    .pipeline_create                    = ngpu_pipeline_mtl_create,
    .pipeline_init                      = ngpu_pipeline_mtl_init,
    .pipeline_freep                     = ngpu_pipeline_mtl_freep,

    .program_create                     = ngpu_program_mtl_create,
    .program_init                       = ngpu_program_mtl_init,
    .program_freep                      = ngpu_program_mtl_freep,

    .rendertarget_create                = ngpu_rendertarget_mtl_create,
    .rendertarget_init                  = ngpu_rendertarget_mtl_init,
    .rendertarget_freep                 = ngpu_rendertarget_mtl_freep,

    .texture_create                     = ngpu_texture_mtl_create,
    .texture_init                       = ngpu_texture_mtl_init,
    .texture_upload                     = ngpu_texture_mtl_upload,
    .texture_upload_with_params         = ngpu_texture_mtl_upload_with_params,
    .texture_generate_mipmap            = ngpu_texture_mtl_generate_mipmap,
    .texture_freep                      = ngpu_texture_mtl_freep,
};
