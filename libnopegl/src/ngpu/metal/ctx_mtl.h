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

#ifndef NGPU_CTX_MTL_H
#define NGPU_CTX_MTL_H

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "ngpu/ctx.h"
#include "utils/darray.h"

/*
 * Command submission model
 *
 * The current command buffer is the update or draw command buffer of the
 * frame. Outside of a frame (resource uploads at init time for example), a
 * transient command buffer is created, committed and waited for, like the
 * Vulkan backend transient command buffers. Command buffers execute in commit
 * order and Metal tracks hazards on the resources they use, so no explicit
 * synchronization is needed between them.
 *
 * At most one encoder is open at a time: the render encoder of the current
 * render pass, or a short lived blit/compute encoder outside of render passes.
 */
struct ngpu_ctx_mtl {
    struct ngpu_ctx parent;

    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    id<MTLCommandBuffer> cmd_buffer;         /* current update or draw command buffer */
    id<MTLCommandBuffer> transient_cmd_buffer;
    id<MTLCommandBuffer> *frame_cmd_buffers; /* last command buffer of each in-flight frame */
    id<MTLCommandBuffer> last_cmd_buffer;    /* last committed command buffer */
    id<MTLRenderCommandEncoder> render_encoder;

    CAMetalLayer *layer;
    id<CAMetalDrawable> drawable;

    uint32_t width;
    uint32_t height;

    struct darray colors;
    struct darray ms_colors;
    struct darray depth_stencils;
    struct darray rts;
    struct darray rts_load;
    id<MTLBuffer> capture_buffer;
    size_t capture_buffer_size;

    struct ngpu_rendertarget *default_rt;
    struct ngpu_rendertarget *default_rt_load;
    struct ngpu_rendertarget_layout default_rt_layout;

    /* Bound to unbound texture slots: the nope.gl API allows them */
    struct ngpu_texture *dummy_texture;
};

/* Encoders outside of render passes, on the current or a transient command buffer */
id<MTLBlitCommandEncoder> ngpu_ctx_mtl_begin_blit(struct ngpu_ctx *s);
void ngpu_ctx_mtl_end_blit(struct ngpu_ctx *s, id<MTLBlitCommandEncoder> encoder);
id<MTLComputeCommandEncoder> ngpu_ctx_mtl_begin_compute(struct ngpu_ctx *s);
void ngpu_ctx_mtl_end_compute(struct ngpu_ctx *s, id<MTLComputeCommandEncoder> encoder);

/* Wait for every committed command buffer to complete */
void ngpu_ctx_mtl_wait_submitted(struct ngpu_ctx *s);

#endif
