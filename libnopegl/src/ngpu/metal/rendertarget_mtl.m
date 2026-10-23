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

#include "rendertarget_mtl.h"
#include "texture_mtl.h"
#include "utils/memory.h"
#include "utils/utils.h"

static const MTLLoadAction load_action_map[] = {
    [NGPU_LOAD_OP_LOAD]      = MTLLoadActionLoad,
    [NGPU_LOAD_OP_CLEAR]     = MTLLoadActionClear,
    [NGPU_LOAD_OP_DONT_CARE] = MTLLoadActionDontCare,
};

static MTLStoreAction get_store_action(enum ngpu_store_op store_op, int resolve)
{
    if (resolve)
        return store_op == NGPU_STORE_OP_STORE ? MTLStoreActionStoreAndMultisampleResolve
                                               : MTLStoreActionMultisampleResolve;
    return store_op == NGPU_STORE_OP_STORE ? MTLStoreActionStore : MTLStoreActionDontCare;
}

/* Layer of a texture: a slice for arrays and cube maps, a depth plane for 3D textures */
static void set_attachment_texture(MTLRenderPassAttachmentDescriptor *desc, const struct ngpu_texture *texture, uint32_t layer)
{
    const struct ngpu_texture_mtl *texture_mtl = (const struct ngpu_texture_mtl *)texture;
    desc.texture = texture_mtl->texture;
    if (texture->params.type == NGPU_TEXTURE_TYPE_3D)
        desc.depthPlane = layer;
    else
        desc.slice = layer;
}

static void set_resolve_texture(MTLRenderPassAttachmentDescriptor *desc, const struct ngpu_texture *texture, uint32_t layer)
{
    const struct ngpu_texture_mtl *texture_mtl = (const struct ngpu_texture_mtl *)texture;
    desc.resolveTexture = texture_mtl->texture;
    if (texture->params.type == NGPU_TEXTURE_TYPE_3D)
        desc.resolveDepthPlane = layer;
    else
        desc.resolveSlice = layer;
}

MTLRenderPassDescriptor *ngpu_rendertarget_mtl_get_descriptor(const struct ngpu_rendertarget *s)
{
    const struct ngpu_rendertarget_params *params = &s->params;
    MTLRenderPassDescriptor *desc = [MTLRenderPassDescriptor renderPassDescriptor];

    for (size_t i = 0; i < params->nb_colors; i++) {
        const struct ngpu_attachment *attachment = &params->colors[i];
        MTLRenderPassColorAttachmentDescriptor *color = desc.colorAttachments[i];
        set_attachment_texture(color, attachment->attachment, attachment->attachment_layer);
        if (attachment->resolve_target)
            set_resolve_texture(color, attachment->resolve_target, attachment->resolve_target_layer);
        color.loadAction = load_action_map[attachment->load_op];
        color.storeAction = get_store_action(attachment->store_op, attachment->resolve_target != NULL);
        const float *c = attachment->clear_value;
        color.clearColor = MTLClearColorMake(c[0], c[1], c[2], c[3]);
    }

    const struct ngpu_attachment *attachment = &params->depth_stencil;
    const struct ngpu_texture *texture = attachment->attachment;
    if (texture) {
        const enum ngpu_format format = texture->params.format;
        const MTLLoadAction load_action = load_action_map[attachment->load_op];
        const MTLStoreAction store_action = get_store_action(attachment->store_op, attachment->resolve_target != NULL);
        if (ngpu_format_has_depth(format)) {
            MTLRenderPassDepthAttachmentDescriptor *depth = desc.depthAttachment;
            set_attachment_texture(depth, texture, attachment->attachment_layer);
            if (attachment->resolve_target)
                set_resolve_texture(depth, attachment->resolve_target, attachment->resolve_target_layer);
            depth.loadAction = load_action;
            depth.storeAction = store_action;
            depth.clearDepth = 1.0;
        }
        if (ngpu_format_has_stencil(format)) {
            MTLRenderPassStencilAttachmentDescriptor *stencil = desc.stencilAttachment;
            set_attachment_texture(stencil, texture, attachment->attachment_layer);
            if (attachment->resolve_target)
                set_resolve_texture(stencil, attachment->resolve_target, attachment->resolve_target_layer);
            stencil.loadAction = load_action;
            stencil.storeAction = store_action;
            stencil.clearStencil = 0;
        }
    }

    return desc;
}

struct ngpu_rendertarget *ngpu_rendertarget_mtl_create(struct ngpu_ctx *gpu_ctx)
{
    struct ngpu_rendertarget_mtl *s = ngli_calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->parent.gpu_ctx = gpu_ctx;
    return (struct ngpu_rendertarget *)s;
}

int ngpu_rendertarget_mtl_init(struct ngpu_rendertarget *s)
{
    return 0;
}

void ngpu_rendertarget_mtl_freep(struct ngpu_rendertarget **sp)
{
    ngli_freep(sp);
}
