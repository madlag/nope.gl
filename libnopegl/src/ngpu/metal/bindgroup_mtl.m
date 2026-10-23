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

#include "bindgroup_mtl.h"
#include "buffer_mtl.h"
#include "ctx_mtl.h"
#include "program_mtl.h"
#include "texture_mtl.h"
#include "utils/memory.h"
#include "utils/utils.h"

struct ngpu_bindgroup_layout *ngpu_bindgroup_layout_mtl_create(struct ngpu_ctx *gpu_ctx)
{
    struct ngpu_bindgroup_layout_mtl *s = ngli_calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->parent.gpu_ctx = gpu_ctx;
    return (struct ngpu_bindgroup_layout *)s;
}

int ngpu_bindgroup_layout_mtl_init(struct ngpu_bindgroup_layout *s)
{
    return 0;
}

void ngpu_bindgroup_layout_mtl_freep(struct ngpu_bindgroup_layout **sp)
{
    ngli_freep(sp);
}

struct ngpu_bindgroup *ngpu_bindgroup_mtl_create(struct ngpu_ctx *gpu_ctx)
{
    struct ngpu_bindgroup_mtl *s = ngli_calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->parent.gpu_ctx = gpu_ctx;
    return (struct ngpu_bindgroup *)s;
}

static void unref_texture_binding(void *user_arg, void *data)
{
    struct texture_binding_mtl *binding = data;
    NGLI_RC_UNREFP(&binding->texture);
}

static void unref_buffer_binding(void *user_arg, void *data)
{
    struct buffer_binding_mtl *binding = data;
    NGLI_RC_UNREFP(&binding->buffer);
}

static int is_dynamic(enum ngpu_type type)
{
    return type == NGPU_TYPE_UNIFORM_BUFFER_DYNAMIC || type == NGPU_TYPE_STORAGE_BUFFER_DYNAMIC;
}

/*
 * Dynamic offsets are given in increasing binding order (Vulkan semantics):
 * the index of a dynamic buffer is the number of dynamic buffers with a lower
 * binding.
 */
static int get_dynamic_index(const struct ngpu_bindgroup_layout *layout, const struct ngpu_bindgroup_layout_entry *entry)
{
    if (!is_dynamic(entry->type))
        return -1;
    int index = 0;
    for (size_t i = 0; i < layout->nb_buffers; i++) {
        const struct ngpu_bindgroup_layout_entry *other = &layout->buffers[i];
        if (is_dynamic(other->type) && other->binding < entry->binding)
            index++;
    }
    return index;
}

int ngpu_bindgroup_mtl_init(struct ngpu_bindgroup *s, const struct ngpu_bindgroup_params *params)
{
    struct ngpu_bindgroup_mtl *s_priv = (struct ngpu_bindgroup_mtl *)s;

    if (params->resources.nb_buffers > 0)
        ngli_assert(params->resources.nb_buffers == params->layout->nb_buffers);
    if (params->resources.nb_textures > 0)
        ngli_assert(params->resources.nb_textures == params->layout->nb_textures);

    s->layout = NGLI_RC_REF(params->layout);

    ngli_darray_init(&s_priv->texture_bindings, sizeof(struct texture_binding_mtl), 0);
    ngli_darray_init(&s_priv->buffer_bindings, sizeof(struct buffer_binding_mtl), 0);
    ngli_darray_set_free_func(&s_priv->texture_bindings, unref_texture_binding, NULL);
    ngli_darray_set_free_func(&s_priv->buffer_bindings, unref_buffer_binding, NULL);

    const struct ngpu_bindgroup_layout *layout = s->layout;
    for (size_t i = 0; i < layout->nb_buffers; i++) {
        const struct ngpu_bindgroup_layout_entry *entry = &layout->buffers[i];
        const struct buffer_binding_mtl binding = {
            .layout_entry  = *entry,
            .dynamic_index = get_dynamic_index(layout, entry),
        };
        if (!ngli_darray_push(&s_priv->buffer_bindings, &binding))
            return NGL_ERROR_MEMORY;
    }

    for (size_t i = 0; i < layout->nb_textures; i++) {
        const struct texture_binding_mtl binding = {.layout_entry = layout->textures[i]};
        if (!ngli_darray_push(&s_priv->texture_bindings, &binding))
            return NGL_ERROR_MEMORY;
    }

    for (size_t i = 0; i < params->resources.nb_buffers; i++) {
        int ret = ngpu_bindgroup_update_buffer(s, (int32_t)i, &params->resources.buffers[i]);
        if (ret < 0)
            return ret;
    }

    for (size_t i = 0; i < params->resources.nb_textures; i++) {
        int ret = ngpu_bindgroup_update_texture(s, (int32_t)i, &params->resources.textures[i]);
        if (ret < 0)
            return ret;
    }

    return 0;
}

int ngpu_bindgroup_mtl_update_texture(struct ngpu_bindgroup *s, uint32_t index, const struct ngpu_texture_binding *binding)
{
    struct ngpu_bindgroup_mtl *s_priv = (struct ngpu_bindgroup_mtl *)s;
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)s->gpu_ctx;

    struct texture_binding_mtl *binding_mtl = ngli_darray_get(&s_priv->texture_bindings, index);
    NGLI_RC_UNREFP(&binding_mtl->texture);

    /* Unbound textures are allowed by the nope.gl pipeline API */
    const struct ngpu_texture *texture = binding->texture ? binding->texture : gpu_ctx_mtl->dummy_texture;
    binding_mtl->texture = NGLI_RC_REF(texture);

    return 0;
}

int ngpu_bindgroup_mtl_update_buffer(struct ngpu_bindgroup *s, uint32_t index, const struct ngpu_buffer_binding *binding)
{
    struct ngpu_bindgroup_mtl *s_priv = (struct ngpu_bindgroup_mtl *)s;

    struct buffer_binding_mtl *binding_mtl = ngli_darray_get(&s_priv->buffer_bindings, index);
    NGLI_RC_UNREFP(&binding_mtl->buffer);

    binding_mtl->buffer = binding->buffer ? NGLI_RC_REF(binding->buffer) : NULL;
    binding_mtl->offset = binding->offset;
    binding_mtl->size   = binding->size;

    return 0;
}

#define BIND_RESOURCES(stage_bit, set_buffer, set_texture, set_sampler) do {                     \
    const struct ngpu_bindgroup_mtl *s_priv = (const struct ngpu_bindgroup_mtl *)s;              \
                                                                                                 \
    const struct buffer_binding_mtl *buffers = ngli_darray_data(&s_priv->buffer_bindings);       \
    for (size_t i = 0; i < ngli_darray_count(&s_priv->buffer_bindings); i++) {                   \
        const struct buffer_binding_mtl *binding = &buffers[i];                                  \
        if (!binding->buffer || !(binding->layout_entry.stage_flags & (stage_bit)))              \
            continue;                                                                            \
        size_t offset = binding->offset;                                                         \
        if (binding->dynamic_index >= 0) {                                                       \
            ngli_assert((size_t)binding->dynamic_index < nb_dynamic_offsets);                    \
            offset += dynamic_offsets[binding->dynamic_index];                                   \
        }                                                                                        \
        const struct ngpu_buffer_mtl *buffer_mtl = (const struct ngpu_buffer_mtl *)binding->buffer; \
        [encoder set_buffer:buffer_mtl->buffer offset:offset atIndex:binding->layout_entry.binding]; \
    }                                                                                            \
                                                                                                 \
    const struct texture_binding_mtl *textures = ngli_darray_data(&s_priv->texture_bindings);    \
    for (size_t i = 0; i < ngli_darray_count(&s_priv->texture_bindings); i++) {                  \
        const struct texture_binding_mtl *binding = &textures[i];                                \
        if (!(binding->layout_entry.stage_flags & (stage_bit)))                                  \
            continue;                                                                            \
        const uint32_t index = binding->layout_entry.binding;                                    \
        const struct ngpu_texture_mtl *texture_mtl = (const struct ngpu_texture_mtl *)binding->texture; \
        [encoder set_texture:texture_mtl->texture atIndex:index];                                \
        const int8_t sampler = index < NGPU_MTL_MAX_BINDINGS ? program->sampler_map[index] : -1; \
        if (sampler >= 0 && texture_mtl->sampler)                                                \
            [encoder set_sampler:texture_mtl->sampler atIndex:(NSUInteger)sampler];              \
    }                                                                                            \
} while (0)

void ngpu_bindgroup_mtl_bind_render(const struct ngpu_bindgroup *s, const struct ngpu_program_mtl *program,
                                    id<MTLRenderCommandEncoder> encoder,
                                    const uint32_t *dynamic_offsets, size_t nb_dynamic_offsets)
{
    BIND_RESOURCES(NGPU_PROGRAM_STAGE_VERTEX_BIT, setVertexBuffer, setVertexTexture, setVertexSamplerState);
    BIND_RESOURCES(NGPU_PROGRAM_STAGE_FRAGMENT_BIT, setFragmentBuffer, setFragmentTexture, setFragmentSamplerState);
}

void ngpu_bindgroup_mtl_bind_compute(const struct ngpu_bindgroup *s, const struct ngpu_program_mtl *program,
                                     id<MTLComputeCommandEncoder> encoder,
                                     const uint32_t *dynamic_offsets, size_t nb_dynamic_offsets)
{
    BIND_RESOURCES(NGPU_PROGRAM_STAGE_COMPUTE_BIT, setBuffer, setTexture, setSamplerState);
}

void ngpu_bindgroup_mtl_freep(struct ngpu_bindgroup **sp)
{
    if (!*sp)
        return;

    struct ngpu_bindgroup *s = *sp;
    struct ngpu_bindgroup_mtl *s_priv = (struct ngpu_bindgroup_mtl *)s;
    ngli_darray_reset(&s_priv->texture_bindings);
    ngli_darray_reset(&s_priv->buffer_bindings);
    NGLI_RC_UNREFP(&s->layout);
    ngli_freep(sp);
}
