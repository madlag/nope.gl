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

#ifndef NGPU_BINDGROUP_MTL_H
#define NGPU_BINDGROUP_MTL_H

#import <Metal/Metal.h>

#include "ngpu/bindgroup.h"
#include "utils/darray.h"

struct ngpu_program_mtl;

struct texture_binding_mtl {
    struct ngpu_bindgroup_layout_entry layout_entry;
    const struct ngpu_texture *texture;
};

struct buffer_binding_mtl {
    struct ngpu_bindgroup_layout_entry layout_entry;
    const struct ngpu_buffer *buffer;
    size_t offset;
    size_t size;
    int dynamic_index; /* index in the dynamic offsets array, -1 if not dynamic */
};

struct ngpu_bindgroup_layout_mtl {
    struct ngpu_bindgroup_layout parent;
};

struct ngpu_bindgroup_mtl {
    struct ngpu_bindgroup parent;
    struct darray texture_bindings; // array of texture_binding_mtl
    struct darray buffer_bindings;  // array of buffer_binding_mtl
};

struct ngpu_bindgroup_layout *ngpu_bindgroup_layout_mtl_create(struct ngpu_ctx *gpu_ctx);
int ngpu_bindgroup_layout_mtl_init(struct ngpu_bindgroup_layout *s);
void ngpu_bindgroup_layout_mtl_freep(struct ngpu_bindgroup_layout **sp);

struct ngpu_bindgroup *ngpu_bindgroup_mtl_create(struct ngpu_ctx *gpu_ctx);
int ngpu_bindgroup_mtl_init(struct ngpu_bindgroup *s, const struct ngpu_bindgroup_params *params);
int ngpu_bindgroup_mtl_update_texture(struct ngpu_bindgroup *s, uint32_t index, const struct ngpu_texture_binding *binding);
int ngpu_bindgroup_mtl_update_buffer(struct ngpu_bindgroup *s, uint32_t index, const struct ngpu_buffer_binding *binding);
void ngpu_bindgroup_mtl_freep(struct ngpu_bindgroup **sp);

/* Bind the resources on an encoder; the program provides the sampler slots */
void ngpu_bindgroup_mtl_bind_render(const struct ngpu_bindgroup *s, const struct ngpu_program_mtl *program,
                                    id<MTLRenderCommandEncoder> encoder,
                                    const uint32_t *dynamic_offsets, size_t nb_dynamic_offsets);
void ngpu_bindgroup_mtl_bind_compute(const struct ngpu_bindgroup *s, const struct ngpu_program_mtl *program,
                                     id<MTLComputeCommandEncoder> encoder,
                                     const uint32_t *dynamic_offsets, size_t nb_dynamic_offsets);

#endif
