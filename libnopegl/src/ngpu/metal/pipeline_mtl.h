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

#ifndef NGPU_PIPELINE_MTL_H
#define NGPU_PIPELINE_MTL_H

#import <Metal/Metal.h>

#include "ngpu/pipeline.h"

/*
 * Vertex buffers are bound at the top of the vertex stage buffer table, away
 * from the resource bindings which start at 0.
 */
#define NGPU_MTL_MAX_BUFFER_INDEX 30
#define NGPU_MTL_VERTEX_BUFFER_INDEX(i) (NGPU_MTL_MAX_BUFFER_INDEX - (i))

struct ngpu_pipeline_mtl {
    struct ngpu_pipeline parent;
    id<MTLRenderPipelineState> render_state;
    id<MTLDepthStencilState> depth_stencil_state;
    id<MTLComputePipelineState> compute_state;
    MTLPrimitiveType primitive_type;
    MTLCullMode cull_mode;
    MTLWinding winding;
    MTLSize threads_per_threadgroup;
};

struct ngpu_pipeline *ngpu_pipeline_mtl_create(struct ngpu_ctx *gpu_ctx);
int ngpu_pipeline_mtl_init(struct ngpu_pipeline *s);
void ngpu_pipeline_mtl_draw(struct ngpu_pipeline *s, uint32_t nb_vertices, uint32_t nb_instances, uint32_t first_vertex);
void ngpu_pipeline_mtl_draw_indexed(struct ngpu_pipeline *s, uint32_t nb_indices, uint32_t nb_instances);
void ngpu_pipeline_mtl_dispatch(struct ngpu_pipeline *s, uint32_t nb_group_x, uint32_t nb_group_y, uint32_t nb_group_z);
void ngpu_pipeline_mtl_freep(struct ngpu_pipeline **sp);

#endif
