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

#ifndef NGPU_PROGRAM_MTL_H
#define NGPU_PROGRAM_MTL_H

#import <Metal/Metal.h>

#include "ngpu/program.h"
#include "shader_mtl.h"

struct ngpu_program_mtl {
    struct ngpu_program parent;
    id<MTLFunction> functions[NGPU_PROGRAM_STAGE_NB];
    int8_t sampler_map[NGPU_MTL_MAX_BINDINGS]; /* binding -> sampler index, -1 if not a sampled texture */
    uint32_t workgroup_size[3];
};

struct ngpu_program *ngpu_program_mtl_create(struct ngpu_ctx *gpu_ctx);
int ngpu_program_mtl_init(struct ngpu_program *s, const struct ngpu_program_params *params);
void ngpu_program_mtl_freep(struct ngpu_program **sp);

#endif
