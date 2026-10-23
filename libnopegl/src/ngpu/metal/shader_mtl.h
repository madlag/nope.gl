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

#ifndef NGPU_SHADER_MTL_H
#define NGPU_SHADER_MTL_H

#include <stddef.h>
#include <stdint.h>

#include "ngpu/program.h"

/*
 * SPIR-V to MSL translation (SPIRV-Cross).
 *
 * Resource mapping: nope.gl shaders use a single binding namespace shared by
 * every resource type and stage (Vulkan style), so a binding number is unique
 * within a program. Buffers and textures keep their binding number as Metal
 * argument index. Samplers are limited to 16 slots per stage in Metal, so each
 * sampled texture gets a dense sampler index, shared by all the stages of the
 * program (sampler_map[binding]).
 */

#define NGPU_MTL_MAX_BINDINGS 64

struct ngpu_msl_shader {
    char *source;
    char *entry_point;
    uint32_t workgroup_size[3];
};

/* Add the bindings of the sampled textures of a SPIR-V module to *mask */
int ngpu_shader_mtl_get_sampled_bindings(const void *spirv, size_t size, uint64_t *mask);

int ngpu_shader_mtl_translate(const void *spirv, size_t size, enum ngpu_program_stage stage, int ios,
                              const int8_t *sampler_map, struct ngpu_msl_shader *shader);

void ngpu_msl_shader_reset(struct ngpu_msl_shader *shader);

#endif
