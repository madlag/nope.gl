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
#include <spirv_cross/spirv_cross_c.h>

#include "config.h"

#include "log.h"
#include "nopegl.h"
#include "shader_mtl.h"
#include "utils/memory.h"
#include "utils/string.h"
#include "utils/utils.h"

static void error_callback(void *userdata, const char *error)
{
    LOG(ERROR, "SPIRV-Cross: %s", error);
}

static int create_compiler(const void *spirv, size_t size, spvc_context *contextp, spvc_compiler *compilerp)
{
    spvc_context context = NULL;
    if (spvc_context_create(&context) != SPVC_SUCCESS)
        return NGL_ERROR_MEMORY;
    spvc_context_set_error_callback(context, error_callback, NULL);

    spvc_parsed_ir ir = NULL;
    spvc_compiler compiler = NULL;
    if (spvc_context_parse_spirv(context, spirv, size / sizeof(SpvId), &ir) != SPVC_SUCCESS ||
        spvc_context_create_compiler(context, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &compiler) != SPVC_SUCCESS) {
        spvc_context_destroy(context);
        return NGL_ERROR_EXTERNAL;
    }

    *contextp = context;
    *compilerp = compiler;
    return 0;
}

int ngpu_shader_mtl_get_sampled_bindings(const void *spirv, size_t size, uint64_t *mask)
{
    spvc_context context;
    spvc_compiler compiler;
    int ret = create_compiler(spirv, size, &context, &compiler);
    if (ret < 0)
        return ret;

    spvc_resources resources = NULL;
    const spvc_reflected_resource *list = NULL;
    size_t count = 0;
    if (spvc_compiler_create_shader_resources(compiler, &resources) != SPVC_SUCCESS ||
        spvc_resources_get_resource_list_for_type(resources, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, &list, &count) != SPVC_SUCCESS) {
        spvc_context_destroy(context);
        return NGL_ERROR_EXTERNAL;
    }

    for (size_t i = 0; i < count; i++) {
        const uint32_t binding = spvc_compiler_get_decoration(compiler, list[i].id, SpvDecorationBinding);
        if (binding >= NGPU_MTL_MAX_BINDINGS) {
            LOG(ERROR, "binding %u exceeds the Metal backend limit (%d)", binding, NGPU_MTL_MAX_BINDINGS);
            spvc_context_destroy(context);
            return NGL_ERROR_GRAPHICS_LIMIT_EXCEEDED;
        }
        *mask |= 1ULL << binding;
    }

    spvc_context_destroy(context);
    return 0;
}

static const SpvExecutionModel execution_models[] = {
    [NGPU_PROGRAM_STAGE_VERT] = SpvExecutionModelVertex,
    [NGPU_PROGRAM_STAGE_FRAG] = SpvExecutionModelFragment,
    [NGPU_PROGRAM_STAGE_COMP] = SpvExecutionModelGLCompute,
};

static const spvc_resource_type mapped_resource_types[] = {
    SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
    SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
    SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
    SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
};

static int add_resource_bindings(spvc_compiler compiler, SpvExecutionModel model, const int8_t *sampler_map)
{
    spvc_resources resources = NULL;
    if (spvc_compiler_create_shader_resources(compiler, &resources) != SPVC_SUCCESS)
        return NGL_ERROR_EXTERNAL;

    for (size_t t = 0; t < NGLI_ARRAY_NB(mapped_resource_types); t++) {
        const spvc_reflected_resource *list = NULL;
        size_t count = 0;
        if (spvc_resources_get_resource_list_for_type(resources, mapped_resource_types[t], &list, &count) != SPVC_SUCCESS)
            return NGL_ERROR_EXTERNAL;

        for (size_t i = 0; i < count; i++) {
            const uint32_t set = spvc_compiler_get_decoration(compiler, list[i].id, SpvDecorationDescriptorSet);
            const uint32_t binding = spvc_compiler_get_decoration(compiler, list[i].id, SpvDecorationBinding);
            if (binding >= NGPU_MTL_MAX_BINDINGS) {
                LOG(ERROR, "binding %u exceeds the Metal backend limit (%d)", binding, NGPU_MTL_MAX_BINDINGS);
                return NGL_ERROR_GRAPHICS_LIMIT_EXCEEDED;
            }

            spvc_msl_resource_binding resource_binding;
            spvc_msl_resource_binding_init(&resource_binding);
            resource_binding.stage       = model;
            resource_binding.desc_set    = set;
            resource_binding.binding     = binding;
            resource_binding.msl_buffer  = binding;
            resource_binding.msl_texture = binding;
            resource_binding.msl_sampler = sampler_map[binding] >= 0 ? (uint32_t)sampler_map[binding] : 0;
            if (spvc_compiler_msl_add_resource_binding(compiler, &resource_binding) != SPVC_SUCCESS)
                return NGL_ERROR_EXTERNAL;
        }
    }

    return 0;
}

int ngpu_shader_mtl_translate(const void *spirv, size_t size, enum ngpu_program_stage stage, int ios,
                              const int8_t *sampler_map, struct ngpu_msl_shader *shader)
{
    memset(shader, 0, sizeof(*shader));

    spvc_context context;
    spvc_compiler compiler;
    int ret = create_compiler(spirv, size, &context, &compiler);
    if (ret < 0)
        return ret;

    const SpvExecutionModel model = execution_models[stage];

    spvc_compiler_options options = NULL;
    if (spvc_compiler_create_compiler_options(compiler, &options) != SPVC_SUCCESS) {
        ret = NGL_ERROR_EXTERNAL;
        goto end;
    }
    spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(2, 3, 0));
    spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_PLATFORM,
                                   ios ? SPVC_MSL_PLATFORM_IOS : SPVC_MSL_PLATFORM_MACOS);
    /* Vulkan clip space has its Y axis pointing down, Metal up: flipping the
     * vertex output Y (like MoltenVK) makes Metal behave exactly like Vulkan,
     * including for the internal passes writing clip space positions directly */
    spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_FLIP_VERTEX_Y, SPVC_TRUE);
    if (spvc_compiler_install_compiler_options(compiler, options) != SPVC_SUCCESS) {
        ret = NGL_ERROR_EXTERNAL;
        goto end;
    }

    ret = add_resource_bindings(compiler, model, sampler_map);
    if (ret < 0)
        goto end;

    const char *source = NULL;
    if (spvc_compiler_compile(compiler, &source) != SPVC_SUCCESS) {
        ret = NGL_ERROR_EXTERNAL;
        goto end;
    }

    shader->source = ngli_strdup(source);
    shader->entry_point = ngli_strdup(spvc_compiler_get_cleansed_entry_point_name(compiler, "main", model));
    if (!shader->source || !shader->entry_point) {
        ret = NGL_ERROR_MEMORY;
        goto end;
    }

    if (stage == NGPU_PROGRAM_STAGE_COMP) {
        for (uint32_t i = 0; i < 3; i++)
            shader->workgroup_size[i] = spvc_compiler_get_execution_mode_argument_by_index(compiler, SpvExecutionModeLocalSize, i);
    }

end:
    if (ret < 0)
        ngpu_msl_shader_reset(shader);
    spvc_context_destroy(context);
    return ret;
}

void ngpu_msl_shader_reset(struct ngpu_msl_shader *shader)
{
    ngli_freep(&shader->source);
    ngli_freep(&shader->entry_point);
}
