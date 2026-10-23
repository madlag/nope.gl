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

#include "ctx_mtl.h"
#include "log.h"
#include "ngpu/vulkan/glslang_utils.h"
#include "nopegl.h"
#include "program_mtl.h"
#include "utils/memory.h"
#include "utils/string.h"
#include "utils/utils.h"

static void log_source(const char *label, const char *kind, const char *source)
{
    char *numbered = ngli_numbered_lines(source);
    if (numbered) {
        LOG(ERROR, "failed to compile %s shader \"%s\":\n%s", kind, label ? label : "", numbered);
        ngli_free(numbered);
    }
}

struct ngpu_program *ngpu_program_mtl_create(struct ngpu_ctx *gpu_ctx)
{
    struct ngpu_program_mtl *s = ngli_calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->parent.gpu_ctx = gpu_ctx;
    return (struct ngpu_program *)s;
}

static int compile_msl(struct ngpu_program *s, const struct ngpu_program_params *params,
                       enum ngpu_program_stage stage, const struct ngpu_msl_shader *shader)
{
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)s->gpu_ctx;
    struct ngpu_program_mtl *s_priv = (struct ngpu_program_mtl *)s;

    @autoreleasepool {
        MTLCompileOptions *options = [[[MTLCompileOptions alloc] init] autorelease];
        options.languageVersion = MTLLanguageVersion2_3;
        /* Keep IEEE semantics: nope.gl output is compared against references */
        options.mathMode = MTLMathModeSafe;

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:shader->source];
        id<MTLLibrary> library = [gpu_ctx_mtl->device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            log_source(params->label, "MSL", shader->source);
            LOG(ERROR, "%s", error.localizedDescription.UTF8String);
            return NGL_ERROR_GRAPHICS_GENERIC;
        }

        NSString *entry_point = [NSString stringWithUTF8String:shader->entry_point];
        s_priv->functions[stage] = [library newFunctionWithName:entry_point];
        [library release];
        if (!s_priv->functions[stage]) {
            LOG(ERROR, "entry point %s not found", shader->entry_point);
            return NGL_ERROR_GRAPHICS_GENERIC;
        }
    }

    return 0;
}

int ngpu_program_mtl_init(struct ngpu_program *s, const struct ngpu_program_params *params)
{
    struct ngpu_program_mtl *s_priv = (struct ngpu_program_mtl *)s;
    const struct ngl_config *config = &s->gpu_ctx->config;

    const char *sources[NGPU_PROGRAM_STAGE_NB] = {
        [NGPU_PROGRAM_STAGE_VERT] = params->vertex,
        [NGPU_PROGRAM_STAGE_FRAG] = params->fragment,
        [NGPU_PROGRAM_STAGE_COMP] = params->compute,
    };
    void *spirv[NGPU_PROGRAM_STAGE_NB] = {0};
    size_t spirv_size[NGPU_PROGRAM_STAGE_NB] = {0};
    int ret = 0;

    /* GLSL -> SPIR-V, and collect the sampled textures of all the stages */
    uint64_t sampled_bindings = 0;
    for (size_t i = 0; i < NGPU_PROGRAM_STAGE_NB; i++) {
        if (!sources[i])
            continue;
        ret = ngli_glslang_compile(i, sources[i], config->debug, &spirv[i], &spirv_size[i]);
        if (ret < 0) {
            log_source(params->label, "GLSL", sources[i]);
            goto end;
        }
        ret = ngpu_shader_mtl_get_sampled_bindings(spirv[i], spirv_size[i], &sampled_bindings);
        if (ret < 0)
            goto end;
    }

    int8_t nb_samplers = 0;
    for (size_t binding = 0; binding < NGPU_MTL_MAX_BINDINGS; binding++)
        s_priv->sampler_map[binding] = sampled_bindings & (1ULL << binding) ? nb_samplers++ : -1;
    if (nb_samplers > 16) {
        LOG(ERROR, "program \"%s\" uses %d samplers, Metal supports 16", params->label ? params->label : "", nb_samplers);
        ret = NGL_ERROR_GRAPHICS_LIMIT_EXCEEDED;
        goto end;
    }

    /* SPIR-V -> MSL -> Metal function */
#if defined(TARGET_IPHONE)
    const int ios = 1;
#else
    const int ios = 0;
#endif
    for (size_t i = 0; i < NGPU_PROGRAM_STAGE_NB; i++) {
        if (!spirv[i])
            continue;
        struct ngpu_msl_shader shader;
        ret = ngpu_shader_mtl_translate(spirv[i], spirv_size[i], i, ios, s_priv->sampler_map, &shader);
        if (ret < 0) {
            log_source(params->label, "GLSL", sources[i]);
            goto end;
        }
        ret = compile_msl(s, params, i, &shader);
        if (i == NGPU_PROGRAM_STAGE_COMP)
            memcpy(s_priv->workgroup_size, shader.workgroup_size, sizeof(s_priv->workgroup_size));
        ngpu_msl_shader_reset(&shader);
        if (ret < 0)
            goto end;
    }

end:
    for (size_t i = 0; i < NGPU_PROGRAM_STAGE_NB; i++)
        ngli_freep(&spirv[i]);
    return ret;
}

void ngpu_program_mtl_freep(struct ngpu_program **sp)
{
    if (!*sp)
        return;
    struct ngpu_program_mtl *s_priv = (struct ngpu_program_mtl *)*sp;
    for (size_t i = 0; i < NGPU_PROGRAM_STAGE_NB; i++)
        [s_priv->functions[i] release];
    ngli_freep(sp);
}
