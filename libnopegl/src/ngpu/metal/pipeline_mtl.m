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
#include "format_mtl.h"
#include "log.h"
#include "nopegl.h"
#include "pipeline_mtl.h"
#include "program_mtl.h"
#include "utils/memory.h"
#include "utils/utils.h"

static const MTLPrimitiveType primitive_type_map[NGPU_PRIMITIVE_TOPOLOGY_NB] = {
    [NGPU_PRIMITIVE_TOPOLOGY_POINT_LIST]     = MTLPrimitiveTypePoint,
    [NGPU_PRIMITIVE_TOPOLOGY_LINE_LIST]      = MTLPrimitiveTypeLine,
    [NGPU_PRIMITIVE_TOPOLOGY_LINE_STRIP]     = MTLPrimitiveTypeLineStrip,
    [NGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST]  = MTLPrimitiveTypeTriangle,
    [NGPU_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP] = MTLPrimitiveTypeTriangleStrip,
};

static const MTLBlendFactor blend_factor_map[NGPU_BLEND_FACTOR_NB] = {
    [NGPU_BLEND_FACTOR_ZERO]                     = MTLBlendFactorZero,
    [NGPU_BLEND_FACTOR_ONE]                      = MTLBlendFactorOne,
    [NGPU_BLEND_FACTOR_SRC_COLOR]                = MTLBlendFactorSourceColor,
    [NGPU_BLEND_FACTOR_ONE_MINUS_SRC_COLOR]      = MTLBlendFactorOneMinusSourceColor,
    [NGPU_BLEND_FACTOR_DST_COLOR]                = MTLBlendFactorDestinationColor,
    [NGPU_BLEND_FACTOR_ONE_MINUS_DST_COLOR]      = MTLBlendFactorOneMinusDestinationColor,
    [NGPU_BLEND_FACTOR_SRC_ALPHA]                = MTLBlendFactorSourceAlpha,
    [NGPU_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA]      = MTLBlendFactorOneMinusSourceAlpha,
    [NGPU_BLEND_FACTOR_DST_ALPHA]                = MTLBlendFactorDestinationAlpha,
    [NGPU_BLEND_FACTOR_ONE_MINUS_DST_ALPHA]      = MTLBlendFactorOneMinusDestinationAlpha,
    [NGPU_BLEND_FACTOR_CONSTANT_COLOR]           = MTLBlendFactorBlendColor,
    [NGPU_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR] = MTLBlendFactorOneMinusBlendColor,
    [NGPU_BLEND_FACTOR_CONSTANT_ALPHA]           = MTLBlendFactorBlendAlpha,
    [NGPU_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA] = MTLBlendFactorOneMinusBlendAlpha,
    [NGPU_BLEND_FACTOR_SRC_ALPHA_SATURATE]       = MTLBlendFactorSourceAlphaSaturated,
};

static const MTLBlendOperation blend_op_map[NGPU_BLEND_OP_NB] = {
    [NGPU_BLEND_OP_ADD]              = MTLBlendOperationAdd,
    [NGPU_BLEND_OP_SUBTRACT]         = MTLBlendOperationSubtract,
    [NGPU_BLEND_OP_REVERSE_SUBTRACT] = MTLBlendOperationReverseSubtract,
    [NGPU_BLEND_OP_MIN]              = MTLBlendOperationMin,
    [NGPU_BLEND_OP_MAX]              = MTLBlendOperationMax,
};

static const MTLCompareFunction compare_func_map[NGPU_COMPARE_OP_NB] = {
    [NGPU_COMPARE_OP_NEVER]            = MTLCompareFunctionNever,
    [NGPU_COMPARE_OP_LESS]             = MTLCompareFunctionLess,
    [NGPU_COMPARE_OP_EQUAL]            = MTLCompareFunctionEqual,
    [NGPU_COMPARE_OP_LESS_OR_EQUAL]    = MTLCompareFunctionLessEqual,
    [NGPU_COMPARE_OP_GREATER]          = MTLCompareFunctionGreater,
    [NGPU_COMPARE_OP_NOT_EQUAL]        = MTLCompareFunctionNotEqual,
    [NGPU_COMPARE_OP_GREATER_OR_EQUAL] = MTLCompareFunctionGreaterEqual,
    [NGPU_COMPARE_OP_ALWAYS]           = MTLCompareFunctionAlways,
};

static const MTLStencilOperation stencil_op_map[NGPU_STENCIL_OP_NB] = {
    [NGPU_STENCIL_OP_KEEP]                = MTLStencilOperationKeep,
    [NGPU_STENCIL_OP_ZERO]                = MTLStencilOperationZero,
    [NGPU_STENCIL_OP_REPLACE]             = MTLStencilOperationReplace,
    [NGPU_STENCIL_OP_INCREMENT_AND_CLAMP] = MTLStencilOperationIncrementClamp,
    [NGPU_STENCIL_OP_DECREMENT_AND_CLAMP] = MTLStencilOperationDecrementClamp,
    [NGPU_STENCIL_OP_INVERT]              = MTLStencilOperationInvert,
    [NGPU_STENCIL_OP_INCREMENT_AND_WRAP]  = MTLStencilOperationIncrementWrap,
    [NGPU_STENCIL_OP_DECREMENT_AND_WRAP]  = MTLStencilOperationDecrementWrap,
};

static const MTLCullMode cull_mode_map[NGPU_CULL_MODE_NB] = {
    [NGPU_CULL_MODE_NONE]      = MTLCullModeNone,
    [NGPU_CULL_MODE_FRONT_BIT] = MTLCullModeFront,
    [NGPU_CULL_MODE_BACK_BIT]  = MTLCullModeBack,
};

static const MTLWinding winding_map[NGPU_FRONT_FACE_NB] = {
    [NGPU_FRONT_FACE_COUNTER_CLOCKWISE] = MTLWindingCounterClockwise,
    [NGPU_FRONT_FACE_CLOCKWISE]         = MTLWindingClockwise,
};

static MTLColorWriteMask get_color_write_mask(uint32_t mask)
{
    return (mask & NGPU_COLOR_COMPONENT_R_BIT ? MTLColorWriteMaskRed   : 0) |
           (mask & NGPU_COLOR_COMPONENT_G_BIT ? MTLColorWriteMaskGreen : 0) |
           (mask & NGPU_COLOR_COMPONENT_B_BIT ? MTLColorWriteMaskBlue  : 0) |
           (mask & NGPU_COLOR_COMPONENT_A_BIT ? MTLColorWriteMaskAlpha : 0);
}

static MTLVertexDescriptor *create_vertex_descriptor(const struct ngpu_vertex_state *state)
{
    MTLVertexDescriptor *desc = [MTLVertexDescriptor vertexDescriptor];
    for (size_t i = 0; i < state->nb_buffers; i++) {
        const struct ngpu_vertex_buffer_layout *buffer = &state->buffers[i];
        const NSUInteger buffer_index = NGPU_MTL_VERTEX_BUFFER_INDEX(i);

        MTLVertexBufferLayoutDescriptor *layout = desc.layouts[buffer_index];
        layout.stride = buffer->stride;
        if (buffer->rate) {
            layout.stepFunction = MTLVertexStepFunctionPerInstance;
            layout.stepRate = buffer->rate;
        } else {
            layout.stepFunction = MTLVertexStepFunctionPerVertex;
            layout.stepRate = 1;
        }

        for (size_t j = 0; j < buffer->nb_attributes; j++) {
            const struct ngpu_vertex_attribute *attribute = &buffer->attributes[j];
            MTLVertexAttributeDescriptor *attr = desc.attributes[attribute->location];
            attr.format = ngpu_format_ngl_to_mtl_vertex(attribute->format);
            attr.offset = attribute->offset;
            attr.bufferIndex = buffer_index;
        }
    }
    return desc;
}

static MTLStencilDescriptor *create_stencil_descriptor(const struct ngpu_stencil_op_state *state)
{
    MTLStencilDescriptor *desc = [[[MTLStencilDescriptor alloc] init] autorelease];
    desc.stencilCompareFunction    = compare_func_map[state->func];
    desc.stencilFailureOperation   = stencil_op_map[state->fail];
    desc.depthFailureOperation     = stencil_op_map[state->depth_fail];
    desc.depthStencilPassOperation = stencil_op_map[state->depth_pass];
    desc.readMask                  = state->read_mask;
    desc.writeMask                 = state->write_mask;
    return desc;
}

static int pipeline_graphics_init(struct ngpu_pipeline *s)
{
    struct ngpu_ctx *gpu_ctx = s->gpu_ctx;
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)gpu_ctx;
    struct ngpu_pipeline_mtl *s_priv = (struct ngpu_pipeline_mtl *)s;
    const struct ngpu_pipeline_graphics *graphics = &s->graphics;
    const struct ngpu_graphics_state *state = &graphics->state;
    const struct ngpu_rendertarget_layout *rt_layout = &graphics->rt_layout;
    const struct ngpu_program_mtl *program = (const struct ngpu_program_mtl *)s->program;

    if (graphics->vertex_state.nb_buffers > NGPU_MTL_MAX_BUFFER_INDEX + 1) {
        LOG(ERROR, "too many vertex buffers: %zu", graphics->vertex_state.nb_buffers);
        return NGL_ERROR_GRAPHICS_LIMIT_EXCEEDED;
    }

    MTLRenderPipelineDescriptor *desc = [[[MTLRenderPipelineDescriptor alloc] init] autorelease];
    desc.vertexFunction = program->functions[NGPU_PROGRAM_STAGE_VERT];
    desc.fragmentFunction = program->functions[NGPU_PROGRAM_STAGE_FRAG];
    desc.vertexDescriptor = create_vertex_descriptor(&graphics->vertex_state);
    desc.rasterSampleCount = NGLI_MAX(rt_layout->samples, 1);

    for (size_t i = 0; i < rt_layout->nb_colors; i++) {
        MTLRenderPipelineColorAttachmentDescriptor *color = desc.colorAttachments[i];
        color.pixelFormat                 = ngpu_format_ngl_to_mtl(rt_layout->colors[i].format);
        color.blendingEnabled             = state->blend ? YES : NO;
        color.sourceRGBBlendFactor        = blend_factor_map[state->blend_src_factor];
        color.destinationRGBBlendFactor   = blend_factor_map[state->blend_dst_factor];
        color.rgbBlendOperation           = blend_op_map[state->blend_op];
        color.sourceAlphaBlendFactor      = blend_factor_map[state->blend_src_factor_a];
        color.destinationAlphaBlendFactor = blend_factor_map[state->blend_dst_factor_a];
        color.alphaBlendOperation         = blend_op_map[state->blend_op_a];
        color.writeMask                   = get_color_write_mask(state->color_write_mask);
    }

    const enum ngpu_format ds_format = rt_layout->depth_stencil.format;
    if (ds_format != NGPU_FORMAT_UNDEFINED) {
        const MTLPixelFormat format = ngpu_format_ngl_to_mtl(ds_format);
        if (ngpu_format_has_depth(ds_format))
            desc.depthAttachmentPixelFormat = format;
        if (ngpu_format_has_stencil(ds_format))
            desc.stencilAttachmentPixelFormat = format;
    }

    NSError *error = nil;
    s_priv->render_state = [gpu_ctx_mtl->device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!s_priv->render_state) {
        LOG(ERROR, "unable to create render pipeline: %s", error.localizedDescription.UTF8String);
        return NGL_ERROR_GRAPHICS_GENERIC;
    }

    /* Like OpenGL, depth writes only happen when the depth test is enabled */
    MTLDepthStencilDescriptor *ds_desc = [[[MTLDepthStencilDescriptor alloc] init] autorelease];
    ds_desc.depthCompareFunction = state->depth_test ? compare_func_map[state->depth_func] : MTLCompareFunctionAlways;
    ds_desc.depthWriteEnabled = state->depth_test && state->depth_write;
    if (state->stencil_test) {
        ds_desc.frontFaceStencil = create_stencil_descriptor(&state->stencil_front);
        ds_desc.backFaceStencil = create_stencil_descriptor(&state->stencil_back);
    }
    s_priv->depth_stencil_state = [gpu_ctx_mtl->device newDepthStencilStateWithDescriptor:ds_desc];
    if (!s_priv->depth_stencil_state)
        return NGL_ERROR_GRAPHICS_GENERIC;

    s_priv->primitive_type = primitive_type_map[graphics->topology];
    s_priv->cull_mode = cull_mode_map[state->cull_mode];
    s_priv->winding = winding_map[state->front_face];

    return 0;
}

static int pipeline_compute_init(struct ngpu_pipeline *s)
{
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)s->gpu_ctx;
    struct ngpu_pipeline_mtl *s_priv = (struct ngpu_pipeline_mtl *)s;
    const struct ngpu_program_mtl *program = (const struct ngpu_program_mtl *)s->program;

    NSError *error = nil;
    s_priv->compute_state = [gpu_ctx_mtl->device newComputePipelineStateWithFunction:program->functions[NGPU_PROGRAM_STAGE_COMP]
                                                                               error:&error];
    if (!s_priv->compute_state) {
        LOG(ERROR, "unable to create compute pipeline: %s", error.localizedDescription.UTF8String);
        return NGL_ERROR_GRAPHICS_GENERIC;
    }

    const uint32_t *size = program->workgroup_size;
    s_priv->threads_per_threadgroup = MTLSizeMake(NGLI_MAX(size[0], 1), NGLI_MAX(size[1], 1), NGLI_MAX(size[2], 1));

    return 0;
}

struct ngpu_pipeline *ngpu_pipeline_mtl_create(struct ngpu_ctx *gpu_ctx)
{
    struct ngpu_pipeline_mtl *s = ngli_calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->parent.gpu_ctx = gpu_ctx;
    return (struct ngpu_pipeline *)s;
}

int ngpu_pipeline_mtl_init(struct ngpu_pipeline *s)
{
    @autoreleasepool {
        if (s->type == NGPU_PIPELINE_TYPE_GRAPHICS)
            return pipeline_graphics_init(s);
        if (s->type == NGPU_PIPELINE_TYPE_COMPUTE)
            return pipeline_compute_init(s);
    }
    ngli_assert(0);
}

static void prepare_draw(struct ngpu_pipeline *s, id<MTLRenderCommandEncoder> encoder)
{
    struct ngpu_ctx *gpu_ctx = s->gpu_ctx;
    struct ngpu_pipeline_mtl *s_priv = (struct ngpu_pipeline_mtl *)s;
    const struct ngpu_graphics_state *state = &s->graphics.state;

    [encoder setRenderPipelineState:s_priv->render_state];
    [encoder setDepthStencilState:s_priv->depth_stencil_state];
    [encoder setCullMode:s_priv->cull_mode];
    [encoder setFrontFacingWinding:s_priv->winding];
    [encoder setStencilFrontReferenceValue:state->stencil_front.ref backReferenceValue:state->stencil_back.ref];

    if (gpu_ctx->bindgroup)
        ngpu_bindgroup_mtl_bind_render(gpu_ctx->bindgroup, (const struct ngpu_program_mtl *)s->program, encoder,
                                       gpu_ctx->dynamic_offsets, gpu_ctx->nb_dynamic_offsets);

    const struct ngpu_vertex_state *vertex_state = &s->graphics.vertex_state;
    for (size_t i = 0; i < vertex_state->nb_buffers; i++) {
        const struct ngpu_buffer_mtl *buffer = (const struct ngpu_buffer_mtl *)gpu_ctx->vertex_buffers[i];
        [encoder setVertexBuffer:buffer->buffer offset:0 atIndex:NGPU_MTL_VERTEX_BUFFER_INDEX(i)];
    }
}

void ngpu_pipeline_mtl_draw(struct ngpu_pipeline *s, uint32_t nb_vertices, uint32_t nb_instances, uint32_t first_vertex)
{
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)s->gpu_ctx;
    struct ngpu_pipeline_mtl *s_priv = (struct ngpu_pipeline_mtl *)s;
    id<MTLRenderCommandEncoder> encoder = gpu_ctx_mtl->render_encoder;
    ngli_assert(encoder);

    prepare_draw(s, encoder);
    [encoder drawPrimitives:s_priv->primitive_type
                vertexStart:first_vertex
                vertexCount:nb_vertices
              instanceCount:nb_instances];
}

void ngpu_pipeline_mtl_draw_indexed(struct ngpu_pipeline *s, uint32_t nb_indices, uint32_t nb_instances)
{
    struct ngpu_ctx *gpu_ctx = s->gpu_ctx;
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)gpu_ctx;
    struct ngpu_pipeline_mtl *s_priv = (struct ngpu_pipeline_mtl *)s;
    id<MTLRenderCommandEncoder> encoder = gpu_ctx_mtl->render_encoder;
    ngli_assert(encoder);

    prepare_draw(s, encoder);

    const struct ngpu_buffer_mtl *index_buffer = (const struct ngpu_buffer_mtl *)gpu_ctx->index_buffer;
    const MTLIndexType index_type = gpu_ctx->index_format == NGPU_FORMAT_R16_UNORM ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
    [encoder drawIndexedPrimitives:s_priv->primitive_type
                        indexCount:nb_indices
                         indexType:index_type
                       indexBuffer:index_buffer->buffer
                 indexBufferOffset:0
                     instanceCount:nb_instances];
}

void ngpu_pipeline_mtl_dispatch(struct ngpu_pipeline *s, uint32_t nb_group_x, uint32_t nb_group_y, uint32_t nb_group_z)
{
    struct ngpu_ctx *gpu_ctx = s->gpu_ctx;
    struct ngpu_pipeline_mtl *s_priv = (struct ngpu_pipeline_mtl *)s;

    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = ngpu_ctx_mtl_begin_compute(gpu_ctx);
        [encoder setComputePipelineState:s_priv->compute_state];
        if (gpu_ctx->bindgroup)
            ngpu_bindgroup_mtl_bind_compute(gpu_ctx->bindgroup, (const struct ngpu_program_mtl *)s->program, encoder,
                                            gpu_ctx->dynamic_offsets, gpu_ctx->nb_dynamic_offsets);
        [encoder dispatchThreadgroups:MTLSizeMake(nb_group_x, nb_group_y, nb_group_z)
                threadsPerThreadgroup:s_priv->threads_per_threadgroup];
        ngpu_ctx_mtl_end_compute(gpu_ctx, encoder);
    }
}

void ngpu_pipeline_mtl_freep(struct ngpu_pipeline **sp)
{
    if (!*sp)
        return;
    struct ngpu_pipeline_mtl *s_priv = (struct ngpu_pipeline_mtl *)*sp;
    [s_priv->render_state release];
    [s_priv->depth_stencil_state release];
    [s_priv->compute_state release];
    ngli_freep(sp);
}
