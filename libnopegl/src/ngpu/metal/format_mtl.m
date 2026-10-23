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

#include "format_mtl.h"
#include "utils/utils.h"

/*
 * Metal has no 3-component pixel formats and no packed 24-bit depth on Apple
 * GPUs: those entries stay MTLPixelFormatInvalid and report no feature, so
 * nope.gl falls back on supported formats.
 */
static const MTLPixelFormat pixel_format_map[NGPU_FORMAT_NB] = {
    [NGPU_FORMAT_R8_UNORM]             = MTLPixelFormatR8Unorm,
    [NGPU_FORMAT_R8_SNORM]             = MTLPixelFormatR8Snorm,
    [NGPU_FORMAT_R8_UINT]              = MTLPixelFormatR8Uint,
    [NGPU_FORMAT_R8_SINT]              = MTLPixelFormatR8Sint,
    [NGPU_FORMAT_R8_SRGB]              = MTLPixelFormatR8Unorm_sRGB,
    [NGPU_FORMAT_R8G8_UNORM]           = MTLPixelFormatRG8Unorm,
    [NGPU_FORMAT_R8G8_SNORM]           = MTLPixelFormatRG8Snorm,
    [NGPU_FORMAT_R8G8_UINT]            = MTLPixelFormatRG8Uint,
    [NGPU_FORMAT_R8G8_SINT]            = MTLPixelFormatRG8Sint,
    [NGPU_FORMAT_R8G8_SRGB]            = MTLPixelFormatRG8Unorm_sRGB,
    [NGPU_FORMAT_R8G8B8A8_UNORM]       = MTLPixelFormatRGBA8Unorm,
    [NGPU_FORMAT_R8G8B8A8_SNORM]       = MTLPixelFormatRGBA8Snorm,
    [NGPU_FORMAT_R8G8B8A8_UINT]        = MTLPixelFormatRGBA8Uint,
    [NGPU_FORMAT_R8G8B8A8_SINT]        = MTLPixelFormatRGBA8Sint,
    [NGPU_FORMAT_R8G8B8A8_SRGB]        = MTLPixelFormatRGBA8Unorm_sRGB,
    [NGPU_FORMAT_B8G8R8A8_UNORM]       = MTLPixelFormatBGRA8Unorm,
    [NGPU_FORMAT_B8G8R8A8_SRGB]        = MTLPixelFormatBGRA8Unorm_sRGB,
    [NGPU_FORMAT_R16_UNORM]            = MTLPixelFormatR16Unorm,
    [NGPU_FORMAT_R16_SNORM]            = MTLPixelFormatR16Snorm,
    [NGPU_FORMAT_R16_UINT]             = MTLPixelFormatR16Uint,
    [NGPU_FORMAT_R16_SINT]             = MTLPixelFormatR16Sint,
    [NGPU_FORMAT_R16_SFLOAT]           = MTLPixelFormatR16Float,
    [NGPU_FORMAT_R16G16_UNORM]         = MTLPixelFormatRG16Unorm,
    [NGPU_FORMAT_R16G16_SNORM]         = MTLPixelFormatRG16Snorm,
    [NGPU_FORMAT_R16G16_UINT]          = MTLPixelFormatRG16Uint,
    [NGPU_FORMAT_R16G16_SINT]          = MTLPixelFormatRG16Sint,
    [NGPU_FORMAT_R16G16_SFLOAT]        = MTLPixelFormatRG16Float,
    [NGPU_FORMAT_R16G16B16A16_UNORM]   = MTLPixelFormatRGBA16Unorm,
    [NGPU_FORMAT_R16G16B16A16_SNORM]   = MTLPixelFormatRGBA16Snorm,
    [NGPU_FORMAT_R16G16B16A16_UINT]    = MTLPixelFormatRGBA16Uint,
    [NGPU_FORMAT_R16G16B16A16_SINT]    = MTLPixelFormatRGBA16Sint,
    [NGPU_FORMAT_R16G16B16A16_SFLOAT]  = MTLPixelFormatRGBA16Float,
    [NGPU_FORMAT_R32_UINT]             = MTLPixelFormatR32Uint,
    [NGPU_FORMAT_R32_SINT]             = MTLPixelFormatR32Sint,
    [NGPU_FORMAT_R32_SFLOAT]           = MTLPixelFormatR32Float,
    [NGPU_FORMAT_R32G32_UINT]          = MTLPixelFormatRG32Uint,
    [NGPU_FORMAT_R32G32_SINT]          = MTLPixelFormatRG32Sint,
    [NGPU_FORMAT_R32G32_SFLOAT]        = MTLPixelFormatRG32Float,
    [NGPU_FORMAT_R32G32B32A32_UINT]    = MTLPixelFormatRGBA32Uint,
    [NGPU_FORMAT_R32G32B32A32_SINT]    = MTLPixelFormatRGBA32Sint,
    [NGPU_FORMAT_R32G32B32A32_SFLOAT]  = MTLPixelFormatRGBA32Float,
    [NGPU_FORMAT_D16_UNORM]            = MTLPixelFormatDepth16Unorm,
    [NGPU_FORMAT_D32_SFLOAT]           = MTLPixelFormatDepth32Float,
    [NGPU_FORMAT_D32_SFLOAT_S8_UINT]   = MTLPixelFormatDepth32Float_Stencil8,
    [NGPU_FORMAT_S8_UINT]              = MTLPixelFormatStencil8,
};

MTLPixelFormat ngpu_format_ngl_to_mtl(enum ngpu_format format)
{
    ngli_assert(format >= 0 && format < NGPU_FORMAT_NB);
    return pixel_format_map[format];
}

static const MTLVertexFormat vertex_format_map[NGPU_FORMAT_NB] = {
    [NGPU_FORMAT_R8_UNORM]            = MTLVertexFormatUCharNormalized,
    [NGPU_FORMAT_R8_SNORM]            = MTLVertexFormatCharNormalized,
    [NGPU_FORMAT_R8_UINT]             = MTLVertexFormatUChar,
    [NGPU_FORMAT_R8_SINT]             = MTLVertexFormatChar,
    [NGPU_FORMAT_R8G8_UNORM]          = MTLVertexFormatUChar2Normalized,
    [NGPU_FORMAT_R8G8_SNORM]          = MTLVertexFormatChar2Normalized,
    [NGPU_FORMAT_R8G8_UINT]           = MTLVertexFormatUChar2,
    [NGPU_FORMAT_R8G8_SINT]           = MTLVertexFormatChar2,
    [NGPU_FORMAT_R8G8B8_UNORM]        = MTLVertexFormatUChar3Normalized,
    [NGPU_FORMAT_R8G8B8_SNORM]        = MTLVertexFormatChar3Normalized,
    [NGPU_FORMAT_R8G8B8_UINT]         = MTLVertexFormatUChar3,
    [NGPU_FORMAT_R8G8B8_SINT]         = MTLVertexFormatChar3,
    [NGPU_FORMAT_R8G8B8A8_UNORM]      = MTLVertexFormatUChar4Normalized,
    [NGPU_FORMAT_R8G8B8A8_SNORM]      = MTLVertexFormatChar4Normalized,
    [NGPU_FORMAT_R8G8B8A8_UINT]       = MTLVertexFormatUChar4,
    [NGPU_FORMAT_R8G8B8A8_SINT]       = MTLVertexFormatChar4,
    [NGPU_FORMAT_R16_UNORM]           = MTLVertexFormatUShortNormalized,
    [NGPU_FORMAT_R16_SNORM]           = MTLVertexFormatShortNormalized,
    [NGPU_FORMAT_R16_UINT]            = MTLVertexFormatUShort,
    [NGPU_FORMAT_R16_SINT]            = MTLVertexFormatShort,
    [NGPU_FORMAT_R16_SFLOAT]          = MTLVertexFormatHalf,
    [NGPU_FORMAT_R16G16_UNORM]        = MTLVertexFormatUShort2Normalized,
    [NGPU_FORMAT_R16G16_SNORM]        = MTLVertexFormatShort2Normalized,
    [NGPU_FORMAT_R16G16_UINT]         = MTLVertexFormatUShort2,
    [NGPU_FORMAT_R16G16_SINT]         = MTLVertexFormatShort2,
    [NGPU_FORMAT_R16G16_SFLOAT]       = MTLVertexFormatHalf2,
    [NGPU_FORMAT_R16G16B16_UNORM]     = MTLVertexFormatUShort3Normalized,
    [NGPU_FORMAT_R16G16B16_SNORM]     = MTLVertexFormatShort3Normalized,
    [NGPU_FORMAT_R16G16B16_UINT]      = MTLVertexFormatUShort3,
    [NGPU_FORMAT_R16G16B16_SINT]      = MTLVertexFormatShort3,
    [NGPU_FORMAT_R16G16B16_SFLOAT]    = MTLVertexFormatHalf3,
    [NGPU_FORMAT_R16G16B16A16_UNORM]  = MTLVertexFormatUShort4Normalized,
    [NGPU_FORMAT_R16G16B16A16_SNORM]  = MTLVertexFormatShort4Normalized,
    [NGPU_FORMAT_R16G16B16A16_UINT]   = MTLVertexFormatUShort4,
    [NGPU_FORMAT_R16G16B16A16_SINT]   = MTLVertexFormatShort4,
    [NGPU_FORMAT_R16G16B16A16_SFLOAT] = MTLVertexFormatHalf4,
    [NGPU_FORMAT_R32_UINT]            = MTLVertexFormatUInt,
    [NGPU_FORMAT_R32_SINT]            = MTLVertexFormatInt,
    [NGPU_FORMAT_R32_SFLOAT]          = MTLVertexFormatFloat,
    [NGPU_FORMAT_R32G32_UINT]         = MTLVertexFormatUInt2,
    [NGPU_FORMAT_R32G32_SINT]         = MTLVertexFormatInt2,
    [NGPU_FORMAT_R32G32_SFLOAT]       = MTLVertexFormatFloat2,
    [NGPU_FORMAT_R32G32B32_UINT]      = MTLVertexFormatUInt3,
    [NGPU_FORMAT_R32G32B32_SINT]      = MTLVertexFormatInt3,
    [NGPU_FORMAT_R32G32B32_SFLOAT]    = MTLVertexFormatFloat3,
    [NGPU_FORMAT_R32G32B32A32_UINT]   = MTLVertexFormatUInt4,
    [NGPU_FORMAT_R32G32B32A32_SINT]   = MTLVertexFormatInt4,
    [NGPU_FORMAT_R32G32B32A32_SFLOAT] = MTLVertexFormatFloat4,
};

MTLVertexFormat ngpu_format_ngl_to_mtl_vertex(enum ngpu_format format)
{
    ngli_assert(format >= 0 && format < NGPU_FORMAT_NB);
    return vertex_format_map[format];
}

static int is_integer_format(MTLPixelFormat format)
{
    switch (format) {
    case MTLPixelFormatR8Uint:     case MTLPixelFormatR8Sint:
    case MTLPixelFormatRG8Uint:    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatRGBA8Uint:  case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatR16Uint:    case MTLPixelFormatR16Sint:
    case MTLPixelFormatRG16Uint:   case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatR32Uint:    case MTLPixelFormatR32Sint:
    case MTLPixelFormatRG32Uint:   case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGBA32Sint:
        return 1;
    default:
        return 0;
    }
}

static int is_snorm_format(MTLPixelFormat format)
{
    switch (format) {
    case MTLPixelFormatR8Snorm:  case MTLPixelFormatRG8Snorm:  case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatR16Snorm: case MTLPixelFormatRG16Snorm: case MTLPixelFormatRGBA16Snorm:
        return 1;
    default:
        return 0;
    }
}

/*
 * Feature table derived from the "Metal Feature Set Tables" for Apple family
 * GPUs (Apple7+: A14/M1 and later), the minimum nope.gl targets on Apple
 * hardware.
 */
uint32_t ngpu_format_mtl_get_features(id<MTLDevice> device, enum ngpu_format format)
{
    const MTLPixelFormat mtl_format = ngpu_format_ngl_to_mtl(format);
    if (mtl_format == MTLPixelFormatInvalid)
        return 0;

    if (ngpu_format_has_depth(format) || ngpu_format_has_stencil(format)) {
        uint32_t features = NGPU_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                            NGPU_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if (mtl_format == MTLPixelFormatDepth16Unorm)
            features |= NGPU_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        return features;
    }

    uint32_t features = NGPU_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                        NGPU_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;

    const int is_integer = is_integer_format(mtl_format);
    const int is_32bit_float = mtl_format == MTLPixelFormatR32Float  ||
                               mtl_format == MTLPixelFormatRG32Float ||
                               mtl_format == MTLPixelFormatRGBA32Float;
    if (!is_integer && (!is_32bit_float || device.supports32BitFloatFiltering))
        features |= NGPU_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if (!is_integer && !is_snorm_format(mtl_format))
        features |= NGPU_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;

    return features;
}
