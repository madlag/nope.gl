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

#ifndef NGPU_TEXTURE_MTL_H
#define NGPU_TEXTURE_MTL_H

#import <Metal/Metal.h>

#include "ngpu/texture.h"

struct ngpu_texture_mtl {
    struct ngpu_texture parent;
    id<MTLTexture> texture;
    id<MTLSamplerState> sampler;
    uint32_t array_layers;
    uint32_t mipmap_levels;
    size_t bytes_per_pixel;
    int wrapped;
};

struct ngpu_texture *ngpu_texture_mtl_create(struct ngpu_ctx *gpu_ctx);
int ngpu_texture_mtl_init(struct ngpu_texture *s, const struct ngpu_texture_params *params);

/*
 * Wrap an existing Metal texture (e.g. a drawable or a CoreVideo backed
 * texture). The texture can be replaced later with ngpu_texture_mtl_set_texture().
 */
int ngpu_texture_mtl_wrap(struct ngpu_texture *s, const struct ngpu_texture_params *params, id<MTLTexture> texture);
void ngpu_texture_mtl_set_texture(struct ngpu_texture *s, id<MTLTexture> texture);

int ngpu_texture_mtl_upload(struct ngpu_texture *s, const uint8_t *data, uint32_t linesize);
int ngpu_texture_mtl_upload_with_params(struct ngpu_texture *s, const uint8_t *data,
                                        const struct ngpu_texture_transfer_params *transfer_params);
int ngpu_texture_mtl_generate_mipmap(struct ngpu_texture *s);
void ngpu_texture_mtl_copy_to_buffer(struct ngpu_texture *s, id<MTLBuffer> buffer);
void ngpu_texture_mtl_freep(struct ngpu_texture **sp);

#endif
