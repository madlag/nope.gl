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

#ifndef NGPU_RENDERTARGET_MTL_H
#define NGPU_RENDERTARGET_MTL_H

#import <Metal/Metal.h>

#include "ngpu/rendertarget.h"

struct ngpu_rendertarget_mtl {
    struct ngpu_rendertarget parent;
};

struct ngpu_rendertarget *ngpu_rendertarget_mtl_create(struct ngpu_ctx *gpu_ctx);
int ngpu_rendertarget_mtl_init(struct ngpu_rendertarget *s);
void ngpu_rendertarget_mtl_freep(struct ngpu_rendertarget **sp);

/*
 * Build the render pass descriptor at the beginning of each pass: attachments
 * may be wrapped textures whose Metal texture changes every frame (drawables).
 */
MTLRenderPassDescriptor *ngpu_rendertarget_mtl_get_descriptor(const struct ngpu_rendertarget *s);

#endif
