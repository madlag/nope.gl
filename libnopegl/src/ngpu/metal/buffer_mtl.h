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

#ifndef NGPU_BUFFER_MTL_H
#define NGPU_BUFFER_MTL_H

#import <Metal/Metal.h>

#include "ngpu/buffer.h"

struct ngpu_buffer_mtl {
    struct ngpu_buffer parent;
    id<MTLBuffer> buffer;
};

struct ngpu_buffer *ngpu_buffer_mtl_create(struct ngpu_ctx *gpu_ctx);
int ngpu_buffer_mtl_init(struct ngpu_buffer *s);
int ngpu_buffer_mtl_wait(struct ngpu_buffer *s);
int ngpu_buffer_mtl_upload(struct ngpu_buffer *s, const void *data, size_t offset, size_t size);
int ngpu_buffer_mtl_map(struct ngpu_buffer *s, size_t offset, size_t size, void **datap);
void ngpu_buffer_mtl_unmap(struct ngpu_buffer *s);
void ngpu_buffer_mtl_freep(struct ngpu_buffer **sp);

#endif
