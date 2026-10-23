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

#include "buffer_mtl.h"
#include "ctx_mtl.h"
#include "log.h"
#include "nopegl.h"
#include "utils/memory.h"
#include "utils/utils.h"

/*
 * All buffers live in shared memory: Apple GPUs use a unified memory
 * architecture, so a private buffer plus staging copies would only add
 * bandwidth. Uploads and maps are plain CPU accesses; synchronization with the
 * GPU is the caller's responsibility (as with persistently mapped Vulkan
 * buffers), and ngpu_buffer_wait() waits for the submitted work.
 */

struct ngpu_buffer *ngpu_buffer_mtl_create(struct ngpu_ctx *gpu_ctx)
{
    struct ngpu_buffer_mtl *s = ngli_calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->parent.gpu_ctx = gpu_ctx;
    return (struct ngpu_buffer *)s;
}

int ngpu_buffer_mtl_init(struct ngpu_buffer *s)
{
    struct ngpu_ctx_mtl *gpu_ctx_mtl = (struct ngpu_ctx_mtl *)s->gpu_ctx;
    struct ngpu_buffer_mtl *s_priv = (struct ngpu_buffer_mtl *)s;

    /* Metal does not allow zero-length buffers */
    const NSUInteger length = NGLI_MAX(s->size, 1);
    s_priv->buffer = [gpu_ctx_mtl->device newBufferWithLength:length options:MTLResourceStorageModeShared];
    if (!s_priv->buffer) {
        LOG(ERROR, "unable to allocate buffer of %zu bytes", s->size);
        return NGL_ERROR_GRAPHICS_MEMORY;
    }
    return 0;
}

int ngpu_buffer_mtl_wait(struct ngpu_buffer *s)
{
    ngpu_ctx_mtl_wait_submitted(s->gpu_ctx);
    return 0;
}

int ngpu_buffer_mtl_upload(struct ngpu_buffer *s, const void *data, size_t offset, size_t size)
{
    struct ngpu_buffer_mtl *s_priv = (struct ngpu_buffer_mtl *)s;
    ngli_assert(offset + size <= s->size);
    memcpy((uint8_t *)s_priv->buffer.contents + offset, data, size);
    return 0;
}

int ngpu_buffer_mtl_map(struct ngpu_buffer *s, size_t offset, size_t size, void **datap)
{
    struct ngpu_buffer_mtl *s_priv = (struct ngpu_buffer_mtl *)s;
    ngli_assert(offset + size <= s->size);
    *datap = (uint8_t *)s_priv->buffer.contents + offset;
    return 0;
}

void ngpu_buffer_mtl_unmap(struct ngpu_buffer *s)
{
}

void ngpu_buffer_mtl_freep(struct ngpu_buffer **sp)
{
    if (!*sp)
        return;
    struct ngpu_buffer_mtl *s_priv = (struct ngpu_buffer_mtl *)*sp;
    [s_priv->buffer release];
    ngli_freep(sp);
}
