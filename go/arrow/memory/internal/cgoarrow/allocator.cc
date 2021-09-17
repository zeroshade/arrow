// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// +build cgo
// +build ccalloc

#include <memory>
#include <iostream>
#include "allocator.h"
#include "arrow/memory_pool.h"

namespace {
    template <typename T>
    uintptr_t create_ref(std::shared_ptr<T> t) {
        std::shared_ptr<T>* retained_ptr = new std::shared_ptr<T>(t);
        return reinterpret_cast<uintptr_t>(retained_ptr);
    }

    template <typename T>
    std::shared_ptr<T> retrieve_instance(uintptr_t ref) {
        std::shared_ptr<T>* retrieved_ptr = reinterpret_cast<std::shared_ptr<T>*>(ref);
        return *retrieved_ptr;
    }

    template <typename T>
    void release_ref(uintptr_t ref) {
        std::shared_ptr<T>* retrieved_ptr = reinterpret_cast<std::shared_ptr<T>*>(ref);
        delete retrieved_ptr;
    }
}

struct mem_holder {
    std::unique_ptr<arrow::MemoryPool> pool;
    arrow::MemoryPool* current_pool;
};

ArrowMemoryPool arrow_create_memory_pool(bool enable_logging) {
    auto holder = std::make_shared<mem_holder>();
    holder->pool = arrow::MemoryPool::CreateDefault();    
    if (enable_logging) {
        holder->current_pool = new arrow::LoggingMemoryPool(holder->pool.get());
    } else {
        holder->current_pool = holder->pool.get();
    }

    return create_ref(holder);
}

void arrow_release_pool(ArrowMemoryPool pool) {
    auto holder = retrieve_instance<mem_holder>(pool);
    if (holder->pool.get() != holder->current_pool) {
        delete holder->current_pool;
    }
    release_ref<mem_holder>(pool);
}

int arrow_pool_allocate(ArrowMemoryPool pool, int64_t size, uint8_t** out) {
    auto holder = retrieve_instance<mem_holder>(pool);
    auto status = holder->current_pool->Allocate(size, out);
    if (!status.ok()) {
        return 1;
    }
    return 0;
}

void arrow_pool_free(ArrowMemoryPool pool, uint8_t* buffer, int64_t size) {
    auto holder = retrieve_instance<mem_holder>(pool);
    holder->current_pool->Free(buffer, size);
}

int arrow_pool_reallocate(ArrowMemoryPool pool, int64_t old_size, int64_t new_size, uint8_t** ptr) {
    auto holder = retrieve_instance<mem_holder>(pool);
    auto status = holder->current_pool->Reallocate(old_size, new_size, ptr);
    if (!status.ok()) {
        return 1;
    }
    return 0;
}

int64_t arrow_pool_bytes_allocated(ArrowMemoryPool pool) {
    auto holder = retrieve_instance<mem_holder>(pool);
    return holder->current_pool->bytes_allocated();
}
