// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transport/transport.h"

#include "error.h"
#include "transfer_engine.h"

namespace mooncake {
thread_local static Transport::ThreadLocalSliceCache tl_slice_cache;

void Transport::markSubmissionFailed(TransferTask &task) {
    if (__atomic_exchange_n(&task.submission_failed, true, __ATOMIC_ACQ_REL)) {
        return;
    }
    __atomic_store_n(&task.is_finished, true, __ATOMIC_RELEASE);

    if (task.batch_id == 0) return;
    auto &batch_desc = toBatchDesc(task.batch_id);
    batch_desc.has_failure.store(true, std::memory_order_release);
#ifdef USE_EVENT_DRIVEN_COMPLETION
    auto previous =
        batch_desc.finished_task_count.fetch_add(1, std::memory_order_relaxed);
    if (previous + 1 == batch_desc.batch_size) {
        {
            std::lock_guard<std::mutex> lock(batch_desc.completion_mutex);
            batch_desc.is_finished.store(true, std::memory_order_release);
        }
        batch_desc.completion_cv.notify_all();
    }
#endif
}

Transport::ThreadLocalSliceCache &Transport::getSliceCache() {
    return tl_slice_cache;
}

Transport::BatchID Transport::allocateBatchID(size_t batch_size) {
    auto batch_desc = new BatchDesc();
    if (!batch_desc) return ERR_MEMORY;
    batch_desc->id = BatchID(batch_desc);
    batch_desc->batch_size = batch_size;
    batch_desc->task_list.reserve(batch_size);
    batch_desc->context = NULL;
#ifdef CONFIG_USE_BATCH_DESC_SET
    batch_desc_lock_.lock();
    batch_desc_set_[batch_desc->id] = batch_desc;
    batch_desc_lock_.unlock();
#endif
    return batch_desc->id;
}

Status Transport::freeBatchID(BatchID batch_id) {
    auto &batch_desc = *((BatchDesc *)(batch_id));
    const size_t task_count = batch_desc.task_list.size();
    for (size_t task_id = 0; task_id < task_count; task_id++) {
        if (!batch_desc.task_list[task_id].is_finished) {
            LOG(ERROR) << "BatchID cannot be freed until all tasks are done";
            return Status::BatchBusy(
                "BatchID cannot be freed until all tasks are done");
        }
    }
    delete &batch_desc;
#ifdef CONFIG_USE_BATCH_DESC_SET
    RWSpinlock::WriteGuard guard(batch_desc_lock_);
    batch_desc_set_.erase(batch_id);
#endif
    return Status::OK();
}

int Transport::install(std::string &local_server_name,
                       std::shared_ptr<TransferMetadata> meta,
                       std::shared_ptr<Topology> topo) {
    local_server_name_ = local_server_name;
    metadata_ = meta;
    return 0;
}
}  // namespace mooncake
