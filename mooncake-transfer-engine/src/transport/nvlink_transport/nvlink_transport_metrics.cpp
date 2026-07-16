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

#include "transport/nvlink_transport/nvlink_transport.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace mooncake {

struct NvlinkTransport::ConsumerMetrics {
    std::array<std::atomic<uint64_t>, 2> mapping_cache_total{};
    std::atomic<uint64_t> lazy_import_duration_us_total{0};
    std::atomic<uint64_t> lazy_import_observations_total{0};
    std::array<std::atomic<uint64_t>, 5> failures_total{};
    std::array<std::atomic<uint64_t>, 4> transfer_results_total{};
};

void NvlinkTransport::ConsumerMetricsDeleter::operator()(
    ConsumerMetrics* metrics) const {
    delete metrics;
}

NvlinkTransport::ConsumerMetricsPtr NvlinkTransport::createConsumerMetrics() {
    return ConsumerMetricsPtr(new ConsumerMetrics());
}

void NvlinkTransport::observeCacheLookup(bool hit) {
    consumer_metrics_->mapping_cache_total[hit ? 0 : 1].fetch_add(
        1, std::memory_order_relaxed);
}

void NvlinkTransport::observeLazyImportLatency(uint64_t duration_us) {
    consumer_metrics_->lazy_import_duration_us_total.fetch_add(
        std::max<uint64_t>(duration_us, 1), std::memory_order_relaxed);
    consumer_metrics_->lazy_import_observations_total.fetch_add(
        1, std::memory_order_relaxed);
}

void NvlinkTransport::observeConsumerFailure(ConsumerFailureStage stage) {
    consumer_metrics_->failures_total[static_cast<size_t>(stage)].fetch_add(
        1, std::memory_order_relaxed);
}

void NvlinkTransport::observeTransferResult(TransferRequest::OpCode operation,
                                            bool success) {
    const size_t operation_offset = operation == TransferRequest::READ ? 0 : 2;
    consumer_metrics_
        ->transfer_results_total[operation_offset + (success ? 0 : 1)]
        .fetch_add(1, std::memory_order_relaxed);
}

bool NvlinkTransport::observeTransferResultOnce(TransferTask& task,
                                                bool success) {
    if (!task.operation_initialized) return false;
    if (__atomic_exchange_n(&task.transport_result_observed, true,
                            __ATOMIC_ACQ_REL)) {
        return false;
    }
    observeTransferResult(task.operation, success);
    return true;
}

void NvlinkTransport::finalizeTransferResult(TransferTask& task, bool success) {
    if (observeTransferResultOnce(task, success) && !success) {
        observeConsumerFailure(ConsumerFailureStage::COPY);
    }
}

void NvlinkTransport::finalizeSubmissionFailure(TransferTask& task,
                                                bool copy_failure) {
    const bool first_result = observeTransferResultOnce(task, false);
    if (copy_failure && first_result) {
        observeConsumerFailure(ConsumerFailureStage::COPY);
    }
    // Publish task/batch completion only after its result and exact failure
    // category are stable. Otherwise the event-driven batch fast path could
    // win the result CAS and misclassify a non-copy failure as copy.
    markSubmissionFailed(task);
}

void NvlinkTransport::appendMetrics(std::string& output) {
    if (!output.empty() && output.back() != '\n') output.push_back('\n');
    auto append_header = [&output](const char* name, const char* help) {
        output.append("# HELP ").append(name).append(" ").append(help).append(
            "\n# TYPE ");
        output.append(name).append(" counter\n");
    };
    auto append_sample = [&output](const char* name, const char* labels,
                                   uint64_t value) {
        output.append(name);
        if (labels != nullptr && labels[0] != '\0') {
            output.append("{").append(labels).append("}");
        }
        output.append(" ").append(std::to_string(value)).append("\n");
    };

    constexpr const char* cache_name =
        "mooncake_nvlink_consumer_mapping_cache_total";
    append_header(cache_name, "NVLink Consumer mapping cache lookups");
    append_sample(cache_name, "result=\"hit\"",
                  consumer_metrics_->mapping_cache_total[0].load(
                      std::memory_order_relaxed));
    append_sample(cache_name, "result=\"miss\"",
                  consumer_metrics_->mapping_cache_total[1].load(
                      std::memory_order_relaxed));

    constexpr const char* duration_name =
        "mooncake_nvlink_consumer_lazy_import_duration_us_total";
    append_header(duration_name,
                  "Cumulative NVLink Consumer lazy import duration in "
                  "microseconds");
    append_sample(duration_name, nullptr,
                  consumer_metrics_->lazy_import_duration_us_total.load(
                      std::memory_order_relaxed));
    constexpr const char* observations_name =
        "mooncake_nvlink_consumer_lazy_import_observations_total";
    append_header(observations_name,
                  "Observed NVLink Consumer lazy import attempts");
    append_sample(observations_name, nullptr,
                  consumer_metrics_->lazy_import_observations_total.load(
                      std::memory_order_relaxed));

    constexpr const char* failure_name =
        "mooncake_nvlink_consumer_failures_total";
    constexpr std::array<const char*, 5> failure_labels = {
        "stage=\"import\"", "stage=\"reserve\"", "stage=\"map\"",
        "stage=\"set_access\"", "stage=\"copy\""};
    append_header(failure_name, "NVLink Consumer failures by stage");
    for (size_t index = 0; index < failure_labels.size(); ++index) {
        append_sample(failure_name, failure_labels[index],
                      consumer_metrics_->failures_total[index].load(
                          std::memory_order_relaxed));
    }

    constexpr const char* transfer_name =
        "mooncake_nvlink_consumer_transfer_results_total";
    constexpr std::array<const char*, 4> transfer_labels = {
        "operation=\"read\",result=\"success\"",
        "operation=\"read\",result=\"failure\"",
        "operation=\"write\",result=\"success\"",
        "operation=\"write\",result=\"failure\""};
    append_header(transfer_name, "NVLink Consumer transfer results");
    for (size_t index = 0; index < transfer_labels.size(); ++index) {
        append_sample(transfer_name, transfer_labels[index],
                      consumer_metrics_->transfer_results_total[index].load(
                          std::memory_order_relaxed));
    }
}

}  // namespace mooncake
