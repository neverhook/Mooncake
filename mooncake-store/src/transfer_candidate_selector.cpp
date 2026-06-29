#include "transfer_candidate_selector.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

namespace mooncake {
namespace {

bool IsComplete(const Replica::Descriptor& replica) {
    return replica.status == ReplicaStatus::COMPLETE;
}

std::vector<std::string> SplitProtocols(const std::string& protocols) {
    std::vector<std::string> result;
    std::stringstream ss(protocols);
    std::string protocol;
    while (std::getline(ss, protocol, ',')) {
        protocol.erase(protocol.begin(),
                       std::find_if(protocol.begin(), protocol.end(),
                                    [](unsigned char ch) {
                                        return !std::isspace(ch);
                                    }));
        protocol.erase(std::find_if(protocol.rbegin(), protocol.rend(),
                                    [](unsigned char ch) {
                                        return !std::isspace(ch);
                                    }).base(),
                       protocol.end());
        std::transform(protocol.begin(), protocol.end(), protocol.begin(),
                       [](unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                       });
        if (!protocol.empty()) {
            result.emplace_back(std::move(protocol));
        }
    }
    return result;
}

bool AdvertisesProtocol(const AllocatedBuffer::Descriptor& descriptor,
                        const std::string& protocol) {
    const auto protocols = SplitProtocols(descriptor.protocol_);
    return std::find(protocols.begin(), protocols.end(), protocol) !=
           protocols.end();
}

bool IsMultiProtocol(const AllocatedBuffer::Descriptor& descriptor) {
    return SplitProtocols(descriptor.protocol_).size() > 1;
}

std::string SelectedProtocolFor(const AllocatedBuffer::Descriptor& descriptor,
                                const std::string& protocol) {
    return IsMultiProtocol(descriptor) ? protocol : "";
}

bool IsLocalMemoryReplica(const Replica::Descriptor& replica,
                          const TransferCandidateContext& context) {
    if (!replica.is_memory_replica()) {
        return false;
    }
    const auto& buffer =
        replica.get_memory_descriptor().buffer_descriptor;
    return context.local_endpoints.count(buffer.transport_endpoint_) > 0;
}

bool IsNvlinkHostNumaCandidate(const Replica::Descriptor& replica,
                               const TransferCandidateContext& context) {
    if (!replica.is_memory_replica() ||
        !context.enable_nvlink_host_numa ||
        !context.destination_is_device) {
        return false;
    }
    const auto& buffer =
        replica.get_memory_descriptor().buffer_descriptor;
    return AdvertisesProtocol(buffer, "nvlink") &&
           buffer.memory_kind_ == "HOST_NUMA" &&
           buffer.scale_up_domain_id_ == context.local_scale_up_domain_id;
}

bool IsRdmaCandidate(const Replica::Descriptor& replica) {
    if (!replica.is_memory_replica()) {
        return false;
    }
    return AdvertisesProtocol(replica.get_memory_descriptor().buffer_descriptor,
                              "rdma");
}

bool IsLegacyMemoryCandidate(const Replica::Descriptor& replica) {
    if (!replica.is_memory_replica()) {
        return false;
    }
    const auto& buffer = replica.get_memory_descriptor().buffer_descriptor;
    return !AdvertisesProtocol(buffer, "nvlink") &&
           !AdvertisesProtocol(buffer, "rdma");
}

TransferCandidate MemoryCandidate(const Replica::Descriptor& replica,
                                  const std::string& selected_protocol) {
    return TransferCandidate{replica, selected_protocol};
}

}  // namespace

std::optional<TransferCandidate> SelectTransferCandidate(
    const std::vector<Replica::Descriptor>& replicas,
    const TransferCandidateContext& context) {
    const Replica::Descriptor* first_memory = nullptr;
    const Replica::Descriptor* first_nof = nullptr;

    for (const auto& replica : replicas) {
        if (!IsComplete(replica)) {
            continue;
        }
        if (IsLocalMemoryReplica(replica, context)) {
            return MemoryCandidate(replica, "");
        }
        if (IsLegacyMemoryCandidate(replica) && first_memory == nullptr) {
            first_memory = &replica;
        } else if (replica.is_nof_replica()) {
            if (first_nof == nullptr) {
                first_nof = &replica;
            }
        }
    }

    for (const auto& replica : replicas) {
        if (!IsComplete(replica) ||
            !IsNvlinkHostNumaCandidate(replica, context)) {
            continue;
        }
        const auto& buffer =
            replica.get_memory_descriptor().buffer_descriptor;
        return MemoryCandidate(replica, SelectedProtocolFor(buffer, "nvlink"));
    }

    for (const auto& replica : replicas) {
        if (!IsComplete(replica) || !IsRdmaCandidate(replica)) {
            continue;
        }
        const auto& buffer =
            replica.get_memory_descriptor().buffer_descriptor;
        return MemoryCandidate(replica, SelectedProtocolFor(buffer, "rdma"));
    }

    if (first_memory != nullptr) {
        return MemoryCandidate(*first_memory, "");
    }
    if (first_nof != nullptr) {
        return TransferCandidate{*first_nof, ""};
    }

    const Replica::Descriptor* best_disk = nullptr;
    for (const auto& replica : replicas) {
        if (!IsComplete(replica)) {
            continue;
        }
        if (replica.is_local_disk_replica()) {
            best_disk = &replica;
        } else if (replica.is_disk_replica() && best_disk == nullptr) {
            best_disk = &replica;
        }
    }
    if (best_disk != nullptr) {
        return TransferCandidate{*best_disk, ""};
    }
    return std::nullopt;
}

}  // namespace mooncake
