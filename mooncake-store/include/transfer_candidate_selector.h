#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "replica.h"

namespace mooncake {

struct TransferCandidateContext {
    std::unordered_set<std::string> local_endpoints;
    bool enable_nvlink_host_numa = false;
    std::string local_scale_up_domain_id;
    bool destination_is_device = false;
};

struct TransferCandidate {
    Replica::Descriptor replica;
    std::string selected_protocol;
};

std::optional<TransferCandidate> SelectTransferCandidate(
    const std::vector<Replica::Descriptor>& replicas,
    const TransferCandidateContext& context);

}  // namespace mooncake
