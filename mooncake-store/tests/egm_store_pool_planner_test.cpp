#include "egm_store_pool.h"

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace mooncake {
namespace {

class FakeEnvironment final : public EgmStorePoolEnvironment {
   public:
    EgmStorePoolResult<std::vector<int>> VisibleCudaDevices() const override {
        return devices;
    }

    EgmStorePoolResult<std::string> PciBdfForCudaDevice(
        int device_id) const override {
        auto it = device_pci.find(device_id);
        if (it == device_pci.end()) {
            return tl::make_unexpected("missing device");
        }
        return it->second;
    }

    EgmStorePoolResult<int> ReadPciNumaNode(
        const std::string& pci_bdf) const override {
        auto it = pci_node.find(pci_bdf);
        if (it == pci_node.end()) {
            return tl::make_unexpected("missing PCI device");
        }
        return it->second;
    }

    bool IsNumaNodeOnline(int node_id) const override {
        return online_nodes.count(node_id) != 0;
    }

    std::vector<int> devices;
    std::map<int, std::string> device_pci;
    std::map<std::string, int> pci_node;
    std::set<int> online_nodes;
};

TEST(EgmStorePoolPlannerTest, AutoDiscoveryDeduplicatesAndSortsNodes) {
    FakeEnvironment environment;
    environment.devices = {2, 0, 1};
    environment.device_pci = {
        {0, "0000:01:00.0"}, {1, "0000:02:00.0"}, {2, "0000:03:00.0"}};
    environment.pci_node = {
        {"0000:01:00.0", 4}, {"0000:02:00.0", 0}, {"0000:03:00.0", 4}};
    environment.online_nodes = {0, 4};

    EgmStorePoolOptions options;
    options.enabled = true;
    auto nodes = DiscoverEgmStorePoolNodes(options, 4096, environment);
    ASSERT_TRUE(nodes);
    EXPECT_EQ(*nodes, (std::vector<int>{0, 4}));
}

TEST(EgmStorePoolPlannerTest, ExplicitNodesBypassCudaDiscovery) {
    FakeEnvironment environment;
    environment.online_nodes = {1, 3};
    EgmStorePoolOptions options;
    options.enabled = true;
    options.auto_nodes = false;
    options.nodes = {3, 1, 3};

    auto nodes = DiscoverEgmStorePoolNodes(options, 4096, environment);
    ASSERT_TRUE(nodes);
    EXPECT_EQ(*nodes, (std::vector<int>{1, 3}));
}

TEST(EgmStorePoolPlannerTest, BalancesAndSplitsAtAlignedMaxMrSize) {
    auto plan =
        PlanEgmStorePoolCapacity(16 * 12, {{4, 3}, {0, 4}}, 5 * 12 + 1, 4);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->common_alignment, 12);
    EXPECT_EQ(plan->effective_total, 16 * 12);
    ASSERT_EQ(plan->nodes.size(), 2);
    EXPECT_EQ(plan->nodes[0].node_id, 0);
    EXPECT_EQ(plan->nodes[1].node_id, 4);
    EXPECT_EQ(plan->nodes[0].effective_bytes, 8 * 12);
    EXPECT_EQ(plan->nodes[1].effective_bytes, 8 * 12);
    ASSERT_EQ(plan->chunks.size(), 4);
    for (size_t index = 0; index < plan->chunks.size(); ++index) {
        EXPECT_EQ(plan->chunks[index].plan_index, index);
        EXPECT_EQ(plan->chunks[index].chunk_bytes % plan->common_alignment, 0);
        EXPECT_LE(plan->chunks[index].chunk_bytes, 5 * 12);
    }
}

TEST(EgmStorePoolPlannerTest, RejectsInvalidCapacityInputs) {
    EXPECT_FALSE(PlanEgmStorePoolCapacity(0, {{0, 4096}}, 4096, 4096));
    EXPECT_FALSE(PlanEgmStorePoolCapacity(4096, {}, 4096, 4096));
    EXPECT_FALSE(
        PlanEgmStorePoolCapacity(4096, {{0, 4096}, {0, 4096}}, 4096, 4096));
    EXPECT_FALSE(PlanEgmStorePoolCapacity(4096, {{0, 4096}}, 2048, 4096));
}

}  // namespace
}  // namespace mooncake
