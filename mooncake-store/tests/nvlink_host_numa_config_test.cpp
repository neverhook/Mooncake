#include <gtest/gtest.h>

#include "nvlink_host_numa.h"

namespace mooncake {
namespace {

TEST(NvlinkHostNumaConfigTest, DefaultsToDisabled) {
    ConfigDict config;
    auto options = ParseNvlinkHostNumaOptions(config, 4096);
    ASSERT_TRUE(options);
    EXPECT_FALSE(options->enabled);
    EXPECT_TRUE(options->auto_nodes);
    EXPECT_TRUE(options->nodes.empty());
}

TEST(NvlinkHostNumaConfigTest, ParsesStrictCaseInsensitiveBooleans) {
    for (const std::string& value : {"true", "TRUE", "TrUe", "1"}) {
        ConfigDict config{{CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA, value}};
        auto options = ParseNvlinkHostNumaOptions(config, 4096);
        ASSERT_TRUE(options) << value;
        EXPECT_TRUE(options->enabled) << value;
    }
    for (const std::string& value : {"false", "FALSE", "FaLsE", "0"}) {
        ConfigDict config{{CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA, value}};
        auto options = ParseNvlinkHostNumaOptions(config, 4096);
        ASSERT_TRUE(options) << value;
        EXPECT_FALSE(options->enabled) << value;
    }
    for (const std::string& value : {"", "yes", "2", " true ", "falsex"}) {
        ConfigDict config{{CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA, value}};
        EXPECT_FALSE(ParseNvlinkHostNumaOptions(config, 4096)) << value;
    }
}

TEST(NvlinkHostNumaConfigTest, ParsesAutoAndSortedDeduplicatedCsv) {
    ConfigDict auto_config{{CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA, "true"},
                           {CONFIG_KEY_NVLINK_HOST_NUMA_NODES, " auto "}};
    auto auto_options = ParseNvlinkHostNumaOptions(auto_config, 4096);
    ASSERT_TRUE(auto_options);
    EXPECT_TRUE(auto_options->auto_nodes);

    ConfigDict csv_config{{CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA, "true"},
                          {CONFIG_KEY_NVLINK_HOST_NUMA_NODES, " 4, 2,4, 0 "}};
    auto csv_options = ParseNvlinkHostNumaOptions(csv_config, 4096);
    ASSERT_TRUE(csv_options);
    EXPECT_FALSE(csv_options->auto_nodes);
    EXPECT_EQ(csv_options->nodes, (std::vector<int>{0, 2, 4}));
}

TEST(NvlinkHostNumaConfigTest, RejectsInvalidCsv) {
    for (const std::string& value :
         {"", ",", "1,", ",1", "1,,2", "-1", "1x", "+1", "2147483648", "1 2"}) {
        ConfigDict config{{CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA, "true"},
                          {CONFIG_KEY_NVLINK_HOST_NUMA_NODES, value}};
        EXPECT_FALSE(ParseNvlinkHostNumaOptions(config, 4096)) << value;
    }
}

TEST(NvlinkHostNumaConfigTest, IgnoresNodeExpressionWhenItCannotApply) {
    ConfigDict disabled{{CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA, "false"},
                        {CONFIG_KEY_NVLINK_HOST_NUMA_NODES, "invalid"}};
    auto disabled_options = ParseNvlinkHostNumaOptions(disabled, 4096);
    ASSERT_TRUE(disabled_options);
    EXPECT_FALSE(disabled_options->enabled);

    ConfigDict no_global{{CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA, "true"},
                         {CONFIG_KEY_NVLINK_HOST_NUMA_NODES, "invalid"}};
    auto no_global_options = ParseNvlinkHostNumaOptions(no_global, 0);
    ASSERT_TRUE(no_global_options);
    EXPECT_TRUE(no_global_options->enabled);
    EXPECT_TRUE(no_global_options->auto_nodes);
    EXPECT_TRUE(no_global_options->nodes.empty());
}

TEST(NvlinkHostNumaConfigTest, ExposesStableConfigKeyNames) {
    EXPECT_STREQ(CONFIG_KEY_ENABLE_NVLINK_HOST_NUMA, "enable_nvlink_host_numa");
    EXPECT_STREQ(CONFIG_KEY_NVLINK_HOST_NUMA_NODES, "nvlink_host_numa_nodes");
}

}  // namespace
}  // namespace mooncake
