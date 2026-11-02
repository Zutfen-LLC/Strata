// Check one actual ordinary GDN Q8_0 role using the existing native-order WO control.
#include "native_q8_wo_fixture.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <vector>

int main(int argc, char** argv) {
    try {
        using namespace native_q8_wo_fixture;
        need(argc == 10, "GDN output control: six shards, layer, packed input and output required");
        std::vector<std::string> paths;
        for (int i = 1; i <= 6; ++i) paths.emplace_back(argv[i]);
        size_t parsed = 0;
        const auto layer = std::stoul(argv[7], &parsed);
        need(parsed == std::string(argv[7]).size() && layer < 48 && layer % 4 != 3 && layer != 1,
             "GDN output control: ordinary layer required");
        strata::GgufModel model(paths);
        const auto name = "blk." + std::to_string(layer) + ".ssm_out.weight";
        size_t shard = 0;
        const auto* role = model.find(name, &shard);
        need(role && model.in_bounds(*role, shard) && role->type == 8 &&
             role->shape == std::vector<uint64_t>({6144, 2560}), "GDN output control: unsupported role");
        const auto* weights = model.shard(shard).tensor_data(*role);
        const auto packed = load(argv[8], q8_wo_input_bytes), output = load(argv[9], 2560 * 4);
        for (size_t row = 0; row < 2560; ++row) {
            const auto value = reference(weights, 2560 * q8_wo_row_bytes, packed.data(), packed.size(), 2560, row).value;
            const uint32_t actual = uint32_t(output[row*4]) | (uint32_t(output[row*4+1]) << 8) |
                                    (uint32_t(output[row*4+2]) << 16) | (uint32_t(output[row*4+3]) << 24);
            if (bits(value) != actual) throw std::runtime_error("GDN output control: projected byte mismatch at row " + std::to_string(row));
        }
        std::printf("{\"role\":\"%s\",\"packed_bytes\":6912,\"output_bytes\":10240,\"exact\":true,\"qualified\":false}\n", name.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
