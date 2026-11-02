// Read the GGUF role directly and check the existing native-order WO control.
#include "native_q8_wo_fixture.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <vector>
int main(int argc,char**argv){try{
    using namespace native_q8_wo_fixture;
    need(argc==9,"WO model control: six shards, packed input and output required");
    std::vector<std::string> paths;for(int i=1;i<=6;++i)paths.emplace_back(argv[i]);strata::GgufModel model(paths);
    size_t shard=0;const auto* role=model.find("blk.3.attn_output.weight",&shard);
    need(role&&model.in_bounds(*role,shard)&&role->type==8&&role->shape==std::vector<uint64_t>({6144,2560}),"WO model control: unsupported role");
    const auto* weights=model.shard(shard).tensor_data(*role);const auto packed=load(argv[7],q8_wo_input_bytes),output=load(argv[8],2560*4);
    for(size_t row=0;row<2560;++row){auto value=reference(weights,2560*q8_wo_row_bytes,packed.data(),packed.size(),2560,row).value;
        uint32_t actual=uint32_t(output[row*4])|(uint32_t(output[row*4+1])<<8)|(uint32_t(output[row*4+2])<<16)|(uint32_t(output[row*4+3])<<24);
        if(bits(value)!=actual)throw std::runtime_error("WO model control: projected byte mismatch at row "+std::to_string(row));
    }
    puts("{\"role\":\"blk.3.attn_output.weight\",\"packed_bytes\":6912,\"output_bytes\":10240,\"exact\":true,\"qualified\":false}");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
