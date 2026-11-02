// Check each mapped GGUF GDN QKV/z role against the existing native-order host control.
#include "native_q8_wo_fixture.hpp"
#include "native_gdn_input_layout.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <array>
#include <vector>
int main(int argc,char**argv){try{
    using namespace native_q8_wo_fixture;
    need(argc==11,"GDN input model control: six shards, layer, packed input and two outputs required");
    std::vector<std::string> paths;for(int i=1;i<=6;++i)paths.emplace_back(argv[i]);strata::GgufModel model(paths);
    const size_t layer=std::stoul(argv[7]);need(layer<48&&layer%4!=3&&layer!=1,"GDN input model control: unsupported layer");
    const auto packed=load(argv[8],gdn_input_packed_bytes);
    const std::array<std::string,2>names{{"blk."+std::to_string(layer)+".attn_qkv.weight","blk."+std::to_string(layer)+".attn_gate.weight"}};
    const std::array<size_t,2>rows{{10240,6144}};
    for(size_t i=0;i<2;++i){size_t shard=0;const auto*role=model.find(names[i],&shard);
        need(role&&model.in_bounds(*role,shard)&&role->type==8&&role->shape==std::vector<uint64_t>({2560,rows[i]}),"GDN input model control: unsupported role");
        const auto*weights=model.shard(shard).tensor_data(*role);const auto output=load(argv[9+i],rows[i]*4);
        for(size_t row=0;row<rows[i];++row){auto value=reference_blocks(weights,rows[i]*gdn_input_row_bytes,packed.data(),packed.size(),rows[i],row,gdn_input_blocks).value;
            uint32_t actual=uint32_t(output[row*4])|(uint32_t(output[row*4+1])<<8)|(uint32_t(output[row*4+2])<<16)|(uint32_t(output[row*4+3])<<24);
            if(bits(value)!=actual)throw std::runtime_error("GDN input model control: projected byte mismatch for "+std::string(names[i])+" row "+std::to_string(row));
        }
    }
    puts("{\"roles\":2,\"packed_bytes\":2880,\"output_bytes\":65536,\"exact\":true,\"qualified\":false}");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
