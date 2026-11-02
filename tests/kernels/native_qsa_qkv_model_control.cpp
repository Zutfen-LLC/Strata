// Check each mapped GGUF Q/K/V role against the existing native-order host control.
#include "native_q8_wo_fixture.hpp"
#include "native_qsa_qkv_layout.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <array>
#include <vector>
int main(int argc,char**argv){try{
    using namespace native_q8_wo_fixture;
    need(argc==11,"QSA QKV model control: six shards, packed input and three outputs required");
    std::vector<std::string> paths;for(int i=1;i<=6;++i)paths.emplace_back(argv[i]);strata::GgufModel model(paths);
    const auto packed=load(argv[7],qsa_qkv_packed_bytes);
    const std::array<const char*,3>names{{"blk.3.attn_q.weight","blk.3.attn_k.weight","blk.3.attn_v.weight"}};
    const std::array<size_t,3>rows{{12288,512,512}};
    for(size_t i=0;i<3;++i){size_t shard=0;const auto*role=model.find(names[i],&shard);
        need(role&&model.in_bounds(*role,shard)&&role->type==8&&role->shape==std::vector<uint64_t>({2560,rows[i]}),"QSA QKV model control: unsupported role");
        const auto*weights=model.shard(shard).tensor_data(*role);const auto output=load(argv[8+i],rows[i]*4);
        for(size_t row=0;row<rows[i];++row){auto value=reference_blocks(weights,rows[i]*qsa_qkv_row_bytes,packed.data(),packed.size(),rows[i],row,qsa_qkv_blocks,InputScaleDomain::qkv_finite).value;
            uint32_t actual=uint32_t(output[row*4])|(uint32_t(output[row*4+1])<<8)|(uint32_t(output[row*4+2])<<16)|(uint32_t(output[row*4+3])<<24);
            if(bits(value)!=actual)throw std::runtime_error("QSA QKV model control: projected byte mismatch for "+std::string(names[i])+" row "+std::to_string(row));
        }
    }
    puts("{\"roles\":3,\"packed_bytes\":2880,\"output_bytes\":53248,\"exact\":true,\"qualified\":false}");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
