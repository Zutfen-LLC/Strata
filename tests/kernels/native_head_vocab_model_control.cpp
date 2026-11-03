// Check the actual GGUF output role using the existing native-order Q8 control.
#include "native_q8_wo_fixture.hpp"
#include "native_head_vocab_layout.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <vector>
int main(int argc,char** argv){try {
    using namespace native_q8_wo_fixture;
    need(argc==9,"head vocabulary control: six shards, packed input and logits required");
    std::vector<std::string> paths;for(int i=1;i<=6;++i)paths.emplace_back(argv[i]);strata::GgufModel model(paths);
    size_t shard=0;const auto* role=model.find("output.weight",&shard);
    need(role&&model.in_bounds(*role,shard)&&role->type==8&&role->shape.size()==2&&role->shape[0]==2560&&role->shape[1]>0&&role->shape[1]<=1048576,"head vocabulary control: unsupported role");
    const size_t rows=size_t(role->shape[1]);const auto* weights=model.shard(shard).tensor_data(*role);
    const auto packed=load(argv[7],head_vocab_packed_bytes);const auto output=load(argv[8],rows*sizeof(float));
    for(size_t row=0;row<rows;++row){const auto expected=reference_blocks(weights,rows*head_vocab_row_bytes,packed.data(),packed.size(),rows,row,80,InputScaleDomain::qkv_finite).value;
        const uint32_t actual=uint32_t(output[row*4])|(uint32_t(output[row*4+1])<<8)|(uint32_t(output[row*4+2])<<16)|(uint32_t(output[row*4+3])<<24);
        if(bits(expected)!=actual)throw std::runtime_error("head vocabulary control: projected byte mismatch at row "+std::to_string(row));
    }
    std::printf("{\"role\":\"output.weight\",\"rows\":%zu,\"packed_bytes\":2880,\"output_bytes\":%zu,\"exact\":true,\"qualified\":false}\n",rows,rows*sizeof(float));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
