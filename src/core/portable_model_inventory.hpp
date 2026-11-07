#pragma once
// Shared portable-model inventory, including unused MTP extent/overlap checks.
#include "portable_graph.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <set>
#include <string>
#include <vector>
namespace strata::core::portable_detail {
inline void portable_model_inventory(const GgufModel& m) {
    const std::set<std::string> globals{"token_embd.weight", "output.weight", "output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", "per_layer_token_embd.weight"};
    const std::set<std::string> dense{"hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight", "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    const std::set<std::string> gdn{"attn_qkv.weight", "attn_gate.weight", "ssm_a", "ssm_dt.bias", "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight"};
    const std::set<std::string> qsa{"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"};
    const std::set<std::string> ple{"ple_key.weight", "ple_value.weight", "ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight"};
    for(size_t s=0;s<m.size();++s){
        std::vector<std::pair<uint64_t,uint64_t>> ranges;
        for(const auto& t:m.shard(s).tensors()){
            require(m.in_bounds(t,s),"portable model: unknown encoding/truncated tensor "+t.name);
            ranges.emplace_back(t.offset,t.offset+tensor_payload_bytes(t));
            if(globals.count(t.name))continue;
            if(t.name.rfind("nextn.",0)==0||t.name.rfind("mtp.",0)==0||t.name.rfind("blk.48.",0)==0)continue;
            require(t.name.rfind("blk.",0)==0,"portable model: unsupported global tensor "+t.name);
            const size_t end=t.name.find('.',4);require(end!=std::string::npos,"portable model: malformed block tensor");
            const auto number=t.name.substr(4,end-4);const int l=std::stoi(number);
            require(l>=0&&l<48&&std::to_string(l)==number,"portable model: invalid block index");
            const auto key=t.name.substr(end+1);
            require(dense.count(key)||(l%4==3?qsa.count(key):gdn.count(key))||(l==1&&ple.count(key)),"portable model: unsupported block tensor "+t.name);
        }
        std::sort(ranges.begin(),ranges.end());
        for(size_t i=1;i<ranges.size();++i)require(ranges[i-1].second<=ranges[i].first,"portable model: overlapping tensor payloads");
    }
}
}
