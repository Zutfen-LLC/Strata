#include "native_qsa_wo_compose_fixture.hpp"
using namespace native_qsa_wo_compose;
int main(int argc,char** argv){try{
#ifdef STRATA_GDN_OUTPUT_SCALE_SUITE
    if(argc==4&&std::string(argv[1])=="--validate-direct"){
        auto packed=w::load(argv[2],q::bytes),output=w::load(argv[3],rows*4);need(packed==direct_packed(),"GDN output: direct packed control changed");direct_controls(output);return 0;
    }
    if(argc==2&&std::string(argv[1])=="--self-test"){
        auto ww=expected_weights(),direct=direct_packed();packed_domain(ww,direct);Bytes out(rows*4);
        for(size_t row:{1u,3u,4u,5u,6u,7u,8u,9u}){auto v=w::reference(ww.data(),ww.size(),direct.data(),direct.size(),rows,row,w::InputScaleDomain::gdn_output_finite).value;std::memcpy(out.data()+row*4,&v,4);}direct_controls(out);
        std::array<bool,1024> seen{};
        for(size_t i=0;i<scale_cases;++i){auto input=scale_input(i);auto packed=native_q8_fixture::quantize(input);scale_words(packed,i);
            auto blocks=scale_blocks();for(size_t j=0;j<blocks.size();++j){auto h=expected_scale(i,j);if(h&&h<1024)seen[h]=true;}}
        for(size_t h=1;h<seen.size();++h)need(seen[h],"GDN output: missing subnormal fixture");
        return 0;
    }
    if(argc==3&&std::string(argv[1])=="--scale-suite"){
        std::filesystem::path dir=argv[2];need(std::filesystem::create_directory(dir),"GDN output: fixture directory exists");
        for(size_t i=0;i<scale_cases;++i)q::write((dir/case_name(i)).string(),scale_input(i));auto ww=expected_weights();save(dir/"weights.bin",ww.data(),ww.size());return 0;
    }
#endif
    if(argc==3 && std::string(argv[1])=="--synthetic-input") {q::write(argv[2],deterministic());return 0;}
    if(argc==5 && std::string(argv[1])=="--contract") {contract(argv[2],argv[3],argv[4]);return 0;}
    if(argc==5 && std::string(argv[1])=="--validate-packed") {
        auto ww=weights(argv[2]);auto input=read_input(argv[3]);auto packed=w::load(argv[4],q::bytes);
        packed_domain(ww,packed);
#ifdef STRATA_GDN_OUTPUT_SCALE_SUITE
        fixture_scale_words(argv[3],input,packed);
#endif
        return 0;
    }
    throw std::runtime_error("composition: invalid validator arguments");
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
