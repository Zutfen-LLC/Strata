#include "native_q8_boundary_input.hpp"
#include <iostream>
static void set(std::array<unsigned char,512>& a,size_t i,uint32_t x){std::memcpy(a.data()+4*i,&x,4);}
static bool rejected(const std::array<unsigned char,512>& a) {try{boundary_input_check(a);return false;}catch(const std::exception&){return true;}}
int main(){
    std::array<unsigned char,512> a{};
    for(size_t i=0;i<4;++i)set(a,32*i,0x04000000u);
    boundary_input_check(a);
    set(a,1,1);set(a,2,0x80000001u);boundary_input_check(a);
    auto bad=a;set(bad,0,0x03ffffffu);if(!rejected(bad))return 1;
    bad=a;set(bad,1,0x7f800000u);if(!rejected(bad))return 2;
    bad=a;set(bad,1,0x7fc00000u);if(!rejected(bad))return 3;
    bad=a;set(bad,1,0x42fe0001u);if(!rejected(bad))return 4;
    std::cout << "boundary domains checked\n";
}
