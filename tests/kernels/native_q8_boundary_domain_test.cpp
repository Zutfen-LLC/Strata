#include "native_q8_boundary_domain.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
static void put(std::array<unsigned char,128>& a,size_t i,uint32_t bits){std::memcpy(a.data()+4*i,&bits,4);}
int main() {
    std::array<unsigned char,128> a{};put(a,0,0x3f800000);put(a,1,0xbf800000);
    if(!boundary_gdn_input_error(a.data()).empty())return 1;
    auto b=a;put(b,2,0x00000001);if(boundary_gdn_input_error(b.data()).empty())return 2;
    b.fill(0);put(b,0,0x00800001);put(b,16,0x80800000);
    if(boundary_gdn_input_error(b.data()).empty())return 3; // subnormal reduction
    std::array<unsigned char,36> q{};q[1]=0x3c;
    if(!boundary_gdn_scale_error(q.data()).empty())return 4;
    q[1]=0;q[0]=1;if(boundary_gdn_scale_error(q.data()).empty())return 5;
    q[0]=0;q[1]=0x7c;if(boundary_gdn_scale_error(q.data()).empty())return 6;
    q.fill(0);if(!boundary_gdn_scale_error(q.data()).empty())return 7; // actual producer allows zero
    std::cout << "GDN input/reduction and stored-scale domain checks verified\n";
}
