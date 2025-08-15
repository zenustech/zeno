//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <vector_types.h>
#include <sutil/vec_math.h>
 
static __host__ __device__ __inline__  float van_der_corput(unsigned int n, unsigned int base=2) {
     // keep only lowest 24 bits — higher bits don't survive float precision
    n &= (1u << 24) - 1u;      // 0xFFFFFF
    // reverse all bits
    n = __brev(n);
    // shift down so reversed 24 bits are in the LSB position
    n >>= (32 - 24);
    // scale to [0,1)
    return n * (1.0f / (1u << 24) ); // 1 / 2^24

    n = n%8192;
    float result = 0.0;
    float invdenom = 1.0;
    float invbase = 0.5;

    while( n > 0 ) {
        const int remainder = n & 1;
        invdenom *= invbase;
        n = n >> 1;
        result += static_cast<float>(remainder) * invdenom;
    }
    return result;
}
const unsigned int V[8*32] = {
    2147483648u,1073741824u,536870912u,268435456u,134217728u,67108864u,33554432u,16777216u,8388608u,4194304u,2097152u,1048576u,524288u,262144u,131072u,65536u,32768u,16384u,8192u,4096u,2048u,1024u,512u,256u,128u,64u,32u,16u,8u,4u,2u,1u,2147483648u,3221225472u,2684354560u,4026531840u,2281701376u,3422552064u,2852126720u,4278190080u,2155872256u,3233808384u,2694840320u,4042260480u,2290614272u,3435921408u,2863267840u,4294901760u,2147516416u,3221274624u,2684395520u,4026593280u,2281736192u,3422604288u,2852170240u,4278255360u,2155905152u,3233857728u,2694881440u,4042322160u,2290649224u,3435973836u,2863311530u,4294967295u,2147483648u,3221225472u,1610612736u,2415919104u,3892314112u,1543503872u,2382364672u,3305111552u,1753219072u,2629828608u,3999268864u,1435500544u,2154299392u,3231449088u,1626210304u,2421489664u,3900735488u,1556135936u,2388680704u,3314585600u,1751705600u,2627492864u,4008611328u,1431684352u,2147543168u,3221249216u,1610649184u,2415969680u,3892340840u,1543543964u,2382425838u,3305133397u,2147483648u,3221225472u,536870912u,1342177280u,4160749568u,1946157056u,2717908992u,2466250752u,3632267264u,624951296u,1507852288u,3872391168u,2013790208u,3020685312u,2181169152u,3271884800u,546275328u,1363623936u,4226424832u,1977167872u,2693105664u,2437829632u,3689389568u,635137280u,1484783744u,3846176960u,2044723232u,3067084880u,2148008184u,3222012020u,537002146u,1342505107u,2147483648u,1073741824u,536870912u,2952790016u,4160749568u,3690987520u,2046820352u,2634022912u,1518338048u,801112064u,2707423232u,4038066176u,3666345984u,1875116032u,2170683392u,1085997056u,579305472u,3016343552u,4217741312u,3719483392u,2013407232u,2617981952u,1510979072u,755882752u,2726789248u,4090085440u,3680870432u,1840435376u,2147625208u,1074478300u,537900666u,2953698205u,2147483648u,1073741824u,1610612736u,805306368u,2818572288u,335544320u,2113929216u,3472883712u,2290089984u,3829399552u,3059744768u,1127219200u,3089629184u,4199809024u,3567124480u,1891565568u,394297344u,3988799488u,920674304u,4193267712u,2950604800u,3977188352u,3250028032u,129093376u,2231568512u,2963678272u,4281226848u,432124720u,803643432u,1633613396u,2672665246u,3170194367u,2147483648u,3221225472u,2684354560u,3489660928u,1476395008u,2483027968u,1040187392u,3808428032u,3196059648u,599785472u,505413632u,4077912064u,1182269440u,1736704000u,2017853440u,2221342720u,3329785856u,2810494976u,3628507136u,1416089600u,2658719744u,864310272u,3863387648u,3076993792u,553150080u,272922560u,4167467040u,1148698640u,1719673080u,2009075780u,2149644390u,3222291575u,2147483648u,1073741824u,2684354560u,1342177280u,2281701376u,1946157056u,436207616u,2566914048u,2625634304u,3208642560u,2720006144u,2098200576u,111673344u,2354315264u,3464626176u,4027383808u,2886631424u,3770826752u,1691164672u,3357462528u,1993345024u,3752330240u,873073152u,2870150400u,1700563072u,87021376u,1097028000u,1222351248u,1560027592u,2977959924u,23268898u,437609937u
};
static __host__ __device__ __inline__  unsigned int grayCode(unsigned int i) {
	return i ^ (i>>1);
}
static __host__ __device__ __inline__ float sobol(unsigned int d, unsigned int i) {
    unsigned int result = 0;
    unsigned int offset = d * 32;
    for(unsigned int j = 0; i!=0; i >>= 1, j++)
        if((i & 1)!=0)
            result ^= V[j+offset];

    return float(result) * (1.0f/float(0xFFFFFFFFU));
}
static __host__ __device__ __inline__ float2 sobolVec2(unsigned int i, unsigned int b) {
    float u = sobol(b*2, grayCode(i));
    float v = sobol(b*2+1, grayCode(i));
    return make_float2(u, v);
}
static __host__ __device__ __inline__ unsigned int wang_hash(unsigned int &seed) {
    seed = (unsigned int )(seed ^ (unsigned int )(61)) ^ (unsigned int )(seed >> (unsigned int )(16));
    seed *= (unsigned int )(9);
    seed = seed ^ (seed >> 4);
    seed *= (unsigned int )(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}
static __host__ __device__ __inline__ float2  CranleyPattersonRotation(unsigned int & seed, float2 p) {


    float u = float(wang_hash(seed)) / 4294967296.0;
    float v = float(wang_hash(seed)) / 4294967296.0;

    p.x += u;
    if(p.x>1) p.x -= 1;
    if(p.x<0) p.x += 1;

    p.y += v;
    if(p.y>1) p.y -= 1;
    if(p.y<0) p.y += 1;

    return p;
}
template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ unsigned int lcg2(unsigned int &prev)
{
  prev = (prev*8121 + 28411)  % 134456;
  return prev;
}

static __host__ __device__ __inline__ unsigned int lcg32(unsigned int &prev)
{
  /* implicit mod 2^32 */
  prev = (1103515245 * (prev) + 12345);
  return prev;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &seed)
{
    auto state = seed;
	seed = seed * 747796405u + 2891336453u;
	auto word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	auto tmp = (word >> 22u) ^ word;

    return (float)tmp / (float)0xffffffff;
//    float val = van_der_corput(seed);
//    return val;
}
static __host__ __device__ __inline__ float vdcrnd(unsigned int &seed)
{
    seed += 1;
    float val = van_der_corput(seed);
    return val;
}

static __host__ __device__ __inline__ unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
    return seed ^ frame;
}

static __device__ __inline__
uint2 Sobol(unsigned int n) {
    uint2 p = make_uint2(0u,0u);
    uint2 d = make_uint2(0x80000000u,0x80000000u);

    for(; n != 0u; n >>= 1u) {
        if((n & 1u) != 0u){
            p.x ^= d.x;
            p.y ^= d.y;
        }
        
        d.x >>= 1u; // 1st dimension Sobol matrix, is same as base 2 Van der Corput
        d.y ^= d.y >> 1u; // 2nd dimension Sobol matrix
    }
    return p;
}


// adapted from: https://www.shadertoy.com/view/3lcczS
static __device__ __inline__
unsigned int ReverseBits(unsigned int x) {
    x = ((x & 0xaaaaaaaau) >> 1) | ((x & 0x55555555u) << 1);
    x = ((x & 0xccccccccu) >> 2) | ((x & 0x33333333u) << 2);
    x = ((x & 0xf0f0f0f0u) >> 4) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x & 0xff00ff00u) >> 8) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}

// EDIT: updated with a new hash that fixes an issue with the old one.
// details in the post linked at the top.
static __device__ __inline__
unsigned int OwenHash(unsigned int x, unsigned int seed) { // works best with random seeds
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return x;
}
static __device__ __inline__
unsigned int OwenScramble(unsigned int p, unsigned int seed) {
    p = ReverseBits(p);
    p = OwenHash(p, seed);
    return ReverseBits(p);
}
static __device__ __inline__
float2 sobolRnd(unsigned int i, unsigned b, unsigned int & seed)
{
    float2 v = sobolVec2(i,b);
    return CranleyPattersonRotation(seed, v);
}

static __device__ __inline__
float2 sobolRnd2(unsigned int & seed)
{

     uint2 ip = Sobol(seed);
     ip.x = OwenScramble(ip.x, 0xe7843fbfu);
     ip.y = OwenScramble(ip.y, 0x8d8fb1e0u);
     seed++;
     seed = seed&0xffffffffu;
     return make_float2(float(ip.x)/float(0xffffffffu), float(ip.y)/float(0xffffffffu));

    //return make_float2(rnd(seed), rnd(seed));
}

__forceinline__ __device__ float fractf(float x)
{
	return x - floorf(x);
}

/* Gradient noise from Jorge Jimenez's presentation: */
/* http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare */
__forceinline__ __device__ float InterleavedGradientNoise(float2 uv)
{
	return fractf(52.9829189 * fractf(dot(uv, float2{0.06711056, 0.00583715})));
}