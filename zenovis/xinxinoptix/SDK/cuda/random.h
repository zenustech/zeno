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
static __host__ __device__ __inline__ unsigned int J(unsigned int i)
{
    i=( i<<16)|(i>>16);
    i=((i&0x00FF00FF)<<8)|((i&0xFF00FF00)>>8);
    i=((i&0x0F0F0F0F)<<4)|((i&0xF0F0F0F0)>>4);
    i=((i&0x33333333)<<2)|((i&0xCCCCCCCC)>>2);
    i=((i&0x55555555)<<1)|((i&0xAAAAAAAA)>>1);
    return i;
}
static __host__ __device__ __inline__ unsigned int P(unsigned int v)
{
    v^=v<<16;
	v^=(v&0x00FF00FF)<<8;
	v^=(v&0x0F0F0F0F)<<4;
	v^=(v&0x33333333)<<2;
	v^=(v&0x55555555)<<1;
	return v;
}

static __host__ __device__ __inline__ unsigned int JPJ(unsigned int v)
{
    v^=v>>16;
    v^=(v&0xFF00FF00)>>8;
    v^=(v&0xF0F0F0F0)>>4;
    v^=(v&0xCCCCCCCC)>>2;
    v^=(v&0xAAAAAAAA)>>1;
    return v;
}
static __host__ __device__ __inline__ unsigned int JPJ(unsigned int v, int m)
{
    return (JPJ(v >> (32 - m)) << (32 - m)) | (v & (0xFFFFFFFF >> m));
}
static __host__ __device__ __inline__ unsigned int mmdX(unsigned int x, int m)
{
    unsigned int v = JPJ(x >> (32 - m));
    int padding = (m - 6) >> 1;
    v ^= (v & (0x10 << padding)) >> 1;
    return (v << (32 - m)) | (x & (0xFFFFFFFF >> m));
}
static __host__ __device__ __inline__ unsigned int mmdY(unsigned int y, int m)
{
    unsigned int v = JPJ(y >> (32 - m));
    int padding = (m - 6) >> 1;
    v ^= (
            ((v & (0x30 << padding)) >> 1) ^
            ((v & (0x08 << padding)) >> 2)
         );
    return (v << (32 - m)) | (y & (0xFFFFFFFF >> m));
}
static __device__ __inline__
float2 hammersley(unsigned int &index)
{
    index &= 0xFFFFFFFF;
    unsigned int x = J(index);
    unsigned int y = P(x);
    uint2 p = {JPJ(x,12), JPJ(y,12)};
    float c = 1.0f/(float)(0xFFFFFFFF);
    index++;
    return make_float2((float)p.x*c, (float)p.y*c);
}
static __device__ __inline__
float2 mmd(unsigned int &index)
{
    index &= 0xFFFFFFFF;
    unsigned int x = J(index);
    unsigned int y = P(x);
    uint2 p = {mmdX(x,12), mmdY(y,12)};
    float c = 1.0f/(float)(0xFFFFFFFF);
    index++;
    return make_float2((float)p.x*c, (float)p.y*c);
}
static __host__ __device__ __inline__
float radicalInverse_VdC(unsigned int bits)
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
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
#include "SobolDirections.h"

static __host__ __device__ __inline__  unsigned int grayCode(unsigned int i) {
	return i ^ (i>>1);
}
static __host__ __device__ __inline__ float sobol(unsigned int d, unsigned int i) {
    unsigned int result = 0;
    unsigned int offset = d * 32;
    for(unsigned int j = 0; i!=0; i >>= 1, j++)
        if((i & 1)!=0)
            result ^= zenoSobolDirections[j+offset];

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

    return (tmp >> 8) * (1.0f / 16777216.0f);

  return (float)tmp / (float)0xffffffff;
}
static __host__ __device__ __inline__ unsigned int getScrumble(unsigned int current_scrumble)
{
    unsigned int s = current_scrumble;
    rnd(s);rnd(s);rnd(s);
    return s;
}
static __host__ __device__ __inline__ float vdcrnd(unsigned int index, unsigned int &seed)
{
    unsigned int scrambled = index ^ seed;
    float val = van_der_corput(scrambled);
    rnd(seed);
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