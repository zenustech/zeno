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

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
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
float2 sobolRnd(unsigned int & seed)
{

     uint2 ip = Sobol(seed);
     ip.x = OwenScramble(ip.x, 0xe7843fbfu);
     ip.y = OwenScramble(ip.y, 0x8d8fb1e0u);
     seed++;
     seed = seed&0xffffffffu;
     return make_float2(float(ip.x)/float(0xffffffffu), float(ip.y)/float(0xffffffffu));

    //return make_float2(rnd(seed), rnd(seed));
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
