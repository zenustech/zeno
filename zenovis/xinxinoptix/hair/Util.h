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

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <sutil/Exception.h>

#include <vector>


template <typename T>
void copyToDevice( const T& source, CUdeviceptr destination )
{
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( destination ), &source, sizeof( T ), cudaMemcpyHostToDevice ) );
}

template <typename T>
void createOnDevice( const T& source, CUdeviceptr* destination )
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( destination ), sizeof( T ) ) );
    copyToDevice( source, *destination );
}

template <typename T>
void copyToDevice( const std::vector<T>& source, CUdeviceptr destination )
{
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( destination ), source.data(), source.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
}

template <typename T>
void createOnDevice( const std::vector<T>& source, CUdeviceptr* destination )
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( destination ), source.size() * sizeof( T ) ) );
    copyToDevice( source, *destination );
}