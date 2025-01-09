
/* 
* SPDX-FileCopyrightText: Copyright (c) 2009 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary 
* 
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual 
* property and proprietary rights in and to this material, related 
* documentation and any modifications thereto. Any use, reproduction, 
* disclosure or distribution of this material and related documentation 
* without an express license agreement from NVIDIA CORPORATION or 
* its affiliates is strictly prohibited. 
*/
/// @file
/// @author NVIDIA Corporation
/// @brief  OptiX public API header
///
/// Includes the host api if compiling host code, includes the cuda api if compiling device code.
/// For the math library routines include optix_math.h

#ifndef OPTIX_OPTIX_H
#define OPTIX_OPTIX_H

/// The OptiX version.
///
/// - major =  OPTIX_VERSION/10000
/// - minor = (OPTIX_VERSION%10000)/100
/// - micro =  OPTIX_VERSION%100
#define OPTIX_VERSION 80100


#ifdef __CUDACC__
#include "optix_device.h"
#else
#include "optix_host.h"
#endif


#endif  // OPTIX_OPTIX_H
