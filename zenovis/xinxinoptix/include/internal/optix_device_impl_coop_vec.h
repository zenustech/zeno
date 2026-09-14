/*
* SPDX-FileCopyrightText: Copyright (c) 2019 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary
*
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
* property and proprietary rights in and to this material, related
* documentation and any modifications thereto. Any use, reproduction,
* disclosure or distribution of this material and related documentation
* without an express license agreement from NVIDIA CORPORATION or
* its affiliates is strictly prohibited.
*/
/// @file optix_device_impl_coopvec.h
/// @author NVIDIA Corporation
/// @brief  OptiX public API header
///

#ifndef OPTIX_OPTIX_DEVICE_IMPL_COOP_VEC_H
#define OPTIX_OPTIX_DEVICE_IMPL_COOP_VEC_H

#if !defined( __OPTIX_INCLUDE_INTERNAL_HEADERS__ )
#error("optix_device_impl.h is an internal header file and must not be used directly.  Please use optix_device.h or optix.h instead.")
#endif

namespace optix_internal {

typedef enum OptixCoopVecOp
{
    OPTIX_COOP_VEC_OP_UNKNOWN = 0x2A20,
    OPTIX_COOP_VEC_OP_EXP2    = 0x2A21,
    OPTIX_COOP_VEC_OP_LOG2    = 0x2A22,
    OPTIX_COOP_VEC_OP_TANH    = 0x2A23,
    OPTIX_COOP_VEC_OP_MAX     = 0x2A24,
    OPTIX_COOP_VEC_OP_MIN     = 0x2A25,
    OPTIX_COOP_VEC_OP_FFMA    = 0x2A26,
    OPTIX_COOP_VEC_OP_MUL     = 0x2A27,
    OPTIX_COOP_VEC_OP_ADD     = 0x2A28,
    OPTIX_COOP_VEC_OP_SUB     = 0x2A29,
    OPTIX_COOP_VEC_OP_CVT     = 0x2A2A,
    OPTIX_COOP_VEC_OP_STEP    = 0x2A2B,
} OptixCoopVecOp;
}  // end namespace optix_internal

#if !defined( OPTIX_DONT_INCLUDE_CUDA )
// If OPTIX_DONT_INCLUDE_CUDA is defined, cuda driver types must be defined through other
// means before including optix headers.
#include <cuda_fp16.h>
#endif


namespace optix_internal {
namespace coop_vec_type_traits {
// clang-format off

// We need to implement code that is available in <type_traits> since nvrtc does not support the <type_traits> header.
// Custom is_float implementation - specialized only for half and float
template <typename T> struct is_float { static const bool value = false; };
template <> struct is_float<float> { static const bool value = true; };
template <> struct is_float<half> { static const bool value = true; };

template <typename T> struct is_integral { static const bool value = !is_float<T>::value; };

template <typename T> struct is_signed_impl { static const bool value = static_cast<T>(-1) < static_cast<T>(0); };

// If it's a float type, it's signed. Otherwise use the generic test.
template <typename T>
struct is_signed { static const bool value = is_float<T>::value ? true : is_signed_impl<T>::value; };
// NVRTC is stricter about template instantiation requirements and requires both branches of a ternary operator
// to be syntactically valid during compilation, so we need to explicitly specialize half to bypass the generic
// is_signed_impl template that uses static_cast, avoiding the ambiguous conversion issue entirely.
template <> struct is_signed<half> { static const bool value = true; };

template <bool is_integral, bool is_signed, size_t byte_size> struct TT;
template <> struct TT<true,  true,  1> { static const OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_INT8; };
template <> struct TT<true,  false, 1> { static const OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_UINT8; };
template <> struct TT<true,  true,  4> { static const OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_INT32; };
template <> struct TT<true,  false, 4> { static const OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_UINT32; };
template <> struct TT<false, true,  4> { static const OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT32; };
template <> struct TT<false, true,  2> { static const OptixCoopVecElemType value = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16; };

template< size_t byte_size > struct TB;
template<> struct TB<1> { using bitType = unsigned char; };
template<> struct TB<2> { using bitType = unsigned short; };
template<> struct TB<4> { using bitType = unsigned int; };
// clang-format on

// The non-specialized template can take advantage of all the built-in types, while for
// other special types like half, will be handled by specialization.
template <typename T>
struct OptixCoopVecElemTypeTrait
{
    static const OptixCoopVecElemType elementType =
        TT<coop_vec_type_traits::is_integral<T>::value, coop_vec_type_traits::is_signed<T>::value, sizeof( T )>::value;
    using bitType = typename TB<sizeof( T )>::bitType;
};
}  // end namespace coop_vec_type_traits
}  // end namespace optix_internal

namespace optix_internal {

template <typename VecTOut>
struct OptixCoopVecLoadASMGenerator
{
    static const OptixCoopVecElemType outputElementType =
        optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTOut::value_type>::elementType;
    using outputBitType =
        typename optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTOut::value_type>::bitType;

    __forceinline__ __device__ static VecTOut generateASMPtr( CUdeviceptr ptr )
    {
        VecTOut result;
        asm( "call"
             "(),"
             "_optix_vector_load_ptr,"
             "(%0,%1,%2,%3);"
             :
             : "r"( outputElementType ), "r"( VecTOut::size ), "l"( ptr ), "l"( result.data() ) );
        return result;
    }
    __forceinline__ __device__ static VecTOut generateASM( CUdeviceptr ptr )
    {
        if( VecTOut::size > 64 || sizeof( typename VecTOut::value_type ) > sizeof( unsigned int ) )
            return generateASMPtr( ptr );
        else
        {
            // This code needs to live in an else, block otherwise the compiler will
            // complain about the loop being unreachable.
            unsigned int O[64];
            if( VecTOut::size <= 16 )
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15),"
                     "_optix_vector_load_16xi32,"
                     "(%16,%17,%18);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] )
                     : "r"( outputElementType ), "r"( VecTOut::size ), "l"( ptr ) );
            else
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                     "26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%"
                     "50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63),"
                     "_optix_vector_load_64xi32,"
                     "(%64,%65,%66);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] ), "=r"( O[16] ), "=r"( O[17] ), "=r"( O[18] ),
                       "=r"( O[19] ), "=r"( O[20] ), "=r"( O[21] ), "=r"( O[22] ), "=r"( O[23] ), "=r"( O[24] ),
                       "=r"( O[25] ), "=r"( O[26] ), "=r"( O[27] ), "=r"( O[28] ), "=r"( O[29] ), "=r"( O[30] ),
                       "=r"( O[31] ), "=r"( O[32] ), "=r"( O[33] ), "=r"( O[34] ), "=r"( O[35] ), "=r"( O[36] ),
                       "=r"( O[37] ), "=r"( O[38] ), "=r"( O[39] ), "=r"( O[40] ), "=r"( O[41] ), "=r"( O[42] ),
                       "=r"( O[43] ), "=r"( O[44] ), "=r"( O[45] ), "=r"( O[46] ), "=r"( O[47] ), "=r"( O[48] ),
                       "=r"( O[49] ), "=r"( O[50] ), "=r"( O[51] ), "=r"( O[52] ), "=r"( O[53] ), "=r"( O[54] ),
                       "=r"( O[55] ), "=r"( O[56] ), "=r"( O[57] ), "=r"( O[58] ), "=r"( O[59] ), "=r"( O[60] ),
                       "=r"( O[61] ), "=r"( O[62] ), "=r"( O[63] )
                     : "r"( outputElementType ), "r"( VecTOut::size ), "l"( ptr ) );

            VecTOut result;
            for( unsigned int i = 0; i < VecTOut::size; ++i )
            {
                outputBitType o = O[i];
                result[i]       = *( reinterpret_cast<typename VecTOut::value_type*>( &( o ) ) );
            }
            return result;
        }
    }
};


template <OptixCoopVecOp VectorOp, typename VecTOut, typename VecTIn>
struct OptixCoopVecASMGenerator
{
    static const OptixCoopVecElemType outputElementType =
        optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTOut::value_type>::elementType;
    using outputBitType =
        typename optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTOut::value_type>::bitType;
    static const OptixCoopVecElemType inputElementType =
        optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTIn::value_type>::elementType;
    using inputBitType =
        typename optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTIn::value_type>::bitType;

    __forceinline__ __device__ static VecTOut generateASMPtr( const VecTIn& vecA )
    {
        VecTOut result;
        asm( "call"
             "(),"
             "_optix_vector_op1_ptr,"
             "(%0,%1,%2,%3,%4,%5,%6);"
             :
             : "r"( VectorOp ), "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ),
               "r"( VecTIn::size ), "l"( vecA.data() ), "l"( result.data() ) );
        return result;
    }

    __forceinline__ __device__ static VecTOut generateASMPtr( const VecTIn& vecA, const VecTIn& vecB )
    {
        VecTOut result;
        asm( "call"
             "(),"
             "_optix_vector_op2_ptr,"
             "(%0,%1,%2,%3,%4,%5,%6,%7);"
             :
             : "r"( VectorOp ), "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ),
               "r"( VecTIn::size ), "l"( vecA.data() ), "l"( vecB.data() ), "l"( result.data() ) );
        return result;
    }

    __forceinline__ __device__ static VecTOut generateASMPtr( const VecTIn& vecA, const VecTIn& vecB, const VecTIn& vecC )
    {
        VecTOut result;
        asm( "call"
             "(),"
             "_optix_vector_op3_ptr,"
             "(%0,%1,%2,%3,%4,%5,%6,%7,%8);"
             :
             : "r"( VectorOp ), "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ),
               "r"( VecTIn::size ), "l"( vecA.data() ), "l"( vecB.data() ), "l"( vecC.data() ), "l"( result.data() ) );
        return result;
    }

    __forceinline__ __device__ static VecTOut generateASM( const VecTIn& vecA )
    {
        if( VecTIn::size > 64 || VecTOut::size > 64 || sizeof( typename VecTIn::value_type ) > sizeof( unsigned int )
            || sizeof( typename VecTOut::value_type ) > sizeof( unsigned int ) )
            return generateASMPtr( vecA );
        else
        {
            // This code needs to live in an else, block otherwise the compiler will
            // complain about the loop being unreachable.
            unsigned int IA[64];
            unsigned int O[64];
            for( unsigned int i = 0; i < VecTIn::size; ++i )
            {
                IA[i] = *( reinterpret_cast<const inputBitType*>( &( vecA[i] ) ) );
            }
            if( VecTOut::size <= 16 && VecTIn::size <= 16 )
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15),"
                     "_optix_vector_op1_16xi32,"
                     "(%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] )
                     : "r"( VectorOp ), "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ),
                       "r"( VecTIn::size ), "r"( IA[0] ), "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ),
                       "r"( IA[5] ), "r"( IA[6] ), "r"( IA[7] ), "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ),
                       "r"( IA[11] ), "r"( IA[12] ), "r"( IA[13] ), "r"( IA[14] ), "r"( IA[15] ) );
            else
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                     "26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%"
                     "50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63),"
                     "_optix_vector_op1_64xi32,"
                     "(%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,"
                     "%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%"
                     "109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127,%"
                     "128,%129,%130,%131,%132);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] ), "=r"( O[16] ), "=r"( O[17] ), "=r"( O[18] ),
                       "=r"( O[19] ), "=r"( O[20] ), "=r"( O[21] ), "=r"( O[22] ), "=r"( O[23] ), "=r"( O[24] ),
                       "=r"( O[25] ), "=r"( O[26] ), "=r"( O[27] ), "=r"( O[28] ), "=r"( O[29] ), "=r"( O[30] ),
                       "=r"( O[31] ), "=r"( O[32] ), "=r"( O[33] ), "=r"( O[34] ), "=r"( O[35] ), "=r"( O[36] ),
                       "=r"( O[37] ), "=r"( O[38] ), "=r"( O[39] ), "=r"( O[40] ), "=r"( O[41] ), "=r"( O[42] ),
                       "=r"( O[43] ), "=r"( O[44] ), "=r"( O[45] ), "=r"( O[46] ), "=r"( O[47] ), "=r"( O[48] ),
                       "=r"( O[49] ), "=r"( O[50] ), "=r"( O[51] ), "=r"( O[52] ), "=r"( O[53] ), "=r"( O[54] ),
                       "=r"( O[55] ), "=r"( O[56] ), "=r"( O[57] ), "=r"( O[58] ), "=r"( O[59] ), "=r"( O[60] ),
                       "=r"( O[61] ), "=r"( O[62] ), "=r"( O[63] )
                     : "r"( VectorOp ), "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ),
                       "r"( VecTIn::size ), "r"( IA[0] ), "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ),
                       "r"( IA[5] ), "r"( IA[6] ), "r"( IA[7] ), "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ),
                       "r"( IA[11] ), "r"( IA[12] ), "r"( IA[13] ), "r"( IA[14] ), "r"( IA[15] ), "r"( IA[16] ),
                       "r"( IA[17] ), "r"( IA[18] ), "r"( IA[19] ), "r"( IA[20] ), "r"( IA[21] ), "r"( IA[22] ),
                       "r"( IA[23] ), "r"( IA[24] ), "r"( IA[25] ), "r"( IA[26] ), "r"( IA[27] ), "r"( IA[28] ),
                       "r"( IA[29] ), "r"( IA[30] ), "r"( IA[31] ), "r"( IA[32] ), "r"( IA[33] ), "r"( IA[34] ),
                       "r"( IA[35] ), "r"( IA[36] ), "r"( IA[37] ), "r"( IA[38] ), "r"( IA[39] ), "r"( IA[40] ),
                       "r"( IA[41] ), "r"( IA[42] ), "r"( IA[43] ), "r"( IA[44] ), "r"( IA[45] ), "r"( IA[46] ),
                       "r"( IA[47] ), "r"( IA[48] ), "r"( IA[49] ), "r"( IA[50] ), "r"( IA[51] ), "r"( IA[52] ),
                       "r"( IA[53] ), "r"( IA[54] ), "r"( IA[55] ), "r"( IA[56] ), "r"( IA[57] ), "r"( IA[58] ),
                       "r"( IA[59] ), "r"( IA[60] ), "r"( IA[61] ), "r"( IA[62] ), "r"( IA[63] ) );

            VecTOut result;
            for( unsigned int i = 0; i < VecTOut::size; ++i )
            {
                outputBitType o = O[i];
                result[i]       = *( reinterpret_cast<typename VecTOut::value_type*>( &( o ) ) );
            }
            return result;
        }
    }

    __forceinline__ __device__ static VecTOut generateASM( const VecTIn& vecA, const VecTIn& vecB )
    {
        if( VecTIn::size > 64 || VecTOut::size > 64 || sizeof( typename VecTIn::value_type ) > sizeof( unsigned int )
            || sizeof( typename VecTOut::value_type ) > sizeof( unsigned int ) )
            return generateASMPtr( vecA, vecB );
        else
        {
            // This code needs to live in an else, block otherwise the compiler will
            // complain about the loop being unreachable.
            unsigned int IA[64];
            unsigned int IB[64];
            unsigned int O[64];
            for( unsigned int i = 0; i < VecTIn::size; ++i )
            {
                IA[i] = *( reinterpret_cast<const inputBitType*>( &( vecA[i] ) ) );
                IB[i] = *( reinterpret_cast<const inputBitType*>( &( vecB[i] ) ) );
            }
            if( VecTOut::size <= 16 && VecTIn::size <= 16 )
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15),"
                     "_optix_vector_op2_16xi32,"
                     "(%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,"
                     "%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] )
                     : "r"( VectorOp ), "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ),
                       "r"( VecTIn::size ), "r"( IA[0] ), "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ),
                       "r"( IA[5] ), "r"( IA[6] ), "r"( IA[7] ), "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ), "r"( IA[11] ),
                       "r"( IA[12] ), "r"( IA[13] ), "r"( IA[14] ), "r"( IA[15] ), "r"( IB[0] ), "r"( IB[1] ), "r"( IB[2] ),
                       "r"( IB[3] ), "r"( IB[4] ), "r"( IB[5] ), "r"( IB[6] ), "r"( IB[7] ), "r"( IB[8] ), "r"( IB[9] ),
                       "r"( IB[10] ), "r"( IB[11] ), "r"( IB[12] ), "r"( IB[13] ), "r"( IB[14] ), "r"( IB[15] ) );
            else
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                     "26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%"
                     "50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63),"
                     "_optix_vector_op2_64xi32,"
                     "(%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,"
                     "%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%"
                     "109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127,%"
                     "128,%129,%130,%131,%132,%133,%134,%135,%136,%137,%138,%139,%140,%141,%142,%143,%144,%145,%146,%"
                     "147,%148,%149,%150,%151,%152,%153,%154,%155,%156,%157,%158,%159,%160,%161,%162,%163,%164,%165,%"
                     "166,%167,%168,%169,%170,%171,%172,%173,%174,%175,%176,%177,%178,%179,%180,%181,%182,%183,%184,%"
                     "185,%186,%187,%188,%189,%190,%191,%192,%193,%194,%195,%196);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] ), "=r"( O[16] ), "=r"( O[17] ), "=r"( O[18] ),
                       "=r"( O[19] ), "=r"( O[20] ), "=r"( O[21] ), "=r"( O[22] ), "=r"( O[23] ), "=r"( O[24] ),
                       "=r"( O[25] ), "=r"( O[26] ), "=r"( O[27] ), "=r"( O[28] ), "=r"( O[29] ), "=r"( O[30] ),
                       "=r"( O[31] ), "=r"( O[32] ), "=r"( O[33] ), "=r"( O[34] ), "=r"( O[35] ), "=r"( O[36] ),
                       "=r"( O[37] ), "=r"( O[38] ), "=r"( O[39] ), "=r"( O[40] ), "=r"( O[41] ), "=r"( O[42] ),
                       "=r"( O[43] ), "=r"( O[44] ), "=r"( O[45] ), "=r"( O[46] ), "=r"( O[47] ), "=r"( O[48] ),
                       "=r"( O[49] ), "=r"( O[50] ), "=r"( O[51] ), "=r"( O[52] ), "=r"( O[53] ), "=r"( O[54] ),
                       "=r"( O[55] ), "=r"( O[56] ), "=r"( O[57] ), "=r"( O[58] ), "=r"( O[59] ), "=r"( O[60] ),
                       "=r"( O[61] ), "=r"( O[62] ), "=r"( O[63] )
                     : "r"( VectorOp ), "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ),
                       "r"( VecTIn::size ), "r"( IA[0] ), "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ),
                       "r"( IA[5] ), "r"( IA[6] ), "r"( IA[7] ), "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ), "r"( IA[11] ),
                       "r"( IA[12] ), "r"( IA[13] ), "r"( IA[14] ), "r"( IA[15] ), "r"( IA[16] ), "r"( IA[17] ),
                       "r"( IA[18] ), "r"( IA[19] ), "r"( IA[20] ), "r"( IA[21] ), "r"( IA[22] ), "r"( IA[23] ),
                       "r"( IA[24] ), "r"( IA[25] ), "r"( IA[26] ), "r"( IA[27] ), "r"( IA[28] ), "r"( IA[29] ),
                       "r"( IA[30] ), "r"( IA[31] ), "r"( IA[32] ), "r"( IA[33] ), "r"( IA[34] ), "r"( IA[35] ),
                       "r"( IA[36] ), "r"( IA[37] ), "r"( IA[38] ), "r"( IA[39] ), "r"( IA[40] ), "r"( IA[41] ),
                       "r"( IA[42] ), "r"( IA[43] ), "r"( IA[44] ), "r"( IA[45] ), "r"( IA[46] ), "r"( IA[47] ),
                       "r"( IA[48] ), "r"( IA[49] ), "r"( IA[50] ), "r"( IA[51] ), "r"( IA[52] ), "r"( IA[53] ),
                       "r"( IA[54] ), "r"( IA[55] ), "r"( IA[56] ), "r"( IA[57] ), "r"( IA[58] ), "r"( IA[59] ),
                       "r"( IA[60] ), "r"( IA[61] ), "r"( IA[62] ), "r"( IA[63] ), "r"( IB[0] ), "r"( IB[1] ), "r"( IB[2] ),
                       "r"( IB[3] ), "r"( IB[4] ), "r"( IB[5] ), "r"( IB[6] ), "r"( IB[7] ), "r"( IB[8] ), "r"( IB[9] ),
                       "r"( IB[10] ), "r"( IB[11] ), "r"( IB[12] ), "r"( IB[13] ), "r"( IB[14] ), "r"( IB[15] ),
                       "r"( IB[16] ), "r"( IB[17] ), "r"( IB[18] ), "r"( IB[19] ), "r"( IB[20] ), "r"( IB[21] ),
                       "r"( IB[22] ), "r"( IB[23] ), "r"( IB[24] ), "r"( IB[25] ), "r"( IB[26] ), "r"( IB[27] ),
                       "r"( IB[28] ), "r"( IB[29] ), "r"( IB[30] ), "r"( IB[31] ), "r"( IB[32] ), "r"( IB[33] ),
                       "r"( IB[34] ), "r"( IB[35] ), "r"( IB[36] ), "r"( IB[37] ), "r"( IB[38] ), "r"( IB[39] ),
                       "r"( IB[40] ), "r"( IB[41] ), "r"( IB[42] ), "r"( IB[43] ), "r"( IB[44] ), "r"( IB[45] ),
                       "r"( IB[46] ), "r"( IB[47] ), "r"( IB[48] ), "r"( IB[49] ), "r"( IB[50] ), "r"( IB[51] ),
                       "r"( IB[52] ), "r"( IB[53] ), "r"( IB[54] ), "r"( IB[55] ), "r"( IB[56] ), "r"( IB[57] ),
                       "r"( IB[58] ), "r"( IB[59] ), "r"( IB[60] ), "r"( IB[61] ), "r"( IB[62] ), "r"( IB[63] ) );

            VecTOut result;
            for( unsigned int i = 0; i < VecTOut::size; ++i )
            {
                outputBitType o = O[i];
                result[i]       = *( reinterpret_cast<typename VecTOut::value_type*>( &( o ) ) );
            }
            return result;
        }
    }

    __forceinline__ __device__ static VecTOut generateASM( const VecTIn& vecA, const VecTIn& vecB, const VecTIn& vecC )
    {
        if( VecTIn::size > 64 || VecTOut::size > 64 || sizeof( typename VecTIn::value_type ) > sizeof( unsigned int )
            || sizeof( typename VecTOut::value_type ) > sizeof( unsigned int ) )
            return generateASMPtr( vecA, vecB, vecC );
        else
        {
            // This code needs to live in an else, block otherwise the compiler will
            // complain about the loop being unreachable.
            unsigned int IA[64];
            unsigned int IB[64];
            unsigned int IC[64];
            unsigned int O[64];
            for( unsigned int i = 0; i < VecTIn::size; ++i )
            {
                IA[i] = *( reinterpret_cast<const inputBitType*>( &( vecA[i] ) ) );
                IB[i] = *( reinterpret_cast<const inputBitType*>( &( vecB[i] ) ) );
                IC[i] = *( reinterpret_cast<const inputBitType*>( &( vecC[i] ) ) );
            }
            if( VecTOut::size <= 16 && VecTIn::size <= 16 )
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15),"
                     "_optix_vector_op3_16xi32,"
                     "(%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,"
                     "%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%"
                     "64,%65,%66,%67,%68);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] )
                     : "r"( VectorOp ), "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ),
                       "r"( VecTIn::size ), "r"( IA[0] ), "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ),
                       "r"( IA[5] ), "r"( IA[6] ), "r"( IA[7] ), "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ),
                       "r"( IA[11] ), "r"( IA[12] ), "r"( IA[13] ), "r"( IA[14] ), "r"( IA[15] ), "r"( IB[0] ),
                       "r"( IB[1] ), "r"( IB[2] ), "r"( IB[3] ), "r"( IB[4] ), "r"( IB[5] ), "r"( IB[6] ), "r"( IB[7] ),
                       "r"( IB[8] ), "r"( IB[9] ), "r"( IB[10] ), "r"( IB[11] ), "r"( IB[12] ), "r"( IB[13] ),
                       "r"( IB[14] ), "r"( IB[15] ), "r"( IC[0] ), "r"( IC[1] ), "r"( IC[2] ), "r"( IC[3] ),
                       "r"( IC[4] ), "r"( IC[5] ), "r"( IC[6] ), "r"( IC[7] ), "r"( IC[8] ), "r"( IC[9] ),
                       "r"( IC[10] ), "r"( IC[11] ), "r"( IC[12] ), "r"( IC[13] ), "r"( IC[14] ), "r"( IC[15] ) );
            else
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                     "26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%"
                     "50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63),"
                     "_optix_vector_op3_64xi32,"
                     "(%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,"
                     "%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%"
                     "109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127,%"
                     "128,%129,%130,%131,%132,%133,%134,%135,%136,%137,%138,%139,%140,%141,%142,%143,%144,%145,%146,%"
                     "147,%148,%149,%150,%151,%152,%153,%154,%155,%156,%157,%158,%159,%160,%161,%162,%163,%164,%165,%"
                     "166,%167,%168,%169,%170,%171,%172,%173,%174,%175,%176,%177,%178,%179,%180,%181,%182,%183,%184,%"
                     "185,%186,%187,%188,%189,%190,%191,%192,%193,%194,%195,%196,%197,%198,%199,%200,%201,%202,%203,%"
                     "204,%205,%206,%207,%208,%209,%210,%211,%212,%213,%214,%215,%216,%217,%218,%219,%220,%221,%222,%"
                     "223,%224,%225,%226,%227,%228,%229,%230,%231,%232,%233,%234,%235,%236,%237,%238,%239,%240,%241,%"
                     "242,%243,%244,%245,%246,%247,%248,%249,%250,%251,%252,%253,%254,%255,%256,%257,%258,%259,%260);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] ), "=r"( O[16] ), "=r"( O[17] ), "=r"( O[18] ),
                       "=r"( O[19] ), "=r"( O[20] ), "=r"( O[21] ), "=r"( O[22] ), "=r"( O[23] ), "=r"( O[24] ),
                       "=r"( O[25] ), "=r"( O[26] ), "=r"( O[27] ), "=r"( O[28] ), "=r"( O[29] ), "=r"( O[30] ),
                       "=r"( O[31] ), "=r"( O[32] ), "=r"( O[33] ), "=r"( O[34] ), "=r"( O[35] ), "=r"( O[36] ),
                       "=r"( O[37] ), "=r"( O[38] ), "=r"( O[39] ), "=r"( O[40] ), "=r"( O[41] ), "=r"( O[42] ),
                       "=r"( O[43] ), "=r"( O[44] ), "=r"( O[45] ), "=r"( O[46] ), "=r"( O[47] ), "=r"( O[48] ),
                       "=r"( O[49] ), "=r"( O[50] ), "=r"( O[51] ), "=r"( O[52] ), "=r"( O[53] ), "=r"( O[54] ),
                       "=r"( O[55] ), "=r"( O[56] ), "=r"( O[57] ), "=r"( O[58] ), "=r"( O[59] ), "=r"( O[60] ),
                       "=r"( O[61] ), "=r"( O[62] ), "=r"( O[63] )
                     : "r"( VectorOp ), "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ),
                       "r"( VecTIn::size ), "r"( IA[0] ), "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ),
                       "r"( IA[5] ), "r"( IA[6] ), "r"( IA[7] ), "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ),
                       "r"( IA[11] ), "r"( IA[12] ), "r"( IA[13] ), "r"( IA[14] ), "r"( IA[15] ), "r"( IA[16] ),
                       "r"( IA[17] ), "r"( IA[18] ), "r"( IA[19] ), "r"( IA[20] ), "r"( IA[21] ), "r"( IA[22] ),
                       "r"( IA[23] ), "r"( IA[24] ), "r"( IA[25] ), "r"( IA[26] ), "r"( IA[27] ), "r"( IA[28] ),
                       "r"( IA[29] ), "r"( IA[30] ), "r"( IA[31] ), "r"( IA[32] ), "r"( IA[33] ), "r"( IA[34] ),
                       "r"( IA[35] ), "r"( IA[36] ), "r"( IA[37] ), "r"( IA[38] ), "r"( IA[39] ), "r"( IA[40] ),
                       "r"( IA[41] ), "r"( IA[42] ), "r"( IA[43] ), "r"( IA[44] ), "r"( IA[45] ), "r"( IA[46] ),
                       "r"( IA[47] ), "r"( IA[48] ), "r"( IA[49] ), "r"( IA[50] ), "r"( IA[51] ), "r"( IA[52] ),
                       "r"( IA[53] ), "r"( IA[54] ), "r"( IA[55] ), "r"( IA[56] ), "r"( IA[57] ), "r"( IA[58] ),
                       "r"( IA[59] ), "r"( IA[60] ), "r"( IA[61] ), "r"( IA[62] ), "r"( IA[63] ), "r"( IB[0] ),
                       "r"( IB[1] ), "r"( IB[2] ), "r"( IB[3] ), "r"( IB[4] ), "r"( IB[5] ), "r"( IB[6] ), "r"( IB[7] ),
                       "r"( IB[8] ), "r"( IB[9] ), "r"( IB[10] ), "r"( IB[11] ), "r"( IB[12] ), "r"( IB[13] ),
                       "r"( IB[14] ), "r"( IB[15] ), "r"( IB[16] ), "r"( IB[17] ), "r"( IB[18] ), "r"( IB[19] ),
                       "r"( IB[20] ), "r"( IB[21] ), "r"( IB[22] ), "r"( IB[23] ), "r"( IB[24] ), "r"( IB[25] ),
                       "r"( IB[26] ), "r"( IB[27] ), "r"( IB[28] ), "r"( IB[29] ), "r"( IB[30] ), "r"( IB[31] ),
                       "r"( IB[32] ), "r"( IB[33] ), "r"( IB[34] ), "r"( IB[35] ), "r"( IB[36] ), "r"( IB[37] ),
                       "r"( IB[38] ), "r"( IB[39] ), "r"( IB[40] ), "r"( IB[41] ), "r"( IB[42] ), "r"( IB[43] ),
                       "r"( IB[44] ), "r"( IB[45] ), "r"( IB[46] ), "r"( IB[47] ), "r"( IB[48] ), "r"( IB[49] ),
                       "r"( IB[50] ), "r"( IB[51] ), "r"( IB[52] ), "r"( IB[53] ), "r"( IB[54] ), "r"( IB[55] ),
                       "r"( IB[56] ), "r"( IB[57] ), "r"( IB[58] ), "r"( IB[59] ), "r"( IB[60] ), "r"( IB[61] ),
                       "r"( IB[62] ), "r"( IB[63] ), "r"( IC[0] ), "r"( IC[1] ), "r"( IC[2] ), "r"( IC[3] ),
                       "r"( IC[4] ), "r"( IC[5] ), "r"( IC[6] ), "r"( IC[7] ), "r"( IC[8] ), "r"( IC[9] ),
                       "r"( IC[10] ), "r"( IC[11] ), "r"( IC[12] ), "r"( IC[13] ), "r"( IC[14] ), "r"( IC[15] ),
                       "r"( IC[16] ), "r"( IC[17] ), "r"( IC[18] ), "r"( IC[19] ), "r"( IC[20] ), "r"( IC[21] ),
                       "r"( IC[22] ), "r"( IC[23] ), "r"( IC[24] ), "r"( IC[25] ), "r"( IC[26] ), "r"( IC[27] ),
                       "r"( IC[28] ), "r"( IC[29] ), "r"( IC[30] ), "r"( IC[31] ), "r"( IC[32] ), "r"( IC[33] ),
                       "r"( IC[34] ), "r"( IC[35] ), "r"( IC[36] ), "r"( IC[37] ), "r"( IC[38] ), "r"( IC[39] ),
                       "r"( IC[40] ), "r"( IC[41] ), "r"( IC[42] ), "r"( IC[43] ), "r"( IC[44] ), "r"( IC[45] ),
                       "r"( IC[46] ), "r"( IC[47] ), "r"( IC[48] ), "r"( IC[49] ), "r"( IC[50] ), "r"( IC[51] ),
                       "r"( IC[52] ), "r"( IC[53] ), "r"( IC[54] ), "r"( IC[55] ), "r"( IC[56] ), "r"( IC[57] ),
                       "r"( IC[58] ), "r"( IC[59] ), "r"( IC[60] ), "r"( IC[61] ), "r"( IC[62] ), "r"( IC[63] ) );
            VecTOut result;
            for( unsigned int i = 0; i < VecTOut::size; ++i )
            {
                outputBitType o = O[i];
                result[i]       = *( reinterpret_cast<typename VecTOut::value_type*>( &o ) );
            }
            return result;
        }
    }
};

}  // end namespace optix_internal

template <typename VecTOut>
static __forceinline__ __device__ VecTOut optixCoopVecLoad( CUdeviceptr ptr )
{
    return optix_internal::OptixCoopVecLoadASMGenerator<VecTOut>::generateASM( ptr );
}

template <typename VecTOut, typename T>
static __forceinline__ __device__ VecTOut optixCoopVecLoad( T* ptr )
{
    return optixCoopVecLoad<VecTOut>( reinterpret_cast<CUdeviceptr>( ptr ) );
}


template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecExp2( const VecT& vec )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_EXP2, VecT, VecT>::generateASM( vec );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecLog2( const VecT& vec )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_LOG2, VecT, VecT>::generateASM( vec );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecTanh( const VecT& vec )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_TANH, VecT, VecT>::generateASM( vec );
}

template <typename VecTOut, typename VecTIn>
static __forceinline__ __device__ VecTOut optixCoopVecCvt( const VecTIn& vec )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_CVT, VecTOut, VecTIn>::generateASM( vec );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMin( const VecT& vecA, const VecT& vecB )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_MIN, VecT, VecT>::generateASM( vecA, vecB );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMin( const VecT& vecA, typename VecT::value_type B )
{
    VecT vecB( B );
    return optixCoopVecMin( vecA, vecB );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMax( const VecT& vecA, const VecT& vecB )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_MAX, VecT, VecT>::generateASM( vecA, vecB );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMax( const VecT& vecA, typename VecT::value_type B )
{
    VecT vecB( B );
    return optixCoopVecMax( vecA, vecB );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecMul( const VecT& vecA, const VecT& vecB )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_MUL, VecT, VecT>::generateASM( vecA, vecB );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecAdd( const VecT& vecA, const VecT& vecB )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_ADD, VecT, VecT>::generateASM( vecA, vecB );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecSub( const VecT& vecA, const VecT& vecB )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_SUB, VecT, VecT>::generateASM( vecA, vecB );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecStep( const VecT& vecA, const VecT& vecB )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_STEP, VecT, VecT>::generateASM( vecA, vecB );
}

template <typename VecT>
static __forceinline__ __device__ VecT optixCoopVecFFMA( const VecT& vecA, const VecT& vecB, const VecT& vecC )
{
    return optix_internal::OptixCoopVecASMGenerator<optix_internal::OPTIX_COOP_VEC_OP_FFMA, VecT, VecT>::generateASM( vecA, vecB, vecC );
}


namespace optix_internal {
template <typename VecTOut, typename VecTIn, OptixCoopVecElemType inputInterpretation, OptixCoopVecMatrixLayout matrixLayout, bool transpose, unsigned int N, unsigned int K, OptixCoopVecElemType matrixElementType, OptixCoopVecElemType biasElementType>
struct OptixCoopVecMatMulASMGenerator
{
    static const OptixCoopVecElemType outputElementType =
        optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTOut::value_type>::elementType;
    using outputBitType =
        typename optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTOut::value_type>::bitType;
    static const OptixCoopVecElemType inputElementType =
        optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTIn::value_type>::elementType;
    using inputBitType =
        typename optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTIn::value_type>::bitType;

    __forceinline__ __device__ static VecTOut generateASMPtr( const VecTIn& inputVector,
                                                              CUdeviceptr   matrix,
                                                              unsigned      matrixOffsetInBytes,
                                                              unsigned      rowColumnStrideInBytes,
                                                              CUdeviceptr   bias,
                                                              unsigned      biasOffsetInBytes )
    {
        VecTOut result;
        // clang-format off
        asm( "call"
             "(),"
             "_optix_matvecmul_ptr,"
             "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17);"
             :
             : "r"( outputElementType ), "r"( VecTOut::size ),
               "r"( inputElementType), "r"( VecTIn::size ), "r"( inputInterpretation ),
               "r"( N ), "r"( K ),
               "l"( matrix ), "r"( matrixOffsetInBytes ), "r"( rowColumnStrideInBytes ),
               "r"( matrixLayout ), "r"( (unsigned)transpose ), "r"( matrixElementType ),
               "l"( bias ), "r"( biasOffsetInBytes ), "r"( biasElementType ),
               "l"( inputVector.data() ), "l"( result.data() )
          );
        // clang-format on
        return result;
    }

    __forceinline__ __device__ static VecTOut generateASM( const VecTIn& inputVector,
                                                           CUdeviceptr   matrix,
                                                           unsigned      matrixOffsetInBytes,
                                                           unsigned      rowColumnStrideInBytes,
                                                           CUdeviceptr   bias,
                                                           unsigned      biasOffsetInBytes )
    {
        // If too many elements or elements too large, fall back to the pointer passing method
        if( VecTIn::size > 64 || VecTOut::size > 64 || sizeof( typename VecTIn::value_type ) > sizeof( unsigned int )
            || sizeof( typename VecTOut::value_type ) > sizeof( unsigned int ) )
            return generateASMPtr( inputVector, matrix, matrixOffsetInBytes, rowColumnStrideInBytes, bias, biasOffsetInBytes );
        else
        {
            // This code needs to live in an else, block otherwise the compiler will
            // complain about the loop being unreachable.
            unsigned int I[64];
            unsigned int O[64];
            for( unsigned int i = 0; i < VecTIn::size; ++i )
            {
                I[i] = *( reinterpret_cast<const inputBitType*>( &( inputVector[i] ) ) );
            }
            if( VecTOut::size <= 16 && VecTIn::size <= 16 )
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15),"
                     "_optix_matvecmul_16xi32,"
                     "(%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,"
                     "%40,%41,%42,%43,%44,%45,%46,%47);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] )
                     : "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ), "r"( VecTIn::size ),
                       "r"( inputInterpretation ), "r"( N ), "r"( K ), "l"( matrix ), "r"( matrixOffsetInBytes ),
                       "r"( rowColumnStrideInBytes ), "r"( matrixLayout ), "r"( (unsigned)transpose ), "r"( matrixElementType ),
                       "l"( bias ), "r"( biasOffsetInBytes ), "r"( biasElementType ), "r"( I[0] ), "r"( I[1] ),
                       "r"( I[2] ), "r"( I[3] ), "r"( I[4] ), "r"( I[5] ), "r"( I[6] ), "r"( I[7] ), "r"( I[8] ),
                       "r"( I[9] ), "r"( I[10] ), "r"( I[11] ), "r"( I[12] ), "r"( I[13] ), "r"( I[14] ), "r"( I[15] ) );
            else
                asm( "call"
                     "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                     "26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%"
                     "50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63),"
                     "_optix_matvecmul_64xi32,"
                     "(%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,"
                     "%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%"
                     "109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127,%"
                     "128,%129,%130,%131,%132,%133,%134,%135,%136,%137,%138,%139,%140,%141,%142,%143);"
                     : "=r"( O[0] ), "=r"( O[1] ), "=r"( O[2] ), "=r"( O[3] ), "=r"( O[4] ), "=r"( O[5] ), "=r"( O[6] ),
                       "=r"( O[7] ), "=r"( O[8] ), "=r"( O[9] ), "=r"( O[10] ), "=r"( O[11] ), "=r"( O[12] ),
                       "=r"( O[13] ), "=r"( O[14] ), "=r"( O[15] ), "=r"( O[16] ), "=r"( O[17] ), "=r"( O[18] ),
                       "=r"( O[19] ), "=r"( O[20] ), "=r"( O[21] ), "=r"( O[22] ), "=r"( O[23] ), "=r"( O[24] ),
                       "=r"( O[25] ), "=r"( O[26] ), "=r"( O[27] ), "=r"( O[28] ), "=r"( O[29] ), "=r"( O[30] ),
                       "=r"( O[31] ), "=r"( O[32] ), "=r"( O[33] ), "=r"( O[34] ), "=r"( O[35] ), "=r"( O[36] ),
                       "=r"( O[37] ), "=r"( O[38] ), "=r"( O[39] ), "=r"( O[40] ), "=r"( O[41] ), "=r"( O[42] ),
                       "=r"( O[43] ), "=r"( O[44] ), "=r"( O[45] ), "=r"( O[46] ), "=r"( O[47] ), "=r"( O[48] ),
                       "=r"( O[49] ), "=r"( O[50] ), "=r"( O[51] ), "=r"( O[52] ), "=r"( O[53] ), "=r"( O[54] ),
                       "=r"( O[55] ), "=r"( O[56] ), "=r"( O[57] ), "=r"( O[58] ), "=r"( O[59] ), "=r"( O[60] ),
                       "=r"( O[61] ), "=r"( O[62] ), "=r"( O[63] )
                     : "r"( outputElementType ), "r"( VecTOut::size ), "r"( inputElementType ), "r"( VecTIn::size ),
                       "r"( inputInterpretation ), "r"( N ), "r"( K ), "l"( matrix ), "r"( matrixOffsetInBytes ),
                       "r"( rowColumnStrideInBytes ), "r"( matrixLayout ), "r"( (unsigned)transpose ),
                       "r"( matrixElementType ), "l"( bias ), "r"( biasOffsetInBytes ), "r"( biasElementType ), "r"( I[0] ),
                       "r"( I[1] ), "r"( I[2] ), "r"( I[3] ), "r"( I[4] ), "r"( I[5] ), "r"( I[6] ), "r"( I[7] ),
                       "r"( I[8] ), "r"( I[9] ), "r"( I[10] ), "r"( I[11] ), "r"( I[12] ), "r"( I[13] ), "r"( I[14] ),
                       "r"( I[15] ), "r"( I[16] ), "r"( I[17] ), "r"( I[18] ), "r"( I[19] ), "r"( I[20] ), "r"( I[21] ),
                       "r"( I[22] ), "r"( I[23] ), "r"( I[24] ), "r"( I[25] ), "r"( I[26] ), "r"( I[27] ), "r"( I[28] ),
                       "r"( I[29] ), "r"( I[30] ), "r"( I[31] ), "r"( I[32] ), "r"( I[33] ), "r"( I[34] ), "r"( I[35] ),
                       "r"( I[36] ), "r"( I[37] ), "r"( I[38] ), "r"( I[39] ), "r"( I[40] ), "r"( I[41] ), "r"( I[42] ),
                       "r"( I[43] ), "r"( I[44] ), "r"( I[45] ), "r"( I[46] ), "r"( I[47] ), "r"( I[48] ), "r"( I[49] ),
                       "r"( I[50] ), "r"( I[51] ), "r"( I[52] ), "r"( I[53] ), "r"( I[54] ), "r"( I[55] ), "r"( I[56] ),
                       "r"( I[57] ), "r"( I[58] ), "r"( I[59] ), "r"( I[60] ), "r"( I[61] ), "r"( I[62] ), "r"( I[63] ) );
            VecTOut result;
            for( unsigned int i = 0; i < VecTOut::size; ++i )
            {
                outputBitType o = O[i];
                result[i]       = *( reinterpret_cast<typename VecTOut::value_type*>( &( o ) ) );
            }
            return result;
        }
    }
};

template <typename VecTIn>
struct OptixCoopVecReduceSumAccumulateASMGenerator
{
    static const OptixCoopVecElemType inputElementType =
        optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTIn::value_type>::elementType;
    using inputBitType =
        typename optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTIn::value_type>::bitType;

    __forceinline__ __device__ static void generateASMPtr( const VecTIn& vecA, CUdeviceptr outputVector, unsigned offsetInBytes )
    {
        asm volatile(
            "call"
            "(),"
            "_optix_reduce_sum_accumulate_ptr,"
            "(%0,%1,%2,%3,%4);"
            :
            : "r"( inputElementType ), "r"( VecTIn::size ), "l"( outputVector ), "r"( offsetInBytes ), "l"( vecA.data() ) );
    }

    __forceinline__ __device__ static void generateASM( const VecTIn& vecA, CUdeviceptr outputVector, unsigned offsetInBytes )
    {
        if( VecTIn::size > 64 || sizeof( typename VecTIn::value_type ) > sizeof( unsigned int ) )
            return generateASMPtr( vecA, outputVector, offsetInBytes );
        else
        {
            // This code needs to live in an else, block otherwise the compiler will
            // complain about the loop being unreachable.
            unsigned int IA[64];
            for( unsigned int i = 0; i < VecTIn::size; ++i )
            {
                IA[i] = *( reinterpret_cast<const inputBitType*>( &( vecA[i] ) ) );
            }
            if( VecTIn::size <= 16 )
                asm volatile(
                    "call"
                    "(),"
                    "_optix_reduce_sum_accumulate_16xi32,"
                    "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19);"
                    :
                    : "r"( inputElementType ), "r"( VecTIn::size ), "l"( outputVector ), "r"( offsetInBytes ),
                      "r"( IA[0] ), "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ), "r"( IA[5] ), "r"( IA[6] ),
                      "r"( IA[7] ), "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ), "r"( IA[11] ), "r"( IA[12] ),
                      "r"( IA[13] ), "r"( IA[14] ), "r"( IA[15] ) );
            else
                asm volatile(
                    "call"
                    "(),"
                    "_optix_reduce_sum_accumulate_64xi32,"
                    "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                    "26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%"
                    "50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
                    "%64,%65,%66,%67);"
                    :
                    : "r"( inputElementType ), "r"( VecTIn::size ), "l"( outputVector ), "r"( offsetInBytes ), "r"( IA[0] ),
                      "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ), "r"( IA[5] ), "r"( IA[6] ), "r"( IA[7] ),
                      "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ), "r"( IA[11] ), "r"( IA[12] ), "r"( IA[13] ), "r"( IA[14] ),
                      "r"( IA[15] ), "r"( IA[16] ), "r"( IA[17] ), "r"( IA[18] ), "r"( IA[19] ), "r"( IA[20] ),
                      "r"( IA[21] ), "r"( IA[22] ), "r"( IA[23] ), "r"( IA[24] ), "r"( IA[25] ), "r"( IA[26] ),
                      "r"( IA[27] ), "r"( IA[28] ), "r"( IA[29] ), "r"( IA[30] ), "r"( IA[31] ), "r"( IA[32] ),
                      "r"( IA[33] ), "r"( IA[34] ), "r"( IA[35] ), "r"( IA[36] ), "r"( IA[37] ), "r"( IA[38] ),
                      "r"( IA[39] ), "r"( IA[40] ), "r"( IA[41] ), "r"( IA[42] ), "r"( IA[43] ), "r"( IA[44] ),
                      "r"( IA[45] ), "r"( IA[46] ), "r"( IA[47] ), "r"( IA[48] ), "r"( IA[49] ), "r"( IA[50] ),
                      "r"( IA[51] ), "r"( IA[52] ), "r"( IA[53] ), "r"( IA[54] ), "r"( IA[55] ), "r"( IA[56] ), "r"( IA[57] ),
                      "r"( IA[58] ), "r"( IA[59] ), "r"( IA[60] ), "r"( IA[61] ), "r"( IA[62] ), "r"( IA[63] ) );
        }
    }
};

template <typename VecTA, typename VecTB, OptixCoopVecMatrixLayout matrixLayout>
struct OptixCoopVecOuterProductAccumulateASMGenerator
{
    static const OptixCoopVecElemType vecAElementType =
        optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTA::value_type>::elementType;
    using vecABitType =
        typename optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTA::value_type>::bitType;
    static const OptixCoopVecElemType vecBElementType =
        optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTB::value_type>::elementType;
    using vecBBitType =
        typename optix_internal::coop_vec_type_traits::OptixCoopVecElemTypeTrait<typename VecTB::value_type>::bitType;

    __forceinline__ __device__ static void generateASMPtr( const VecTA& vecA,
                                                           const VecTB& vecB,
                                                           CUdeviceptr  outputMatrix,
                                                           unsigned     offsetInBytes,
                                                           unsigned     rowColumnStrideInBytes )
    {
        asm volatile(
            "call"
            "(),"
            "_optix_outer_product_accumulate_ptr,"
            "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9);"
            :
            : "r"( vecAElementType ), "r"( VecTA::size ), "r"( vecBElementType ), "r"( VecTB::size ), "l"( outputMatrix ),
              "r"( offsetInBytes ), "r"( matrixLayout ), "r"( rowColumnStrideInBytes ), "l"( vecA.data() ), "l"( vecB.data() ) );
    }

    __forceinline__ __device__ static void generateASM( const VecTA& vecA,
                                                        const VecTB& vecB,
                                                        CUdeviceptr  outputMatrix,
                                                        unsigned     offsetInBytes,
                                                        unsigned     rowColumnStrideInBytes )
    {
        if( VecTA::size > 64 || VecTB::size > 64 || sizeof( typename VecTA::value_type ) > sizeof( unsigned int )
            || sizeof( typename VecTB::value_type ) > sizeof( unsigned int ) )
            return generateASMPtr( vecA, vecB, outputMatrix, offsetInBytes, rowColumnStrideInBytes );
        else
        {
            // This code needs to live in an else, block otherwise the compiler will
            // complain about the loop being unreachable.
            unsigned int IA[64];
            unsigned int IB[64];
            for( unsigned int i = 0; i < VecTA::size; ++i )
            {
                IA[i] = *( reinterpret_cast<const vecABitType*>( &( vecA[i] ) ) );
            }
            for( unsigned int i = 0; i < VecTB::size; ++i )
            {
                IB[i] = *( reinterpret_cast<const vecBBitType*>( &( vecB[i] ) ) );
            }
            if( VecTB::size <= 16 && VecTA::size <= 16 )
                asm volatile(
                    "call"
                    "(),"
                    "_optix_outer_product_accumulate_16xi32,"
                    "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                    "26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39);"
                    :
                    : "r"( vecAElementType ), "r"( VecTA::size ), "r"( vecBElementType ), "r"( VecTB::size ),
                      "l"( outputMatrix ), "r"( offsetInBytes ), "r"( matrixLayout ), "r"( rowColumnStrideInBytes ),
                      "r"( IA[0] ), "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ), "r"( IA[5] ), "r"( IA[6] ),
                      "r"( IA[7] ), "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ), "r"( IA[11] ), "r"( IA[12] ),
                      "r"( IA[13] ), "r"( IA[14] ), "r"( IA[15] ), "r"( IB[0] ), "r"( IB[1] ), "r"( IB[2] ),
                      "r"( IB[3] ), "r"( IB[4] ), "r"( IB[5] ), "r"( IB[6] ), "r"( IB[7] ), "r"( IB[8] ), "r"( IB[9] ),
                      "r"( IB[10] ), "r"( IB[11] ), "r"( IB[12] ), "r"( IB[13] ), "r"( IB[14] ), "r"( IB[15] ) );
            else
                asm volatile(
                    "call"
                    "(),"
                    "_optix_outer_product_accumulate_64xi32,"
                    "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%"
                    "26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%"
                    "50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%"
                    "74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%"
                    "98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,"
                    "%118,%119,%120,%121,%122,%123,%124,%125,%126,%127,%128,%129,%130,%131,%132,%133,%134,%135);"
                    :
                    : "r"( vecAElementType ), "r"( VecTA::size ), "r"( vecBElementType ), "r"( VecTB::size ),
                      "l"( outputMatrix ), "r"( offsetInBytes ), "r"( matrixLayout ), "r"( rowColumnStrideInBytes ),
                      "r"( IA[0] ), "r"( IA[1] ), "r"( IA[2] ), "r"( IA[3] ), "r"( IA[4] ), "r"( IA[5] ), "r"( IA[6] ),
                      "r"( IA[7] ), "r"( IA[8] ), "r"( IA[9] ), "r"( IA[10] ), "r"( IA[11] ), "r"( IA[12] ),
                      "r"( IA[13] ), "r"( IA[14] ), "r"( IA[15] ), "r"( IA[16] ), "r"( IA[17] ), "r"( IA[18] ),
                      "r"( IA[19] ), "r"( IA[20] ), "r"( IA[21] ), "r"( IA[22] ), "r"( IA[23] ), "r"( IA[24] ),
                      "r"( IA[25] ), "r"( IA[26] ), "r"( IA[27] ), "r"( IA[28] ), "r"( IA[29] ), "r"( IA[30] ),
                      "r"( IA[31] ), "r"( IA[32] ), "r"( IA[33] ), "r"( IA[34] ), "r"( IA[35] ), "r"( IA[36] ),
                      "r"( IA[37] ), "r"( IA[38] ), "r"( IA[39] ), "r"( IA[40] ), "r"( IA[41] ), "r"( IA[42] ),
                      "r"( IA[43] ), "r"( IA[44] ), "r"( IA[45] ), "r"( IA[46] ), "r"( IA[47] ), "r"( IA[48] ),
                      "r"( IA[49] ), "r"( IA[50] ), "r"( IA[51] ), "r"( IA[52] ), "r"( IA[53] ), "r"( IA[54] ),
                      "r"( IA[55] ), "r"( IA[56] ), "r"( IA[57] ), "r"( IA[58] ), "r"( IA[59] ), "r"( IA[60] ),
                      "r"( IA[61] ), "r"( IA[62] ), "r"( IA[63] ), "r"( IB[0] ), "r"( IB[1] ), "r"( IB[2] ),
                      "r"( IB[3] ), "r"( IB[4] ), "r"( IB[5] ), "r"( IB[6] ), "r"( IB[7] ), "r"( IB[8] ), "r"( IB[9] ),
                      "r"( IB[10] ), "r"( IB[11] ), "r"( IB[12] ), "r"( IB[13] ), "r"( IB[14] ), "r"( IB[15] ),
                      "r"( IB[16] ), "r"( IB[17] ), "r"( IB[18] ), "r"( IB[19] ), "r"( IB[20] ), "r"( IB[21] ),
                      "r"( IB[22] ), "r"( IB[23] ), "r"( IB[24] ), "r"( IB[25] ), "r"( IB[26] ), "r"( IB[27] ),
                      "r"( IB[28] ), "r"( IB[29] ), "r"( IB[30] ), "r"( IB[31] ), "r"( IB[32] ), "r"( IB[33] ),
                      "r"( IB[34] ), "r"( IB[35] ), "r"( IB[36] ), "r"( IB[37] ), "r"( IB[38] ), "r"( IB[39] ),
                      "r"( IB[40] ), "r"( IB[41] ), "r"( IB[42] ), "r"( IB[43] ), "r"( IB[44] ), "r"( IB[45] ),
                      "r"( IB[46] ), "r"( IB[47] ), "r"( IB[48] ), "r"( IB[49] ), "r"( IB[50] ), "r"( IB[51] ),
                      "r"( IB[52] ), "r"( IB[53] ), "r"( IB[54] ), "r"( IB[55] ), "r"( IB[56] ), "r"( IB[57] ),
                      "r"( IB[58] ), "r"( IB[59] ), "r"( IB[60] ), "r"( IB[61] ), "r"( IB[62] ), "r"( IB[63] ) );
        }
    }
};
}  // end namespace optix_internal


template <typename VecTOut,  //
          typename VecTIn,
          OptixCoopVecElemType     inputInterpretation,
          OptixCoopVecMatrixLayout matrixLayout,
          bool                     transpose,
          unsigned int             N,
          unsigned int             K,
          OptixCoopVecElemType     matrixElementType,
          OptixCoopVecElemType     biasElementType>
static __forceinline__ __device__ VecTOut optixCoopVecMatMul( const VecTIn& inputVector,
                                                              CUdeviceptr matrix,  // 64 byte aligned, Array of KxN elements
                                                              unsigned    matrixOffsetInBytes,  // 64 byte aligned
                                                              CUdeviceptr bias,  // 16 byte aligned, Array of N elements
                                                              unsigned    biasOffsetInBytes,  // 16 byte aligned
                                                              unsigned    rowColumnStrideInBytes )
{
    return optix_internal::OptixCoopVecMatMulASMGenerator<VecTOut, VecTIn, inputInterpretation, matrixLayout, transpose, N, K, matrixElementType, biasElementType>::generateASM(
        inputVector, matrix, matrixOffsetInBytes, rowColumnStrideInBytes, bias, biasOffsetInBytes );
}

template <typename VecTOut,  //
          typename VecTIn,
          OptixCoopVecElemType     inputInterpretation,
          OptixCoopVecMatrixLayout matrixLayout,
          bool                     transpose,
          unsigned int             N,
          unsigned int             K,
          OptixCoopVecElemType     matrixElementType>
static __forceinline__ __device__ VecTOut optixCoopVecMatMul( const VecTIn& inputVector,
                                                              CUdeviceptr matrix,  // 64 byte aligned, Array of KxN elements
                                                              unsigned matrixOffsetInBytes,  // 64 byte aligned
                                                              unsigned rowColumnStrideInBytes )
{
    return optix_internal::OptixCoopVecMatMulASMGenerator<VecTOut, VecTIn, inputInterpretation, matrixLayout, transpose, N, K, matrixElementType,
                                                          OPTIX_COOP_VEC_ELEM_TYPE_UNKNOWN>::generateASM( inputVector, matrix,
                                                                                                          matrixOffsetInBytes,
                                                                                                          rowColumnStrideInBytes,
                                                                                                          0, 0 );
}

template <typename VecTIn>
static __forceinline__ __device__ void optixCoopVecReduceSumAccumulate( const VecTIn& inputVector, CUdeviceptr outputVector, unsigned offsetInBytes )
{
    optix_internal::OptixCoopVecReduceSumAccumulateASMGenerator<VecTIn>::generateASM( inputVector, outputVector, offsetInBytes );
}

template <typename VecTA, typename VecTB, OptixCoopVecMatrixLayout matrixLayout>
static __forceinline__ __device__ void optixCoopVecOuterProductAccumulate( const VecTA& vecA,
                                                                           const VecTB& vecB,
                                                                           CUdeviceptr  outputMatrix,
                                                                           unsigned     offsetInBytes,
                                                                           unsigned     rowColumnStrideInBytes )
{
    optix_internal::OptixCoopVecOuterProductAccumulateASMGenerator<VecTA, VecTB, matrixLayout>::generateASM(
        vecA, vecB, outputMatrix, offsetInBytes, rowColumnStrideInBytes );
}


template <unsigned int N, unsigned int K, OptixCoopVecElemType elementType, OptixCoopVecMatrixLayout layout, unsigned int rowColumnStrideInBytes>
static __forceinline__ __device__ unsigned int optixCoopVecGetMatrixSize()
{
    unsigned int size;
    asm( "call"
         "(%0),"
         "_optix_coop_vec_get_matrix_size,"
         "(%1,%2,%3,%4,%5);"
         : "=r"( size )
         : "r"( N ), "r"( K ), "r"( elementType ), "r"( layout ), "r"( rowColumnStrideInBytes ) );
    return size;
}

#endif  // #ifndef OPTIX_OPTIX_DEVICE_IMPL_COOP_VEC_H
