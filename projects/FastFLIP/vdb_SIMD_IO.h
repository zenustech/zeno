#pragma once
#include "openvdb/openvdb.h"
#include <immintrin.h>
struct vdb_SIMD_IO {
	static void get_simd_vector(
		__m256& out_vector,
		const float* in_data,
		uint32_t load_offset,
		const __m256& default_vector) {
		if (!in_data) {
			out_vector = default_vector;
		}
		else {
			out_vector = _mm256_loadu_ps(in_data + load_offset);
		}
	}

	static void get_simd_vector_unsafe(
		__m256& out_vector,
		const float* in_data,
		uint32_t load_offset) {
		out_vector = _mm256_loadu_ps(in_data + load_offset);
	}

	static void get_neg_z_simd_vector(
		__m256& out_vector,
		const float* this_data,
		const float* neg_neib_z_data,
		uint32_t load_offset,
		const  __m256& default_vector) {
		//the beginning of the vector in a leaf
		//this vector
		//| 8 9 ... 15 |
		__m256 original_this_vector;
		if (this_data) {
			original_this_vector = _mm256_loadu_ps(this_data + load_offset);
		}
		else {
			original_this_vector = default_vector;
			if (!neg_neib_z_data) {
				out_vector = default_vector;
				return;
			}
		}

		//the vector in the negative neighbor
		//|0 1 2 3 4 5 6 7|
		__m256 neg_neib_vector;
		if (neg_neib_z_data) {
			neg_neib_vector = _mm256_set1_ps(*(neg_neib_z_data + load_offset + 7));
		}
		else {
			neg_neib_vector = default_vector;
		}

		//(1),(2)
		//permute each vector, this happens within each 128 half of the 256 vector
		//lowz  highz 
		//|0 1 2 3| -> |3 0 1 2|
		//hence imm8[1:0]=3
		//imm8[3:2] = 0
		//imm8[5:4] = 1
		//imm8[7:6] = 2
		//use _mm256_permute_ps
		const uint8_t permute_imm8 = 0b10010011;
		original_this_vector = _mm256_permute_ps(original_this_vector, permute_imm8);


		//(3)
		//the lower part of this is the high part of the permuted neg neiber
		//the higher part of this is the lower part of the permuted original 
		//use _mm256_permute2f128_ps
		/*
		DEFINE SELECT4(src1, src2, control) {
		CASE(control[1:0]) OF
		0:	tmp[127:0] := src1[127:0]
		1:	tmp[127:0] := src1[255:128]
		2:	tmp[127:0] := src2[127:0]
		3:	tmp[127:0] := src2[255:128]
		ESAC
		IF control[3]
			tmp[127:0] := 0
		FI
		RETURN tmp[127:0]
		}
		dst[127:0] := SELECT4(a[255:0], b[255:0], imm8[3:0])
		dst[255:128] := SELECT4(a[255:0], b[255:0], imm8[7:4])
		dst[MAX:256] := 0
		*/

		//we use src1 = neg neib
		//       src2 = original 
		//lower 128 is high src1, imm8[3:0] = 1
		//higher 128 is low src2, imm8[7:4] = 2
		__m256 high_neg_neib_low_original;

		const uint8_t interweave_imm8 = 0b00100001;
		high_neg_neib_low_original = _mm256_permute2f128_ps(
			neg_neib_vector, original_this_vector,
			interweave_imm8);

		//blend the permuted original lane with the 
		//the 0 and 4 position will be from the high_neg_neib
		//the rest bits are from the permuted original
		const uint8_t blend_imm8 = 0b00010001;
		out_vector = _mm256_blend_ps(original_this_vector, high_neg_neib_low_original, blend_imm8);
	}

	static void get_pos_z_simd_vector(__m256& out_vector,
		const float* this_data,
		const float* pos_neib_z_data,
		uint32_t load_offset,
		const __m256& default_vector) {
		//this vector
		//| 8 9 ... 15 |
		__m256 original_this_vector;
		if (this_data) {
			original_this_vector = _mm256_loadu_ps(this_data + load_offset);
		}
		else {
			original_this_vector = default_vector;
			if (!pos_neib_z_data) {
				out_vector = default_vector;
				return;
			}
		}

		//the vector in the negative neighbor
		//|0 1 2 3 4 5 6 7|
		__m256 pos_neib_vector;
		if (pos_neib_z_data) {
			pos_neib_vector = _mm256_set1_ps(*(pos_neib_z_data + load_offset));
		}
		else {
			pos_neib_vector = default_vector;
		}

		//(1),(2)
		//permute each vector, this happens within each 128 half of the 256 vector
		//lowz  highz 
		//|0 1 2 3| -> |1 2 3 0|
		//hence imm8[1:0]=1
		//imm8[3:2] = 2
		//imm8[5:4] = 3
		//imm8[7:6] = 0
		//use _mm256_permute_ps
		const uint8_t permute_imm8 = 0b00111001;
		original_this_vector = _mm256_permute_ps(original_this_vector, permute_imm8);
		//pos_neib_lane = _mm256_permute_ps(pos_neib_lane, permute_imm8);


		//(3)
		//the lower part of this is the high part of the permuted neg neiber
		//the higher part of this is the lower part of the permuted original 
		//use _mm256_permute2f128_ps
		/*
		DEFINE SELECT4(src1, src2, control) {
		CASE(control[1:0]) OF
		0:	tmp[127:0] := src1[127:0]
		1:	tmp[127:0] := src1[255:128]
		2:	tmp[127:0] := src2[127:0]
		3:	tmp[127:0] := src2[255:128]
		ESAC
		IF control[3]
			tmp[127:0] := 0
		FI
		RETURN tmp[127:0]
		}
		dst[127:0] := SELECT4(a[255:0], b[255:0], imm8[3:0])
		dst[255:128] := SELECT4(a[255:0], b[255:0], imm8[7:4])
		dst[MAX:256] := 0
		*/

		//we use src1 = original
		//       src2 = high neib
		//lower 128 is high src1 imm8[3:0] = 1
		//higher 128 is low src2, imm8[7:4] = 2
		__m256 high_original_low_pos_neib;

		//
		const uint8_t interweave_imm8 = 0b00100001;
		high_original_low_pos_neib = _mm256_permute2f128_ps(
			original_this_vector, pos_neib_vector,
			interweave_imm8);

		//blend the permuted original lane with the 
		//the 3 and 7 position will be from the high_neg_neib
		//the rest bits are from the permuted original
		const uint8_t blend_imm8 = 0b10001000;
		out_vector = _mm256_blend_ps(original_this_vector, high_original_low_pos_neib, blend_imm8);
	}

};