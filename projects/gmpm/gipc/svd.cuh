#pragma once
#ifndef _SVD_CUH_
#define _SVD_CUH_
#include <cuda_runtime.h>
namespace __GEIGEN__ {
	__device__
		void svd(
			float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33,			// input A     
			float& u11, float& u12, float& u13, float& u21, float& u22, float& u23, float& u31, float& u32, float& u33,	// output U      
			float& s11,
			//float &s12, float &s13, float &s21, 
			float& s22,
			//float &s23, float &s31, float &s32, 
			float& s33,	// output S
			float& v11, float& v12, float& v13, float& v21, float& v22, float& v23, float& v31, float& v32, float& v33	// output V
		);
}

#endif