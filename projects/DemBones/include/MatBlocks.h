///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////



/** @file "DemBones/MatBlocks.h"
	@brief Defines some macros to access sub-blocks of packing transformation/position matrices for convenience

	@details These definitions are not included by default although they are used in @p DemBones and @p DemBonesExt 
		(they are undefined at the end of the files).
*/

#ifndef DEM_BONES_MAT_BLOCKS
#define DEM_BONES_MAT_BLOCKS

//! A 4*4 sub-block that represents a transformation matrix, typically @p k denotes frame number and @p j denotes bone index 
#define blk4(k, j) template block<4, 4>((k)*4, (j)*4)

//! The 3*3 rotation part or the transformation matrix #blk4(@p k, @p j)
#define rotMat(k, j) template block<3, 3>((k)*4, (j)*4)

//! The 3*1 translation vector part or the transformation matrix #blk4(@p k, @p j)
#define transVec(k, j) col((j)*4+3).template segment<3>((k)*4)

//! A 3*1 sub-block that represents position of a vertex, typically @p k denotes frame number and @p i denotes vertex index 
#define vec3(k, i) col(i).template segment<3>((k)*3)

#endif
