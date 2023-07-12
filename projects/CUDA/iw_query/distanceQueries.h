// ======================================================================== //
// Copyright 2009-2017 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

/*! \file distanceQueries.h C99/Fortran style API to performing
    distance queries of the sort "given point P and triangles T[],
    find closest point P' \in t, among all triangle t in T */

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  typedef void *distance_query_scene;

  distance_query_scene rtdqNewTriangleMeshdi(const double  *vertex_x,
                                             const double  *vertex_y,
                                             const double  *vertex_z,
                                             const size_t   vertex_stride,
                                             const int32_t *index_x,
                                             const int32_t *index_y,
                                             const int32_t *index_z,
                                             const size_t   index_stride,
                                             const size_t   numTriangles);
  
  distance_query_scene rtdqNewTriangleMeshfi(const float   *vertex_x,
                                             const float   *vertex_y,
                                             const float   *vertex_z,
                                             const size_t   vertex_stride,
                                             const int32_t *index_x,
                                             const int32_t *index_y,
                                             const int32_t *index_z,
                                             const size_t   index_stride,
                                             const size_t    numTriangles);

  /*! destroy a scene created with rtdqNew...() */
  void rtdqDestroy(distance_query_scene scene);
  
  /*! for each point in in_query_point[] array, find the closest point
    P' among all the triangles in the given scene, and store position
    of closest triangle point, primID of the triangle that this point
    belongs to, and distance to that closest point, in the
    corresponding output arrays.
    
    for fullest flexibilty we are passing individual base pointers and
    strides for every individual array member, thus allowing to use
    both C++-style array of structs as well as fortran-suyle list of
    arrays data layouts
  */
  void rtdqComputeClosestPointsfi(distance_query_scene scene,
                                 float   *out_closest_point_pos_x,
                                 float   *out_closest_point_pos_y,
                                 float   *out_closest_point_pos_z,
                                 size_t   out_closest_point_pos_stride,
                                 float   *out_closest_point_dist,
                                 size_t   out_closest_point_dist_stride,
                                 int32_t *out_closest_point_primID,
                                 size_t   out_closest_point_primID_stride,
                                 const float *in_query_point_x,
                                 const float *in_query_point_y,
                                 const float *in_query_point_z,
                                 const size_t in_query_point_stride,
                                 const size_t numQueryPoints);

  void rtdqComputeClosestPointsdi(distance_query_scene scene,
                                 double   *out_closest_point_pos_x,
                                 double   *out_closest_point_pos_y,
                                 double   *out_closest_point_pos_z,
                                 size_t   out_closest_point_pos_stride,
                                 double   *out_closest_point_dist,
                                 size_t   out_closest_point_dist_stride,
                                 int32_t *out_closest_point_primID,
                                 size_t   out_closest_point_primID_stride,
                                 const double *in_query_point_x,
                                 const double *in_query_point_y,
                                 const double *in_query_point_z,
                                 const size_t in_query_point_stride,
                                 const size_t numQueryPoints);
  
#ifdef __cplusplus
}
#endif
