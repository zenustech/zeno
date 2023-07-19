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

#include "distanceQueries.h"
#include "bvh.h"

#include <vector>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <queue>

namespace bvhlib {

  /*! result of a single closest-point query */
  struct QueryResult {
    
    /*! closest point on given triangle */
    vec3fa  point;
    
    /*! distance to query point */
    float   distance;
    
    /* primitmive it in scene object */
    int32_t primID;
  };

  inline std::ostream &operator<<(std::ostream &o, const __m128 &v)
  {
    float *f = (float*)&v;
    o << "{" << f[0] << "," << f[1] << "," << f[2] << "," << f[3] << "}";
    return o;
  }

  inline float computeDistance(const BVH::Node *node, const vec3fa &P)
  {
#if 0
    // vectorized version - but doesn't cost enough to actually gain anything :-(
    const __m128 lo = (__m128&)node->lower;
    const __m128 hi = (__m128&)node->upper;
    const __m128 p  = (__m128&)P;
    const __m128 pp = _mm_min_ps(_mm_max_ps(p,lo),hi);
    const __m128 d  = _mm_sub_ps(pp,p);
    const __m128 dd = _mm_mul_ps(d,d);
    float xx,yy,zz;
    (int&)xx = _mm_extract_ps(dd,0);
    (int&)yy = _mm_extract_ps(dd,1);
    (int&)zz = _mm_extract_ps(dd,2);
    const float vec = sqrtf(xx+yy+zz);
    return vec;
#else
    const vec3fa Pclamped = min(max(P,(const vec3fa&)node->lower),(const vec3fa&)node->upper);
    return length(P-Pclamped);
#endif
  }
  
  inline vec3fa projectToEdge(const vec3fa &P, const vec3fa &v0, const vec3fa &e0)
  {
    const float den = dot(e0,e0);
    if (den == 0.f) return v0;

#if 0
    float f = dot(P-v0,e0) / den;
#else
    float f = dot(P-v0,e0) / (double)den;
#endif
    f = std::max(0.f,std::min(1.f,f));
    return v0+f*e0;
  }

  inline vec3fa projectToPlane(float &dist, const vec3fa &P, const vec3fa &N, const vec3fa &A)
  {
    const float den = dot(N,N);
    if (den == 0.f) {
      const vec3fa PP = A;
      dist = length(PP-P);
      return PP;
    } else {
      const vec3fa PP = P - float((dot(P-A,N)/(double)den)) * N;
      dist = length(PP-P);
      return PP;
    }
  }

  inline void checkEdge(vec3fa &closestPoint, float &closestDist,
                        const vec3fa &queryPoint,
                        const vec3fa &v0, const vec3fa &e0)
  {
    const vec3fa PP = projectToEdge(queryPoint,v0,e0);
    const float dist = length(PP-queryPoint);
    if (dist < closestDist) {
      closestDist = dist;
      closestPoint = PP;
    }
  }


  struct QueryObject : public bvhlib::Geometry {
    
    /*! destructor - clean up */
    virtual ~QueryObject() {};
    
    /*! test new point, and update 'result' if it's closer */
    virtual void testPrimAndUpdateIfCloser(QueryResult &result,
                                           const vec3fa &P,
                                           const size_t primID) = 0;

    BVH       bvh;
  };

  struct Triangle {
    vec3fa v0;
    vec3fa v1;
    vec3fa v2;
  };

  template<typename coord_t, typename index_t, bool has_strides>
  struct GeneralTriangleMesh : public QueryObject {

    /*! construct a new general triangle mesh */
    GeneralTriangleMesh(const coord_t *vtx_x,
                        const coord_t *vtx_y,
                        const coord_t *vtx_z,
                        const size_t   vtx_stride,
                        const index_t *idx_x, 
                        const index_t *idx_y,
                        const index_t *idx_z,
                        const size_t   idx_stride,
                        const size_t   numTriangles);

    /*! destructor - clean up */
    virtual ~GeneralTriangleMesh()
    { /* nothing to do - we don't own any of the arrays, so nothing to free */ }

    /*! get a triangle's vertices to operate on, hiding thinke like addressing */
    inline Triangle getTriangle(const size_t primID) const;
    
    /*! number of total primitmives (inherited from Geometry: the BVH
      builder needs to know this) */
    virtual size_t numPrimitives() const
    { return numTriangles; }

    /*! test a new point, and update 'result' if it's closer */
    virtual void initBuildPrim(BuildPrim &bp, const size_t primID) const;

    /*! test a new point, and update 'result' if it's closer */
    virtual void testPrimAndUpdateIfCloser(QueryResult &result,
                                           const vec3fa &queryPoint,
                                           const size_t primID);

    const coord_t *const vtx_x;
    const coord_t *const vtx_y;
    const coord_t *const vtx_z;
    const size_t   vtx_stride;
    const index_t *const idx_x;
    const index_t *const idx_y;
    const index_t *const idx_z;
    const size_t   idx_stride;
    const size_t   numTriangles;
    
  };


  template<typename coord_t, typename index_t, bool has_strides>
  GeneralTriangleMesh<coord_t,index_t,has_strides>::GeneralTriangleMesh(const coord_t *vtx_x,
                                                                        const coord_t *vtx_y,
                                                                        const coord_t *vtx_z,
                                                                        const size_t   vtx_stride,
                                                                        const index_t *idx_x,
                                                                        const index_t *idx_y,
                                                                        const index_t *idx_z,
                                                                        const size_t   idx_stride,
                                                                        const size_t   numTriangles)
  : vtx_x(vtx_x),
    vtx_y(vtx_y),
    vtx_z(vtx_z),
    vtx_stride(vtx_stride),
    idx_x(idx_x),
    idx_y(idx_y),
    idx_z(idx_z),
    idx_stride(idx_stride),
    numTriangles(numTriangles)
  {
    bvh.build(this);
  }

  /*! get a triangle's vertices to operate on, hiding thinke like addressing */
  template<typename coord_t, typename index_t, bool has_strides>
  inline Triangle GeneralTriangleMesh<coord_t,index_t,has_strides>::getTriangle(const size_t primID) const
  {
    Triangle t;
    if (has_strides) {
      const size_t i0 = this->idx_x[primID*idx_stride];
      const size_t i1 = this->idx_y[primID*idx_stride];
      const size_t i2 = this->idx_z[primID*idx_stride];
    
      t.v0 = vec3fa(this->vtx_x[i0*vtx_stride],
                    this->vtx_y[i0*vtx_stride],
                    this->vtx_z[i0*vtx_stride]);
      t.v1 = vec3fa(this->vtx_x[i1*vtx_stride],
                    this->vtx_y[i1*vtx_stride],
                    this->vtx_z[i1*vtx_stride]);
      t.v2 = vec3fa(this->vtx_x[i2*vtx_stride],
                    this->vtx_y[i2*vtx_stride],
                    this->vtx_z[i2*vtx_stride]);
    } else {
      const size_t i0 = this->idx_x[primID];
      const size_t i1 = this->idx_y[primID];
      const size_t i2 = this->idx_z[primID];
    
      t.v0 = vec3fa(this->vtx_x[i0],
                    this->vtx_y[i0],
                    this->vtx_z[i0]);
      t.v1 = vec3fa(this->vtx_x[i1],
                    this->vtx_y[i1],
                    this->vtx_z[i1]);
      t.v2 = vec3fa(this->vtx_x[i2],
                    this->vtx_y[i2],
                    this->vtx_z[i2]);
    }
    return t;
  }
  
  /*! test a new point, and update 'result' if it's closer */
  template<typename coord_t, typename index_t, bool has_strides>
  void GeneralTriangleMesh<coord_t,index_t,has_strides>
  ::testPrimAndUpdateIfCloser(QueryResult &result,
                              const vec3fa &queryPoint,
                              const size_t primID)
  {
    const Triangle tri = getTriangle(primID);

    const vec3fa e0 = tri.v1-tri.v0;
    const vec3fa e1 = tri.v2-tri.v1;
    const vec3fa e2 = tri.v0-tri.v2;
    const vec3fa N  = cross(e2,e0);

    const vec3fa Na = cross(N,e1);
    const vec3fa Nb = cross(N,e2);
    const vec3fa Nc = cross(N,e0);
    
    const float a = dot(queryPoint-tri.v1,Na);
    const float b = dot(queryPoint-tri.v2,Nb);
    const float c = dot(queryPoint-tri.v0,Nc);
      
    vec3fa closestPoint;
    float closestDist;
    if (std::min(std::min(a,b),c) >= 0.f)
      closestPoint = projectToPlane(closestDist,queryPoint,N,tri.v0);
    else {
      closestDist = std::numeric_limits<float>::infinity();
      if (a <= 0.f) 
        checkEdge(closestPoint,closestDist,queryPoint,tri.v1,e1);
      if (b <= 0.f) 
        checkEdge(closestPoint,closestDist,queryPoint,tri.v2,e2);
      if (c <= 0.f) 
        checkEdge(closestPoint,closestDist,queryPoint,tri.v0,e0);
    }
      
    if (closestDist >= result.distance) return;
      
    result.distance = closestDist;
    result.point    = closestPoint;
    result.primID   = primID;
  }
  

  /*! test a new point, and update 'result' if it's closer */
  template<typename coord_t, typename index_t, bool has_strides>
  void GeneralTriangleMesh<coord_t,index_t,has_strides>
  ::initBuildPrim(BuildPrim &bp, const size_t primID) const
  {
    const Triangle tri = getTriangle(primID);
    bp.lower = min(min(tri.v0,tri.v1),tri.v2);
    bp.upper = max(max(tri.v0,tri.v1),tri.v2);
  }

  extern "C"
  distance_query_scene rtdqNewTriangleMeshfi(const float   *vertex_x,
                                             const float   *vertex_y,
                                             const float   *vertex_z,
                                             const size_t   vertex_stride,
                                             const int32_t *index_x,
                                             const int32_t *index_y,
                                             const int32_t *index_z,
                                             const size_t   index_stride,
                                             const size_t    numTriangles)
  {
    if (index_stride == 1 && vertex_stride == 1)
      return (distance_query_scene)
        new GeneralTriangleMesh<float,int,0>(vertex_x,vertex_y,vertex_z,vertex_stride,
                                             index_x,index_y,index_z,index_stride,
                                             numTriangles);
    else
      return (distance_query_scene)
        new GeneralTriangleMesh<float,int,1>(vertex_x,vertex_y,vertex_z,vertex_stride,
                                             index_x,index_y,index_z,index_stride,
                                             numTriangles);

  }

  extern "C"
  distance_query_scene rtdqNewTriangleMeshdi(const double   *vertex_x,
                                             const double   *vertex_y,
                                             const double   *vertex_z,
                                             const size_t   vertex_stride,
                                             const int32_t *index_x,
                                             const int32_t *index_y,
                                             const int32_t *index_z,
                                             const size_t   index_stride,
                                             const size_t    numTriangles)
  {
    if (index_stride == 1 && vertex_stride == 1)
      return (distance_query_scene)
        new GeneralTriangleMesh<double,int,0>(vertex_x,vertex_y,vertex_z,vertex_stride,
                                              index_x,index_y,index_z,index_stride,
                                              numTriangles);
    else
      return (distance_query_scene)
        new GeneralTriangleMesh<double,int,1>(vertex_x,vertex_y,vertex_z,vertex_stride,
                                              index_x,index_y,index_z,index_stride,
                                              numTriangles);

  }

  void oneQuery(QueryResult &result,
                QueryObject *qo,
                const vec3fa &point)
  {
    std::priority_queue<std::pair<float,const BVH::Node *>,
                        std::vector<std::pair<float,const BVH::Node *>>,
                        std::greater<std::pair<float,const BVH::Node *>>
                        > traversalQueue;
    
    // push sentinel, so we never have to check if the queue runs dry.
    traversalQueue.push(std::pair<float,const BVH::Node *>
                        (std::numeric_limits<float>::infinity(),NULL));
    
    const std::vector<BVH::Node> &nodeList = qo->bvh.nodeList;
    const BVH::Node *node = &nodeList[0];
    while (node) {
      if (!node->isLeaf) {
        // this is a inner node ...
        const BVH::Node *const child0 = &nodeList[node->child+0];
        const BVH::Node *const child1 = &nodeList[node->child+1];
        const float dist0 = computeDistance(child0,point);
        const float dist1 = computeDistance(child1,point);

        /* fast path: check if closer of the two children is already
           as good as anything the queue has to offer - because if so,
           we can save the pushing and popping of this node, and set
           to it right away */
        if (dist0 < dist1) {
          if (dist0 <= std::min(traversalQueue.top().first,result.distance)) {
            node = child0;
            if (dist1 < result.distance)
              traversalQueue.push(std::pair<float,const BVH::Node*>(dist1,child1));
            continue;
          }
        } else {
          if (dist1 <= std::min(traversalQueue.top().first,result.distance)) {
            node = child1;
            if (dist0 < result.distance)
              traversalQueue.push(std::pair<float,const BVH::Node*>(dist0,child0));
            continue;
          }
        }

        /* fast-path optimization didn't apply - push both children
           (if they can't be culled, of course), and let the queue
           provide the next node */
        if (dist0 < result.distance)
          traversalQueue.push(std::pair<float,const BVH::Node*>(dist0,child0));
        if (dist1 < result.distance)
          traversalQueue.push(std::pair<float,const BVH::Node*>(dist1,child1));
      } else {
        qo->testPrimAndUpdateIfCloser(result,point,node->child);
      }


      /* need to get a next node to traverse, from the trv queue: */
      
      /* note - _not_ checking for empty, since we know to have a sentinel, anyway */
      // if (traversalQueue.empty()) break;
      
      // closest candidate is already too far?
      if (traversalQueue.top().first >= result.distance) break;

      // closest candidate might be closer: pop it and use it
      node = traversalQueue.top().second;


      if (node == NULL) {
        PING;
        PRINT(node);
        PRINT(result.distance);
        PRINT(traversalQueue.top().first);
      }

      traversalQueue.pop();
    }
  }
  
#define CACHE_LAST_RESULT 0

  extern "C"
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
                                  const size_t numQueryPoints)
  {
    QueryObject *qo = (QueryObject *)scene;
    if (!qo)
      return;

#if CACHE_LAST_RESULT
    vec3f lastPoint;
#endif
    for (size_t qpi=0;qpi<numQueryPoints;qpi++) {
      const vec3fa queryPoint(in_query_point_x[qpi*in_query_point_stride],
                              in_query_point_y[qpi*in_query_point_stride],
                              in_query_point_z[qpi*in_query_point_stride]);

      QueryResult qr;
      qr.distance = std::numeric_limits<float>::infinity();
      qr.primID   = -1;
#if CACHE_LAST_RESULT
      if (qpi > 0) {
        qr.distance = length(queryPoint-lastPoint)*(((1<<22)+1)/float(1<<22));
      }
#endif
      oneQuery(qr,qo,queryPoint);
#if CACHE_LAST_RESULT
      lastPoint = qr.point;
#endif
      if (out_closest_point_pos_x)
        out_closest_point_pos_x[qpi*out_closest_point_pos_stride] = qr.point.x;
      if (out_closest_point_pos_y)
        out_closest_point_pos_y[qpi*out_closest_point_pos_stride] = qr.point.y;
      if (out_closest_point_pos_z)
        out_closest_point_pos_z[qpi*out_closest_point_pos_stride] = qr.point.z;
      if (out_closest_point_primID)
        out_closest_point_primID[qpi*out_closest_point_primID_stride] = qr.primID;
      if (out_closest_point_dist)
        out_closest_point_dist[qpi*out_closest_point_dist_stride] = qr.distance;
    }
  }
  
  extern "C"
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
                                  const size_t numQueryPoints)
  {
    QueryObject *qo = (QueryObject *)scene;
    if (!qo)
      return; 

#if CACHE_LAST_RESULT
    vec3f lastPoint;
#endif
    for (size_t qpi=0;qpi<numQueryPoints;qpi++) {
      const vec3fa queryPoint(in_query_point_x[qpi*in_query_point_stride],
                              in_query_point_y[qpi*in_query_point_stride],
                              in_query_point_z[qpi*in_query_point_stride]);
      
      QueryResult qr;
      qr.distance = std::numeric_limits<float>::infinity();
      qr.primID   = -1;
  
#if CACHE_LAST_RESULT
      if (qpi > 0) {
        qr.distance = length(queryPoint-lastPoint)*(((1<<22)+1)/float(1<<22));
      }
#endif
      oneQuery(qr,qo,queryPoint);
#if CACHE_LAST_RESULT
      lastPoint = qr.point;
#endif
      
      if (out_closest_point_pos_x)
        out_closest_point_pos_x[qpi*out_closest_point_pos_stride] = qr.point.x;
      if (out_closest_point_pos_y)
        out_closest_point_pos_y[qpi*out_closest_point_pos_stride] = qr.point.y;
      if (out_closest_point_pos_z)
        out_closest_point_pos_z[qpi*out_closest_point_pos_stride] = qr.point.z;
      if (out_closest_point_primID)
        out_closest_point_primID[qpi*out_closest_point_primID_stride] = qr.primID;
      if (out_closest_point_dist)
        out_closest_point_dist[qpi*out_closest_point_dist_stride] = qr.distance;
    }
  }
  
  
  /*! destroy a scene created with rtdqNew...() */
  extern "C"
  void rtdqDestroy(distance_query_scene scene)
  {
    if (scene) delete (QueryObject*)scene;
  }

  
}
 
