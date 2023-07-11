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

#include "vec.h"
#include "box.h"
#include <mutex>
#include <vector>

namespace bvhlib {

  using ospcommon::vec3f;
  using ospcommon::vec3fa;
  using ospcommon::box3f;
  using ospcommon::box3fa;

  struct BuildPrim : public box3fa
  {
    inline vec3fa center() const { return 0.5f*(lower+upper); }
  };

  /*! abstraction of a geometric object that can multiple primitmives */
  struct Geometry {
    virtual size_t numPrimitives() const = 0;
    virtual void   initBuildPrim(BuildPrim &bp, const size_t primID) const = 0;
  };

  /*! a binary, axis-aligned bounding box based, bounding volume
      hierarchy */
  struct BVH {
    struct Node {
      vec3f    lower;
      uint32_t isLeaf;
      vec3f    upper;
      uint32_t child;
    };

    void build(const Geometry *object);
    
    /*! build recursively, given the bounding box of the primitives() centroids */
    void buildRec(size_t nodeID,
                  const box3fa &centBounds,
                  const BuildPrim *buildPrim,
                  std::vector<size_t> &primID);
        
    std::vector<Node>   nodeList;
    std::mutex          mutex;
  };

}
