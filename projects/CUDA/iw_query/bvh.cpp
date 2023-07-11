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

#include "bvh.h"
#include <atomic>

namespace bvhlib {


  inline size_t widestDimension(const box3fa &box) 
  {
    const vec3fa width = box.upper - box.lower;
    size_t widest = width.y > width.x;
    if (width.z > width[widest]) widest = 2;
    return widest;
  };

  // # pragma omp parallel for schedule(dynamic)
  
  void BVH::buildRec(size_t nodeID,
                     const box3fa &centBounds,
                     const BuildPrim *const buildPrim,
                     std::vector<size_t> &primList)
  {
    if (primList.size() == 1) {
      size_t primID = primList[0];
      std::lock_guard (this->mutex);
      nodeList[nodeID].lower       = buildPrim[primID].lower;
      nodeList[nodeID].upper       = buildPrim[primID].upper;
      nodeList[nodeID].isLeaf      = 1;
      nodeList[nodeID].child       = primID;
      return;
    } else {
      std::mutex mutex;

      std::atomic<size_t> lCount(0);
      std::atomic<size_t> rCount(0);
      std::vector<size_t> lPrim; lPrim.resize(primList.size());
      std::vector<size_t> rPrim; rPrim.resize(primList.size());
      box3fa lCentBounds = ospcommon::empty;
      box3fa rCentBounds = ospcommon::empty;

      size_t splitDim = widestDimension(centBounds);
      float  splitPos = centBounds.center()[splitDim];
      // bool   zeroVolume = (centBounds.lower == centBounds.upper);
        
      size_t blockSize = 1000;
      size_t numBlocks = (primList.size()+blockSize-1)/blockSize;
      for (size_t blockID=0;blockID<numBlocks;blockID++) {
        size_t begin   = blockID * blockSize;
        size_t end     = std::min(begin+blockSize,primList.size());
        box3fa lBounds = ospcommon::empty;
        box3fa rBounds = ospcommon::empty;
        for (size_t i=begin;i<end;i++) {
          const size_t primID = primList[i];
          const vec3f center = buildPrim[primID].center();
          if (center[splitDim] < splitPos) {
          // if ((center[splitDim] < splitPos) || (zeroVolume && (i&1))) {
            lBounds.extend(center);
            lPrim[lCount++] = primID;
          } else {
            rBounds.extend(center);
            rPrim[rCount++] = primID;
          }
        }
        std::lock_guard lock(mutex);
        lCentBounds.extend(lBounds);
        rCentBounds.extend(rBounds);
      }

      // sanity-check: if we got at least one empty side, simply split
      // randomly
      if (lCount == 0 || rCount == 0) {
        // std::cout << "abnormal case detected - all prims falling on same side of partitioning place (fixing right now...)" << std::endl;
        lCount = 0;
        rCount = 0;
        lCentBounds = ospcommon::empty;
        rCentBounds = ospcommon::empty;
        for (int i=0;i<primList.size();i++) {
          const size_t primID = primList[i];
          const vec3f center = buildPrim[primID].center();
          if ((i&1) == 0) {
            lCentBounds.extend(center);
            lPrim[lCount++] = primID;
          } else {
            rCentBounds.extend(center);
            rPrim[rCount++] = primID;
          }
        }
      }
      
      // OK, we've done the split ...

      // resize the lists so we don't unnecessarily waste memory
      primList.clear();
      lPrim.resize(lCount);
      rPrim.resize(rCount);

      size_t childID=0;
      { std::lock_guard lock(mutex);
        childID = nodeList.size();
        nodeList.push_back(Node());
        nodeList.push_back(Node());
      }

      buildRec(childID+0,lCentBounds,buildPrim,lPrim);
      buildRec(childID+1,rCentBounds,buildPrim,rPrim);
      { std::lock_guard lock(mutex);
        nodeList[nodeID].lower = min(nodeList[childID+0].lower,
                                     nodeList[childID+1].lower);
        nodeList[nodeID].upper = max(nodeList[childID+0].upper,
                                     nodeList[childID+1].upper);
        nodeList[nodeID].isLeaf   = 0;
        nodeList[nodeID].child    = childID;
      }
    }
  }
  
  void BVH::build(const Geometry *geometry)
  {
    assert(geometry);
    size_t numPrimitives = geometry->numPrimitives();
    std::vector<BuildPrim> buildPrim;
    std::vector<size_t>    primID;
    buildPrim.resize(numPrimitives);
    primID.resize(numPrimitives);
    
    size_t blockSize = 1000;
    size_t numBlocks = (numPrimitives+blockSize-1) / blockSize;
    std::mutex mutex;
    box3fa centBounds  = ospcommon::empty;

// #pragma omp for
    for (size_t blockID=0;blockID<numBlocks;blockID++) {
      size_t begin   = blockID * blockSize;
      size_t end     = std::min(begin+blockSize,numPrimitives);
      box3fa blockBounds  = ospcommon::empty;
      for (size_t i=begin;i<end;i++) {
        BuildPrim bp;
        geometry->initBuildPrim(bp,i);
        blockBounds.extend(bp.center());
        buildPrim[i] = bp;
        primID[i] = i;
      }

      std::lock_guard lock(mutex);
      centBounds.extend(blockBounds);
    }
    nodeList.push_back(Node());
    buildRec(0,centBounds,buildPrim.data(),primID);
  }

}
