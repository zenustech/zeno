#include <mcut/internal/bvh.h>
#include <mcut/internal/utils.h>

#include <cmath> // see: if it is possible to remove thsi header

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt

// https://stackoverflow.com/questions/355967/how-to-use-msvc-intrinsics-to-get-the-equivalent-of-this-gcc-code
unsigned int __inline clz_(unsigned int value)
{
    unsigned long leading_zero = 0;

    if (_BitScanReverse(&leading_zero, value))
    {
        return 31 - leading_zero;
    }
    else
    {
        // Same remarks as above
        return 32;
    }
}

#endif

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

namespace mcut
{
    namespace bvh
    {

#if defined(USE_OIBVH)
        // count leading zeros in 32 bit bitfield
        unsigned int clz(unsigned int x) // stub
        {
#ifdef _MSC_VER
            return clz_(x);
#else
            return __builtin_clz(x); // only tested with gcc!!!
#endif
        }

        // next power of two from x
        int next_power_of_two(int x)
        {
            x--;
            x |= x >> 1;
            x |= x >> 2;
            x |= x >> 4;
            x |= x >> 8;
            x |= x >> 16;
            x++;
            return x;
        }

        // check if "x" is a power of two
        bool is_power_of_two(int x)
        {
            return (x != 0) && !(x & (x - 1));
        }

        // compute log-base-2 of "x"
        int ilog2(unsigned int x)
        {
            return sizeof(unsigned int) * CHAR_BIT - clz(x) - 1;
        }

        // compute index (0...N-1) of the leaf level from the number of leaves
        int get_leaf_level_from_real_leaf_count(const int t)
        {
            const int np2 = next_power_of_two(t); // todo
            const int tLeafLev = ilog2(np2);
            return tLeafLev;
        }

        // compute tree-level index from implicit index of a node
        int get_level_from_implicit_idx(const int bvhNodeImplicitIndex)
        {
            return ilog2(bvhNodeImplicitIndex + 1);
        }

        // compute previous power of two
        unsigned int flp2(unsigned int x) // prev pow2
        {
            x = x | (x >> 1);
            x = x | (x >> 2);
            x = x | (x >> 4);
            x = x | (x >> 8);
            x = x | (x >> 16);
            return x - (x >> 1);
        }

        // compute size of of Oi-BVH give number of triangles
        int get_ostensibly_implicit_bvh_size(const int t)
        {
            return 2 * t - 1 + __builtin_popcount(next_power_of_two(t) - t);
        }

        // compute left-most node on a given level
        int get_level_leftmost_node(const int node_level)
        {
            return (1 << node_level) - 1;
        }

        // compute right-most leaf node in tree
        int get_rightmost_real_leaf(const int bvhLeafLevelIndex, const int num_real_leaf_nodes_in_bvh)
        {
            return (get_level_leftmost_node(bvhLeafLevelIndex) + num_real_leaf_nodes_in_bvh) - 1;
        }

        // check if node is a "real node"
        bool is_real_implicit_tree_node_id(const int bvhNodeImplicitIndex, const int num_real_leaf_nodes_in_bvh)
        {

            const int t = num_real_leaf_nodes_in_bvh;
            //const int q = bvhNodeImplicitIndex; // queried node
            const int li = get_leaf_level_from_real_leaf_count(t);
            const int i = get_rightmost_real_leaf(li, t);
            const int lq = get_level_from_implicit_idx(bvhNodeImplicitIndex);
            const int p = (int)((1.0f / (1 << (li - lq))) + ((float)i / (1 << (li - lq))) - 1);

            return bvhNodeImplicitIndex <= p || p == 0; // and p is not the root
        }

        // get the right most real node on a given tree level
        int get_level_rightmost_real_node(
            const int rightmostRealLeafNodeImplicitIndex,
            const int bvhLeafLevelIndex,
            const int ancestorLevelIndex)
        {
            using namespace std;
            const int level_dist = (bvhLeafLevelIndex - ancestorLevelIndex);
            const int implicit_index_of_ancestor = (int)((1.0f / (1 << level_dist)) + ((float)rightmostRealLeafNodeImplicitIndex / (1 << level_dist)) - 1);
            return implicit_index_of_ancestor;
        }

        // compute implicit index of a node's ancestor
        int get_node_ancestor(
            const int nodeImplicitIndex,
            const int nodeLevelIndex,
            const int ancestorLevelIndex)
        {
            using namespace std;
            const int levelDistance = nodeLevelIndex - ancestorLevelIndex;
            return (int)((1.0f / (1 << levelDistance)) + ((float)nodeImplicitIndex / (1 << levelDistance)) - 1); /*trunc((1.0f / pow(bvhDegree, level_dist)) + (rightmostRealLeafNodeImplicitIndex / pow(bvhDegree, level_dist)) - 1)*/
        }

        // calculate linear memory index of a real node
        int get_node_mem_index(
            const int nodeImplicitIndex,
            const int leftmostImplicitIndexOnNodeLevel,
            const int bvh_data_base_offset,
            const int rightmostRealNodeImplicitIndexOnNodeLevel)
        {
            return bvh_data_base_offset + get_ostensibly_implicit_bvh_size((rightmostRealNodeImplicitIndexOnNodeLevel - leftmostImplicitIndexOnNodeLevel) + 1) - 1 - (rightmostRealNodeImplicitIndexOnNodeLevel - nodeImplicitIndex);
        }

        // Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
        unsigned int expandBits(unsigned int v)
        {
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        };

        // Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
        unsigned int morton3D(float x, float y, float z)
        {
            x = std::fmin(std::fmax(x * 1024.0f, 0.0f), 1023.0f);
            y = std::fmin(std::fmax(y * 1024.0f, 0.0f), 1023.0f);
            z = std::fmin(std::fmax(z * 1024.0f, 0.0f), 1023.0f);

            unsigned int xx = expandBits((unsigned int)x);
            unsigned int yy = expandBits((unsigned int)y);
            unsigned int zz = expandBits((unsigned int)z);

            return (xx * 4 + yy * 2 + zz);
        };
#else
        BoundingVolumeHierarchy::BoundingVolumeHierarchy()
        {
        }

        BoundingVolumeHierarchy::~BoundingVolumeHierarchy() {}

        // three stages to BVH construction
        void BoundingVolumeHierarchy::buildTree(const mesh_t &mesh_,
                                                const math::fixed_precision_number_t &enlargementEps_,
                                                uint32_t mp_,
                                                const SplitMethod &sm_)
        {
            SCOPED_TIMER(__FUNCTION__);
            mesh = &(mesh_); ///
            MCUT_ASSERT(mesh->number_of_faces() >= 1);
            maxPrimsInNode = (std::min(255u, mp_)); //
            splitMethod = (sm_);                    //
            enlargementEps = (enlargementEps_);

            buildData.clear();
            primitives.clear();
            primitiveOrderedBBoxes.clear();
            nodes.clear();

            // First, bounding information about each primitive is computed and stored in an array
            // that will be used during tree construction.

            // initialize buildData array for primitives

            buildData.reserve(mesh->number_of_faces());

            primitiveOrderedBBoxes.resize(mesh->number_of_faces());
            primitives.resize(mesh->number_of_faces());

            // TODO make this parallel
            // for each face in mesh
            for (mcut::mesh_t::face_iterator_t f = mesh->faces_begin(); f != mesh->faces_end(); ++f)
            {
                const int i = static_cast<int>(*f);
                primitives[i] = *f;

                const std::vector<mcut::vd_t> vertices_on_face = mesh->get_vertices_around_face(*f);

                mcut::geom::bounding_box_t<mcut::math::fast_vec3> bbox;
                // for each vertex on face
                for (std::vector<mcut::vd_t>::const_iterator v = vertices_on_face.cbegin(); v != vertices_on_face.cend(); ++v)
                {
                    const mcut::math::fast_vec3 coords = mesh->vertex(*v);
                    bbox.expand(coords);
                }

                if (enlargementEps > 0.0)
                {
                    bbox.enlarge(enlargementEps);
                }

                primitiveOrderedBBoxes[i] = bbox;

                buildData.push_back(BVHPrimitiveInfo(i, primitiveOrderedBBoxes[i]));
            }

            // Next, the tree is built via a procedure that splits the primitives into subsets and
            // recursively builds BVHs for the subsets. The result is a binary tree where each
            // interior node holds pointers to its children and each leaf node holds references to
            // one or more primitives.

            uint32_t totalNodes = 0;
            std::vector<fd_t> orderedPrims;
            //orderedPrims.reserve(mesh->number_of_faces());

            std::shared_ptr<BVHBuildNode> root = recursiveBuild(
                buildData,
                0,
                /*primitives.size()*/ mesh->number_of_faces(),
                &totalNodes,
                orderedPrims);

            primitives.swap(orderedPrims);

            // Finally, this tree is converted to a more compact (and thus more efficient) pointerless
            // representation for use during rendering

            nodes.resize(totalNodes);
            for (uint32_t i = 0; i < totalNodes; ++i)
            {
                //new (&nodes[i]) LinearBVHNode;
                nodes[i] = std::make_shared<LinearBVHNode>();
            }
            uint32_t offset = 0;
            flattenBVHTree(root, &offset);
        }

        const BBox &BoundingVolumeHierarchy::GetPrimitiveBBox(int primitiveIndex) const
        {
            MCUT_ASSERT(primitiveIndex < mesh->number_of_faces());
            return primitiveOrderedBBoxes[primitiveIndex];
        }

        uint32_t BoundingVolumeHierarchy::flattenBVHTree(std::shared_ptr<BVHBuildNode> node, uint32_t *offset)
        {
            MCUT_ASSERT(*offset < nodes.size());
            std::shared_ptr<LinearBVHNode> linearNode = nodes[*offset];
            linearNode->bounds = node->bounds;
            uint32_t myOffset = (*offset)++;
            if (node->nPrimitives > 0)
            {
                linearNode->primitivesOffset = node->firstPrimOffset;
                linearNode->nPrimitives = node->nPrimitives;
            }
            else
            {
                //Creater interior flattened BVH node
                linearNode->axis = node->splitAxis;
                linearNode->nPrimitives = 0;

                flattenBVHTree(node->children[0], offset);

                linearNode->secondChildOffset = flattenBVHTree(node->children[1],
                                                               offset);
            }
            return myOffset;
        }

        std::shared_ptr<BVHBuildNode> BoundingVolumeHierarchy::recursiveBuild(
            std::vector<BVHPrimitiveInfo> &buildData,
            uint32_t start,
            uint32_t end,
            uint32_t *totalNodes,
            std::vector<fd_t> &orderedPrims)
        {
            (*totalNodes)++;

            std::shared_ptr<BVHBuildNode> node = std::make_shared<BVHBuildNode>();

            //Compute bounds of all primitives in BVH node
            uint32_t nPrimitives = end - start;
            MCUT_ASSERT((nPrimitives - 1) < (uint32_t)mesh->number_of_faces());

            BBox bbox;
            for (uint32_t i = start; i < end; ++i)
            {
                MCUT_ASSERT(i < buildData.size());
                bbox = Union(bbox, buildData[i].bounds);
            }
            if (nPrimitives == 1)
            {
                //Create leaf BVHBuildNode
                uint32_t firstPrimOffset = orderedPrims.size();
                for (uint32_t i = start; i < end; ++i)
                {
                    MCUT_ASSERT(i < buildData.size());
                    uint32_t primNum = buildData[i].primitiveNumber;
                    orderedPrims.push_back(primitives[primNum]);
                }
                node->InitLeaf(firstPrimOffset, nPrimitives, bbox);
            }
            else
            {
                //Compute bound of primitive centroids, choose split dimension dim
                BBox centroidBounds;
                for (uint32_t i = start; i < end; ++i)
                {
                    MCUT_ASSERT(i < buildData.size());
                    centroidBounds = Union(centroidBounds, buildData[i].centroid);
                }

                int dim = centroidBounds.MaximumExtent();
                MCUT_ASSERT(dim < 3);

                //
                // Partition primitives into two sets and build children
                //
                uint32_t mid = (start + end) / 2;
                switch (this->splitMethod)
                {
                case SplitMethod::SPLIT_MIDDLE:
                {
                    // Partition primitives through node’s midpoint
                    math::fixed_precision_number_t pmid = (centroidBounds.minimum()[dim] + centroidBounds.maximum()[dim]) * .5;
#if 1
                    MCUT_ASSERT(start < buildData.size());
                    BVHPrimitiveInfo *midPtr = std::partition(&buildData[start],
                                                              &buildData[end - 1] + 1,
                                                              /*CompareToMid(dim, pmid)*/
                                                              [dim, pmid](const BVHPrimitiveInfo &pi)
                                                              {
                                                                  return pi.centroid[dim] < pmid;
                                                              });
                    mid = midPtr - &buildData[0];
#else
                    std::vector<BVHPrimitiveInfo>::iterator midPtr = std::partition(buildData.begin() + start,
                                                                                    buildData.end(),
                                                                                    CompareToMid(dim, pmid));
                    mid = std::distance(buildData.begin(), midPtr);
#endif
                }
                break;
                case SplitMethod::SPLIT_EQUAL_COUNTS:
                {
                    // Partition primitives into equally-sized subsets
                    mid = (start + end) / 2;
                    std::nth_element(&buildData[start], &buildData[mid],
                                     &buildData[end - 1] + 1, ComparePoints(dim));
                }
                break;
                case SplitMethod::SPLIT_SAH:
                {
                    // Partition primitives using approximate SAH
                    if (nPrimitives <= 4)
                    {
                        //Partition primitives into equally-sized subsets
                        mid = (start + end) / 2;
                        std::nth_element(&buildData[start], &buildData[mid],
                                         &buildData[end - 1] + 1, ComparePoints(dim));
                    }
                    else
                    {
                        //Allocate BucketInfo for SAH partition buckets
                        const int nBuckets = 12;
                        BucketInfo buckets[nBuckets];

                        //Initialize BucketInfo for SAH partition buckets
                        for (uint32_t i = start; i < end; ++i)
                        {
                            int b = nBuckets *
                                    ((buildData[i].centroid[dim] - centroidBounds.minimum()[dim]) /
                                     (centroidBounds.maximum()[dim] - centroidBounds.minimum()[dim]));
                            if (b == nBuckets)
                                b = nBuckets - 1;
                            buckets[b].count++;
                            buckets[b].bounds = Union(buckets[b].bounds, buildData[i].bounds);
                        }
                        //Compute costs for splitting after each bucket

                        math::fixed_precision_number_t cost[nBuckets - 1];

                        for (int i = 0; i < nBuckets - 1; ++i)
                        {
                            BBox b0, b1;
                            int count0 = 0, count1 = 0;
                            for (int j = 0; j <= i; ++j)
                            {
                                b0 = Union(b0, buckets[j].bounds);
                                count0 += buckets[j].count;
                            }
                            for (int j = i + 1; j < nBuckets; ++j)
                            {
                                b1 = Union(b1, buckets[j].bounds);
                                count1 += buckets[j].count;
                            }
                            cost[i] = .125f + (count0 * b0.SurfaceArea() + count1 * b1.SurfaceArea()) /
                                                  bbox.SurfaceArea();
                        }
                        //Find bucket to split at that minimizes SAH metric
                        float minCost = cost[0];
                        uint32_t minCostSplit = 0;
                        for (int i = 1; i < nBuckets - 1; ++i)
                        {
                            if (cost[i] < minCost)
                            {
                                minCost = cost[i];
                                minCostSplit = i;
                            }
                        }
                        //Either create leaf or split primitives at selected SAH bucket

                        if (nPrimitives > (uint32_t)maxPrimsInNode ||
                            minCost < nPrimitives)
                        {
                            const BVHPrimitiveInfo *pmid = std::partition(&buildData[start],
                                                                          &buildData[end - 1] + 1,
                                                                          CompareToBucket(minCostSplit, nBuckets, dim, centroidBounds));
                            mid = pmid - &buildData[0];
                        }
                        else
                        {
                            // Create leaf BVHBuildNode
                            uint32_t firstPrimOffset = orderedPrims.size();
                            for (uint32_t i = start; i < end; ++i)
                            {
                                uint32_t primNum = buildData[i].primitiveNumber;
                                orderedPrims.push_back(primitives[primNum]);
                            }
                            node->InitLeaf(firstPrimOffset, nPrimitives, bbox);
                        }
                    }
                }
                break;
                default:
                    fprintf(stderr, "[MCUT]: error, unknown split method\n");
                    break;
                }

                mid = (start + end) / 2;
                if (centroidBounds.maximum()[dim] == centroidBounds.minimum()[dim])
                {
                    // Create leaf BVHBuildNode
                    int32_t firstPrimOffset = orderedPrims.size();
                    for (uint32_t i = start; i < end; ++i)
                    {
                        uint32_t primNum = buildData[i].primitiveNumber;
                        orderedPrims.push_back(primitives[primNum]);
                    }
                    node->InitLeaf(firstPrimOffset, nPrimitives, bbox);
                    return node;
                }

                std::shared_ptr<BVHBuildNode> leftSubTree = recursiveBuild(buildData, start, mid,
                                                                           totalNodes, orderedPrims);
                std::shared_ptr<BVHBuildNode> rightSubTree = recursiveBuild(buildData, mid, end,
                                                                            totalNodes, orderedPrims);
                // Partition primitives based on splitMethod〉
                node->InitInterior(dim,
                                   leftSubTree,
                                   rightSubTree);
            }
            return node;
        }

        int BoundingVolumeHierarchy::GetNodeCount() const
        {
            return (int)nodes.size();
        }

        const std::shared_ptr<LinearBVHNode> &BoundingVolumeHierarchy::GetNode(int idx) const
        {
            return nodes[idx];
        }

        const fd_t &BoundingVolumeHierarchy::GetPrimitive(int index) const
        {
            MCUT_ASSERT(index < (int)primitives.size());
            return primitives[index];
        }

        void BoundingVolumeHierarchy::intersectBVHTrees(
#if defined(MCUT_MULTI_THREADED)
            thread_pool &scheduler,
#endif
            std::map<mcut::fd_t, std::vector<mcut::fd_t>> &symmetric_intersecting_pairs,
            const BoundingVolumeHierarchy &bvhA,
            const BoundingVolumeHierarchy &bvhB,
            const uint32_t primitiveOffsetA,
            const uint32_t primitiveOffsetB)
        {
            SCOPED_TIMER(__FUNCTION__);
            MCUT_ASSERT(bvhA.GetNodeCount() > 0);
            MCUT_ASSERT(bvhB.GetNodeCount() > 0);
            
            auto fn_intersectBVHTrees = [&bvhA, &bvhB, &primitiveOffsetA, &primitiveOffsetB](
                                            std::vector<std::pair<int, int>> &worklist_,
                                            std::map<mcut::fd_t, std::vector<mcut::fd_t>> &symmetric_intersecting_pairs_,
                                            const uint32_t maxWorklistSize)
            {
                // Simultaneous DFS traversal
                while (worklist_.size() > 0 && worklist_.size() < maxWorklistSize)
                {
                    //maxTodoSz = std::max(maxTodoSz, (int)worklist_.size());
                    //std::cout << "worklist_.size()="<<worklist_.size()<<std::endl;
                    std::pair<int, int> cur = worklist_.back();
                    // TODO: try to keep an additional counter that allows us to minimize pushing and popping
                    // Might require a wrapper class over std::vector "lazy vector"
                    worklist_.pop_back();

                    const uint32_t nodeAIndex = cur.first;
                    const uint32_t nodeBIndex = cur.second;
                    const std::shared_ptr<LinearBVHNode> nodeA = bvhA.GetNode(nodeAIndex);
                    const std::shared_ptr<LinearBVHNode> nodeB = bvhB.GetNode(nodeBIndex);

                    if (!geom::intersect_bounding_boxes(nodeA->bounds, nodeB->bounds))
                    {
                        continue;
                    }

                    bool nodeAIsLeaf = nodeA->nPrimitives > 0;
                    bool nodeBIsLeaf = nodeB->nPrimitives > 0;

                    if (nodeAIsLeaf)
                    {
                        if (nodeBIsLeaf)
                        {
                            for (int i = 0; i < nodeA->nPrimitives; ++i)
                            {
                                const fd_t faceA = bvhA.GetPrimitive((uint32_t)(nodeA->primitivesOffset + i));
                                const fd_t faceAOffsetted(primitiveOffsetA + faceA);

                                for (int j = 0; j < nodeB->nPrimitives; ++j)
                                {
                                    const fd_t faceB = bvhB.GetPrimitive((uint32_t)(nodeB->primitivesOffset + j));
                                    const fd_t faceBOffsetted(primitiveOffsetB + faceB);

                                    symmetric_intersecting_pairs_[faceAOffsetted].push_back(faceBOffsetted);
                                    symmetric_intersecting_pairs_[faceBOffsetted].push_back(faceAOffsetted);
                                }
                            }
                        }
                        else
                        {
                            const uint32_t nodeBLeftChild = nodeBIndex + 1;
                            const uint32_t nodeBRightChild = nodeB->secondChildOffset;
                            worklist_.emplace_back(nodeAIndex, nodeBLeftChild);
                            worklist_.emplace_back(nodeAIndex, nodeBRightChild);
                        }
                    }
                    else
                    {
                        if (nodeBIsLeaf)
                        {
                            const uint32_t nodeALeftChild = nodeAIndex + 1;
                            const uint32_t nodeARightChild = nodeA->secondChildOffset;
                            worklist_.emplace_back(nodeALeftChild, nodeBIndex);
                            worklist_.emplace_back(nodeARightChild, nodeBIndex);
                        }
                        else
                        {
                            const uint32_t nodeALeftChild = nodeAIndex + 1;
                            const uint32_t nodeARightChild = nodeA->secondChildOffset;

                            const uint32_t nodeBLeftChild = nodeBIndex + 1;
                            const uint32_t nodeBRightChild = nodeB->secondChildOffset;

                            worklist_.emplace_back(nodeALeftChild, nodeBLeftChild);
                            worklist_.emplace_back(nodeALeftChild, nodeBRightChild);

                            worklist_.emplace_back(nodeARightChild, nodeBLeftChild);
                            worklist_.emplace_back(nodeARightChild, nodeBRightChild);
                        }
                    }
                }
            };

            // start with pair of root nodes
            std::vector<std::pair<int, int>> todo(1, std::make_pair(0,0));

#if defined(MCUT_MULTI_THREADED)
            {
                // master thread intersects the BVHs until the number of node pairs
                // reaches a threshold (or workload was small enough that traversal
                // is finished)
                const uint32_t threshold = scheduler.get_num_threads();
                fn_intersectBVHTrees(todo, symmetric_intersecting_pairs, threshold);

                uint32_t remainingWorkloadCount = (uint32_t)todo.size(); // how much work do we still have left

                if (remainingWorkloadCount > 0)
                { // do parallel traversal by distributing blocks of node-pairs across worker threads
                // NOTE: we do not manage load-balancing (too complex for the perf gain)
                    typedef std::vector<std::pair<int, int>>::const_iterator InputStorageIteratorType;
                    typedef std::map<mcut::fd_t, std::vector<mcut::fd_t>> OutputStorageType; // symmetric_intersecting_pairs (local)

                    auto fn_intersect = [&](InputStorageIteratorType block_start_, InputStorageIteratorType block_end_) -> OutputStorageType
                    {
                        OutputStorageType symmetric_intersecting_pairs_local;

                        std::vector<std::pair<int, int>> todo_local(block_start_, block_end_);

                        fn_intersectBVHTrees(
                            todo_local,
                            symmetric_intersecting_pairs_local,
                            // traverse until leaves
                            std::numeric_limits<uint32_t>::max());

                        return symmetric_intersecting_pairs_local;
                    };

                    std::vector<std::future<OutputStorageType>> futures;
                    OutputStorageType partial_res;

                    parallel_fork_and_join(
                        scheduler,
                        todo.cbegin(),
                        todo.cend(),
                        (1 << 1),
                        fn_intersect,
                        partial_res, // output of master thread
                        futures);

                    symmetric_intersecting_pairs.insert(partial_res.cbegin(), partial_res.cend());

                    for (int i = 0; i < (int)futures.size(); ++i)
                    {
                        std::future<OutputStorageType> &f = futures[i];
                        MCUT_ASSERT(f.valid());
                        OutputStorageType future_res = f.get();

                        symmetric_intersecting_pairs.insert(future_res.cbegin(), future_res.cend());
                    }
                }
            }
#else
            fn_intersectBVHTrees(todo, symmetric_intersecting_pairs, std::numeric_limits<uint32_t>::max());
#endif // #if defined(MCUT_MULTI_THREADED)
        }

#endif
    } // namespace bvh {
} // namespace mcut {
