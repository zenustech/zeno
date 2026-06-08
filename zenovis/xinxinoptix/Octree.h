#include <openvdb/openvdb.h>
#include <zeno/utils/vec.h>

#include "optixCommon.h"
#include "TypeCaster.h"

struct VolumeAggregate {

    using Vec3F = openvdb::Vec3f;
    using Box3F = openvdb::math::BBox<Vec3F>;

    template<typename T>
    struct _MinMax {
        using ValueType = typename openvdb::VecTraits<T>::ElementType;
        T mini = T(+std::numeric_limits<ValueType>::max());
        T maxi = T(-std::numeric_limits<ValueType>::max());

        inline void update(const T& v) {
            mini = zeno::min(mini, v);
            maxi = zeno::max(maxi, v);
        }
        inline void merge(const _MinMax<T>& that) {
            mini = zeno::min(mini, that.mini);
            maxi = zeno::max(maxi, that.maxi);
        }
        inline bool valid() const { return mini <= maxi; }
    };

    using FloatMinMax  = _MinMax<float>;
    using Float3MinMax = _MinMax<Vec3F>;

    static Vec3F min3(const Vec3F& a, const Vec3F& b) {
        return { fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()) };
    }
    static Vec3F max3(const Vec3F& a, const Vec3F& b) {
        return { fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()) };
    }

    struct NodeCache {
        FloatMinMax minmax;
        Box3F bbox;

        uint32_t kidCount;
        uint32_t kidOffset;
		// NEW: only valid for leaf nodes
    	const openvdb::tree::LeafNode<float, 3>* leaf = nullptr;
    };
    struct GridMinMax {
        std::vector<NodeCache> cache;
        std::vector<uint32_t> allkids;

		Box3F cBox3F(const openvdb::CoordBBox& raw) const {
			const auto& min = raw.min();
			const auto& max = raw.max();
			return Box3F(min.asVec3s(), max.asVec3s());
		}

        // Base case: Leaf Nodes
        FloatMinMax precompute(const openvdb::tree::LeafNode<float, 3>& leaf, float bgvalue, uint32_t& pos) {
            FloatMinMax minmax;
			if (leaf.isInactive()) {
				minmax.update(bgvalue);
				return minmax;
			}
            for (auto it = leaf.cbeginValueOn(); it; ++it) {
                float val = *it;
                minmax.update(val);
            }
            pos = cache.size();
            auto bbox = leaf.getNodeBoundingBox();
            cache.emplace_back( NodeCache{ minmax, cBox3F(bbox), 0u, 0u, &leaf} );
            return minmax;
        }

        template<typename ChildT, openvdb::Index Log2Dim>
        FloatMinMax precompute(const openvdb::tree::InternalNode<ChildT, Log2Dim>& node, float bgvalue, uint32_t& pos) {
            
			FloatMinMax minmax;
			pos = cache.size();
            cache.emplace_back(NodeCache{});

			uint32_t kcount = 0;
            std::vector<uint32_t> kids;
            kids.reserve(node.childCount() + node.onTileCount());
			//const uint32_t childDim = 1 << ChildT::TOTAL;
    		const uint32_t childDim = 1 << ChildT::LOG2DIM;
            // 1. Process Active TILES (Constant regions)
            for (auto it = node.cbeginValueAll(); it; ++it) {
                float val = it.isValueOn()? it.getValue() : bgvalue;
				minmax.update(val);

				// Calculate the tile's specific BBox
				openvdb::Coord minCoord = it.getCoord();
				// Snap to the tile's origin based on the child dimension
				minCoord &= ~(childDim - 1); 
				openvdb::Coord maxCoord = minCoord.offsetBy(childDim - 1);
				openvdb::CoordBBox tileBox(minCoord, maxCoord);

				uint32_t kid_pos = cache.size();
				cache.emplace_back( NodeCache{ {val, val}, cBox3F(tileBox), 0u, 0u} );
				kids.push_back(kid_pos); ++kcount;
            }
            for (auto it = node.cbeginChildOn(); it; ++it) {
                uint32_t kid_pos;
                auto tmp = precompute(*it, bgvalue, kid_pos); // Compiler picks Leaf or Internal overload
                minmax.merge(tmp);

                kids.push_back(kid_pos); ++kcount;
            }
            
            const uint32_t kidOffset = allkids.size();
            allkids.insert(allkids.end(), kids.begin(), kids.end());

            auto bbox = node.getNodeBoundingBox();
            cache[pos] = NodeCache{ minmax, cBox3F(bbox), kcount, kidOffset};
            return minmax;
        }

        void precompute(const openvdb::FloatGrid& vgrid) {

            const auto& root = vgrid.tree().root();
			const uint32_t childDim = root.getChildDim();

			cache.clear();
            cache.emplace_back(NodeCache{});
			allkids.clear();

			uint32_t kcount = 0;
            std::vector<uint32_t> kids;
            kids.reserve(root.childCount() + root.onTileCount());

            const float bgvalue = root.background(); 
            FloatMinMax minmax;
            // 1. Root Tiles
            for (auto it = root.cbeginValueAll(); it; ++it) {
                float val = it.isValueOn()? it.getValue() : bgvalue;
				minmax.update(val);

				openvdb::Coord minCoord = it.getCoord();
    			minCoord &= ~(childDim - 1);
    			openvdb::Coord maxCoord = minCoord.offsetBy(childDim - 1);
				auto bbox = openvdb::CoordBBox(minCoord, maxCoord);

				uint32_t kid_pos = cache.size();
				cache.emplace_back( NodeCache{ {val, val}, cBox3F(bbox), 0u, 0u} );
				kids.push_back(kid_pos); ++kcount;
            }

            // 2. Root Children
            for (auto it = root.cbeginChildOn(); it; ++it) {
                uint32_t pos;
                auto tmp = precompute(*it, bgvalue, pos);   
                minmax.merge(tmp);

                kids.push_back(pos); ++kcount;
            }
            const uint32_t kidOffset = allkids.size();
            allkids.insert(allkids.end(), kids.begin(), kids.end());

			openvdb::CoordBBox bbox = vgrid.evalActiveVoxelBoundingBox();
            cache[0] = NodeCache{ minmax, cBox3F(bbox), kcount, kidOffset};
        }

        inline const NodeCache& query(uint32_t i) const {
            return cache[i];
        }

        FloatMinMax queryMinMax(const uint32_t idx, const Box3F& queryBox) const
        {
            const auto& cached = query(idx);
            if ( !cached.bbox.hasOverlap(queryBox) ) return {};
            if ( queryBox.isInside(cached.bbox) )
                return cached.minmax;
			
			FloatMinMax minmax;
			if(cached.kidCount == 0) {
				auto leaf = cached.leaf;
				if (leaf==nullptr) { // tile
					return cached.minmax;
				}
				for (auto it = leaf->cbeginValueAll(); it; ++it) {
					const openvdb::Coord& c = it.getCoord();
					if (!queryBox.isInside(c.asVec3s())) continue;
					float v = *it;
					minmax.update(v);
				}
				return minmax;
			} 
            for (int i=0; i<cached.kidCount; ++i) {
                const auto& kid = allkids[cached.kidOffset+i];
                auto tmp = queryMinMax(kid, queryBox);
                if (!tmp.valid()) continue;
                minmax.merge(tmp);
            }
            return minmax;
        }

        FloatMinMax queryMinMax(const openvdb::FloatGrid& vgrid, const Box3F& queryBox) const {

            const auto& root = vgrid.tree().root();
            // using ManagerT = openvdb::tree::NodeManager<const openvdb::FloatTree>;
            // std::unique_ptr<ManagerT> manager = std::make_unique<ManagerT>(vgrid.m_grid->tree());
            // manager->nodeCount();
            using RootT = openvdb::FloatTree::RootNodeType;
            using InternalT = RootT::ChildNodeType;
            const uint32_t DIM = InternalT::DIM;

            const auto& cached = query(0);
            FloatMinMax minmax;

            for (int i=0; i<cached.kidCount; ++i) {
                const auto& kid = allkids[cached.kidOffset+i];
                auto tmp = queryMinMax(kid, queryBox);
                if (!tmp.valid()) continue;
                minmax.merge(tmp);
            }
            return minmax;
        }
    };

    FloatMinMax fetchMinMax(const Box3F& queryBox, const openvdb::FloatGrid& vgrid, const GridMinMax& gmm) {
        return gmm.queryMinMax(vgrid, queryBox);
    }
    
    struct OcNode {
        uint32_t data;
        uint16_t min_d, max_d;

        inline uint8_t childMask() const {
            return (uint8_t)(data >> 24);
        }

        inline uint8_t& childMaskRef() const {
            return *((uint8_t*)&data + 3);
        }

        inline uint32_t childOffset() const {
            return (uint32_t)(data & 0x00FFFFFF);
        }

        inline void setChildMask(uint8_t mask) {
            auto tmp = (uint32_t)mask << 24;
            data |= tmp;
        }

        inline void setChildOffset(uint32_t offset) {
            assert(offset <= 0x00FFFFFF);
            data = (data & 0xFF000000) | offset;
        }
    };

    void subdivide(const Box3F& bbox, std::vector<OcNode>& octree, uint32_t octidx, float T, int depth,
    			   const openvdb::FloatGrid& vgrid, const GridMinMax& gmm) 
    {
        assert(!bbox.empty() && bbox.hasVolume());
        const auto& octbox = bbox;
        const auto ext = octbox.extents()/2;
		if (depth>=6) return;
        
        std::vector<std::pair<uint32_t, Box3F>> tasks;
        tasks.reserve(8);

		octree[octidx].setChildOffset( octree.size() );

        for (int i=0; i<8; ++i) {
            auto p = octbox.min();
            if (i & 1) p.x() += ext.x();
            if (i & 2) p.y() += ext.y();
            if (i & 4) p.z() += ext.z();

            auto subox = Box3F(p, p+ext);

			FloatMinMax min_max = fetchMinMax(subox, vgrid, gmm);
            float min_r = min_max.mini;
            float max_r = min_max.maxi;

			auto& octnode = octree[octidx];
        	auto& child_mask = octnode.childMaskRef();
        	// auto& child_offset = octnode.child_offset;
            if(max_r > 0) {
                child_mask |= 1<<(i); // 8 children bit mask
            } else { continue; } // empty node

            const uint32_t node_idx = octree.size();
            octree.emplace_back(OcNode());
            auto& new_node = octree.back();
            new_node.data = 0;
            new_node.min_d = toHalf(min_r);
            new_node.max_d = toHalf(max_r);
            
            tasks.emplace_back( std::pair{node_idx, subox} );
        } // subox

        for (const auto& [nidx, box] : tasks) {
            subdivide(box, octree, nidx, T, depth+1, vgrid, gmm);
        }
    }

    Box3F octbox;
    std::vector<OcNode> octree;
        
    void aggregate(openvdb::FloatGrid& vgrid, float T=1.0f) {

        octbox = {};
        octree = {};

        const auto bbox = vgrid.evalActiveVoxelBoundingBox();
		octbox = Box3F(bbox.min().asVec3s(), bbox.max().asVec3s());

        octree.emplace_back(OcNode());
        OcNode& rootNode = octree.back();

        GridMinMax gmm;
		gmm.precompute(vgrid);
        
        auto minmax = gmm.cache[0].minmax;
        rootNode.min_d = toHalf(minmax.mini);
        rootNode.max_d = toHalf(minmax.maxi);

        CppTimer timer;
        timer.tick();
        subdivide(octbox, octree, 0, T, 0, vgrid, gmm);
        timer.tock("subdivide cost");
        
        printf("octree size = %zu \n", octree.size());
    }
};