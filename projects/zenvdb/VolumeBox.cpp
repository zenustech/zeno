#include <string>
#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/PrimitiveObject.h>
// #include <zeno/types/DictObject.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/tools/Interpolation.h>

#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/UserData.h>
#include <zeno/logger.h>

#include <zeno/utils/vec.h>
#include <zeno/utils/eulerangle.h>

#include <glm/mat4x4.hpp>

struct GreedyVoxel {

    std::vector<int8_t> storage;
    uint32_t _X, _Y, _Z;

    GreedyVoxel(uint32_t x, uint32_t y, uint32_t z) {
        init(x, y, z);
    }

    void init(uint32_t x, uint32_t y, uint32_t z) {

        _X = x; _Y = y; _Z = z;
        
        uint64_t size = x * y * z;
        storage.resize(size, 0);
    }

    inline uint64_t indexSafe(uint32_t x, uint32_t y, uint32_t z) const {
        uint64_t idx = x + y * _X + _X * _Y * z;
        return std::min(storage.size()-1, idx);
    }

    void fill(uint32_t x, uint32_t y, uint32_t z) {
        uint64_t index = indexSafe(x, y, z);
        storage[index] = 1;
    }

    void take(uint32_t x, uint32_t y, uint32_t z) {
        uint64_t index = indexSafe(x, y, z);
        storage[index] = 0;
    }

    bool available(uint32_t x, uint32_t y, uint32_t z) const {
        uint64_t index = indexSafe(x, y, z);
        return storage[index] > 0;
    }

    auto greedy() {

        openvdb::CoordBBox bounds(openvdb::Coord{0,0,0}, openvdb::Coord{(int)_X+1, (int)_Y+1, (int)_Z+1});

        const auto check = [this](openvdb::CoordBBox& testVoxel){

            auto mini = testVoxel.min();
            auto maxi = testVoxel.max();

            for (int x= mini.x(); x<maxi.x(); ++x)
                for (int y = mini.y(); y<maxi.y(); ++y)
                    for (int z = mini.z(); z<maxi.z(); ++z)
                        if (available(x, y, z) != true) {
                            return false;
                        }
            return true;
        };

        const auto combine = [this, &check](openvdb::CoordBBox& tempVoxel, openvdb::CoordBBox& testVoxel) {

            bool good = check(testVoxel);
            if (!good) return false;

            auto mini = testVoxel.min();
            auto maxi = testVoxel.max();

            for (int x= mini.x(); x<maxi.x(); ++x)
                for (int y = mini.y(); y<maxi.y(); ++y)
                    for (int z = mini.z(); z<maxi.z(); ++z) {
                        take(x, y, z);
                    }
            tempVoxel.expand(testVoxel);
            return true;
        };

        auto extend = [&bounds, &combine](openvdb::CoordBBox& tempVoxel){

            for (auto axis=0; axis<3; ++axis) {

                auto testVoxel = tempVoxel;
                auto lowBound = testVoxel.min()[axis] - bounds.min()[axis];

                for (int i=0; i<lowBound; ++i)
                {   //negative Direction
                    testVoxel.min()[axis] = testVoxel.min()[axis]-1;
                    testVoxel.max()[axis] = testVoxel.min()[axis]+1;

                    if (!combine(tempVoxel, testVoxel)) break;
                }

                testVoxel = tempVoxel;
                auto highBound = bounds.max()[axis] - testVoxel.max()[axis];

                for (int i=0; i<highBound; ++i)
                {   //positiveX Direction
                    testVoxel.min()[axis] = testVoxel.max()[axis];
                    testVoxel.max()[axis] = testVoxel.min()[axis]+1;

                    if (!combine(tempVoxel, testVoxel)) break;
                }
            } // foreach axis
        };

        std::vector<openvdb::CoordBBox> voxels;
        openvdb::CoordBBox tempVoxel;

        for (int x=0; x<_X; ++x) {
            for (int y=0; y<_Y; ++y) {
                for (int z=0; z<_Z; ++z) {

                    if (!available(x, y, z)) continue;
                    
                    if (tempVoxel.empty()) {
                        auto pos = openvdb::Coord{x,y,z};
                        tempVoxel = openvdb::CoordBBox(pos, openvdb::Coord(1)+pos);
                        
                        extend(tempVoxel);
                        voxels.push_back(tempVoxel);
                        tempVoxel.reset();
                    }
                } //Z
            } //Y
        } //X

        return voxels;
    }
};

namespace zeno {

struct CreateVolumeBox : zeno::INode {
    virtual void apply() override {

        auto pos = get_input2<zeno::vec3f>("pos");
        auto scale = get_input2<zeno::vec3f>("scale");
        auto rotate = get_input2<zeno::vec3f>("rotate");

        const auto greedy = get_input2<bool>("greedy");
        auto bounds = get_input2<std::string>("Bounds:");

        auto order = get_input2<std::string>("EulerRotationOrder:");
        auto orderTyped = magic_enum::enum_cast<EulerAngle::RotationOrder>(order).value_or(EulerAngle::RotationOrder::YXZ);

        auto measure = get_input2<std::string>("EulerAngleMeasure:");
        auto measureTyped = magic_enum::enum_cast<EulerAngle::Measure>(measure).value_or(EulerAngle::Measure::Radians);

        glm::vec3 eularAngleXYZ = glm::vec3(rotate[0], rotate[1], rotate[2]);
        glm::mat4 rotation = EulerAngle::rotate(orderTyped, measureTyped, eularAngleXYZ);

        glm::mat4 world_matrix(1.0f);
        std::vector<openvdb::CoordBBox> voxels;

        if (has_input2<VDBGrid>("vdbGrid")) {

            auto grid = get_input2<VDBGrid>("vdbGrid");

            auto float_grid = std::dynamic_pointer_cast<VDBFloatGrid>(grid);
            auto root = float_grid->m_grid->tree().root();

            using GridType = openvdb::FloatGrid;
            using TreeType = GridType::TreeType;
            using RootType = TreeType::RootNodeType;   // level 3 RootNode
            assert(RootType::LEVEL == 3);
            using Int1Type = RootType::ChildNodeType;  // level 2 InternalNode
            using Int2Type = Int1Type::ChildNodeType;  // level 1 InternalNode

            world_matrix = [&]() -> auto {
                auto tmp = grid->getTransform().baseMap()->getAffineMap()->getMat4();
                glm::mat4 result;
                for (size_t i=0; i<16; ++i) {
                    auto ele = *(tmp[0]+i);
                    result[i/4][i%4] = ele;
                }
                return result;
            } ();

            if (greedy) {

                openvdb::CoordBBox allbox;
                std::vector<openvdb::CoordBBox> nodeboxs;

                openvdb::Coord voxelSize;   
                for (TreeType::NodeIter iter = float_grid->m_grid->tree().beginNode(); iter; ++iter) {

                    if (iter.getDepth() == 2) {
                        auto box = iter.getBoundingBox();

                        voxelSize = box.dim();
                        allbox.expand(box);
                        nodeboxs.push_back(box);
                    }
                }

                const auto bdim = allbox.dim();

                openvdb::Coord ndim;
                ndim.x() = bdim.x()/voxelSize.x();
                ndim.y() = bdim.y()/voxelSize.y();
                ndim.z() = bdim.z()/voxelSize.z();

                GreedyVoxel greedyVoxel(ndim.x(), ndim.y(), ndim.z());

                for (const auto& box : nodeboxs) {
                    auto diff = box.min() - allbox.min();
                    auto offset = diff.asVec3i() / voxelSize.asVec3i();  
                    
                    greedyVoxel.fill(offset.x(), offset.y(), offset.z());
                }

                voxels = greedyVoxel.greedy();

                for (auto& box : voxels) {

                    auto mini = box.min().asVec3i() * voxelSize.asVec3i() + allbox.min();
                    auto maxi = box.max().asVec3i() * voxelSize.asVec3i() + allbox.min();

                    openvdb::Coord box_min = openvdb::Coord(mini.x(), mini.y(), mini.z());
                    openvdb::Coord box_max = openvdb::Coord(maxi.x(), maxi.y(), maxi.z());
                    box = openvdb::CoordBBox(box_min, box_max);
                }
            }
            else {
                auto box = grid->evalActiveVoxelBoundingBox();
                voxels = { box };
            }

        } else {

            auto box = openvdb::CoordBBox(openvdb::Coord(0), openvdb::Coord(1));
            voxels = { box };

            glm::mat4 transform(1.0);
            transform = glm::translate(transform, glm::vec3(pos[0], pos[1], pos[2]));
            transform = transform * rotation;
            transform = glm::scale(transform, glm::vec3(scale[0], scale[1], scale[2]));
            transform = glm::translate(transform, glm::vec3(-0.5f));

            world_matrix = transform;
        }

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto& ud = prim->userData();
        ud.set2("mtlid", get_input2<std::string>("vol_mat", ""));
        
        prim->verts->reserve(voxels.size() * 8);
        prim->quads->reserve(voxels.size() * 4);
        auto& raw = prim->add_attr<zeno::vec3f>("raw");
        
        size_t offset = 0;
        for (auto& voxel : voxels) {

            openvdb::Coord dummy[2];
            dummy[0] = voxel.min();
            dummy[1] = voxel.max();

            for (int i=0; i<=1; ++i) {
                for (int j=0; j<=1; ++j) {
                    for (int k=0; k<=1; ++k) {

                        auto x = dummy[i][0];
                        auto y = dummy[j][1];
                        auto z = dummy[k][2];
                        raw.push_back(zeno::vec3f(x, y, z));

                        auto p = glm::vec4(x, y, z, 1.0f);
                        p = world_matrix * p;
                        prim->verts.push_back(zeno::vec3f(p.x, p.y, p.z));
                    }
                }
            }
            // enough to draw box wire frame
            prim->quads->push_back(zeno::vec4i(offset+0, offset+1, offset+3, offset+2));
            prim->quads->push_back(zeno::vec4i(offset+4, offset+5, offset+7, offset+6));
            prim->quads->push_back(zeno::vec4i(offset+0, offset+1, offset+5, offset+4));
            prim->quads->push_back(zeno::vec4i(offset+3, offset+2, offset+6, offset+7));
            offset += 8;
        }
        primWireframe(prim.get(), true);

        auto transform_ptr = glm::value_ptr(world_matrix);
        ud.set2("_transform_row0", *(zeno::vec4f*)(transform_ptr+0));
        ud.set2("_transform_row1", *(zeno::vec4f*)(transform_ptr+4));
        ud.set2("_transform_row2", *(zeno::vec4f*)(transform_ptr+8));
        ud.set2("_transform_row3", *(zeno::vec4f*)(transform_ptr+12));
        
        ud.set2("bounds", bounds);
        ud.set2("vbox", true);

        set_output("prim", prim);
    }
};

ZENDEFNODE(CreateVolumeBox, {
    {
        {"vec3f", "pos", "0, 0, 0"},
        {"vec3f", "scale", "1, 1, 1"},
        {"vec3f", "rotate", "0, 0, 0"},
        {"bool", "greedy", "0"},
        {"vdbGrid" },
        {"string", "vol_mat", ""},
    },
    {"prim"},
    {
        {"enum " + EulerAngle::RotationOrderListString(), "EulerRotationOrder", "XYZ"},
        {"enum " + EulerAngle::MeasureListString(), "EulerAngleMeasure", "Degree"},
        {"enum Box Sphere HemiSphere", "Bounds", "Box"}
    },
    {"create"}
});

struct VolumeAggregate {

    using Vec3F = openvdb::Vec3f;
    using BBoxF = openvdb::math::BBox<Vec3F>;

    template<typename T>
    struct _MinMax {
        using ValueType = typename openvdb::VecTraits<T>::ElementType;
        T mini = T(+std::numeric_limits<ValueType>::max());
        T maxi = T(-std::numeric_limits<ValueType>::max());

        inline void update(const T& v) {
            mini = min(mini, v);
            maxi = max(maxi, v);
        }
        inline void merge(const _MinMax<T>& that) {
            mini = min(mini, that.mini);
            maxi = max(maxi, that.maxi);
        }
        inline bool valid() const { return mini <= maxi; }
    };

    using FloatMinMax  = _MinMax<float>;
    using Float3MinMax = _MinMax<Vec3F>;

    static FloatMinMax merge(const FloatMinMax& a, const FloatMinMax& b) {
        return FloatMinMax{ min(a.mini, b.mini), max(a.maxi, b.maxi) };
    }
    static Vec3F min3(const Vec3F& a, const Vec3F& b) {
        return { min(a.x(), b.x()), min(a.y(), b.y()), min(a.z(), b.z()) };
    }
    static Vec3F max3(const Vec3F& a, const Vec3F& b) {
        return { max(a.x(), b.x()), max(a.y(), b.y()), max(a.z(), b.z()) };
    }
    static Float3MinMax merge(const Float3MinMax& a, const Float3MinMax& b) {
        return Float3MinMax{ min3(a.mini, b.mini), max3(a.maxi, b.maxi) };
    }

    struct NodeCache {
        FloatMinMax minmax;
        openvdb::CoordBBox bbox;

        uint32_t kidCount;
        uint32_t kidOffset;
    };
    struct GridMinMax {
        std::vector<NodeCache> cache;
        std::vector<uint32_t> allkids;

        // Base case: Leaf Nodes
        FloatMinMax precompute(const openvdb::tree::LeafNode<float, 3>& leaf, float bgvalue, uint32_t& pos) {
            FloatMinMax minmax;
            minmax.update(bgvalue);
            // Manual iteration over active voxels in the leaf
            for (auto it = leaf.cbeginValueOn(); it; ++it) {
                float val = *it;
                minmax.update(val);
            }
            // If no active voxels, VDB leaves usually have a background value
            if (minmax.mini == std::numeric_limits<float>::max()) {
                minmax.update(bgvalue); // Fallback to background
            }
            pos = cache.size();
            auto bbox = leaf.getNodeBoundingBox();
            cache.emplace_back( NodeCache{ minmax, bbox, 0u, 0u} );
            return minmax;
        }

        template<typename ChildT, openvdb::Index Log2Dim>
        FloatMinMax precompute(const openvdb::tree::InternalNode<ChildT, Log2Dim>& node, float bgvalue, uint32_t& pos) {
            FloatMinMax minmax;
            minmax.update(bgvalue);
            // 1. Process Active TILES (Constant regions)
            for (auto it = node.cbeginValueOn(); it; ++it) {
                float val = it.getValue();
                minmax.update(val);
            }

            pos = cache.size();
            cache.emplace_back(NodeCache{});

            uint32_t kcount = 0;
            std::vector<uint32_t> kids;
            kids.reserve(node.childCount());

            for (auto it = node.cbeginChildOn(); it; ++it) {
                uint32_t kid_pos;
                auto tmp = precompute(*it, bgvalue, kid_pos); // Compiler picks Leaf or Internal overload
                minmax.merge(tmp);

                kids.push_back(kid_pos); ++kcount;
            }
            
            const uint32_t kidOffset = allkids.size();
            allkids.insert(allkids.end(), kids.begin(), kids.end());

            auto bbox = node.getNodeBoundingBox();
            cache[pos] = NodeCache{ minmax, bbox, kcount, kidOffset};
            return minmax;
        }

        void precompute(const VDBFloatGrid& volume) {

            const auto& root = volume.m_grid->tree().root();            
            const float bgvalue = root.background(); 
            FloatMinMax minmax;
            minmax.update(bgvalue);
            // 1. Root Tiles
            for (auto it = root.cbeginValueOn(); it; ++it) {
                float val = it.getValue();
                minmax.update(val);
            }

            cache.clear();
            cache.emplace_back(NodeCache{});
            
            uint32_t kcount = 0;
            std::vector<uint32_t> kids;
            kids.reserve(root.childCount());
            // 2. Root Children
            for (auto it = root.cbeginChildOn(); it; ++it) {
                uint32_t pos;
                auto tmp = precompute(*it, bgvalue, pos);   
                minmax.merge(tmp);

                kids.push_back(pos); ++kcount;
            }
            const uint32_t kidOffset = allkids.size();
            allkids.insert(allkids.end(), kids.begin(), kids.end());

            auto bbox = root.getNodeBoundingBox();
            cache[0] = NodeCache{ minmax, bbox, kcount, kidOffset};
        }

        const NodeCache dummy{};

        inline const NodeCache& query(uint32_t i) const {
            return cache[i];
        }

        FloatMinMax queryMinMax(const uint32_t idx, const openvdb::CoordBBox& queryBBox) const
        {
            const auto& cached = query(idx);
            if ( !cached.bbox.hasOverlap(queryBBox) ) return {};

            if ( queryBBox.isInside(cached.bbox) || cached.kidCount==0 )
                return cached.minmax;

            FloatMinMax minmax;
            for (int i=0; i<cached.kidCount; ++i) {
                const auto& kid = allkids[cached.kidOffset+i];
                auto tmp = queryMinMax(kid, queryBBox);
                if (!tmp.valid()) continue;
                minmax.merge(tmp);
            }
            return minmax;
        }

        FloatMinMax queryMinMax(const VDBFloatGrid& vgrid, BBoxF& query_box) const {
            // Conservative conversion
            openvdb::Coord minCoord = openvdb::Coord::floor(query_box.min());
            openvdb::Coord maxCoord = openvdb::Coord::ceil(query_box.max());
            openvdb::CoordBBox queryBBox {minCoord, maxCoord}; 

            const auto& root = vgrid.m_grid->tree().root();
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
                auto tmp = queryMinMax(kid, queryBBox);
                if (!tmp.valid()) continue;
                minmax.merge(tmp);
            }
            return minmax;
        }
    };

    FloatMinMax fetchMinMax(const BBoxF& overlap_box, const VDBFloatGrid& vgrid, const glm::mat4& transform, const GridMinMax& gmm) {
        glm::mat4 inverse = glm::inverse(transform); // world to indexed space

        auto mini = Vec3F(+FLT_MAX);
        auto maxi = Vec3F(-FLT_MAX);

        auto ext = overlap_box.extents();
        auto tmp = overlap_box.min();

        for (int i=0; i<8; ++i) {
            auto p = glm::vec4(tmp.x(), tmp.y(), tmp.z(), 1.0);
            if (i & 1) p.x += ext.x();
            if (i & 2) p.y += ext.y();
            if (i & 4) p.z += ext.z();

            p = inverse * p;
            mini = min(mini, reinterpret_cast<Vec3F&>(p));
            maxi = max(maxi, reinterpret_cast<Vec3F&>(p));  
        }
        /*openvdb::CoordBBox cbb;
        cbb.max() = openvdb::Coord::ceil(maxi);
        cbb.min() = openvdb::Coord::floor(mini);*/
        BBoxF bbox = { mini, maxi };
        return gmm.queryMinMax(vgrid, bbox);
    }
    
    struct OctNode {
        uint8_t child_mask = 0b00000000;
        uint32_t child_offset = UINT_MAX;

        uint8_t vlist_size;
        uint32_t vlist_offset =0;
        float min_r, max_r;  
    };

    void subdivide(const BBoxF& bbox, std::vector<OctNode>& octree, uint32_t octidx, float T, std::vector<uint32_t>& grid_sel,
        std::vector<BBoxF>& vboxes, const std::vector<VDBFloatGrid*>& vgrids, const std::vector<glm::mat4>& transforms, const std::vector<GridMinMax>& gmms) 
    {
        assert(!bbox.empty() && bbox.hasVolume());
        const auto& octbox = bbox;
        const auto ext = octbox.extents()/2;
        // if (ext.length() < 2) return;
        
        std::vector<std::pair<uint32_t, BBoxF>> tasks;
        tasks.reserve(8);

        for (int i=0; i<8; ++i) {
            auto p = octbox.min();
            if (i & 1) p.x() += ext.x();
            if (i & 2) p.y() += ext.y();
            if (i & 4) p.z() += ext.z();

            auto subox = BBoxF(p, p+ext);
            //overlap volumes of this subox
            std::vector<uint32_t> overlaps;
            float min_r = 0.f;
            float max_r = 0.f;

            bool gap = true;
            BBoxF coverbox {};

            auto& octnode = octree[octidx];
            auto& child_mask = octnode.child_mask;
            auto& child_offset = octnode.child_offset;

            for (int k=0; k<octnode.vlist_size; ++k) {
                auto offset = octnode.vlist_offset + k;
                const auto idx = grid_sel[offset];

                const auto& vbox = vboxes[idx];
                BBoxF overlap_box {};

                if( subox.hasOverlap(vbox) ) {
                    auto overlap_min = max(subox.min(), vbox.min());
                    auto overlap_max = min(subox.max(), vbox.max());
                    overlap_box = BBoxF(overlap_min, overlap_max);

                    overlaps.push_back(idx);

                    FloatMinMax min_max = fetchMinMax(overlap_box, *vgrids[idx], transforms[idx], gmms[idx]);
                    min_r += min_max.mini;
                    max_r += min_max.maxi;
                } else { continue; }

                if ( !coverbox.hasVolume() ) {
                    coverbox = overlap_box;
                } else {
                    coverbox.min() = min(coverbox.min(), overlap_box.min());
                    coverbox.max() = max(coverbox.max(), overlap_box.max());
                }

                if ( subox.isInside(vbox) || subox.isInside(coverbox)) {
                    gap = false;
                }
            }

            if (gap) min_r = 0.0f;
            if(overlaps.size()>0) {
                child_mask |= 1<<(i); // 8 children bit mask
            } else { continue; } // empty node

            if (UINT_MAX == child_offset)
                child_offset = octree.size();

            const uint32_t node_idx = octree.size();
            octree.emplace_back(OctNode());
            auto& new_node = octree.back();
            new_node.child_mask = 0;
            new_node.child_offset = UINT32_MAX;
            new_node.min_r = min_r;
            new_node.max_r = max_r;
            new_node.vlist_size = overlaps.size();
            new_node.vlist_offset = grid_sel.size();

            grid_sel.insert(grid_sel.end(), overlaps.begin(), overlaps.end());
            if (overlaps.size()==1) continue;

            auto diag = ext.length() * 2; //T ∈ {1, 4}
            bool should_divide = (max_r - min_r) * diag > T;
            if (!should_divide) { continue; }
            
            tasks.emplace_back( std::pair{node_idx, subox} );
        } // subox

        for (const auto& [nidx, box] : tasks) {
            subdivide(box, octree, nidx, T, grid_sel, vboxes, vgrids, transforms, gmms);
        }
    }

    BBoxF octbox;
    std::vector<OctNode> octree;
    std::vector<uint32_t> grid_sel;
        
    void aggregate(const std::vector<VDBFloatGrid*>& vgrids, const std::vector<glm::mat4>& transforms, float T=1.0f) {

        octbox = {};
        octree = {};
        grid_sel = {};

        octbox.min() = openvdb::Vec3f(+FLT_MAX);
        octbox.max() = openvdb::Vec3f(-FLT_MAX);

        std::vector<BBoxF> vboxes;

        for (int n=0; n<vgrids.size(); ++n) {
            auto& grid = const_cast<VDBFloatGrid&>(*vgrids[n]);
            const auto bbox = grid.evalActiveVoxelBoundingBox();

            auto min = bbox.min();
            auto max = bbox.max();

            glm::vec4 dummy[2];
            dummy[0] = glm::vec4(min.x(), min.y(), min.z(), 1.0);
            dummy[1] = glm::vec4(max.x(), max.y(), max.z(), 1.0);

            glm::vec3 mini(+FLT_MAX);
            glm::vec3 maxi(-FLT_MAX);

            for (int i=0; i<=1; ++i)
                for (int j=0; j<=1; ++j)
                    for (int k=0; k<=1; ++k) {

                        auto x = dummy[i][0];
                        auto y = dummy[j][1];
                        auto z = dummy[k][2];

                        auto p = glm::vec4(x, y, z, 1.0f);
                        p = transforms[n] * p;

                        mini.x = fminf(mini.x, p.x); mini.y = fminf(mini.y, p.y); mini.z = fminf(mini.z, p.z);
                        maxi.x = fmaxf(maxi.x, p.x); maxi.y = fmaxf(maxi.y, p.y); maxi.z = fmaxf(maxi.z, p.z);
                    }

            openvdb::math::BBox<openvdb::Vec3f> box;
            box.min() = openvdb::Vec3f(mini.x, mini.y, mini.z);
            box.max() = openvdb::Vec3f(maxi.x, maxi.y, maxi.z);
            vboxes.push_back(box);

            octbox.expand(box);
        }

        octree.emplace_back(OctNode());
        OctNode& rootNode = octree.back();
        rootNode.vlist_size = vgrids.size();
        rootNode.vlist_offset = 0;

        std::vector<GridMinMax> gmms;
        gmms.reserve(vgrids.size());
        for (int n=0; n<vgrids.size(); ++n) {
            grid_sel.emplace_back(n);

            gmms.emplace_back(GridMinMax());
            auto& gmm = gmms.back();
            gmm.precompute(*vgrids[n]);
        }

        subdivide(octbox, octree, 0, T, grid_sel, vboxes, vgrids, transforms, gmms);
    }
};

struct CreateVolumeAggregate : zeno::INode {

    glm::mat4 index_to_world(VDBFloatGrid* grid) {
        auto tmp = grid->getTransform().baseMap()->getAffineMap()->getMat4();
        glm::mat4 result;
        for (size_t i=0; i<16; ++i) {
            auto ele = *(tmp[0]+i);
            result[i/4][i%4] = ele;
        }
        return result;
    }

    void octreeAsGrids(uint32_t idx, const std::vector<VolumeAggregate::OctNode>& octree, const VolumeAggregate::BBoxF& bbox, zeno::PrimitiveObject* prim)
    {   
        const auto& octnode = octree[idx];
        {
            size_t offset = prim->verts.size();
            openvdb::Vec3f dummy[2];
            dummy[0] = bbox.min();
            dummy[1] = bbox.max();

            if (0==octnode.vlist_size) return;

            for (int i=0; i<=1; ++i)
                for (int j=0; j<=1; ++j)
                    for (int k=0; k<=1; ++k) {
                        auto x = dummy[i][0];
                        auto y = dummy[j][1];
                        auto z = dummy[k][2];
                        prim->verts.push_back(zeno::vec3f(x, y, z));
                    }
            // enough to draw box wire frame
            prim->quads->push_back(zeno::vec4i(offset+0, offset+1, offset+3, offset+2));
            prim->quads->push_back(zeno::vec4i(offset+4, offset+5, offset+7, offset+6));
            prim->quads->push_back(zeno::vec4i(offset+0, offset+1, offset+5, offset+4));
            prim->quads->push_back(zeno::vec4i(offset+3, offset+2, offset+6, offset+7));
        }

        const auto ext = bbox.extents()/2;
        uint32_t kid = 0u;
        for (int i=0; i<8; ++i) {
            auto mask = 1<<(i);
            bool valid = i & octnode.child_mask;
            if (!valid) continue;
            if (octnode.vlist_size==0) continue;

            auto p = bbox.min();
            if (i & 1) p.x() += ext.x();
            if (i & 2) p.y() += ext.y();
            if (i & 4) p.z() += ext.z();

            auto subbox = VolumeAggregate::BBoxF(p, p+ext);
            octreeAsGrids(octnode.child_offset+kid, octree, subbox, prim);
            ++kid;
        }
    }

    virtual void apply() override {
        // auto grid = get_input2<VDBGrid>("vdbGrid");
        // auto float_grid = std::dynamic_pointer_cast<VDBFloatGrid>(grid);
        // auto root = float_grid->m_grid->tree().root();

        std::vector<VDBFloatGrid*> vgrids;
        std::vector<glm::mat4>     vtrans;
        
        if (has_input("vgrids")) {
            auto raw = get_input<ListObject>("vgrids")->getRaw();
            if (raw.empty()) {
                throw zeno::makeError("empty list vgrids");
            }
            for (const auto& grid_ptr : raw) {

                const auto ele = dynamic_cast<VDBFloatGrid*>(grid_ptr);
                if (ele == nullptr) {
                    throw zeno::makeError("cast failed vgrids");
                }
                vgrids.push_back(ele);
                vtrans.push_back(index_to_world(ele));
            }                    
        }

        auto threshold = get_input2<float>("threshold");

        VolumeAggregate vagg;
        vagg.aggregate(vgrids, vtrans, threshold);
        
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        prim->verts->reserve(vagg.octree.size() * 8);
        prim->quads->reserve(vagg.octree.size() * 4);

        auto octbox = vagg.octbox;
        // auto octnode = vagg.octree[0];
        octreeAsGrids(0, vagg.octree, octbox, prim.get());
        primWireframe(prim.get(), true);
        set_output("prim", prim);
    }
};

ZENDEFNODE(CreateVolumeAggregate, {
    {
        {"float", "threshold", "1.0"},
        {"list", "vgrids"},
    },
    {"prim"},
    {},
    {"create"}
});

} // namespace