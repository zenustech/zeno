#include <string>
#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/types/PrimitiveObject.h>
// #include <zeno/types/DictObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/UserData.h>

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

        std::vector<glm::mat4> transforms;

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

            auto AsTras = [&grid](openvdb::CoordBBox& box) {

                glm::vec3 bmax = glm::vec3(box.max().x(), box.max().y(), box.max().z()) + 1.0f;
                glm::vec3 bmin = glm::vec3(box.min().x(), box.min().y(), box.min().z());
    
                auto diff = bmax - bmin;
                auto center = bmin + diff / 2.0f;
    
                auto trans = glm::mat4(1.0f);
    
                trans = glm::translate(trans, center);
                trans = glm::scale(trans, diff);
    
                const auto world_matrix = [&]() -> auto {
    
                    auto tmp = grid->getTransform().baseMap()->getAffineMap()->getMat4();
                    glm::mat4 result;
                    for (size_t i=0; i<16; ++i) {
                        auto ele = *(tmp[0]+i);
                        result[i/4][i%4] = ele;
                    }
                    return result;
                }();
    
                return world_matrix * trans;
                //transform = world_matrix * trans;
                //transform = trans;
            };

            if (greedy) {

                openvdb::CoordBBox allbox;
                std::vector<openvdb::CoordBBox> nodeboxs;

                openvdb::Coord voxelSize;   
                for (TreeType::NodeIter iter = float_grid->m_grid->tree().beginNode(); iter; ++iter) {

                    if (iter.getDepth() == 2) {
                        auto box = iter.getBoundingBox();

                        transforms.push_back( AsTras(box) );
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

                GreedyVoxel greedyVoxle(ndim.x(), ndim.y(), ndim.z());

                for (const auto& box : nodeboxs) {
                    auto diff = box.min() - allbox.min();
                    auto offset = diff.asVec3i() / voxelSize.asVec3i();  
                    
                    greedyVoxle.fill(offset.x(), offset.y(), offset.z());
                }

                auto voxels = greedyVoxle.greedy();
                transforms.clear();
                transforms.reserve(voxels.size());

                for (auto& box : voxels) {

                    auto mini = box.min().asVec3i() * voxelSize.asVec3i() + allbox.min();
                    auto maxi = box.max().asVec3i() * voxelSize.asVec3i() + allbox.min();

                    openvdb::Coord box_min = openvdb::Coord(mini.x(), mini.y(), mini.z());
                    openvdb::Coord box_max = openvdb::Coord(maxi.x(), maxi.y(), maxi.z());
                    auto nbox = openvdb::CoordBBox(box_min, box_max);

                    transforms.push_back(AsTras(nbox));
                }
            }
            else {
                auto box = grid->evalActiveVoxelBoundingBox();
                transforms.push_back( AsTras(box) );
            }

        } else {
            glm::mat4 transform(1.0);
            transform = glm::translate(transform, glm::vec3(pos[0], pos[1], pos[2]));
            transform = transform * rotation;
            transform = glm::scale(transform, glm::vec3(scale[0], scale[1], scale[2]));
            transforms.push_back(transform);
        }

        auto list = std::make_shared<zeno::ListObject>();

        for (auto& transform : transforms) {

            auto prim = std::make_shared<zeno::PrimitiveObject>();
            prim->userData().set2("vol_mat", get_input2<std::string>("vol_mat", ""));

            float dummy[] = {-0.5f, 0.5f};

            for (int i=0; i<=1; ++i) {
                for (int j=0; j<=1; ++j) {
                    for (int k=0; k<=1; ++k) {
                        auto p = glm::vec4(dummy[i], dummy[j], dummy[k], 1.0f);
                        p = transform * p;
                        prim->verts.push_back(zeno::vec3f(p.x, p.y, p.z));
                    }
                }
            }

            // enough to draw box wire frame
            prim->quads->push_back(zeno::vec4i(0, 1, 3, 2));
            prim->quads->push_back(zeno::vec4i(4, 5, 7, 6));
            prim->quads->push_back(zeno::vec4i(0, 1, 5, 4));
            prim->quads->push_back(zeno::vec4i(3, 2, 6, 7));

            primWireframe(prim.get(), true);
            prim->userData().set2("bounds", bounds);

            auto transform_ptr = glm::value_ptr(transform);

            zeno::vec4f row0, row1, row2, row3;
            memcpy(row0.data(), transform_ptr, sizeof(float)*4);
            memcpy(row1.data(), transform_ptr+4, sizeof(float)*4);
            memcpy(row2.data(), transform_ptr+8, sizeof(float)*4);
            memcpy(row3.data(), transform_ptr+12, sizeof(float)*4);

            prim->userData().set2("_transform_row0", row0);
            prim->userData().set2("_transform_row1", row1);
            prim->userData().set2("_transform_row2", row2);
            prim->userData().set2("_transform_row3", row3);
            prim->userData().set2("vbox", true);

            list->arr.push_back(prim);
        }

        if (list->arr.size()==1) {
            set_output("prim", std::move(list->arr.front()));
            return;
        }

        set_output("prim", std::move(list));

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

} // namespace