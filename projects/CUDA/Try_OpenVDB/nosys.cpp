#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

// #include "skinning_iobject.h"

#include <iostream>
#include <openvdb/openvdb.h>
#include <openvdb/Metadata.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/ChangeBackground.h>
#include "openvdb/tools/MeshToVolume.h"
#include "openvdb/tools/Morphology.h"
#include "openvdb/tree/LeafManager.h"
#include <openvdb/tools/Interpolation.h>

#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tools/LevelSetSphere.h>

#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"

namespace{
using namespace zeno;

struct HelloWorld : zeno::INode {
    virtual void apply() override {
        // the zeno has already initialize the openvdb the time the program start
        // openvdb::initialize();

        openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
        std::cout << "Testing random access:" << std::endl;
        // Get an accessor for coordinate-based access to voxels.
        openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
        // Define a coordinate with large signed indices.
        openvdb::Coord xyz(1000, -200000000, 30000000);
        // Set the voxel value at (1000, -200000000, 30000000) to 1.
        accessor.setValue(xyz, 1.0);
        // Verify that the voxel value at (1000, -200000000, 30000000) is 1.
        std::cout << "Grid" << xyz << " = " << accessor.getValue(xyz) << std::endl;
        // Reset the coordinates to those of a different voxel.
        xyz.reset(1000, 200000000, -30000000);
        // Verify that the voxel value at (1000, 200000000, -30000000) is
        // the background value, 0.
        std::cout << "Grid" << xyz << " = " << accessor.getValue(xyz) << std::endl;
        // Set the voxel value at (1000, 200000000, -30000000) to 2.
        accessor.setValue(xyz, 2.0);
        // Set the voxels at the two extremes of the available coordinate space.
        // For 32-bit signed coordinates these are (-2147483648, -2147483648, -2147483648)
        // and (2147483647, 2147483647, 2147483647).
        accessor.setValue(openvdb::Coord::min(), 3.0f);
        accessor.setValue(openvdb::Coord::max(), 4.0f);
        std::cout << "Testing sequential access:" << std::endl;
        // Print all active ("on") voxels by means of an iterator.
        for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter) {
            std::cout << "Grid" << iter.getCoord() << " = " << *iter << std::endl;
        }

    }
}; 

ZENDEFNODE(HelloWorld, {
    {},
    {},
    {},
    {"TEST_OPENVDB"},
});


struct CreatingAndWritingGrid : zeno::INode {
    template<class GridType>
    void makeSphere(GridType& grid,float radius,const openvdb::Vec3f& c) {
        using ValueT = typename GridType::ValueType;
        const ValueT outside = grid.background();
        const ValueT inside = -outside;

        int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));
        int dim = int(radius + padding);

        typename GridType::Accessor accessor = grid.getAccessor();

        openvdb::Coord ijk;
        int &i = ijk[0],&j = ijk[1],&k = ijk[2];
        for(i = c[0] - dim;i < c[0] + dim;++i){
            const float x2 = openvdb::math::Pow2(i - c[0]);
            for(j = c[1] - dim;j < c[1] + dim;++j){
                const float x2y2 = openvdb::math::Pow2(j - c[1]) + x2;
                for(k = c[2] - dim;k < c[2] + dim;++k){
                    float dist = openvdb::math::Pow2(k - c[2])  + x2y2;
                    dist = openvdb::math::Sqrt(dist) - radius;

                    ValueT val = ValueT(dist);
                    if(val < inside || outside < val) continue;

                    accessor.setValue(ijk,val);
                }
            }
        }

        openvdb::tools::signedFloodFill(grid.tree());
        // openvdb::tools::signedFloodFill(grid.tree());
    }


    virtual void apply() override {
        // the zeno has already initialize the openvdb the time the program start
        // openvdb::initialize();
        auto output_filename = get_input<StringObject>("output_filename")->get();

        openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(2.0);
        makeSphere(*grid,/*radius=*/50.0f,/*center=*/openvdb::Vec3f(1.5,2,3));
        // grid->insertMeta("radius",openvdb::FloatMetaData(50.0));
        grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
        grid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.5));
        
        grid->setGridClass(openvdb::GRID_LEVEL_SET);
        grid->setName("LevelSetSphere");

        openvdb::io::File file(output_filename.c_str());
        openvdb::GridPtrVec grids;
        grids.push_back(grid);

        file.write(grids);
        file.close();
    }
}; 

ZENDEFNODE(CreatingAndWritingGrid, {
    {{"writepath","output_filename"}},
    {},
    {},
    {"TEST_OPENVDB"},
});


struct ReadingAndModifyingAGrid : zeno::INode {
    virtual void apply() override {
        auto input_filename = get_input<StringObject>("input_filename")->get();
        openvdb::io::File file(input_filename.c_str());
        file.open();

        openvdb::GridBase::Ptr baseGrid;
        for(auto nameIter = file.beginName();nameIter != file.endName();++nameIter) {
            if(nameIter.gridName() == "LevelSetSphere") {
                baseGrid = file.readGrid(nameIter.gridName());
            } else {
                std::cout << "skipped grid " << nameIter.gridName() << std::endl;
            }
        }

        file.close();

        openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

        const float outside = grid->background();
        const float width = 2.0 * outside;

        for(openvdb::FloatGrid::ValueOnIter iter = grid->beginValueOn();iter;++iter) {
            float dist = iter.getValue();
            // iter.setValue((outside - dist) / width);
            iter.setValue(dist / outside);
        }

        for(openvdb::FloatGrid::ValueOffIter iter = grid->beginValueOff();iter;++iter) {
            if(iter.getValue() < 0.0){
                iter.setValue(0.0);
                iter.setValueOff();
            }
        }
        openvdb::tools::changeBackground(grid->tree(),1.0);

        // std::cout << "output the metadata" << std::endl;
        // for(openvdb::MetaMap::MetaIterator iter = grid->beginMeta();iter != grid->endMeta();++iter) {
        //     const std::string& name = iter->first;
        //     openvdb::Metadata::Ptr value = iter->second;
        //     std::string valueAsString = value->str();
        //     std::cout << name << " = " << valueAsString << std::endl;
        // }
        // grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
        // grid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.5));
        
        grid->setGridClass(openvdb::GRID_FOG_VOLUME);
        grid->setName("FogVolumeSphere");

        auto output_filename = get_input<StringObject>("output_filename")->get();
        openvdb::io::File(output_filename.c_str()).write({grid});
    }
};

ZENDEFNODE(ReadingAndModifyingAGrid, {
    {{"readpath","input_filename"},{"writepath","output_filename"}},
    {},
    {},
    {"TEST_OPENVDB"},
});


struct NodeIterator_OpenVDB : zeno::INode {
    virtual void apply() override {
        using GridType = openvdb::FloatGrid;
        using TreeType = GridType::TreeType;
        using RootType = TreeType::RootNodeType;
        assert(RootType::LEVEL == 3);// make sure it it a four level hiararchy tree structure 
        using Int1Type = RootType::ChildNodeType;
        using Int2Type = Int1Type::ChildNodeType;
        using LeafType = Int2Type::ChildNodeType;

        auto input_filename = get_input<StringObject>("input_filename")->get();
        openvdb::io::File file(input_filename.c_str());
        file.open();

        openvdb::GridBase::Ptr baseGrid;
        for(openvdb::io::File::NameIterator nameIter = file.beginName();nameIter != file.endName();++nameIter) {
            if(nameIter.gridName() == "LevelSetSphere")
                baseGrid = file.readGrid(nameIter.gridName());
            else
                std::cout << "skipping grid" << nameIter.gridName() << std::endl;
        }

        openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        // output the meta data
        for(openvdb::MetaMap::MetaIterator iter = grid->beginMeta();iter != grid->endMeta();++iter) {
            const std::string& name = iter->first;
            openvdb::Metadata::Ptr value = iter->second;
            std::string valueAsString = value->str();

            std::cout << name << " = " << valueAsString << std::endl;
        }

        // output all the tile data and voxel data
        std::cout << "Node Iterator" << std::endl;
        for(TreeType::NodeIter iter = grid->tree().beginNode();iter;++iter) {
            switch(iter.getDepth()) {
                case 0: {RootType* node = nullptr;iter.getNode(node);if(node) std::cout << "detect root node " << iter.getBoundingBox() << std::endl;break;}
                case 1: {Int1Type* node = nullptr;iter.getNode(node);if(node) std::cout << "detect int1 node " << iter.getBoundingBox() << std::endl;break;}
                case 2: {Int2Type* node = nullptr;iter.getNode(node);if(node) std::cout << "detect int2 node " << iter.getBoundingBox() << std::endl;break;}
                // case 3: {LeafType* node = nullptr;iter.getNode(node);if(node) std::cout << "detect leaf node " << iter.getBoundingBox() << std::endl;break;}
            }
        }

        std::cout << "Leaf Node Iterator" << std::endl;
        
    }
};

ZENDEFNODE(NodeIterator_OpenVDB,{
    {{"readpath","input_filename"}},
    {},
    {},
    {"TEST_OPENVDB"},
});


struct ViewVDBBoundingBox : zeno::INode {
    template<typename GridPtr>
    void add_bounding_colored_box(GridPtr grid,const openvdb::CoordBBox& bbox,
            int id,const zeno::vec3f& clr,std::vector<zeno::vec2i>& lines,
            std::vector<zeno::vec3f>& pos,std::vector<zeno::vec3f>& clrs) {
        int voffset = id * 8;
        int toffset = id * 12;
        lines[toffset++] = zeno::vec2i{0, 4} + voffset;
        lines[toffset++] = zeno::vec2i{1, 5} + voffset;
        lines[toffset++] = zeno::vec2i{2, 6} + voffset;
        lines[toffset++] = zeno::vec2i{3, 7} + voffset;
        lines[toffset++] = zeno::vec2i{0, 2} + voffset;
        lines[toffset++] = zeno::vec2i{1, 3} + voffset;

        lines[toffset++] = zeno::vec2i{4, 6} + voffset;
        lines[toffset++] = zeno::vec2i{5, 7} + voffset;
        lines[toffset++] = zeno::vec2i{0, 1} + voffset;
        lines[toffset++] = zeno::vec2i{2, 3} + voffset;
        lines[toffset++] = zeno::vec2i{4, 5} + voffset;
        lines[toffset++] = zeno::vec2i{6, 7} + voffset;

        openvdb::Vec3f wmin = grid->transform().indexToWorld(bbox.min().asVec3s());
        openvdb::Vec3f wmax = grid->transform().indexToWorld(bbox.max().asVec3s());
        
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmin.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmin.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmax.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmax.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmin.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmin.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmax.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmax.z());
    }

    virtual void apply() override {
        using GridType = openvdb::FloatGrid;
        using TreeType = GridType::TreeType;
        using RootType = TreeType::RootNodeType;
        using Int1Type = RootType::ChildNodeType;
        using Int2Type = Int1Type::ChildNodeType;
        using LeafType = Int2Type::ChildNodeType;

        auto input_filename = get_input<StringObject>("input_filename")->get();
        openvdb::io::File file(input_filename);
        file.open();
        openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<GridType>(file.readGrid("LevelSetSphere"));
        
        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::vector<int> nm_nodes{0,0,0,0};

        for(TreeType::NodeIter iter = grid->tree().beginNode();iter;++iter)
            nm_nodes[iter.getDepth()] += 1;

        int total_nm_nodes = nm_nodes[0] + nm_nodes[1] + nm_nodes[2] + nm_nodes[3];
        auto& verts = prim->verts;
        auto& segs = prim->lines;
        verts.resize(total_nm_nodes * 8);
        segs.resize(total_nm_nodes * 12);

        std::array<zeno::vec3f,4> level_clrs{
            zeno::vec3f{1.0,0.0,0.0},
            zeno::vec3f{0.0,1.0,0.0},
            zeno::vec3f{0.0,0.0,1.0},
            zeno::vec3f{1.0,1.0,1.0}
        };

        int idx = 0;
        auto& clrs = prim->add_attr<zeno::vec3f>("clr");
        for(TreeType::NodeIter iter = grid->tree().beginNode();iter;++iter)
            add_bounding_colored_box(grid,iter.getBoundingBox(),idx++,level_clrs[iter.getDepth()],segs.values,verts.values,clrs);



        // idx = 0;
        // for(TreeType::NodeIter iter = grid->tree().beginNode();iter;++iter)
        //     add_bounding_box(grid,iter.getBoundingBox(),idx++,segs.values,verts.values);       

        set_output("prim",std::move(prim));
    }
};

ZENDEFNODE(ViewVDBBoundingBox,{
    {{"readpath","input_filename"}},
    {"prim"},
    {},
    {"TEST_OPENVDB"},
});


struct ViewLeafNodeBoundingBox : zeno::INode {
    template<typename GridPtr>
    void add_bounding_colored_box(GridPtr grid,const openvdb::CoordBBox& bbox,
            int id,const zeno::vec3f& clr,std::vector<zeno::vec2i>& lines,
            std::vector<zeno::vec3f>& pos,std::vector<zeno::vec3f>& clrs) {
        int voffset = id * 8;
        int toffset = id * 12;
        lines[toffset++] = zeno::vec2i{0, 4} + voffset;
        lines[toffset++] = zeno::vec2i{1, 5} + voffset;
        lines[toffset++] = zeno::vec2i{2, 6} + voffset;
        lines[toffset++] = zeno::vec2i{3, 7} + voffset;
        lines[toffset++] = zeno::vec2i{0, 2} + voffset;
        lines[toffset++] = zeno::vec2i{1, 3} + voffset;

        lines[toffset++] = zeno::vec2i{4, 6} + voffset;
        lines[toffset++] = zeno::vec2i{5, 7} + voffset;
        lines[toffset++] = zeno::vec2i{0, 1} + voffset;
        lines[toffset++] = zeno::vec2i{2, 3} + voffset;
        lines[toffset++] = zeno::vec2i{4, 5} + voffset;
        lines[toffset++] = zeno::vec2i{6, 7} + voffset;

        openvdb::Vec3f wmin = grid->transform().indexToWorld(bbox.min().asVec3s());
        openvdb::Vec3f wmax = grid->transform().indexToWorld(bbox.max().asVec3s());
        
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmin.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmin.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmax.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmax.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmin.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmin.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmax.z());
        clrs[voffset] = clr;
        pos[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmax.z());
    }

    virtual void apply() override {
        using GridType = openvdb::FloatGrid;
        using TreeType = GridType::TreeType;
        using RootType = TreeType::RootNodeType;
        using Int1Type = RootType::ChildNodeType;
        using Int2Type = Int1Type::ChildNodeType;
        using LeafType = Int2Type::ChildNodeType;


        auto input_filename = get_input<StringObject>("input_filename")->get();
        openvdb::io::File file(input_filename);
        file.open();

        openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid("LevelSetSphere"));
        
        int nm_leaf_nodes = 0;
        for(TreeType::LeafCIter iter = grid->tree().cbeginLeaf();iter;++iter) 
            nm_leaf_nodes++;

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto& verts = prim->verts;
        auto& segs = prim->lines;
        verts.resize(nm_leaf_nodes * 8);
        segs.resize(nm_leaf_nodes * 12);
        auto& clrs = prim->add_attr<zeno::vec3f>("clr");

        int idx = 0;
        for(TreeType::LeafCIter iter = grid->tree().cbeginLeaf();iter;++iter) {
            const LeafType& leaf = *iter;
            auto aabb = leaf.getNodeBoundingBox();
            add_bounding_colored_box(grid,aabb,idx++,zeno::vec3f{1.0},segs.values,verts.values,clrs);
        }

        set_output("prim",std::move(prim));


    }
};


ZENDEFNODE(ViewLeafNodeBoundingBox,{
    {{"readpath","input_filename"}},
    {"prim"},
    {},
    {"TEST_OPENVDB"},
});

// // Value Iterator
// struct ViewValue : zeno::INode {
//     virtual void apply() override {

//     }
// }

// Interpolation of grid values


// Scaling the level set
struct TransformLevelSet : zeno::INode {
    virtual void apply() override {
        using GridType = openvdb::FloatGrid;
        using TreeType = GridType::TreeType;
        using RootType = TreeType::RootNodeType;
        using Int1Type = RootType::ChildNodeType;
        using Int2Type = Int1Type::ChildNodeType;
        using LeafType = Int2Type::ChildNodeType;

        // std::cout << "Try Transform LevelSet" << std::endl;
        std::cout << "LeafType Dim : " << LeafType::LOG2DIM << std::endl;

        auto input_filename = get_input<StringObject>("input_filename")->get();
        openvdb::io::File file(input_filename);
        file.open();

        auto source_grid = openvdb::gridPtrCast<GridType>(file.readGrid("LevelSetSphere"));
        openvdb::FloatGrid::Ptr target_grid = openvdb::FloatGrid::create();

        // std::cout << "Finish read source grid" << std::endl;


        const openvdb::math::Transform &sourceXform = source_grid->transform();

        auto scale = get_input<zeno::NumericObject>("scale")->get<zeno::vec3f>();


        openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(sourceXform.baseMap()->getAffineMap()->getMat4());
        // std::cout << "before scale" << std::endl;
        // transform->print();
        transform->preScale({scale[0],scale[1],scale[2]});
        // std::cout << "after scale" << std::endl;
        // transform->print();
        // std::cout << "voxel_size : " << transform->voxelSize() << std::endl;

        // target_grid->setTransform(source_grid->transformPtr());
        target_grid->setTransform(transform);
        const openvdb::math::Transform &targetXform = target_grid->transform();

        openvdb::Mat4R scale_M;
        scale_M.setToScale(openvdb::Vec3R{scale[0],scale[1],scale[2]});
        
        openvdb::Mat4R xform = sourceXform.baseMap()->getAffineMap()->getMat4() * scale_M;
        xform = xform * targetXform.baseMap()->getAffineMap()->getMat4().inverse();

        openvdb::tools::GridTransformer transformer(xform);

        transformer.transformGrid<openvdb::tools::QuadraticSampler,GridType>(*source_grid,*target_grid);
        target_grid->tree().prune();

        target_grid->setName("LevelSetSphere");
        target_grid->setGridClass(openvdb::GRID_LEVEL_SET);
        auto output_filename = get_input<StringObject>("output_filename")->get();
        // std::cout << "input voxel size : " << source_grid->voxelSize() << "\t" << "output voxel size : " << target_grid->voxelSize() << std::endl;
        openvdb::io::File(output_filename.c_str()).write({target_grid});
    }
};

ZENDEFNODE(TransformLevelSet,{
    {{"readpath","input_filename"},{"writepath","output_filename"},"scale"},
    {},
    {},
    {"TEST_OPENVDB"},
});

struct ViewVDBPoints : zeno::INode {
    void add_colored_aabb(const openvdb::Vec3f& wmin,
        const openvdb::Vec3f& wmax,
        const zeno::vec3f& clr,
        int id,
        zeno::PrimitiveObject& prim) const {
            int voffset = id * 8;
            int toffset = id * 12;
            auto& lines = prim.lines.values;
            auto& verts = prim.verts.values;
            auto& clrs = prim.attr<zeno::vec3f>("clr");

            std::cout << "add : " << wmin << "\t" << wmax << std::endl;

            lines[toffset++] = zeno::vec2i{0, 4} + voffset;
            lines[toffset++] = zeno::vec2i{1, 5} + voffset;
            lines[toffset++] = zeno::vec2i{2, 6} + voffset;
            lines[toffset++] = zeno::vec2i{3, 7} + voffset;
            lines[toffset++] = zeno::vec2i{0, 2} + voffset;
            lines[toffset++] = zeno::vec2i{1, 3} + voffset;

            lines[toffset++] = zeno::vec2i{4, 6} + voffset;
            lines[toffset++] = zeno::vec2i{5, 7} + voffset;
            lines[toffset++] = zeno::vec2i{0, 1} + voffset;
            lines[toffset++] = zeno::vec2i{2, 3} + voffset;
            lines[toffset++] = zeno::vec2i{4, 5} + voffset;
            lines[toffset++] = zeno::vec2i{6, 7} + voffset;   

            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmax.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmax.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmax.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmax.z());            
    }


    virtual void apply() override {
        std::vector<openvdb::Vec3R> positions;

        auto points = get_input<zeno::ListObject>("points")->getLiterial<zeno::vec3f>();
        for(int i = 0;i < points.size();++i)
            positions.push_back(openvdb::Vec3R(points[i][0],points[i][1],points[i][2]));
        // positions.push_back(openvdb::Vec3R(-0.6,1,0));
        // positions.push_back(openvdb::Vec3R(1.5,3.5,1.0));
        // positions.push_back(openvdb::Vec3R(-1,6,-2));
        // positions.push_back(openvdb::Vec3R(1.1,1.25,0.06));

        openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper(positions);
        openvdb::Index pointsPerVoxel = 8;
        openvdb::Index voxelsPerLeaf = openvdb::points::PointDataGrid::TreeType::LeafNodeType::SIZE;
        openvdb::Index pointsPerLeaf = pointsPerVoxel * voxelsPerLeaf;

        float voxelSize = openvdb::points::computeVoxelSize(positionsWrapper,pointsPerVoxel);

        std::cout << "voxelSize = " << voxelSize << std::endl;
        openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(1);
        // the internal offset information is automatically allocated here
        openvdb::points::PointDataGrid::Ptr grid = openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
            openvdb::points::PointDataGrid>(positions,*transform);
        
        grid->setName("Points");
        openvdb::tools::dilateActiveValues(grid->tree(),3,
            openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);

        int nm_leaf_nodes = grid->tree().leafCount();


        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto& verts = prim->verts;
        auto& segs = prim->lines;
        verts.resize((nm_leaf_nodes + positions.size()) * 8);
        segs.resize((nm_leaf_nodes + positions.size()) * 12);
        auto& clrs = prim->add_attr<zeno::vec3f>("clr");

        auto point_width = get_input<zeno::NumericObject>("point_width")->get<float>();
        auto delta = openvdb::Vec3f(point_width);

        int idx = 0;

        std::vector<zeno::vec3f> clrs_graph{zeno::vec3f{1.0,0.0,0.0},zeno::vec3f{0.0,1.0,0.0},{0.0,0.0,1.0}};


        int leaf_idx = 0;

        for(auto leafIter = grid->tree().cbeginLeaf();leafIter;++leafIter,++leaf_idx) {
            auto aabb = leafIter->getNodeBoundingBox();
            openvdb::Vec3f wmin = grid->transform().indexToWorld(aabb.min().asVec3s());
            openvdb::Vec3f wmax = grid->transform().indexToWorld(aabb.max().asVec3s());
            std::cout << "AABB : " << aabb.min() << "\t" << aabb.max() << std::endl;

            // the offset serve as a memory-safer role, the voxels contain no information should not 
            // be allocated with any data-memory and thus have zero offset shifts
            // std::cout << "check internal offsets: " << std::endl;
            // for(int i = 0;i < voxelsPerLeaf;++i)
            //     std::cout << "offset : " << leafIter->getValue(i) << std::endl;

            add_colored_aabb(wmin,wmax,clrs_graph[leaf_idx],idx++,*prim);

            const openvdb::points::AttributeArray& array = leafIter->constAttributeArray("P");
            const openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(array);
            for(auto indexIter = leafIter->beginIndexOn();indexIter;++indexIter) {
                openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);

                const openvdb::Vec3f xyz = indexIter.getCoord().asVec3s();
                openvdb::Vec3f worldPosition = grid->transform().indexToWorld(voxelPosition + xyz);

                auto pbmin = worldPosition - delta;
                auto pbmax = worldPosition + delta;

                // std::cout << "add point : " << voxelPosition + xyz << std::endl;
                add_colored_aabb(pbmin,pbmax,clrs_graph[leaf_idx],idx++,*prim);
            }
        }
        
        set_output("prim",std::move(prim));
    }
};

ZENDEFNODE(ViewVDBPoints,{
    {"points","point_width"},
    {"prim"},
    {},
    {"TEST_OPENVDB"},
});

struct ViewVDBPointsWithAttribute : zeno::INode {

void add_colored_aabb(const openvdb::Vec3f& wmin,
        const openvdb::Vec3f& wmax,
        const zeno::vec3f& clr,
        int id,
        zeno::PrimitiveObject& prim) const {
            int voffset = id * 8;
            int toffset = id * 12;
            auto& lines = prim.lines.values;
            auto& verts = prim.verts.values;
            auto& clrs = prim.attr<zeno::vec3f>("clr");

            std::cout << "add : " << wmin << "\t" << wmax << std::endl;

            lines[toffset++] = zeno::vec2i{0, 4} + voffset;
            lines[toffset++] = zeno::vec2i{1, 5} + voffset;
            lines[toffset++] = zeno::vec2i{2, 6} + voffset;
            lines[toffset++] = zeno::vec2i{3, 7} + voffset;
            lines[toffset++] = zeno::vec2i{0, 2} + voffset;
            lines[toffset++] = zeno::vec2i{1, 3} + voffset;

            lines[toffset++] = zeno::vec2i{4, 6} + voffset;
            lines[toffset++] = zeno::vec2i{5, 7} + voffset;
            lines[toffset++] = zeno::vec2i{0, 1} + voffset;
            lines[toffset++] = zeno::vec2i{2, 3} + voffset;
            lines[toffset++] = zeno::vec2i{4, 5} + voffset;
            lines[toffset++] = zeno::vec2i{6, 7} + voffset;   

            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmax.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmax.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmax.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmax.z());            
    }

    virtual void apply() override {
        auto points = get_input<ListObject>("points")->getLiterial<zeno::vec3f>();
        auto radius_ = get_input<ListObject>("radius")->getLiterial<float>();
        std::vector<openvdb::Vec3R> positions(points.size());
        for(int i = 0;i < points.size();++i)
            positions[i] = openvdb::Vec3R(points[i][0],points[i][1],points[i][2]);
        
        std::vector<float> radius(radius_.size());
        for(size_t i = 0;i < radius.size();++i)
            radius[i] = radius_[i];

        openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper(positions);

        float voxelSize = 1.f;
        openvdb::math::Transform::Ptr transform = 
            openvdb::math::Transform::createLinearTransform(voxelSize);
        
        openvdb::tools::PointIndexGrid::Ptr pointIndexGrid = 
            openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(positionsWrapper,*transform);


        openvdb::points::PointDataGrid::Ptr grid = 
            openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
                openvdb::points::PointDataGrid>(*pointIndexGrid,positionsWrapper,*transform);
        // for(auto leafIter = pointIndexGrid->tree().cbeginLeaf();leafIter;++leafIter) {
        //     std::cout << "Leaf" << leafIter->origin() << std::endl;
        //     std::cout << "bounding_box : " << leafIter->getNodeBoundingBox() << std::endl;
        //     auto indices = leafIter->indices();
        //     for(int i = 0;i < indices.size();++i)
        //         std::cout << indices[i] << "\t";
        //     std::cout << std::endl;

        // }
        
        using Codec = openvdb::points::FixedPointCodec<false,openvdb::points::UnitRange>;
        openvdb::points::TypedAttributeArray<float,Codec>::registerType();
        openvdb::NamePair radiusAttributeTag = 
            openvdb::points::TypedAttributeArray<float,Codec>::attributeType();
        
        openvdb::points::appendAttribute(grid->tree(),"pscale",radiusAttributeTag);
        openvdb::points::PointAttributeVector<float> radiusWrapper(radius);
        openvdb::points::populateAttribute<openvdb::points::PointDataTree,
            openvdb::tools::PointIndexTree,openvdb::points::PointAttributeVector<float>>(
                grid->tree(),pointIndexGrid->tree(),"pscale",radiusWrapper);
                
        grid->setName("Points");

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        int nm_leaf_nodes = 0;
        for(auto iter = grid->tree().cbeginLeaf();iter;++iter)
            nm_leaf_nodes++;

        prim->add_attr<zeno::vec3f>("clr"); 
        prim->verts.resize((nm_leaf_nodes + positions.size()) * 8);
        prim->lines.resize((nm_leaf_nodes + positions.size()) * 12);
       

        int idx = 0;

        // auto point_width = get_input2<float>("point_width");

        for(auto leafIter = grid->tree().cbeginLeaf();leafIter;++leafIter) {
            std::cout << "Leaf" << leafIter->origin() << std::endl;
            const openvdb::points::AttributeArray& positionArray = 
                leafIter->constAttributeArray("P");

            const openvdb::points::AttributeArray& radiusArray = 
                leafIter->constAttributeArray("pscale");
            
            openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(positionArray);
            openvdb::points::AttributeHandle<float> radiusHandle(radiusArray);

            auto leafnode_aabb = leafIter->getNodeBoundingBox();
            auto wmin = grid->transform().indexToWorld(leafnode_aabb.min().asVec3s());
            auto wmax = grid->transform().indexToWorld(leafnode_aabb.max().asVec3s());

            add_colored_aabb(wmin,wmax,zeno::vec3f{1.0,1.0,1.0},idx++,*prim);

            for(auto indexIter = leafIter->beginIndexOn();indexIter;++indexIter) {
                openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
                openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();

                openvdb::Vec3f worldPosition = grid->transform().indexToWorld(voxelPosition + xyz);
                float radius = radiusHandle.get(*indexIter);

                auto wmin = worldPosition - openvdb::Vec3f{radius};
                auto wmax = worldPosition + openvdb::Vec3f{radius};

                add_colored_aabb(wmin,wmax,zeno::vec3f{1.0,1.0,0.0},idx++,*prim);

                std::cout << "voxelPosition = [" << voxelPosition << "] ";
                std::cout << "xyz = [" << xyz << "] "; 
                std::cout << "* PointIndex=[" << *indexIter << "] ";
                std::cout << "WorldPosition=" << worldPosition << " ";
                std::cout << "Radius=" << radius << std::endl;
            }
            
        }

        set_output("prim",prim);
    }
};

ZENDEFNODE(ViewVDBPointsWithAttribute,{
    {"points","radius"},
    {"prim"},
    {},
    {"TEST_OPENVDB"},
});

struct VDBRandomPointGeneration : zeno::INode {
    virtual void apply() override {
        openvdb::FloatGrid::Ptr sphereGrid =
            openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(/*radius=*/20.0,
                /*center=*/openvdb::Vec3f(1.5, 2, 3), /*voxel size=*/0.5);
        
        openvdb::Index leafCount = sphereGrid->tree().leafCount();
        openvdb::points::PointDataTree::Ptr pointTree(
            new openvdb::points::PointDataTree(sphereGrid->tree(),0,openvdb::TopologyCopy())
        );

        pointTree->voxelizeActiveTiles();
        using PositionAttribute = openvdb::points::TypedAttributeArray<openvdb::Vec3f,
            openvdb::points::FixedPointCodec<false>>;
        
        openvdb::NamePair positionType = PositionAttribute::attributeType();
        openvdb::points::AttributeSet::Descriptor::Ptr descriptor(
            openvdb::points::AttributeSet::Descriptor::create(positionType));      

        openvdb::Index pointsPerVoxel = 8;
        openvdb::Index voxelsPerLeaf = openvdb::points::PointDataGrid::TreeType::LeafNodeType::SIZE;
        openvdb::Index pointsPerLeaf = pointsPerVoxel * voxelsPerLeaf;            

        // Iterate over the leaf nodes in the point tree.
        for (auto leafIter = pointTree->beginLeaf(); leafIter; ++leafIter) {
            // Initialize the attributes using the descriptor and point count.
            leafIter->initializeAttributes(descriptor, pointsPerLeaf);
            // Initialize the voxel offsets
            // explicitely set the offset here and uniformly set to 8
            // but more generally different voxel might have different number of points, and thus different offset shift
            openvdb::Index offset(0);
            for (openvdb::Index index = 0; index < voxelsPerLeaf; ++index) {
                offset += pointsPerVoxel;
                leafIter->setOffsetOn(index, offset);
            }
        }

        // Create the points grid.
        openvdb::points::PointDataGrid::Ptr points =
            openvdb::points::PointDataGrid::create(pointTree);
        points->setName("Points");
        points->setTransform(sphereGrid->transform().copy());


        // Randomize the point positions.
        std::mt19937 generator(/*seed=*/0);
        std::uniform_real_distribution<> distribution(-0.5, 0.5);        
        // Iterate over the leaf nodes in the point tree.
        for (auto leafIter = points->tree().beginLeaf(); leafIter; ++leafIter) {
            // Create an AttributeWriteHandle for position.
            // Note that the handle only requires the value type, not the codec.
            openvdb::points::AttributeArray& array = leafIter->attributeArray("P");
            openvdb::points::AttributeWriteHandle<openvdb::Vec3f> handle(array);
            // Iterate over the point indices in the leaf.
            for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
                // Compute a new random position (in the range -0.5 => 0.5).
                openvdb::Vec3f positionVoxelSpace(distribution(generator));
                // Set the position of this point.
                // As point positions are stored relative to the voxel center, it is
                // not necessary to convert these voxel space values into
                // world-space during this process.
                handle.set(*indexIter, positionVoxelSpace);
            }
        }

        // Verify the point count.
        openvdb::Index count = openvdb::points::pointCount(points->tree());
        std::cout << "LeafCount=" << leafCount << std::endl;
        std::cout << "PointCount=" << count << std::endl;

    }
};

ZENDEFNODE(VDBRandomPointGeneration,{
    {},
    {},
    {},
    {"TEST_OPENVDB"},
});

struct TestDilation : zeno::INode {
    void add_colored_aabb(const openvdb::Vec3f& wmin,
        const openvdb::Vec3f& wmax,
        const zeno::vec3f& clr,
        int id,
        zeno::PrimitiveObject& prim) const {
            int voffset = id * 8;
            int toffset = id * 12;
            auto& lines = prim.lines.values;
            auto& verts = prim.verts.values;
            auto& clrs = prim.attr<zeno::vec3f>("clr");

            // std::cout << "add : " << wmin << "\t" << wmax << std::endl;

            lines[toffset++] = zeno::vec2i{0, 4} + voffset;
            lines[toffset++] = zeno::vec2i{1, 5} + voffset;
            lines[toffset++] = zeno::vec2i{2, 6} + voffset;
            lines[toffset++] = zeno::vec2i{3, 7} + voffset;
            lines[toffset++] = zeno::vec2i{0, 2} + voffset;
            lines[toffset++] = zeno::vec2i{1, 3} + voffset;

            lines[toffset++] = zeno::vec2i{4, 6} + voffset;
            lines[toffset++] = zeno::vec2i{5, 7} + voffset;
            lines[toffset++] = zeno::vec2i{0, 1} + voffset;
            lines[toffset++] = zeno::vec2i{2, 3} + voffset;
            lines[toffset++] = zeno::vec2i{4, 5} + voffset;
            lines[toffset++] = zeno::vec2i{6, 7} + voffset;   

            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmin.y(),wmax.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmin.x(),wmax.y(),wmax.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmin.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmin.y(),wmax.z());
            clrs[voffset] = clr;
            verts[voffset++] = zeno::vec3f(wmax.x(),wmax.y(),wmax.z());            
    }

    virtual void apply() override {
        auto points = get_input<zeno::ListObject>("points")->getLiterial<zeno::vec3f>();
        std::vector<openvdb::Vec3R> positions(points.size());

        for(size_t i = 0;i < points.size();++i)
            positions[i] = openvdb::Vec3R{points[i][0],points[i][1],points[i][2]};

        openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper(positions);
        openvdb::Index pointsPerVoxel = 8;
        openvdb::Index voxelsPerLeaf = openvdb::points::PointDataGrid::TreeType::LeafNodeType::SIZE;
        openvdb::Index pointsPerLeaf = pointsPerVoxel * voxelsPerLeaf;
    
        float voxelSize = openvdb::points::computeVoxelSize(positionsWrapper,pointsPerVoxel);

        // std::cout << "voxelSize = " << voxelSize << std::endl;

        openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(1);

        openvdb::points::PointDataGrid::Ptr grid = openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
            openvdb::points::PointDataGrid>(positions,*transform);

        grid->setName("Points");

        int iter = get_param<int>("dilate");
        openvdb::tools::dilateActiveValues(grid->tree(),iter,openvdb::tools::NearestNeighbors::NN_FACE_EDGE_VERTEX, openvdb::tools::TilePolicy::EXPAND_TILES);        

        int nm_leaf_nodes = grid->treePtr()->leafCount();

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        auto& verts = prim->verts;
        auto& segs = prim->lines;

        int nm_active_voxel = grid->tree().activeLeafVoxelCount();

        std::cout << "nm_leaf_nodes : " << nm_leaf_nodes << std::endl;
        std::cout << "nm_active_voxels : " << nm_active_voxel << std::endl;

        verts.resize((nm_leaf_nodes + positions.size()) * 8 + nm_active_voxel * 8);
        segs.resize((nm_leaf_nodes + positions.size()) * 12 + nm_active_voxel * 12);
        auto& clrs = prim->add_attr<zeno::vec3f>("clr");

        // auto point_width = get_input<zeno::NumericObject>("point_width")->get<float>();
        auto point_width = get_param<float>("radius");
        // auto delta = openvdb::Vec3f(point_width);

        int idx = 0;

        std::vector<zeno::vec3f> clrs_graph{{1.0,0.0,0.0},
                    {0.0,1.0,0.0},
                    {0.0,0.0,1.0},
                    {1.0,1.0,0.0},
                    {1.0,0.0,1.0},
                    {0.0,1.0,1.0}};

        int leaf_idx = 0;

        auto voxel_delta = grid->transform().indexToWorld(openvdb::Vec3f{0.5});
        auto point_delta = grid->transform().indexToWorld(openvdb::Vec3f{point_width});


        int view_active_voxels_only = get_param<int>("view_av_only");

        for(auto leafIter = grid->tree().cbeginLeaf();leafIter;++leafIter,++leaf_idx) {
            auto aabb = leafIter->getNodeBoundingBox();
            openvdb::Vec3f wmin = grid->transform().indexToWorld(aabb.min().asVec3s());
            openvdb::Vec3f wmax = grid->transform().indexToWorld(aabb.max().asVec3s());

            // std::cout < "AABB : " << aabb.min() << "\t" << aabb.max() << std::endl;

            add_colored_aabb(wmin,wmax,clrs_graph[leaf_idx % clrs_graph.size()],idx++,*prim);

            // for(auto indexIter = leafIter->beginIndexOn();indexIter;++indexIter) {
            //     // openvdb::Vec3f voxelPosition = position；
            // }

            openvdb::Coord ijk{};
            auto& i = ijk.x();
            auto& j = ijk.y();
            auto& k = ijk.z();


            // std::cout << "check here" << std::endl;
            for(int i = 0;i < leafIter->size();++i){
                if(leafIter->isValueOn(i)){
                    auto is_voxel_pos = leafIter->offsetToGlobalCoord(i);
                    auto ws_voxel_pos = grid->transform().indexToWorld(leafIter->offsetToGlobalCoord(i));
                    auto wsmax = ws_voxel_pos + voxel_delta;
                    auto wsmin = ws_voxel_pos - voxel_delta;

                    add_colored_aabb(wsmin,wsmax,zeno::vec3f(1.0),idx++,*prim);
                    if(view_active_voxels_only)
                        continue;

                    if(!leafIter->hasAttribute("P"))
                        continue;

                    // std::cout << "use leaf handles" << std::endl;
                    const openvdb::points::AttributeArray& array = leafIter->constAttributeArray("P");
                    const openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(array);
                    // std::cout << "use leaf handles" << std::endl;

                    for(auto indexIter = leafIter->beginIndexVoxel(leafIter->offsetToLocalCoord(i));indexIter;++indexIter) {
                        openvdb::Vec3f point_position = positionHandle.get(*indexIter);
                        openvdb::Vec3f worldPosition = grid->transform().indexToWorld(point_position + is_voxel_pos);

                        auto pbmin = worldPosition - point_delta;
                        auto pbmax = worldPosition + point_delta;

                        // std::cout << "add point : " << voxelPosition + xyz << std::endl;
                        add_colored_aabb(pbmin,pbmax,clrs_graph[leaf_idx % clrs_graph.size()],idx++,*prim);                        
                    }
                }
            }
        }


        set_output("prim",prim);
    }
};


ZENDEFNODE(TestDilation,{
    {"points"},
    {"prim"},
    {{"float","radius","0.1"},{"int","dilate","1"},{"int","view_av_only","1"}},
    {"TEST_OPENVDB"},
});

};