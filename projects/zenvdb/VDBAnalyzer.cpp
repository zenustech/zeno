#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <vector>
#include <zeno/VDBGrid.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/VolumeToSpheres.h>

namespace zeno {

struct ScalarFieldAnalyzer : zeno::INode {
    virtual void apply() override {
        auto inSDF = get_input("InVDB")->as<VDBFloatGrid>();
        auto grid = inSDF->m_grid;
        auto OpType = std::get<std::string>(get_param("Operator"));
        if (OpType == "Gradient") {
            auto result = std::make_shared<VDBFloat3Grid>(openvdb::tools::gradient(*grid));
            set_output("OutVDB", std::move(result));
        }
        else if (OpType == "Curvature") {
            auto result = std::make_shared<VDBFloatGrid>(openvdb::tools::meanCurvature(*grid));
            set_output("OutVDB", std::move(result));
        }
        else if (OpType == "Laplacian") {
            auto result = std::make_shared<VDBFloatGrid>(openvdb::tools::laplacian(*grid));
            set_output("OutVDB", std::move(result));
        }
        else if (OpType == "ClosestPoint") {
            auto x = grid->tree();
            openvdb::Vec3STree::Ptr resultTree(new openvdb::Vec3STree(grid->tree(), openvdb::Vec3f(0.0), openvdb::TopologyCopy()));
            auto closestSurface = openvdb::tools::ClosestSurfacePoint<openvdb::FloatGrid>::create(*grid);
            std::vector<openvdb::Vec3R> activeVoxels(1);
            std::vector<float> distances(1);
            const openvdb::math::Transform& transform = grid->transform();
            for (openvdb::Vec3SGrid::ValueAllIter iter = resultTree->beginValueAll(); iter; ++iter)
            {
                activeVoxels[0] = transform.indexToWorld(iter.getCoord().asVec3d() + 0.5f);
                (*closestSurface).searchAndReplace(activeVoxels, distances);
                iter.setValue(activeVoxels[0]);
                iter.setActiveState(true);
            }
            openvdb::Vec3fGrid::Ptr resultGrid(new openvdb::Vec3fGrid(resultTree));
            resultGrid->setTransform(transform.copy());
            auto result = std::make_shared<VDBFloat3Grid>(std::move(resultGrid));
            set_output("OutVDB", std::move(result));
        }
        else {
            throw zeno::Exception("wrong parameter for ScalarFieldAnalyzer Operator: " + OpType);
        }  
    }
};

ZENO_DEFNODE(ScalarFieldAnalyzer)(
    { /* inputs: */ {
        "InVDB",   
    }, /* outputs: */ {
        "OutVDB"
    }, /* params: */ {
        {"enum Gradient Curvature Laplacian ClosestPoint", "Operator", "Gradient"},
    }, /* category: */ {
        "openvdb",
    } });


struct VectorFieldAnalyzer : zeno::INode {
    virtual void apply() override {
        auto inSDF = get_input("InVDB")->as<VDBFloat3Grid>();
        auto grid = inSDF->m_grid;
        auto OpType = std::get<std::string>(get_param("Operator"));
        if (OpType == "Divergence") {
            auto result = std::make_shared<VDBFloatGrid>(openvdb::tools::divergence(*grid));
            set_output("OutVDB", std::move(result));
        }
        else if (OpType == "Curl") {
            auto result = std::make_shared<VDBFloat3Grid>(openvdb::tools::curl(*grid));
            set_output("OutVDB", std::move(result));
        }
        else if (OpType == "Magnitude") {
            auto result = std::make_shared<VDBFloatGrid>(openvdb::tools::magnitude(*grid));
            set_output("OutVDB", std::move(result));
        }
        else if (OpType == "Normalize") {
            auto result = std::make_shared<VDBFloat3Grid>(openvdb::tools::normalize(*grid));
            set_output("OutVDB", std::move(result));
        }
        else {
            throw zeno::Exception("wrong parameter for VectorFieldAnalyzer Operator: " + OpType);
        }
    }
};

ZENO_DEFNODE(VectorFieldAnalyzer)(
    { /* inputs: */ {
        "InVDB",
    }, /* outputs: */ {
        "OutVDB"
    }, /* params: */ {
        {"enum Divergence Curl Magnitude Normalize", "Operator", "Divergence"},
    }, /* category: */ {
        "openvdb",
    } });

}
