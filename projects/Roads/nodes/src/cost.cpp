#include "roads/roads.h"
#include <limits>
#include "zeno/PrimitiveObject.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/PropertyVisitor.h"
#include "zeno/utils/logger.h"
#include "zeno/zeno.h"
#include "boost/graph/dijkstra_shortest_paths.hpp"
#include <iostream>

template<typename... Args>
inline void RoadsAssert(const bool Expr, const std::string &InMsg = "[Roads] Assert Failed", Args... args) {
    if (!Expr) {
        zeno::log_error(InMsg, args...);
        std::quick_exit(-1);
    }
}

roads::DynamicGrid<roads::AdvancePoint> BuildGridFromPrimitive(const zeno::AttrVector<zeno::vec3f> &PositionSource, const zeno::AttrVector<float> &CurvatureSource, int32_t Nx, int32_t Ny) {
    RoadsAssert(Nx * Ny <= PositionSource.size());
    RoadsAssert(Nx * Ny <= CurvatureSource.size());

    roads::DynamicGrid<roads::AdvancePoint> Grid(Nx, Ny);
    for (size_t i = 0; i < Nx * Ny; ++i) {
        Grid[i].Position = PositionSource[i];
        Grid[i].Gradient = CurvatureSource[i];
    }

    return Grid;
}

namespace {
    using namespace zeno;
    using namespace roads;

    struct PrimCalcSlope : public zeno::reflect::IParameterAutoNode<PrimCalcSlope> {
        ZENO_GENERATE_NODE_BODY(PrimCalcSlope);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        ZENO_DECLARE_INPUT_FIELD(Primitive, "Prim");
        ZENO_DECLARE_OUTPUT_FIELD(Primitive, "Prim");

        std::string SizeXChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeXChannel, "UserData_NxChannel", false, "", "nx");

        std::string SizeYChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeYChannel, "UserData_NyChannel", false, "", "ny");

        int Nx = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Primitive, Nx, SizeXChannel, false);

        int Ny = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Primitive, Ny, SizeYChannel, false);

        std::string HeightChannel;
        ZENO_DECLARE_INPUT_FIELD(HeightChannel, "Vert_PositionChannel", false, "", "pos");

        std::string OutputChannel;
        ZENO_DECLARE_INPUT_FIELD(OutputChannel, "Vert_OutputChannel", false, "", "gradient");

        zeno::AttrVector<float> HeightList{};
        ZENO_BINDING_PRIMITIVE_ATTRIBUTE(Primitive, HeightList, HeightChannel, zeno::reflect::EZenoPrimitiveAttr::VERT);

        void apply() override {
            RoadsAssert(AutoParameter->Nx * AutoParameter->Ny <= AutoParameter->HeightList.size(), "Bad size in userdata! Check your nx ny.");

            DynamicGrid<HeightPoint> HeightField(AutoParameter->Nx, AutoParameter->Ny);
            for (size_t i = 0; i < HeightField.size(); ++i) {
                HeightField[i] = AutoParameter->HeightList[i];
            }

            DynamicGrid<SlopePoint> SlopeField = CalculateSlope(HeightField);
            if (!AutoParameter->Primitive->verts.has_attr(AutoParameter->OutputChannel)) {
                AutoParameter->Primitive->verts.add_attr<float>(AutoParameter->OutputChannel);
            }
            std::vector<float>& SlopeAttr = AutoParameter->Primitive->verts.attr<float>(AutoParameter->OutputChannel);
            SlopeAttr.insert(SlopeAttr.begin(), SlopeField.begin(), SlopeField.end());
        }
    };

    struct ZENO_CRTP(CalcPathCost_Simple, zeno::reflect::IParameterAutoNode) {
    //struct CalcPathCost_Simple : public zeno::reflect::IParameterAutoNode<CalcPathCost_Simple> {
        ZENO_GENERATE_NODE_BODY(CalcPathCost_Simple);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        ZENO_DECLARE_INPUT_FIELD(Primitive, "Prim");
        ZENO_DECLARE_OUTPUT_FIELD(Primitive, "Prim");

        std::string OutputChannel;
        ZENO_DECLARE_INPUT_FIELD(OutputChannel, "Output Channel (Vertex Attr)", false, "", "path_cost");

        std::string SizeXChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeXChannel, "Nx Channel (UserData)", false, "", "nx");

        std::string SizeYChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeYChannel, "Ny Channel (UserData)", false, "", "ny");

        std::string PositionChannel;
        ZENO_DECLARE_INPUT_FIELD(PositionChannel, "Position Channel (Vertex Attr)", false, "", "pos");

        std::string GradientChannel;
        ZENO_DECLARE_INPUT_FIELD(GradientChannel, "Gradient Channel (Vertex Attr)", false, "", "gradient");

        bool bRemoveTriangles;
        ZENO_DECLARE_INPUT_FIELD(bRemoveTriangles, "Remove Triangles", false, "", "true");

        int Nx = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Primitive, Nx, SizeXChannel, false);

        int Ny = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Primitive, Ny, SizeYChannel, false);

        zeno::AttrVector<vec3f> PositionList{};
        ZENO_BINDING_PRIMITIVE_ATTRIBUTE(Primitive, PositionList, PositionChannel, zeno::reflect::EZenoPrimitiveAttr::VERT);

        zeno::AttrVector<float> GradientList{};
        ZENO_BINDING_PRIMITIVE_ATTRIBUTE(Primitive, GradientList, GradientChannel, zeno::reflect::EZenoPrimitiveAttr::VERT);

        void apply() override {
            //auto Grid = BuildGridFromPrimitive(AutoParameter->PositionList, AutoParameter->GradientList, AutoParameter->Nx, AutoParameter->Ny);
            // TODO [darc] : Change cost function, now just simply use gradient value :
            AutoParameter->Nx;
            RoadsAssert(AutoParameter->Nx * AutoParameter->Ny <= AutoParameter->GradientList.size(), "Bad nx ny.");

            DynamicGrid<CostPoint> CostGrid(AutoParameter->Nx, AutoParameter->Ny);
            CostGrid.insert(CostGrid.begin(), AutoParameter->GradientList.begin(), AutoParameter->GradientList.begin() + (AutoParameter->Nx * AutoParameter->Ny));
            auto Graph = CreateWeightGraphFromCostGrid(CostGrid, ConnectiveType::FOUR);
            zeno::log_info("cccc: {}", boost::num_vertices(Graph));

            using VertexDescriptor = boost::graph_traits<WeightedGridUndirectedGraph>::vertex_descriptor;

            std::vector<VertexDescriptor> p(boost::num_vertices(Graph));
            std::vector<double> d(boost::num_vertices(Graph));
            VertexDescriptor Start { 1 };
            VertexDescriptor Goal { 999999 };

            boost::dijkstra_shortest_paths(Graph, Start, boost::predecessor_map(&p[0]).distance_map(&d[0]));

            std::vector<boost::graph_traits<WeightedGridUndirectedGraph>::vertex_descriptor > path;
            boost::graph_traits<WeightedGridUndirectedGraph>::vertex_descriptor current = Goal;

            while(current!=Start)
            {
                path.push_back(current);
                current = p[current];
            }
            path.push_back(Start);

//            AutoParameter->Primitive = std::make_shared<zeno::PrimitiveObject>();
            if (AutoParameter->bRemoveTriangles) {
                AutoParameter->Primitive->tris.clear();
            }
            AutoParameter->Primitive->lines.resize(path.size() - 1);
            for (size_t i = 0; i < path.size() - 1; ++i) {
                AutoParameter->Primitive->lines[i] = zeno::vec2i(path[i], path[i+1]);
            }
        }
    };
}// namespace
