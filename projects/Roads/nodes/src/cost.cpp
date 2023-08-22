#include "boost/graph/dijkstra_shortest_paths.hpp"
#include "boost/graph/astar_search.hpp"
#include "roads/roads.h"
#include "zeno/PrimitiveObject.h"
#include "zeno/types/UserData.h"
#include "zeno/types/CurveObject.h"
#include "zeno/utils/PropertyVisitor.h"
#include "zeno/utils/logger.h"
#include "zeno/zeno.h"
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stack>

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

namespace zeno::reflect {

    struct ConnectiveTypeInput : std::string {
        using std::string::string;
    };

    struct PathAlgorithmTypeInput : std::string {
        using std::string::string;
    };

    template<>
    struct ValueTypeToString<ConnectiveTypeInput> {
        inline static std::string TypeName = "enum 4 8 16 40";
    };

    template<>
    struct ValueTypeToString<PathAlgorithmTypeInput> {
        inline static std::string TypeName = "enum Dijkstra A*";
    };

}

namespace {
    using namespace zeno;
    using namespace roads;

    struct ZENO_CRTP(PrimCalcSlope, zeno::reflect::IParameterAutoNode) {
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

    template <typename Graph, typename CostType>
    class PathDistanceHeuristic : public boost::astar_heuristic<Graph, CostType> {
        typedef typename boost::graph_traits< Graph >::vertex_descriptor Vertex;

    public:
        PathDistanceHeuristic(Vertex Goal) : m_Goal(Goal) {}

        CostType operator()(Vertex u) {
            return std::abs<CostType>(m_Goal - u);
        }

    private:
        Vertex m_Goal;
    };

    struct FoundGoal {};

    template < class Vertex >
    class PathDistanceVisitor : public boost::default_astar_visitor
    {
    public:
        PathDistanceVisitor(Vertex goal) : m_goal(goal) {}
        template < class Graph > void examine_vertex(Vertex u, Graph& g)
        {
            if (u == m_goal)
                throw FoundGoal();
        }

    private:
        Vertex m_goal;
    };

    struct ZENO_CRTP(CalcPathCost_Simple, zeno::reflect::IParameterAutoNode) {
    //struct CalcPathCost_Simple : public zeno::reflect::IParameterAutoNode<CalcPathCost_Simple> {
        ZENO_GENERATE_NODE_BODY(CalcPathCost_Simple);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        ZENO_DECLARE_INPUT_FIELD(Primitive, "Prim");
        ZENO_DECLARE_OUTPUT_FIELD(Primitive, "Prim");

        std::string SizeXChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeXChannel, "Nx Channel (UserData)", false, "", "nx");

        std::string SizeYChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeYChannel, "Ny Channel (UserData)", false, "", "ny");

        std::string PositionChannel;
        ZENO_DECLARE_INPUT_FIELD(PositionChannel, "Position Channel (Vertex Attr)", false, "", "pos");

        std::string GradientChannel;
        ZENO_DECLARE_INPUT_FIELD(GradientChannel, "Gradient Channel (Vertex Attr)", false, "", "gradient");

        zeno::reflect::ConnectiveTypeInput PathConnective;
        ZENO_DECLARE_INPUT_FIELD(PathConnective, "Path Connective", false, "", "16");

        zeno::reflect::PathAlgorithmTypeInput Algorithm;
        ZENO_DECLARE_INPUT_FIELD(Algorithm, "Path Finding Algorithm", false, "", "Dijkstra");

        std::shared_ptr<zeno::CurveObject> GradientCurve = nullptr;
        ZENO_DECLARE_INPUT_FIELD(GradientCurve, "Gradient Cost Control", true);

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

            ConnectiveType Connective = ConnectiveType::SIXTEEN;
            if (AutoParameter->PathConnective == "4") {
                Connective = ConnectiveType::FOUR;
            } else if (AutoParameter->PathConnective == "8") {
                Connective = ConnectiveType::EIGHT;
            } else if (AutoParameter->PathConnective == "16") {
                Connective = ConnectiveType::SIXTEEN;
            } else if (AutoParameter->PathConnective == "40") {
                Connective = ConnectiveType::FOURTY;
            }

            auto GradientMapFunc = [Curve = AutoParameter->GradientCurve] (double In) -> double {
                if (Curve) {
                    return Curve->eval(float(In));
                }
                return In;
            };

            DynamicGrid<CostPoint> CostGrid(AutoParameter->Nx, AutoParameter->Ny);
            CostGrid.insert(CostGrid.begin(), AutoParameter->GradientList.begin(), AutoParameter->GradientList.begin() + (AutoParameter->Nx * AutoParameter->Ny));
            auto Graph = CreateWeightGraphFromCostGrid(CostGrid, Connective, GradientMapFunc);
            zeno::log_info("cccc: {}", boost::num_vertices(Graph));

            using VertexDescriptor = boost::graph_traits<WeightedGridUndirectedGraph>::vertex_descriptor;

            std::vector<VertexDescriptor> p(boost::num_vertices(Graph));
            std::vector<double> d(boost::num_vertices(Graph));
            VertexDescriptor Start { 1 };
            VertexDescriptor Goal { 933333 };

            if (AutoParameter->Algorithm == "Dijkstra") {
                boost::dijkstra_shortest_paths(Graph, Start, boost::predecessor_map(&p[0]).distance_map(&d[0]));
            } else if (AutoParameter->Algorithm == "A*") {
                try {
                    boost::astar_search_tree(Graph, Start, PathDistanceHeuristic<WeightedGridUndirectedGraph, double>(Goal), boost::predecessor_map(&p[0]).distance_map(&d[0]).visitor(PathDistanceVisitor(Goal)));
                } catch (FoundGoal) {
                }
            }

            std::vector<boost::graph_traits<WeightedGridUndirectedGraph>::vertex_descriptor > path;
            boost::graph_traits<WeightedGridUndirectedGraph>::vertex_descriptor current = Goal;

            while(current!=Start)
            {
                path.push_back(current);
                current = p[current];
            }
            path.push_back(Start);

            if (AutoParameter->bRemoveTriangles) {
                AutoParameter->Primitive->tris.clear();
            }
            AutoParameter->Primitive->lines.resize(path.size() - 1);
            for (size_t i = 0; i < path.size() - 1; ++i) {
                AutoParameter->Primitive->lines[i] = zeno::vec2i(int(path[i]), int(path[i+1]));
            }
        }
    };

    struct ZENO_CRTP(HeightFieldFlowPath_Simple, zeno::reflect::IParameterAutoNode) {
        ZENO_GENERATE_NODE_BODY(HeightFieldFlowPath_Simple);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        ZENO_DECLARE_INPUT_FIELD(Primitive, "Prim");
        ZENO_DECLARE_OUTPUT_FIELD(Primitive, "Prim");

        bool bShouldSmooth;
        ZENO_DECLARE_INPUT_FIELD(bShouldSmooth, "Enable Smooth", false, "", "false");

        float DeltaAltitudeThreshold;
        ZENO_DECLARE_INPUT_FIELD(DeltaAltitudeThreshold, "Delta Altitude Threshold", false, "", "1e-06");

        float HeuristicRatio;
        ZENO_DECLARE_INPUT_FIELD(HeuristicRatio, "Heuristic Ratio (0 - 1)", false, "", "0.3");

        std::string SizeXChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeXChannel, "Nx Channel (UserData)", false, "", "nx");

        std::string SizeYChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeYChannel, "Ny Channel (UserData)", false, "", "ny");

        int Nx = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Primitive, Nx, SizeXChannel, false);

        int Ny = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Primitive, Ny, SizeYChannel, false);

        std::string RiverChannel;
        ZENO_DECLARE_INPUT_FIELD(RiverChannel, "Output River Channel (Vertex Attr)", false, "", "is_river");

        std::string LakeChannel;
        ZENO_DECLARE_INPUT_FIELD(LakeChannel, "Output Lake Channel (Vertex Attr)", false, "", "is_lake");

        std::string HeightChannel;
        ZENO_DECLARE_INPUT_FIELD(HeightChannel, "Height Channel (Vertex Attr)", false, "", "height");

        std::string WaterChannel;
        ZENO_DECLARE_INPUT_FIELD(WaterChannel, "Water Channel (Vertex Attr)", false, "", "water");

        zeno::AttrVector<float> Heightmap{};
        ZENO_BINDING_PRIMITIVE_ATTRIBUTE(Primitive, Heightmap, HeightChannel, zeno::reflect::EZenoPrimitiveAttr::VERT);

        zeno::AttrVector<float> WaterMask{};
        ZENO_BINDING_PRIMITIVE_ATTRIBUTE(Primitive, WaterMask, WaterChannel, zeno::reflect::EZenoPrimitiveAttr::VERT);

        void apply() override {
            auto& Prim = AutoParameter->Primitive;
            auto& HeightField = AutoParameter->Heightmap;
            auto& Water = AutoParameter->WaterMask;
            const auto SizeX = AutoParameter->Nx;
            const auto SizeY = AutoParameter->Ny;
            const size_t NumVert = SizeX * SizeY;

            auto& River = Prim->add_attr<float>(AutoParameter->RiverChannel);

            static const std::array<IntPoint2D, 8> SDirection {
                IntPoint2D { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 },
                { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, 1 }
            };

            std::stack<size_t> Stack;

            const float MaxHeight = *std::max_element(std::begin(HeightField), std::end(HeightField));
            const float MinHeight = *std::min_element(std::begin(HeightField), std::end(HeightField));
            const float RiverHeightMax = (MaxHeight - MinHeight) * AutoParameter->HeuristicRatio;

            std::set<size_t> Visited;
            for (size_t i = 0; i < NumVert; ++i) {
                if (Water[i] < 1e-7) {
                    continue;
                }

                Stack.push(i);
                while (!Stack.empty()) {
                    long idx = long(Stack.top());
                    Visited.insert(idx);
                    Stack.pop();
                    if (HeightField[idx] > RiverHeightMax) continue;
                    River[idx] = 1;
                    long y = idx / SizeX;
                    long x = idx % SizeX;
                    for (const IntPoint2D& Dir : SDirection) {
                        long ix = x + Dir[0];
                        long iy = y + Dir[1];
                        if (ix > 0 && iy > 0 && ix < SizeX && iy < SizeX) {
                            size_t nidx = iy * SizeX + ix;
                            if (Visited.find(nidx) == std::end(Visited) && std::abs(HeightField[nidx] - HeightField[idx]) < AutoParameter->DeltaAltitudeThreshold) {
                                Stack.push(nidx);
                            }
                        }
                    }
                }
            }

        }
    };
}// namespace
