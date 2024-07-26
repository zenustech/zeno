#include "boost/geometry.hpp"
#include "boost/geometry/index/rtree.hpp"
#include "boost/graph/astar_search.hpp"
#include "roads/roads.h"
#include "zeno/PrimitiveObject.h"
#include "zeno/types/CurveObject.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/PropertyVisitor.h"
#include "zeno/utils/logger.h"
#include "zeno/zeno.h"
#include <Eigen/Core>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stack>

#include "roads/thirdparty/tinysplinecxx.h"

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

}// namespace zeno::reflect

namespace boost::geometry {
    template<>
    struct point_type<zeno::vec3f> {
        using type = zeno::vec3f;
    };
}// namespace boost::geometry

namespace boost::geometry::index {
    template<typename Ptr>
    struct indexable<Ptr *> {

        typedef Ptr *V;
        typedef Ptr const &result_type;

        result_type operator()(V const &v) const { return *v; }
    };

    template<typename Box>
    struct indexable<std::shared_ptr<Box>> {
        typedef std::shared_ptr<Box> V;

        typedef Box const &result_type;
        result_type operator()(V const &v) const { return *v; }
    };
}// namespace boost::geometry::index

namespace zeno {
    struct RoadBSplineObject : public IObject {

        explicit RoadBSplineObject(const tinyspline::BSpline &Lhs) : Spline(Lhs) {}
        explicit RoadBSplineObject(tinyspline::BSpline &&LhsToMove) : Spline(std::forward<tinyspline::BSpline &&>(LhsToMove)) {}

        tinyspline::BSpline Spline;
    };
}// namespace zeno

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
            std::vector<float> &SlopeAttr = AutoParameter->Primitive->verts.attr<float>(AutoParameter->OutputChannel);
            SlopeAttr.insert(SlopeAttr.begin(), SlopeField.begin(), SlopeField.end());
        }
    };

    template<typename Graph, typename CostType>
    class PathDistanceHeuristic : public boost::astar_heuristic<Graph, CostType> {
        typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;

    public:
        PathDistanceHeuristic(Vertex Goal) : m_Goal(Goal) {}

        CostType operator()(Vertex u) {
            return std::abs<CostType>(m_Goal - u);
        }

    private:
        Vertex m_Goal;
    };

    struct FoundGoal {};

    template<class Vertex>
    class PathDistanceVisitor : public boost::default_astar_visitor {
    public:
        PathDistanceVisitor(Vertex goal) : m_goal(goal) {}
        template<class Graph>
        void examine_vertex(Vertex u, Graph &g) {
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

        //std::string CurvatureChannel;
        //ZENO_DECLARE_INPUT_FIELD(CurvatureChannel, "Curvature Channel (Vertex Attr)", false, "", "curvature");

        int ConnectiveMask;
        ZENO_DECLARE_INPUT_FIELD(ConnectiveMask, "Connective Mask", false, "", "4");

        int AngleMask;
        ZENO_DECLARE_INPUT_FIELD(AngleMask, "Angle Mask", false, "", "4");

        float WeightHeuristic;
        ZENO_DECLARE_INPUT_FIELD(WeightHeuristic, "Weight of Heuristic Function", false, "", "1.0");

        float CurvatureThreshold;
        ZENO_DECLARE_INPUT_FIELD(CurvatureThreshold, "Curvature Threshold", false, "", "0.2");

        zeno::vec2f Start;
        ZENO_DECLARE_INPUT_FIELD(Start, "Start Point");

        zeno::vec2f Goal;
        ZENO_DECLARE_INPUT_FIELD(Goal, "Goal Point");

        zeno::CurvesData HeightCurve;
        ZENO_DECLARE_INPUT_FIELD(HeightCurve, "Height Cost Control", true);

        zeno::CurvesData GradientCurve;
        ZENO_DECLARE_INPUT_FIELD(GradientCurve, "Gradient Cost Control", true);

        zeno::CurvesData CurvatureCurve;
        ZENO_DECLARE_INPUT_FIELD(CurvatureCurve, "Curvature Cost Control", true);

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

        //zeno::AttrVector<float> CurvatureList{};
        //ZENO_BINDING_PRIMITIVE_ATTRIBUTE(Primitive, CurvatureList, CurvatureChannel, zeno::reflect::EZenoPrimitiveAttr::VERT);

        //std::unordered_map<size_t, float> CurvatureCache;

        void apply() override {
            RoadsAssert(AutoParameter->Nx * AutoParameter->Ny <= AutoParameter->GradientList.size(), "Bad nx ny.");

            CostPoint GoalPoint{static_cast<size_t>(AutoParameter->Goal[0]), static_cast<size_t>(AutoParameter->Goal[1]), 0};
            CostPoint StartPoint{static_cast<size_t>(AutoParameter->Start[0]), static_cast<size_t>(AutoParameter->Start[1]), 0};

            DefaultedHashMap<CostPoint, CostPoint> Predecessor;
            DefaultedHashMap<CostPoint, float> CostMap;

            size_t Nx = AutoParameter->Nx, Ny = AutoParameter->Ny;

            auto MapFuncGen = [](const zeno::CurvesData &Curve, float Threshold) -> std::function<float(float)> {
                return [Curve, Threshold](float In) -> float {
                    if (Threshold > 0 && In > Threshold) {
                        return 9e06f;
                    }
                    return Curve.eval(float(In));
                };
            };

            auto HeightCostFunc = MapFuncGen(AutoParameter->HeightCurve, -1.0f);
            auto GradientCostFunc = MapFuncGen(AutoParameter->GradientCurve, -1.0f);
            auto CurvatureCostFunc = MapFuncGen(AutoParameter->CurvatureCurve, AutoParameter->CurvatureThreshold);

            DynamicGrid<CostPoint> CostGrid(AutoParameter->Nx, AutoParameter->Ny);
            CostGrid.resize(AutoParameter->Nx * AutoParameter->Ny);
            //#pragma omp parallel for
            for (size_t i = 0; i < AutoParameter->Nx * AutoParameter->Ny; ++i) {
                size_t x = i % AutoParameter->Nx;
                size_t y = i / AutoParameter->Ny;
                CostGrid[i] = (CostPoint{x, y, 0, AutoParameter->PositionList[i].at(1), AutoParameter->GradientList[i]});
            }

            auto CalcCurvature = [&CostGrid, Nx](const CostPoint &A, const CostPoint &B, const CostPoint &C) -> float {
                float Height_A = float(CostGrid[A[0] + A[1] * Nx].Height);
                float Height_B = float(CostGrid[B[0] + B[1] * Nx].Height);
                float Height_C = float(CostGrid[C[0] + C[1] * Nx].Height);

                Eigen::Vector4f BA = {float(A[0] - B[0]), float(A[1] - B[1]), float(Height_A - Height_B), float(A[2] - B[2])};
                Eigen::Vector4f BC = {float(C[0] - B[0]), float(C[1] - B[1]), float(Height_C - Height_B), float(C[2] - B[2])};
                float Magnitude_BC = BC.norm();

                Eigen::Vector4f BA_Normalized = BA.normalized();
                Eigen::Vector4f BC_Normalized = BC.normalized();

                float Magnitude_Change = (BC_Normalized - BA_Normalized).norm();
                return Magnitude_Change / (Magnitude_BC * Magnitude_BC) * BC.z();
            };

            const size_t IndexMax = AutoParameter->Nx * AutoParameter->Ny - 1;
            std::function<float(const CostPoint &, const CostPoint &)> CostFunc = [&CalcCurvature, &HeightCostFunc, &GradientCostFunc, &CurvatureCostFunc, &CostGrid, &Predecessor, IndexMax, Nx, Ny](const CostPoint &A, const CostPoint &B) mutable -> float {
                size_t ia = A[0] + A[1] * Nx;
                size_t ib = B[0] + B[1] * Nx;

                //constexpr size_t Seed = 12306;
                //size_t Hash = (ia + 0x9e3779b9 + (Seed << 4) + (Seed >> 2)) ^ (ib * 0x9e3779b9 + (Seed << 2) + (Seed >> 4));

                // We assume that A already searched and have a predecessor
                const CostPoint &PrevPoint = Predecessor[A];

                // Calc curvature
                float Curvature = CalcCurvature(PrevPoint, A, B);

                float Cost = HeightCostFunc(float(std::abs(CostGrid[ia].Height - CostGrid[ib].Height))) + GradientCostFunc(float(std::abs(CostGrid[ia].Gradient - CostGrid[ib].Gradient))) + CurvatureCostFunc(float(std::abs(Curvature)));
                return Cost;
            };

            zeno::log_info("[Roads] Generating trajectory...");

            ROADS_TIMING_PRE_GENERATED;

            ROADS_TIMING_BLOCK("AStar Extended", roads::energy::RoadsShortestPath(StartPoint, GoalPoint, CostPoint{static_cast<size_t>(AutoParameter->Nx), static_cast<size_t>(AutoParameter->Ny)}, AutoParameter->ConnectiveMask, AutoParameter->AngleMask, AutoParameter->WeightHeuristic, Predecessor, CostMap, CostFunc));

            zeno::log_info("[Roads] Result Predecessor Size: {}; CostMap Size: {}", Predecessor.size(), CostMap.size());

            CostPoint Current = GoalPoint;

            ArrayList<size_t> Path;
            while (Current != StartPoint) {
                Path.push_back(Current[0] + Current[1] * AutoParameter->Nx);
                Current = Predecessor[Current];
            }
            Path.push_back(StartPoint[0] + StartPoint[1] * AutoParameter->Nx);

            if (AutoParameter->bRemoveTriangles) {
                AutoParameter->Primitive->tris.clear();
            }

            AutoParameter->Primitive->lines.resize(Path.size() - 1);
            for (size_t i = 0; i < Path.size() - 1; ++i) {
                AutoParameter->Primitive->lines[i] = zeno::vec2i(int(Path[i]), int(Path[i + 1]));
            }
        }
    };

    struct ZENO_CRTP(HeightFieldFlowPath_Simple, zeno::reflect::IParameterAutoNode) {
        ZENO_GENERATE_NODE_BODY(HeightFieldFlowPath_Simple);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        ZENO_DECLARE_INPUT_FIELD(Primitive, "Prim");
        ZENO_DECLARE_OUTPUT_FIELD(Primitive, "Prim");

        bool bShouldSmooth = false;
        ZENO_DECLARE_INPUT_FIELD(bShouldSmooth, "Enable Smooth", false, "", "false");

        float DeltaAltitudeThreshold = 1e-06;
        ZENO_DECLARE_INPUT_FIELD(DeltaAltitudeThreshold, "Delta Altitude Threshold", false, "", "1e-06");

        float HeuristicRatio = 0.3;
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
            auto &Prim = AutoParameter->Primitive;
            auto &HeightField = AutoParameter->Heightmap;
            auto &Water = AutoParameter->WaterMask;
            const auto SizeX = AutoParameter->Nx;
            const auto SizeY = AutoParameter->Ny;
            const size_t NumVert = SizeX * SizeY;

            auto &River = Prim->add_attr<float>(AutoParameter->RiverChannel);

            static const std::array<IntPoint2D, 8> SDirection{
                IntPoint2D{0, -1},
                {0, 1},
                {-1, 0},
                {1, 0},
                {-1, -1},
                {1, -1},
                {-1, 1},
                {1, 1}};

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
                    for (const IntPoint2D &Dir: SDirection) {
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

    struct ZENO_CRTP(PrimLineClothoidSmooth, zeno::reflect::IParameterAutoNode) {
        ZENO_GENERATE_NODE_BODY(PrimLineClothoidSmooth);

        std::shared_ptr<zeno::PrimitiveObject> Primitive;
        ZENO_DECLARE_INPUT_FIELD(Primitive, "Prim");
        ZENO_DECLARE_OUTPUT_FIELD(Primitive, "Prim");

        int SampleNum = 1;
        ZENO_DECLARE_INPUT_FIELD(SampleNum, "Sample Points", false, "", "1000");

        std::shared_ptr<zeno::RoadBSplineObject> Spline;
        ZENO_DECLARE_OUTPUT_FIELD(Spline, "Spline");

        void apply() override {
            auto &Prim = AutoParameter->Primitive;

            ArrayList<std::array<float, 3>> Vertices(Prim->verts.begin(), Prim->verts.end());
            ArrayList<std::array<int, 2>> Lines(Prim->lines.begin(), Prim->lines.end());

            ROADS_TIMING_PRE_GENERATED;
            ROADS_TIMING_BLOCK("Spline Creation", auto Spline = spline::GenerateBSplineFromSegment(Vertices, Lines));
            ROADS_TIMING_BLOCK("Resample segments", auto Result = spline::GenerateAndSamplePointsFromSegments(Spline, AutoParameter->SampleNum));

            AutoParameter->Spline = std::make_shared<zeno::RoadBSplineObject>(Spline);

            Prim = std::make_shared<zeno::PrimitiveObject>();

            Prim->verts.reserve(Result.size());
            for (const auto &line: Result) {
                Prim->verts.emplace_back(line.x(), line.y(), line.z());
            }

            Prim->lines.resize(Result.size() - 1);
            for (int i = 0; i < Result.size() - 1; i++) {
                Prim->lines[i] = zeno::vec2i{i, i + 1};
            }

            zeno::log_info("[Roads] Vertices Num: {}, Lines Num: {}", Prim->verts.size(), Prim->lines.size());
        }
    };

    struct ZENO_CRTP(RoadCalcMask, zeno::reflect::IParameterAutoNode) {
        ZENO_GENERATE_NODE_BODY(RoadCalcMask);

        std::shared_ptr<zeno::PrimitiveObject> Mesh;
        ZENO_DECLARE_INPUT_FIELD(Mesh, "Mesh Prim");
        ZENO_DECLARE_OUTPUT_FIELD(Mesh, "Mesh Prim");

        //std::shared_ptr<zeno::PrimitiveObject> Lines;
        //ZENO_DECLARE_INPUT_FIELD(Lines, "Line Prim");

        std::shared_ptr<zeno::RoadBSplineObject> Spline;
        ZENO_DECLARE_INPUT_FIELD(Spline, "Spline");

        std::string OutputChannel;
        ZENO_DECLARE_INPUT_FIELD(OutputChannel, "Road Distance Channel (Vert)", false, "", "roadDis");

        float MaxDistance = 3;
        ZENO_DECLARE_INPUT_FIELD(MaxDistance, "Road Radius", false, "", "5");

        void apply() override {
            using namespace boost::geometry;

            auto &Points = AutoParameter->Mesh->verts;

            std::vector<std::array<float, 3>> New(Points.begin(), Points.end());

            tinyspline::BSpline& SplineQwQ = AutoParameter->Spline->Spline;
            auto DistanceAttr = spline::CalcRoadMask(New, SplineQwQ, AutoParameter->MaxDistance);

            auto& DisAttr = AutoParameter->Mesh->verts.add_attr<float>(AutoParameter->OutputChannel);
            DisAttr.swap(DistanceAttr);
        }
    };

    struct ZENO_CRTP(RoadBDozerSimulate, zeno::reflect::IParameterAutoNode) {
        ZENO_GENERATE_NODE_BODY(RoadBDozerSimulate);

        std::shared_ptr<zeno::PrimitiveObject> Mesh;
        ZENO_DECLARE_INPUT_FIELD(Mesh, "Mesh Prim");
        ZENO_DECLARE_OUTPUT_FIELD(Mesh, "Mesh Prim");

        int SmoothRadius = 3;
        ZENO_DECLARE_INPUT_FIELD(SmoothRadius, "Smooth Radius", false, "", "3");

        int Epochs = 5;
        ZENO_DECLARE_INPUT_FIELD(Epochs, "Epochs", false, "", "5");

        float SlopeThreshold = 0.5f;
        ZENO_DECLARE_INPUT_FIELD(SlopeThreshold, "Slope Threshold", false, "", "0.5");

        float OverThresholdWeightRatio = 0.1f;
        ZENO_DECLARE_INPUT_FIELD(OverThresholdWeightRatio, "Over Threshold Weight Ratio", false, "", "0.1");

        std::string RoadChannel;
        ZENO_DECLARE_INPUT_FIELD(RoadChannel, "Road Distance Channel (Vert)", false, "", "roadMask");

        std::string SizeXChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeXChannel, "Nx Channel (UserData)", false, "", "nx");

        std::string SizeYChannel;
        ZENO_DECLARE_INPUT_FIELD(SizeYChannel, "Ny Channel (UserData)", false, "", "ny");

        int Nx = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Mesh, Nx, SizeXChannel, false);

        int Ny = 0;
        ZENO_BINDING_PRIMITIVE_USERDATA(Mesh, Ny, SizeYChannel, false);

        zeno::AttrVector<float> RoadMask{};
        ZENO_BINDING_PRIMITIVE_ATTRIBUTE(Mesh, RoadMask, RoadChannel, zeno::reflect::EZenoPrimitiveAttr::VERT);

        void apply() override {
            Mesh = AutoParameter->Mesh;
            SmoothRadius = AutoParameter->SmoothRadius;
            Epochs = AutoParameter->Epochs;
            RoadChannel = AutoParameter->RoadChannel;
            Nx = AutoParameter->Nx;
            Ny = AutoParameter->Ny;
            SlopeThreshold = AutoParameter->SlopeThreshold;
            OverThresholdWeightRatio = AutoParameter->OverThresholdWeightRatio;

            zeno::AttrVector<zeno::vec3f>& PositionAttr = Mesh->verts;
            zeno::AttrVector<float>& RoadMaskk = AutoParameter->RoadMask;

            std::vector<float> UpdatedHeightField(PositionAttr.size());

            for (int32_t Epoch = 0; Epoch < Epochs; ++Epoch) {
#pragma omp parallel for
                for (int32_t y = 0; y < Ny; ++y) {
                    for (int32_t x = 0; x < Nx; ++x) {
                        int32_t Idx = y * Nx + x;
                        if (0 != RoadMaskk[Idx]) {
                            float HeightSummary = 0.0f;
                            float WeightSummary = 0;

                            for (int32_t dx = -SmoothRadius; dx <= SmoothRadius; ++dx) {
                                for (int32_t dy = -SmoothRadius; dy <= SmoothRadius; ++dy) {
                                    int32_t nx = x + dx;
                                    int32_t ny = y + dy;

                                    if (nx >= 0 && ny >= 0 && nx < Nx && ny < Ny) {
                                        float Distance = std::sqrt(dx * dx + dy * dy);
                                        float Slope = 0.0f;

                                        if (Distance > 1e-3) {
                                            Slope = std::abs<float>(PositionAttr[y*Nx + x][1] - PositionAttr[ny*Nx + nx][1]) / Distance;
                                        }

                                        float Weight = std::exp(-(dx*dx + dy*dy) / (2.0f * SmoothRadius * SmoothRadius));

                                        if (Slope > SlopeThreshold) {
                                            Weight *= OverThresholdWeightRatio;
                                        }

                                        HeightSummary += PositionAttr[ny * Nx + nx].at(1) * Weight;
                                        WeightSummary += Weight;
                                    }
                                }
                            }

                            if (WeightSummary > 0) {
                                UpdatedHeightField[Idx] = HeightSummary / WeightSummary;
                            }
                        }
                    }
                }
            }

#pragma omp parallel for
            for (int32_t i = 0; i < UpdatedHeightField.size(); ++i) {
                if (0 != RoadMaskk[i])
                    PositionAttr[i][1] = UpdatedHeightField[i];
            }
        }
    };

#if _MSC_VER
#include "Windows.h"
    // Windows Debug
    struct ZENO_CRTP(WindowsWaitForDebugger, zeno::reflect::IParameterAutoNode) {
        ZENO_GENERATE_NODE_BODY(WindowsWaitForDebugger);

        void apply() override {
            while (!IsDebuggerPresent()) {
                ;
            }
        }
    };
#endif
}// namespace
