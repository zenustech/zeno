#pragma once

#include "pch.h"
#include <functional>
#include <numeric>
#include <queue>
#include <unordered_map>

//#include "boost/graph/use_mpi.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_concepts.hpp"
#include "boost/graph/graph_traits.hpp"

namespace tinyspline {
    class BSpline;
}

namespace roads {

    struct AdvancePoint {
        Point Position;
        double Gradient = 0.0;
    };

    using HeightPoint = double;
    using SlopePoint = double;
    struct CostPoint : std::array<size_t, 3> {
        CostPoint() = default;

        CostPoint(size_t x, size_t y, size_t a = 0, double InHeight = 0.0, double InGradient = 0.0 /*, double InCurvature = 0.0**/)
            : std::array<size_t, 3>(), Height(InHeight), Gradient(InGradient) {
            at(0) = x;
            at(1) = y;
            at(2) = a;
        }

        bool operator==(const CostPoint &Rhs) {
            return at(0) == Rhs.at(0) && at(1) == Rhs.at(1) && at(2) == Rhs.at(2);
        }

        double Height;
        double Gradient;// slope

        // We need directional curvature, we cannot pre calculate it
        //double Curvature;
    };

    using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
    using WeightedGridUndirectedGraph = boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, EdgeWeightProperty>;
    using Edge = std::pair<size_t, size_t>;

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point &Point1, const Point &Point2);

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point2D &Point1, const Point2D &Point2);

    ROADS_API DynamicGrid<SlopePoint> CalculateSlope(const DynamicGrid<HeightPoint> &InHeightField);

    namespace energy {
        template<typename IntLikeT>
        inline IntLikeT GreatestCommonDivisor(IntLikeT i, IntLikeT j) {
            if (0 == j) {
                return i;
            }
            return GreatestCommonDivisor(j, i % j);
        }

        template<typename PointType = IntPoint2D>
        void RoadsShortestPath(const PointType &StartPoint, const PointType &GoalPoint, const PointType &Bounds, const int32_t MaskK, const int32_t MaskA, const float WeightHeuristic, DefaultedHashMap<PointType, PointType> &PredecessorList, DefaultedHashMap<PointType, float> &CostTo, const std::function<float(const PointType &, const PointType &)> &CostFunction) {
            auto NewCostFunc = [&CostFunction, &GoalPoint, &Bounds](const PointType &a, const PointType &b) -> float {
                float Result = CostFunction(a, b);
                if (Result < 0) {
                    printf("[Roads] Minus cost P1(%d,%d) P2(%d,%d) Cost=%f.", a[0], a[1], b[0], b[1], Result);
                    throw std::runtime_error("[Roads] Minus edge weight detected.");
                }
                return Result;
                //return CostFunction(a, b) + std::abs<float>(GoalPoint[0] - b[0]) + std::abs<float>(GoalPoint[1] - b[1]);
            };

            std::unordered_map<std::pair<PointType, PointType>, float> SimpleCost;
            std::function<float(const PointType &, const PointType &, bool)> StraightCost = [&NewCostFunc, MaskK, &SimpleCost, &StraightCost](const PointType &From, const PointType &To, bool bMoveX = true) mutable -> float {
                auto CurrentPair = std::make_pair(From, To);
                if (SimpleCost.count(CurrentPair)) {
                    return SimpleCost[CurrentPair];
                }

                if (From[0] == To[0] && From[1] == To[1]) {
                    SimpleCost[CurrentPair] = 0.0f;
                    return 0.0f;
                }

                PointType NextPoint = From;

                if (To[0] != From[0]) {
                    if (std::abs<int32_t>(From[0] - To[0]) <= MaskK) {
                        NextPoint[0] = To[0];
                    } else if (From[0] < To[0]) {
                        NextPoint[0] += MaskK;
                    } else {
                        NextPoint[0] -= MaskK;
                    }
                }
                if (To[1] != From[1]) {
                    if (std::abs<int32_t>(From[1] - To[1]) <= MaskK) {
                        NextPoint[1] = To[1];
                    } else if (From[1] < To[1]) {
                        NextPoint[1] += MaskK;
                    } else {
                        NextPoint[1] -= MaskK;
                    }
                }

                //printf("---- A(%d,%d) B(%d,%d)\n", NextPoint[0], NextPoint[0], To[1], To[1]);

                float Cost = NewCostFunc(From, NextPoint) + StraightCost(NextPoint, To, !bMoveX);
                SimpleCost[CurrentPair] = Cost;
                return Cost;
            };

            // Adding Chebyshev distance as heuristic function
            auto Comparator = [&CostTo, &GoalPoint, &Bounds, &StraightCost, WeightHeuristic](const PointType &Point1, const PointType &Point2) {
                //float DeltaA = std::max(std::abs<int32_t>(Point1[0] - GoalPoint[0]) / Bounds[0], std::abs<int32_t>(Point1[1] - GoalPoint[1]) / Bounds[1]);
                //float DeltaB = std::max(std::abs<int32_t>(Point2[0] - GoalPoint[0]) / Bounds[0], std::abs<int32_t>(Point2[1] - GoalPoint[1]) / Bounds[1]);
                //float DeltaA = std::abs<int32_t>(Point1[0] - GoalPoint[0]) + std::abs<int32_t>(Point1[1] - GoalPoint[1]);
                //float DeltaB = std::abs<int32_t>(Point2[0] - GoalPoint[0]) + std::abs<int32_t>(Point2[1] - GoalPoint[1]);
                //float DeltaA = std::sqrt(std::pow(Point1[0] - GoalPoint[0], 2) + std::pow(Point1[1] - GoalPoint[1], 2));
                //float DeltaB = std::sqrt(std::pow(Point2[0] - GoalPoint[0], 2) + std::pow(Point2[1] - GoalPoint[1], 2));

                float DeltaA = StraightCost(Point1, GoalPoint, true);
                float DeltaB = StraightCost(Point2, GoalPoint, true);

                float CostA = CostTo.DefaultAt(Point1, std::numeric_limits<float>::max()) + WeightHeuristic * DeltaA;
                float CostB = CostTo.DefaultAt(Point2, std::numeric_limits<float>::max()) + WeightHeuristic * DeltaB;
                return CostA > CostB;
            };

            // initialize a priority queue Q with the initial point StartPoint
            CostTo.reserve(Bounds[0] * Bounds[1] * MaskA);
            std::priority_queue<PointType, ArrayList<PointType>, decltype(Comparator)> Q(Comparator);
            for (size_t a = 0; a < MaskA; a++) {
                PointType NewPoint = StartPoint;
                NewPoint[2] = a;
                CostTo[StartPoint] = 0.f;
            }

            PredecessorList[StartPoint] = StartPoint;
            Q.push(StartPoint);

            std::unordered_map<PointType, bool> Visual;

            // while Q is not empty
            while (!Q.empty()) {
                // select the point p_ij from the priority queue with the smallest cost value c(p_ij)
                PointType Point = Q.top();
                Q.pop();

                // if destination has been found p_ij = b, stop the algorithm
                if (GoalPoint == Point) {
                    break;
                }

                if (Visual.find(Point) != std::end(Visual)) {
                    continue;
                }
                Visual[Point] = 1;

                //printf("-- P(%d,%d) %f\n", Point[0], Point[1], CostTo[Point]);

                // extended mask for angle
                for (size_t angle = 0; angle < MaskA; ++angle) {
                    // step3. for all points q ∈ M_k(p_ij)
                    for (int32_t dx = -MaskK; dx <= MaskK; ++dx) {
                        for (int32_t dy = -MaskK; dy <= MaskK; ++dy) {
                            if (GreatestCommonDivisor(std::abs(dx), std::abs(dy)) == 1) {
                                PointType NeighbourPoint{Point[0] + dx, Point[1] + dy, angle};
                                if (NeighbourPoint[0] < 0 || NeighbourPoint[1] < 0 || NeighbourPoint[0] >= Bounds[0] || NeighbourPoint[1] >= Bounds[1]) continue;

                                float NewCost = CostTo[Point] + NewCostFunc(Point, NeighbourPoint);
                                if (NewCost < 0) {
                                    throw std::runtime_error("[Roads] Graph should not have negative weight. Check your curve !");
                                }
                                //printf("---- N(%d,%d) %f\n", NeighbourPoint[0], NeighbourPoint[1], NewCost);

                                if (NewCost < CostTo.DefaultAt(NeighbourPoint, std::numeric_limits<float>::max())) {
                                    PredecessorList[NeighbourPoint] = Point;
                                    CostTo[NeighbourPoint] = NewCost;
                                    Q.push(NeighbourPoint);
                                }
                            }
                        }
                    }
                }
            }
        }
    }// namespace energy

    namespace spline {
        //template<typename PointContainerType = ArrayList<std::array<float, 3>>, typename SegmentContainerType = std::array<int, 2>>
        class tinyspline::BSpline GenerateBSplineFromSegment(const ArrayList<std::array<float, 3>> &InPoints, const ArrayList<std::array<int, 2>> &Segments);

        //template<typename PointContainerType = ArrayList<std::array<float, 3>>, typename SegmentContainerType = ArrayList<std::array<int, 2>>>
        ArrayList<Eigen::Vector3f> GenerateAndSamplePointsFromSegments(const class tinyspline::BSpline& Spline, int32_t SamplePoints = 1000);

        float Distance(const Eigen::Vector3d &point, const tinyspline::BSpline &bSpline, float t);

        float FindNearestPoint(const Eigen::Vector3d &point, const tinyspline::BSpline &bSpline, float& t, float step = 0.01, float tolerance = 1e-6);

        float FindNearestPointSA(const Eigen::Vector3d &Point, const tinyspline::BSpline &Spline);

        ArrayList<float> CalcRoadMask(const std::vector<std::array<float, 3>>& Points, const tinyspline::BSpline& SplineQwQ, float MaxDistance);
    }// namespace spline

}// namespace roads


namespace std {
    template<>
    struct hash<roads::CostPoint> {
        size_t operator()(const roads::CostPoint &rhs) const {
            constexpr size_t Seed = 10086;
            size_t Result = (rhs[0] + 0x9e3779b9 + (Seed << 4) + (Seed >> 2)) ^ (rhs[1] + 0x9e3779b9 + (Seed << 6) + (Seed >> 4)) ^ (rhs[2] + 0x9e3779b9 + (Seed << 8) + (Seed >> 6));
            return Result;
        }
    };

    template<>
    struct hash<std::pair<roads::CostPoint, roads::CostPoint>> {
        size_t operator()(const std::pair<roads::CostPoint, roads::CostPoint> &rhs) const {
            constexpr size_t Seed = 10086;
            auto h = hash<roads::CostPoint>();
            return (h(rhs.first) + (Seed << 4) + (Seed >> 2)) ^ (h(rhs.second) + (Seed << 6) + (Seed >> 4));
        }
    };
}// namespace std
