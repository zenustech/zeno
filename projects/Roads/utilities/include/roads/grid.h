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

namespace roads {

    struct AdvancePoint {
        Point Position;
        double Gradient = 0.0;
    };

    using HeightPoint = double;
    using SlopePoint = double;
    struct CostPoint : std::array<size_t, 2> {
        CostPoint() = default;

        CostPoint(size_t x, size_t y, double InHeight = 0.0, double InGradient = 0.0, double InCurvature = 0.0)
            : std::array<size_t, 2>(), Height(InHeight), Gradient(InGradient), Curvature(InCurvature) {
            at(0) = x;
            at(1) = y;
        }

        bool operator==(const CostPoint &Rhs) {
            return at(0) == Rhs.at(0) && at(1) == Rhs.at(1);
        }

        double Height;
        double Gradient;// slope
        double Curvature;
    };

    using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
    using WeightedGridUndirectedGraph = boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, EdgeWeightProperty>;
    using WeightedGridUndirectedGraphIterator = boost::graph_traits<WeightedGridUndirectedGraph>::edge_iterator;
    using Edge = std::pair<size_t, size_t>;

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point &Point1, const Point &Point2);

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point2D &Point1, const Point2D &Point2);

    ROADS_API DynamicGrid<SlopePoint> CalculateSlope(const DynamicGrid<HeightPoint> &InHeightField);

    ROADS_API WeightedGridUndirectedGraph CreateWeightGraphFromCostGrid(
        const DynamicGrid<CostPoint> &InCostGrid, ConnectiveType Type, const std::function<double(double)> &HeightMappingFunc = [](double v) { return v; }, const std::function<double(double)> &GradientMappingFunc = [](double v) { return v; }, const std::function<double(double)> &CurvatureMappingFunc = [](double v) { return v; });

    ROADS_API ArrayList<ArrayList<double>> FloydWarshallShortestPath(WeightedGridUndirectedGraph &InGraph);

    template<typename T>
    ROADS_API void AverageSmooth(ArrayList<T> &InOutContainer, size_t Iteration = 1) {
        auto num_vertices = InOutContainer.size();
        ArrayList<float> NewHeights(num_vertices);

        for (unsigned int iteration = 0; iteration < 4; ++iteration) {
#pragma omp parallel for
            for (size_t i = 0; i < num_vertices; ++i) {
                // Include the vertex itself and its neighbors.
                ArrayList<T *> Points = InOutContainer[i].Neighbors;
                Points.push_back(&InOutContainer[i]);

                // Calculate the average height.
                NewHeights[i] = std::accumulate(begin(Points), end(Points), 0.0f,
                                                [](float sum, const T *v) { return sum + v->Height; }) /
                                Points.size();
            }

            // Update the heights for the next iteration or the final result.
#pragma omp parallel for
            for (size_t i = 0; i < num_vertices; ++i) {
                InOutContainer[i].Height = NewHeights[i];
            }
        }
    }

    namespace energy {
        ROADS_API double CalculateStepSize(const Point2D &Pt, const Point2D &P, const Point2D &PrevX, const Point2D &CurrX);

        template<typename IntLikeT>
        inline IntLikeT GreatestCommonDivisor(IntLikeT i, IntLikeT j) {
            if (0 == j) {
                return i;
            }
            return GreatestCommonDivisor(j, i % j);
        }

        template<typename PointType = IntPoint2D>
        void RoadsShortestPath(const PointType &StartPoint, const PointType &GoalPoint, const PointType &Bounds, const int32_t MaskK, std::unordered_map<PointType, PointType> &PredecessorList, std::unordered_map<PointType, float> &CostTo, const std::function<float(const PointType &, const PointType &)> &CostFunction) {
            auto NewCostFunc = [CostFunction, GoalPoint, Bounds](const PointType &a, const PointType &b) -> float {
                return CostFunction(a, b);
                //return CostFunction(a, b) + std::abs<float>(GoalPoint[0] - b[0]) + std::abs<float>(GoalPoint[1] - b[1]);
            };

            std::unordered_map<std::pair<PointType, PointType>, float> SimpleCost;
            std::function<float(const PointType &, const PointType &, bool)> StraightCost = [NewCostFunc, MaskK, &SimpleCost, &StraightCost](const PointType &From, const PointType &To, bool bMoveX = true) mutable -> float {
                auto CurrentPair = std::make_pair(From, To);
                if (SimpleCost.count(CurrentPair)) {
                    return SimpleCost[CurrentPair];
                }

                if (From == To) {
                    SimpleCost[CurrentPair] = 0.0f;
                    return 0.0f;
                }

                PointType NextPoint = From;

                if (bMoveX) {
                    if (To[0] != From[0]) {
                        if (std::abs<int32_t>(From[0] - To[0]) <= MaskK) {
                            NextPoint[0] = To[0];
                        } else if (From[0] < To[0]) {
                            NextPoint[0] += 1;
                        } else {
                            NextPoint[0] -= 1;
                        }
                    }
                } else {
                    if (To[1] != From[1]) {
                        if (std::abs<int32_t>(From[1] - To[1]) <= MaskK) {
                            NextPoint[1] = To[1];
                        } else if (From[1] < To[1]) {
                            NextPoint[1] += 1;
                        } else {
                            NextPoint[1] -= 1;
                        }
                    }
                }

                float Cost = NewCostFunc(From, NextPoint) + StraightCost(NextPoint, To, !bMoveX);
                SimpleCost[CurrentPair] = Cost;
                return Cost;
            };

            // Adding Chebyshev distance as heuristic function
            auto Comparator = [&CostTo, &GoalPoint, &Bounds, &StraightCost](const PointType &Point1, const PointType &Point2) {
                //float DeltaA = std::max(std::abs<int32_t>(Point1[0] - GoalPoint[0]) / Bounds[0], std::abs<int32_t>(Point1[1] - GoalPoint[1]) / Bounds[1]);
                //float DeltaB = std::max(std::abs<int32_t>(Point2[0] - GoalPoint[0]) / Bounds[0], std::abs<int32_t>(Point2[1] - GoalPoint[1]) / Bounds[1]);
                //float DeltaA = std::abs<int32_t>(Point1[0] - GoalPoint[0]) + std::abs<int32_t>(Point1[1] - GoalPoint[1]);
                //float DeltaB = std::abs<int32_t>(Point2[0] - GoalPoint[0]) + std::abs<int32_t>(Point2[1] - GoalPoint[1]);
                //float DeltaA = std::sqrt(std::pow(Point1[0] - GoalPoint[0], 2) + std::pow(Point1[1] - GoalPoint[1], 2));
                //float DeltaB = std::sqrt(std::pow(Point2[0] - GoalPoint[0], 2) + std::pow(Point2[1] - GoalPoint[1], 2));

                float DeltaA = StraightCost(Point1, GoalPoint, true);
                float DeltaB = StraightCost(Point2, GoalPoint, true);

                float CostA = CostTo[Point1] + DeltaA;
                float CostB = CostTo[Point2] + DeltaB;
                return CostA > CostB;
            };

            // initialize a priority queue Q with the initial point StartPoint
            std::priority_queue<PointType, ArrayList<PointType>, decltype(Comparator)> Q(Comparator);
            Q.push(StartPoint);
            CostTo[StartPoint] = 0.f;
            for (size_t x = 0; x < Bounds[0]; x++) {
                for (size_t y = 0; y < Bounds[1]; y++) {
                    PointType Point{x, y};
                    if (Point != StartPoint) { CostTo[Point] = std::numeric_limits<float>::max(); }
                }
            }

            // while Q is not empty
            while (!Q.empty()) {
                // select the point p_ij from the priority queue with the smallest cost value c(p_ij)
                PointType Point = Q.top();
                Q.pop();

                // if destination has been found p_ij = b, stop the algorithm
                if (GoalPoint == Point) {
                    break;
                }

                //printf("-- P(%d,%d) %f\n", Point[0], Point[1], CostTo[Point]);

                // step3. for all points q ∈ M_k(p_ij)
                for (int32_t dx = -MaskK; dx <= MaskK; ++dx) {
                    for (int32_t dy = -MaskK; dy <= MaskK; ++dy) {
                        if (GreatestCommonDivisor(std::abs(dx), std::abs(dy)) == 1) {
                            PointType NeighbourPoint{Point[0] + dx, Point[1] + dy};
                            if (NeighbourPoint[0] < 0 || NeighbourPoint[1] < 0 || NeighbourPoint[0] >= Bounds[0] || NeighbourPoint[1] >= Bounds[1]) continue;

                            float NewCost = CostTo[Point] + NewCostFunc(Point, NeighbourPoint);
                            if (NewCost < 0) {
                                throw std::runtime_error("[Roads] Graph should not have negative weight. Check your curve !");
                            }
                            //printf("---- N(%d,%d) %f\n", NeighbourPoint[0], NeighbourPoint[1], NewCost);

                            if (CostTo.find(NeighbourPoint) == CostTo.end() || NewCost < CostTo[NeighbourPoint]) {
                                PredecessorList[NeighbourPoint] = Point;
                                CostTo[NeighbourPoint] = NewCost;
                                Q.push(NeighbourPoint);
                            }
                        }
                    }
                }
            }
        }
    }// namespace energy

}// namespace roads


namespace std {
    template<>
    struct hash<roads::CostPoint> {
        size_t operator()(const roads::CostPoint &rhs) const {
            constexpr size_t Seed = 10086;
            return (rhs[0] + 0x9e3779b9 + (Seed << 4) + (Seed >> 2)) ^ (rhs[1] + 0x9e3779b9 + (Seed << 6) + (Seed >> 2));
        }
    };

    template<>
    struct hash<std::pair<roads::CostPoint, roads::CostPoint>> {
        size_t operator()(const std::pair<roads::CostPoint, roads::CostPoint> &rhs) const {
            constexpr size_t Seed = 10086;
            return (rhs.first[0] + 0x9e3779b9 + (Seed << 4) + (Seed >> 2)) ^ (rhs.first[1] + 0x9e3779b9 + (Seed << 6) + (Seed >> 2)) ^ (rhs.second[0] + 0x9e3779b9 + (Seed << 4) + (Seed >> 2)) ^ (rhs.second[1] + 0x9e3779b9 + (Seed << 6) + (Seed >> 2));
        }
    };
}// namespace std
