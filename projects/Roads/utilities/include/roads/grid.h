#pragma once

#include "pch.h"
#include <numeric>
#include <functional>
#include <queue>

//#include "boost/graph/use_mpi.hpp"
#include "boost/graph/graph_traits.hpp"
#include "boost/graph/graph_concepts.hpp"
#include "boost/graph/adjacency_list.hpp"

namespace roads {

    struct AdvancePoint {
        Point Position;
        double Gradient = 0.0;
    };

    using HeightPoint = double;
    using SlopePoint = double;
    struct CostPoint {
        double Height;
        double Gradient; // slope
        double Curvature;
    };

    using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
    using WeightedGridUndirectedGraph = boost::adjacency_list<boost::listS,boost::vecS,boost::directedS,boost::no_property,EdgeWeightProperty>;
    using WeightedGridUndirectedGraphIterator = boost::graph_traits<WeightedGridUndirectedGraph>::edge_iterator;
    using Edge = std::pair<size_t, size_t>;

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point &Point1, const Point &Point2);

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point2D &Point1, const Point2D &Point2);

    ROADS_API DynamicGrid<SlopePoint> CalculateSlope(const DynamicGrid<HeightPoint>& InHeightField);

    ROADS_API WeightedGridUndirectedGraph CreateWeightGraphFromCostGrid(const DynamicGrid<CostPoint >& InCostGrid, ConnectiveType Type, const std::function<double(double)>& HeightMappingFunc = [] (double v) { return v; }, const std::function<double(double)>& GradientMappingFunc = [] (double v) { return v; }, const std::function<double(double)>& CurvatureMappingFunc = [] (double v) { return v; });

    ROADS_API ArrayList<ArrayList<double>> FloydWarshallShortestPath(WeightedGridUndirectedGraph &InGraph);

    template <typename T>
    ROADS_API void AverageSmooth(ArrayList<T>& InOutContainer, size_t Iteration = 1) {
        auto num_vertices = InOutContainer.size();
        ArrayList<float> NewHeights(num_vertices);

        for (unsigned int iteration = 0; iteration < 4; ++iteration) {
#pragma omp parallel for
            for (size_t i = 0; i < num_vertices; ++i) {
                // Include the vertex itself and its neighbors.
                ArrayList<T*> Points = InOutContainer[i].Neighbors;
                Points.push_back(&InOutContainer[i]);

                // Calculate the average height.
                NewHeights[i] = std::accumulate(begin(Points), end(Points), 0.0f,
                                                [](float sum, const T* v) { return sum + v->Height; }) / Points.size();
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

        template <typename IntLikeT>
        IntLikeT GreatestCommonDivisor(IntLikeT i, IntLikeT j) {
            if (0 == j)
                return i;
            return GreatestCommonDivisor(j, i % j);
        }

        template <typename PointType = IntPoint2D>
        void RoadsShortestPath(const PointType& StartPoint, const PointType& GoalPoint, size_t MaskK, std::map<const PointType&, const PointType&>& PredecessorList, std::map<PointType, float>& CostTo, const std::function<float(PointType, IntPoint2D)>& CostFunction) {
            // Adding Chebyshev distance as heuristic function
            auto NewCostFunc = [CostFunction] (const PointType& a, const PointType& b) -> float {
                return CostFunction(a, b) + std::max(std::abs(a[0] - b[0]), std::abs(a[1] - b[1]));
            };

            // initialize a priority queue Q with the initial point StartPoint
            auto Comparator = [&CostTo](const PointType& Point1, const PointType& Point2) { return CostTo[Point1] < CostTo[Point2]; };

            std::priority_queue<PointType, ArrayList<PointType>, decltype(Comparator)> Q(Comparator);
            Q.push(StartPoint);

            // while Q is not empty
            while (!Q.empty()) {
                // select the point p_ij from the priority queue with the smallest cost value c(p_ij)
                PointType Point = Q.top();
                Q.pop();

                // if destination has been found p_ij = b, stop the algorithm
                if (GoalPoint == Point) {
                    break;
                }

                // for all points q ∈ M_k(p_ij)
                for (int32_t dx = -int32_t(MaskK); dx <= MaskK; ++dx) {
                    for (int32_t dy = -int32_t(MaskK); dy <= MaskK; ++dy) {
                        if (GreatestCommonDivisor(std::abs(dx), std::abs(dy)) == 1) {
                            PointType NeighbourPoint { Point[0] + dx, Point[1] + dy };
                            float NewCost = CostTo[Point] + NewCostFunc(Point, NeighbourPoint);

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
