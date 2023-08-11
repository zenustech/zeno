#pragma once

#include "pch.h"

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
    using CostPoint = double;

    using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
    using WeightedGridUndirectedGraph = boost::adjacency_list<boost::listS,boost::vecS,boost::directedS,boost::no_property,EdgeWeightProperty>;
    using WeightedGridUndirectedGraphIterator = boost::graph_traits<WeightedGridUndirectedGraph>::edge_iterator;
    using Edge = std::pair<size_t, size_t>;

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point &Point1, const Point &Point2);

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point2D &Point1, const Point2D &Point2);

    ROADS_API DynamicGrid<SlopePoint> CalculateSlope(const DynamicGrid<HeightPoint>& InHeightField);

    ROADS_API WeightedGridUndirectedGraph CreateWeightGraphFromCostGrid(const DynamicGrid<CostPoint >& InCostGrid, const ConnectiveType Type);

    ROADS_API ArrayList<ArrayList<double>> FloydWarshallShortestPath(WeightedGridUndirectedGraph &InGraph);

    namespace energy {
        ROADS_API double CalculateStepSize(const Point2D &Pt, const Point2D &P, const Point2D &PrevX, const Point2D &CurrX);
    }// namespace energy

}// namespace roads
