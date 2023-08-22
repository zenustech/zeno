#include "roads/grid.h"
#include "Eigen/Eigen"
#include "boost/graph/floyd_warshall_shortest.hpp"

using namespace roads;

double roads::EuclideanDistance(const Point &Point1, const Point &Point2) {
    return std::sqrt(std::pow(Point2.x() - Point1.x(), 2) + std::pow(Point2.y() - Point1.y(), 2) + std::pow(Point2.z() - Point1.z(), 2));
}

double roads::EuclideanDistance(const Point2D &Point1, const Point2D &Point2) {
    return std::sqrt(std::pow(Point2.x() - Point1.x(), 2) + std::pow(Point2.y() - Point1.y(), 2));
}

DynamicGrid<SlopePoint> roads::CalculateSlope(const DynamicGrid<HeightPoint> &InHeightField) {
    constexpr static std::array<int32_t, 4> XDirection4 = { -1, 0, 1, 0 };
    constexpr static std::array<int32_t, 4> YDirection4 = { 0, 1, 0, -1 };
    constexpr static size_t DirectionSize = std::max(XDirection4.size(), YDirection4.size());

    const size_t SizeX = InHeightField.Nx;
    const size_t SizeY = InHeightField.Ny;

    DynamicGrid<SlopePoint> Result(SizeX, SizeY);

#pragma omp parallel for
    for (int32_t y = 0; y < SizeY; ++y) {
        for (int32_t x = 0; x < SizeX; ++x) {
            const size_t OriginIdx = x + y * SizeX;
            double MaxSlope = 0.0;
            for (int32_t Direction = 0; Direction < DirectionSize; ++Direction) {
                const int32_t nx = x + XDirection4[Direction];
                const int32_t ny = y + YDirection4[Direction];
                if (nx < 0 || ny < 0) continue;

                const size_t idx = nx + ny * SizeX;
                if (idx >= InHeightField.size()) continue;

                const double HeightDiff = InHeightField[OriginIdx] - InHeightField[idx];
                const double Distance = std::sqrt(1 + HeightDiff * HeightDiff);
                MaxSlope = std::max(MaxSlope, HeightDiff / Distance);
            }
            Result[OriginIdx] = MaxSlope;
        }
    }

    return Result;
}

WeightedGridUndirectedGraph roads::CreateWeightGraphFromCostGrid(const DynamicGrid<CostPoint> &InCostGrid, const ConnectiveType Type, float PowParam/*=1.5f*/) {
    ArrayList<IntPoint2D> Directions = { { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 } };
    if (Type >= ConnectiveType::EIGHT) {
        Directions.insert(Directions.end(), { { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, 1 } });
    }
    if (Type >= ConnectiveType::SIXTEEN) {
        Directions.insert(Directions.end(), { { -1, -2 }, { 1, -2 }, { -2, -1 }, { 2, -1 }, { -2, 1 }, { 2, 1 }, { -1, 2 }, { 1, 2 } });
    }
    if (Type >= ConnectiveType::FOURTY) {
    }

    WeightedGridUndirectedGraph NewGraph { InCostGrid.size() };

    // boost graph library seem not provide thread safe
#pragma omp parallel for
    for (int32_t y = 0; y < InCostGrid.Ny; ++y) {
        for (int32_t x = 0; x < InCostGrid.Nx; ++x) {
            const size_t OriginIdx = y * InCostGrid.Nx + x;
            boost::property_map<WeightedGridUndirectedGraph, boost::edge_weight_t>::type WeightMap = boost::get(boost::edge_weight, NewGraph);
            for (auto & Direction : Directions) {
                const size_t ix = x + Direction[0];
                const size_t iy = y + Direction[1];
                if (ix >= InCostGrid.Nx || iy >= InCostGrid.Ny) continue;
                const size_t TargetIdx = iy * InCostGrid.Nx + ix;
                using EdgeDescriptor = boost::graph_traits<WeightedGridUndirectedGraph>::edge_descriptor;
                auto [edge1, _] = boost::add_edge(OriginIdx, TargetIdx, NewGraph);
                auto [edge2, _2] = boost::add_edge(TargetIdx, OriginIdx,  NewGraph);
//                WeightMap[edge1] = std::pow(InCostGrid[TargetIdx] - InCostGrid[OriginIdx] > 0 ? std::min(InCostGrid[TargetIdx] - InCostGrid[OriginIdx] - 20.0, InCostGrid[TargetIdx] - InCostGrid[OriginIdx] - 10.0) : InCostGrid[TargetIdx] - InCostGrid[OriginIdx], 2);
//                WeightMap[edge2] = std::pow(InCostGrid[OriginIdx] - InCostGrid[TargetIdx] > 0 ? std::min(InCostGrid[OriginIdx] - InCostGrid[TargetIdx] - 20.0, InCostGrid[OriginIdx] - InCostGrid[TargetIdx] - 10.0) : InCostGrid[OriginIdx] - InCostGrid[TargetIdx], 2);
                WeightMap[edge1] = std::pow(InCostGrid[TargetIdx] - InCostGrid[OriginIdx], PowParam);
                WeightMap[edge2] = std::pow(InCostGrid[OriginIdx] - InCostGrid[TargetIdx], PowParam);
            }
        }
    }

    return NewGraph;
}

ArrayList<ArrayList<double>> roads::FloydWarshallShortestPath(WeightedGridUndirectedGraph &InGraph) {
    ArrayList<ArrayList<double>> D { InGraph.m_vertices.size() };
    ArrayList<double> d (InGraph.m_vertices.size(), (std::numeric_limits<double>::max)());
    printf("%llu", InGraph.m_vertices.size());
    boost::floyd_warshall_all_pairs_shortest_paths(InGraph, D, boost::distance_map(&d[0]));
    return D;
}

double roads::energy::CalculateStepSize(const Point2D &Pt, const Point2D &P, const Point2D &PrevX, const Point2D &CurrX) {
    const double LengthPtToXc = EuclideanDistance(Pt, CurrX);
    const double LengthPtToP = EuclideanDistance(Pt, P);
    const double LengthPrevXToP = EuclideanDistance(PrevX, P);
    const double LengthPrevXToCurrX = EuclideanDistance(PrevX, CurrX);

    return 0.0;
}
