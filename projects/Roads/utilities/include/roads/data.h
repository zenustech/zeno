#pragma once

#include "Eigen/Eigen"
#include <array>
#include <cassert>
#include <cmath>
#include <vector>
#include <functional>

namespace roads {

    enum class TriangleType {
        ACUTE = 0,
        RIGHT = 1,
        OBTUSE = 2,
    };

    struct Point : public Eigen::Vector3d {};

    struct Triangle : public std::array<Point, 3> {
        bool AnyAngleLargerThan(const double Degree) {
            assert(Degree <= 180.0f);

            const double Rad = Degree * (1.0 / 180.0);

            const Eigen::Vector3d E1 = (at(1) - at(0)).normalized();
            const Eigen::Vector3d E2 = (at(2) - at(1)).normalized();
            const Eigen::Vector3d E3 = (at(0) - at(2)).normalized();
            return asin(E1.dot(E2)) > Rad || asin(E2.dot(E3)) > Rad || asin(E3.dot(E1)) > Rad;
        }
    };

    template <typename T>
    struct ArrayList : public std::vector<T> {};

    template <size_t X, size_t Y, typename GridPointType = Point>
    struct Grid : public std::array<GridPointType, X * Y> {};

    template <typename GridPointType = Point>
    struct DynamicGrid : public std::vector<GridPointType> {
        size_t Nx, Ny;

        DynamicGrid(const size_t InNx, const size_t InNy) : std::vector<GridPointType>(InNx * InNy), Nx(InNx), Ny(InNy) {}
        DynamicGrid(DynamicGrid&& OtherGridToMove)  noexcept : std::vector<GridPointType>(std::forward(OtherGridToMove)) {
            Nx = OtherGridToMove.Nx;
            Ny = OtherGridToMove.Ny;
        }
    };

}// namespace roads
