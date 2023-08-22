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

    enum class ConnectiveType {
        FOUR = 4,
        EIGHT = 8,
        SIXTEEN = 16,
        FOURTY = 40,
    };

    struct Point : public Eigen::Vector3d {
        Point(const std::array<float, 3>& InArray) : Eigen::Vector3d(InArray[0], InArray[1], InArray[2]) {}

        using Eigen::Vector3d::Vector3d;
    };

    struct Point2D : public Eigen::Vector2d {
        using Eigen::Vector2d::Vector2d;
    };

    struct IntPoint2D : public std::array<long, 2> {};

    struct IntPoint : public std::array<long, 3> {};

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
    struct ArrayList : public std::vector<T> {
        using std::vector<T>::vector;
        using std::vector<T>::size;

        bool IsValidIndex(size_t Index) const {
            return Index < size();
        }
    };

    template <size_t X, size_t Y, typename GridPointType = Point>
    struct Grid : public std::array<GridPointType, X * Y> {};

    template<typename GridPointType>
    struct CustomGridBase : public ArrayList<GridPointType> {
        size_t Nx, Ny;

        CustomGridBase(const size_t InNx, const size_t InNy) : ArrayList<GridPointType>(InNx * InNy), Nx(InNx), Ny(InNy) { }
        CustomGridBase(CustomGridBase&& OtherGridToMove)  noexcept : ArrayList<GridPointType>(std::forward<ArrayList<GridPointType>>(OtherGridToMove)) {
            Nx = OtherGridToMove.Nx;
            Ny = OtherGridToMove.Ny;
        }
    };

    template <typename GridPointType = Point>
    struct DynamicGrid : public CustomGridBase<GridPointType> {
        using CustomGridBase<GridPointType>::CustomGridBase;
    };

}// namespace roads
