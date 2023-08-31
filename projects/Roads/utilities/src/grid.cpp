#include "roads/grid.h"
#include "Eigen/Eigen"
#include "boost/graph/floyd_warshall_shortest.hpp"

#include "roads/thirdparty/tinysplinecxx.h"

using namespace roads;

double roads::EuclideanDistance(const Point &Point1, const Point &Point2) {
    return std::sqrt(std::pow(Point2.x() - Point1.x(), 2) + std::pow(Point2.y() - Point1.y(), 2) + std::pow(Point2.z() - Point1.z(), 2));
}

double roads::EuclideanDistance(const Point2D &Point1, const Point2D &Point2) {
    return std::sqrt(std::pow(Point2.x() - Point1.x(), 2) + std::pow(Point2.y() - Point1.y(), 2));
}

DynamicGrid<SlopePoint> roads::CalculateSlope(const DynamicGrid<HeightPoint> &InHeightField) {
    constexpr static std::array<int32_t, 4> XDirection4 = {-1, 0, 1, 0};
    constexpr static std::array<int32_t, 4> YDirection4 = {0, 1, 0, -1};
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

tinyspline::BSpline spline::GenerateBSplineFromSegment(const ArrayList<std::array<float, 3>> &InPoints, const ArrayList<std::array<int, 2>> &Segments) {
    using namespace tinyspline;

    ArrayList<float> Points;

    for (const auto &Seg: Segments) {
        Points.insert(std::end(Points), {float(InPoints[Seg[0]][0]), float(InPoints[Seg[0]][1]), float(InPoints[Seg[0]][2])});
    }
    Points.insert(std::end(Points), {float(InPoints[Segments[Segments.size() - 1][1]][0]), float(InPoints[Segments[Segments.size() - 1][1]][1]), float(InPoints[Segments[Segments.size() - 1][1]][2])});

    return BSpline::interpolateCatmullRom(Points, 3);
}

ArrayList<Eigen::Vector3f> spline::GenerateAndSamplePointsFromSegments(const class tinyspline::BSpline& Spline, int32_t SamplePoints) {
    float Step = 1.0f / float(SamplePoints);

    ArrayList<Eigen::Vector3f> Result;
    Result.resize(SamplePoints);

#pragma omp parallel for
    for (int32_t i = 0; i < SamplePoints; i++) {
        auto Point = Spline.eval(float(i) * Step).resultVec3();
        Result[i] = Eigen::Vector3f{Point.x(), Point.y(), Point.z()};
    }

    return Result;
}

float spline::Distance(const Eigen::Vector3d &point, const tinyspline::BSpline &bSpline, float t) {
    tinyspline::DeBoorNet net = bSpline.eval(t);
    auto Result = net.resultVec3();

    Eigen::Vector3d splinePoint(Result.x(), Result.y(), Result.z());
    return float((splinePoint - point).norm());
}

float spline::FindNearestPoint(const Eigen::Vector3d &point, const tinyspline::BSpline &bSpline, float& t, float step, float tolerance) {
    // initial guess
    t = 0.0;

    while (step > tolerance) {
        float f_current = Distance(point, bSpline, t);
        float nextT = t + step;
        if (nextT > 1) {
            nextT = 1;
        }
        float f_next = Distance(point, bSpline, nextT);

        if (f_next < f_current) {
            t = nextT;
        } else {
            step *= 0.5;
        }
    }

    return Distance(point, bSpline, t);
}

ArrayList<float> spline::CalcRoadMask(const std::vector<std::array<float, 3>> &Points, const tinyspline::BSpline &SplineQwQ, float MaxDistance) {
    ArrayList<float> Result;
    Result.resize(Points.size(), std::numeric_limits<float>::max() - 1);

#pragma omp parallel for
    for (int32_t i = 0; i < Points.size(); ++i) {
        float t = 0;
        const std::array<float, 3>& zp = Points[i];
        Eigen::Vector3d ep(zp[0], zp[1], zp[2]);
        float Distance = spline::FindNearestPoint(ep, SplineQwQ, t);

        if (std::abs<float>(Distance) < MaxDistance) {
            Result[i] = t;
        }
    }

    return Result;
}
