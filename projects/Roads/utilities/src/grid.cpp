#include "roads/grid.h"
#include "Eigen/Eigen"
#include "roads/roads.h"

using namespace roads;

DynamicGrid<CostPoint> roads::CalcCost_Simple(const DynamicGrid<AdvancePoint> &BaseGrid) {
    return {0, 0};
}

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

double roads::energy::CalculateStepSize(const Point2D &Pt, const Point2D &P, const Point2D &PrevX, const Point2D &CurrX) {
    const double LengthPtToXc = EuclideanDistance(Pt, CurrX);
    const double LengthPtToP = EuclideanDistance(Pt, P);
    const double LengthPrevXToP = EuclideanDistance(PrevX, P);
    const double LengthPrevXToCurrX = EuclideanDistance(PrevX, CurrX);

    return 0.0;
}
