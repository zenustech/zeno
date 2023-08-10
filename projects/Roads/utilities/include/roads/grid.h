#pragma once

#include "pch.h"

namespace roads {

    struct CostPoint {
        size_t Cost = 0;
        Point Position;
    };

    struct AdvancePoint {
        Point Position;
        double Curvature = 0.0;
    };

    using HeightPoint = double;
    using SlopePoint = double;

    ROADS_API DynamicGrid<CostPoint> CalcCost_Simple(const DynamicGrid<AdvancePoint> &BaseGrid);

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point &Point1, const Point &Point2);

    ROADS_API ROADS_INLINE double EuclideanDistance(const Point2D &Point1, const Point2D &Point2);

    ROADS_API DynamicGrid<SlopePoint> CalculateSlope(const DynamicGrid<HeightPoint>& InHeightField);

    namespace energy {
        ROADS_API double CalculateStepSize(const Point2D &Pt, const Point2D &P, const Point2D &PrevX, const Point2D &CurrX);
    }// namespace energy

}// namespace roads
