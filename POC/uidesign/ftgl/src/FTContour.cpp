/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2001-2004 Henry Maddocks <ftgl@opengl.geek.nz>
 * Copyright (c) 2008 Sam Hocevar <sam@hocevar.net>
 * Copyright (c) 2008 Éric Beets <ericbeets@free.fr>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "config.h"

#include "FTContour.h"

#include <math.h>

static const unsigned int BEZIER_STEPS = 5;


void FTContour::AddPoint(FTPoint point)
{
    if(pointList.empty() || (point != pointList[pointList.size() - 1]
                              && point != pointList[0]))
    {
        pointList.push_back(point);
    }
}


void FTContour::AddOutsetPoint(FTPoint point)
{
    outsetPointList.push_back(point);
}


void FTContour::AddFrontPoint(FTPoint point)
{
    frontPointList.push_back(point);
}


void FTContour::AddBackPoint(FTPoint point)
{
    backPointList.push_back(point);
}


void FTContour::evaluateQuadraticCurve(FTPoint A, FTPoint B, FTPoint C)
{
    for(unsigned int i = 1; i < BEZIER_STEPS; i++)
    {
        float t = static_cast<float>(i) / BEZIER_STEPS;

        FTPoint U = (1.0f - t) * A + t * B;
        FTPoint V = (1.0f - t) * B + t * C;

        AddPoint((1.0f - t) * U + t * V);
    }
}


void FTContour::evaluateCubicCurve(FTPoint A, FTPoint B, FTPoint C, FTPoint D)
{
    for(unsigned int i = 0; i < BEZIER_STEPS; i++)
    {
        float t = static_cast<float>(i) / BEZIER_STEPS;

        FTPoint U = (1.0f - t) * A + t * B;
        FTPoint V = (1.0f - t) * B + t * C;
        FTPoint W = (1.0f - t) * C + t * D;

        FTPoint M = (1.0f - t) * U + t * V;
        FTPoint N = (1.0f - t) * V + t * W;

        AddPoint((1.0f - t) * M + t * N);
    }
}


// This function is a bit tricky. Given a path ABC, it returns the
// coordinates of the outset point facing B on the left at a distance
// of 64.0.
//                                         M
//                            - - - - - - X
//                             ^         / '
//                             | 64.0   /   '
//  X---->-----X     ==>    X--v-------X     '
// A          B \          A          B \   .>'
//               \                       \<'  64.0
//                \                       \                  .
//                 \                       \                 .
//                C X                     C X
//
FTPoint FTContour::ComputeOutsetPoint(FTPoint A, FTPoint B, FTPoint C)
{
    // If the angle between 'ab' and 'bc' approaches 180 degrees,
    // the outset point goes to infinity, giving an invalid result.
    // Even for angles near 180 degrees, the point will be quite
    // far away from A, B and C. To avoid ugly results, limit
    // its distance to 64.0 * OutsetMax.
    static const FTGL_DOUBLE OutsetMax = 5;

    /* Build the rotation matrix from 'ba' vector */
    FTPoint ba = (A - B).Normalise();
    FTPoint bc = C - B;

    /* Rotate bc to the left */
    FTPoint tmp(bc.X() * -ba.X() + bc.Y() * -ba.Y(),
                bc.X() * ba.Y() + bc.Y() * -ba.X());

    /* Compute the vector bisecting 'abc' */
    FTGL_DOUBLE norm = sqrt(tmp.X() * tmp.X() + tmp.Y() * tmp.Y());
    FTGL_DOUBLE dist;
    if (norm - tmp.X() > (norm + tmp.X()) * OutsetMax * OutsetMax)
      dist = 64.0 * OutsetMax;
    else
      dist = 64.0 * sqrt((norm - tmp.X()) / (norm + tmp.X()));
    tmp.X(tmp.Y() < 0.0 ? dist : -dist);
    tmp.Y(64.0);

    /* Rotate the new bc to the right */
    return FTPoint(tmp.X() * -ba.X() + tmp.Y() * ba.Y(),
                   tmp.X() * -ba.Y() + tmp.Y() * -ba.X());
}


void FTContour::SetParity(int parity)
{
    size_t size = PointCount();
    FTPoint vOutset;

    if(((parity & 1) && clockwise) || (!(parity & 1) && !clockwise))
    {
        // Contour orientation is wrong! We must reverse all points.
        // FIXME: could it be worth writing FTVector::reverse() for this?
        for(size_t i = 0; i < size / 2; i++)
        {
            FTPoint tmp = pointList[i];
            pointList[i] = pointList[size - 1 - i];
            pointList[size - 1 -i] = tmp;
        }

        clockwise = !clockwise;
    }

    for(size_t i = 0; i < size; i++)
    {
        size_t prev, cur, next;

        prev = (i + size - 1) % size;
        cur = i;
        next = (i + size + 1) % size;

        vOutset = ComputeOutsetPoint(Point(prev), Point(cur), Point(next));
        AddOutsetPoint(vOutset);
    }
}


FTContour::FTContour(FT_Vector* contour, char* tags, unsigned int n)
{
    FTPoint prev, cur(contour[(n - 1) % n]), next(contour[0]);
    FTPoint a, b = next - cur;
    double olddir, dir = atan2((next - cur).Y(), (next - cur).X());
    double angle = 0.0;

    // See http://freetype.sourceforge.net/freetype2/docs/glyphs/glyphs-6.html
    // for a full description of FreeType tags.
    for(unsigned int i = 0; i < n; i++)
    {
        prev = cur;
        cur = next;
        next = FTPoint(contour[(i + 1) % n]);
        olddir = dir;
        dir = atan2((next - cur).Y(), (next - cur).X());

        // Compute our path's new direction.
        double t = dir - olddir;
        if(t < -M_PI) t += 2 * M_PI;
        if(t > M_PI) t -= 2 * M_PI;
        angle += t;

        // Only process point tags we know.
        if(n < 2 || FT_CURVE_TAG(tags[i]) == FT_Curve_Tag_On)
        {
            AddPoint(cur);
        }
        else if(FT_CURVE_TAG(tags[i]) == FT_Curve_Tag_Conic)
        {
            FTPoint prev2 = prev, next2 = next;

            // Previous point is either the real previous point (an "on"
            // point), or the midpoint between the current one and the
            // previous "conic off" point.
            if(FT_CURVE_TAG(tags[(i - 1 + n) % n]) == FT_Curve_Tag_Conic)
            {
                prev2 = (cur + prev) * 0.5;
                AddPoint(prev2);
            }

            // Next point is either the real next point or the midpoint.
            if(FT_CURVE_TAG(tags[(i + 1) % n]) == FT_Curve_Tag_Conic)
            {
                next2 = (cur + next) * 0.5;
            }

            evaluateQuadraticCurve(prev2, cur, next2);
        }
        else if(FT_CURVE_TAG(tags[i]) == FT_Curve_Tag_Cubic
                 && FT_CURVE_TAG(tags[(i + 1) % n]) == FT_Curve_Tag_Cubic)
        {
            evaluateCubicCurve(prev, cur, next,
                               FTPoint(contour[(i + 2) % n]));
        }
    }

    // If final angle is positive (+2PI), it's an anti-clockwise contour,
    // otherwise (-2PI) it's clockwise.
    clockwise = (angle < 0.0);
}


void FTContour::buildFrontOutset(float outset)
{
    frontPointList.clear();

    for(size_t i = 0; i < PointCount(); ++i)
    {
        AddFrontPoint(Point(i) + Outset(i) * outset);
    }
}


void FTContour::buildBackOutset(float outset)
{
    backPointList.clear();

    for(size_t i = 0; i < PointCount(); ++i)
    {
        AddBackPoint(Point(i) + Outset(i) * outset);
    }
}

