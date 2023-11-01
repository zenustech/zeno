/* Copyright © 2018 Øystein Myrmo (oystein.myrmo@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once
#include <cassert>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <array>

#define BEZIER_FUZZY_EPSILON 0.0001
#define BEZIER_DEFAULT_INTERVALS 10
#define BEZIER_DEFAULT_MAX_ITERATIONS 15

namespace Bezier
{
    namespace Math
    {
        constexpr float PI = 3.14159265358979f;

        inline size_t binomial(size_t n, size_t k)
        {
            assert(k <= n);
            size_t val = 1;
            for (size_t i = 1; i <= k; i++)
            {
                val *= n + 1 - i;
                val /= i;
            }
            return val;
        }

        inline bool isWithinZeroAndOne(float x)
        {
            return x >= -BEZIER_FUZZY_EPSILON && x <= (1.0 + BEZIER_FUZZY_EPSILON);
        }
    }

    template<size_t N>
    class BinomialCoefficients
    {
    public:
        BinomialCoefficients()
        {
            size_t center = N / 2;
            size_t k = 0;

            while (k <= center)
            {
                mCoefficients[k] = Math::binomial(N, k);
                k++;
            }

            // Utilize the symmetrical nature of the binomial coefficients.
            while (k <= N)
            {
                mCoefficients[k] = mCoefficients[N - k];
                k++;
            }
        }

        static constexpr size_t size()
        {
            return N + 1;
        }

        const size_t operator [](size_t idx) const
        {
            assert(idx < size());
            return mCoefficients[idx];
        }

    private:
        size_t mCoefficients[size()]{0};
    };

    struct PolynomialPair
    {
        size_t t = 0;
        size_t one_minus_t = 0;

        float valueAt(float t) const
        {
            return float(pow(1.0f - t, one_minus_t) * pow(t, float(this->t)));
        }
    };

    template<size_t N>
    class PolynomialCoefficients
    {
    public:
        PolynomialCoefficients()
        {
            for (size_t i = 0; i <= N; i++)
            {
                mPolynomialPairs[i].t = i;
                mPolynomialPairs[i].one_minus_t = N - i;
                assert(mPolynomialPairs[i].t + mPolynomialPairs[i].one_minus_t == N);
            }
        }

        float valueAt(size_t pos, float t) const
        {
            assert(pos < size());
            return mPolynomialPairs[pos].valueAt(t);
        }

        static constexpr size_t size()
        {
            return N + 1;
        }

        const PolynomialPair& operator [](size_t idx) const
        {
            assert(idx < size());
            return mPolynomialPairs[idx];
        }

    private:
        PolynomialPair mPolynomialPairs[size()];
    };

    class Vec2
    {
    public:
        Vec2()
            : x(0)
            , y(0)
        {}

        Vec2(float x, float y)
            : x(x)
            , y(y)
        {}

        Vec2(float x, float y, bool normalize)
            : x(x)
            , y(y)
        {
            if (normalize)
                this->normalize();
        }

        Vec2(const Vec2& other)
            : x(other.x)
            , y(other.y)
        {}

        Vec2(const Vec2& other, bool normalize)
            : Vec2(other.x, other.y, normalize)
        {}

        void set(float x, float y)
        {
            this->x = x;
            this->y = y;
        }

        void set(const Vec2& other)
        {
            this->x = other.x;
            this->y = other.y;
        }

        float length() const
        {
            return sqrt(x*x + y*y);
        }

        void normalize()
        {
            float len = length();
            x /= len;
            y /= len;
        }

        void translate(float dx, float dy)
        {
            x += dx;
            y += dy;
        }

        void translate(const Vec2& distance)
        {
            x += distance.x;
            y += distance.y;
        }

        void rotate(float angle, const Vec2& pivot = Vec2(0, 0))
        {
            float s = sin(angle);
            float c = cos(angle);

            x -= pivot.x;
            y -= pivot.y;

            float xnew = x * c - y * s;
            float ynew = x * s + y * c;

            x = xnew + pivot.x;
            y = ynew + pivot.y;
        }

        float angle() const
        {
            return atan2f(y, x);
        }

        float angleDeg() const
        {
            return angle() * 180.0f / Math::PI;
        }

        float operator[](size_t axis) const
        {
            assert(axis < Vec2::size);
            switch (axis)
            {
                case 0:
                    return x;
                case 1:
                    return y;
                default:
                    return 0;
            }
        }

        float& operator[](size_t axis)
        {
            assert(axis < Vec2::size);
            switch (axis)
            {
                case 0:
                    return x;
                case 1:
                    return y;
                default:
                    return x;
            }
        }

        Vec2 operator+(const Vec2& other) const
        {
            return Vec2(x + other.x, y + other.y);
        }

        Vec2 operator-(const Vec2& other) const
        {
            return Vec2(x - other.x, y - other.y);
        }

        Vec2 operator-() const
        {
            return Vec2(-x, -y);
        }

        Vec2 operator*(float scale) const
        {
            return Vec2(x * scale, y * scale);
        }

        Vec2 operator/(float scale) const
        {
            return Vec2(x / scale, y / scale);
        }

        Vec2 operator/(const Vec2& other) const
        {
            return Vec2(x / other.x, y / other.y);
        }

        bool fuzzyEquals(const Vec2& other) const
        {
            bool equals = true;
            for (size_t axis = 0; axis < Vec2::size; axis++)
            {
                if (fabs((*this)[axis] - other[axis]) >= BEZIER_FUZZY_EPSILON)
                {
                    equals = false;
                    break;
                }
            }
            return equals;
        }

        bool isWithinZeroAndOne() const
        {
            return Math::isWithinZeroAndOne(x) && Math::isWithinZeroAndOne(y);
        }

        float x;
        float y;
        static constexpr size_t size = 2;
    };

    typedef Vec2 Point;
    typedef Vec2 Normal;
    typedef Vec2 Tangent;

    struct ExtremeValue
    {
        ExtremeValue(float t, size_t axis)
            : t(t)
            , axis(axis)
        {}

        bool fuzzyEquals(const ExtremeValue& other) const
        {
            return axis == other.axis && fabs(t - other.t) < BEZIER_FUZZY_EPSILON;
        }

        const float t;
        const size_t axis;
    };

    class ExtremeValues
    {
    public:
        bool add(float t, size_t axis)
        {
            return add(ExtremeValue(t, axis));
        }

        bool add(const ExtremeValue& val)
        {
            assert(Math::isWithinZeroAndOne(val.t));
            for (auto const &v : values)
            {
                if (val.fuzzyEquals(v))
                    return false;
            }
            values.push_back(val);
            return true;
        }

        size_t size() const
        {
            return values.size();
        }

        ExtremeValue& operator[](size_t idx)
        {
            assert(idx < values.size());
            return values[idx];
        }

        ExtremeValue operator[](size_t idx) const
        {
            assert(idx < values.size());
            return values[idx];
        }

    private:
        std::vector<ExtremeValue> values;
    };

    class ExtremePoints
    {
    public:
        bool add(float x, float y)
        {
            return add(Point(x, y));
        }

        bool add(const Point& extremePoint)
        {
            for (auto const &ep : points)
            {
                if (extremePoint.fuzzyEquals(ep))
                    return false;
            }
            points.push_back(extremePoint);
            return true;
        }

        size_t size() const
        {
            return points.size();
        }

        bool empty() const
        {
            return !size();
        }

        Point& operator[](size_t idx)
        {
            assert(idx < size());
            return points[idx];
        }

        Point operator[](size_t idx) const
        {
            assert(idx < size());
            return points[idx];
        }

    private:
        std::vector<Point> points;
    };

    class AxisAlignedBoundingBox
    {
    public:
        AxisAlignedBoundingBox(const Point& p0, const Point& p1, const Point& p2, const Point& p3)
            : points{{p0}, {p1}, {p2}, {p3}}
        {}

        AxisAlignedBoundingBox(const ExtremePoints& xPoints)
        {
            float minX = std::numeric_limits<float>::max();
            float maxX = -std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float maxY = -std::numeric_limits<float>::max();

            for (size_t i = 0; i < xPoints.size(); i++)
            {
                if (xPoints[i].x > maxX)
                    maxX = xPoints[i].x;
                if (xPoints[i].x < minX)
                    minX = xPoints[i].x;
                if (xPoints[i].y > maxY)
                    maxY = xPoints[i].y;
                if (xPoints[i].y < minY)
                    minY = xPoints[i].y;
            }

            points[0].set(minX, minY);
            points[1].set(minX, maxY);
            points[2].set(maxX, maxY);
            points[3].set(maxX, minY);
        }

        static constexpr size_t size()
        {
            return 4;
        }

        float minX() const
        {
            return points[0].x;
        }

        float maxX() const
        {
            return points[2].x;
        }

        float minY() const
        {
            return points[0].y;
        }

        float maxY() const
        {
            return points[2].y;
        }

        float width() const
        {
            return maxX() - minX();
        }

        float height() const
        {
            return maxY() - minY();
        }

        float area() const
        {
            return width() * height();
        }

        Point& operator[](size_t idx)
        {
            assert(idx < size());
            return points[idx];
        }

        Point operator[](size_t idx) const
        {
            assert(idx < size());
            return points[idx];
        }

    private:
        Point points[4]; // Starting in lower left corner, going clock-wise.
    };

    typedef AxisAlignedBoundingBox AABB;

    class TightBoundingBox
    {
    public:
        // Takes the ExtremePoints of the Bezier curve moved to origo and rotated to align the x-axis
        // as arguments as well as the translation/rotation used to calculate it.
        TightBoundingBox(const ExtremePoints& xPoints, const Vec2& translation, float rotation)
        {
            float minX = std::numeric_limits<float>::max();
            float maxX = -std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float maxY = -std::numeric_limits<float>::max();

            for (size_t i = 0; i < xPoints.size(); i++)
            {
                if (xPoints[i].x > maxX)
                    maxX = xPoints[i].x;
                if (xPoints[i].x < minX)
                    minX = xPoints[i].x;
                if (xPoints[i].y > maxY)
                    maxY = xPoints[i].y;
                if (xPoints[i].y < minY)
                    minY = xPoints[i].y;
            }

            points[0].set(minX, minY);
            points[1].set(minX, maxY);
            points[2].set(maxX, maxY);
            points[3].set(maxX, minY);

            if (xPoints.empty())
                return;

            for (size_t i = 0; i < 4; i++)
            {
                points[i].rotate(-rotation);
                points[i].translate(-translation);
            }
        }

        static constexpr size_t size()
        {
            return 4;
        }

        float minX() const
        {
            return std::min({points[0].x, points[1].x, points[2].x, points[3].x});
        }

        float maxX() const
        {
            return std::max({points[0].x, points[1].x, points[2].x, points[3].x});
        }

        float minY() const
        {
            return std::min({points[0].y, points[1].y, points[2].y, points[3].y});
        }

        float maxY() const
        {
            return std::max({points[0].y, points[1].y, points[2].y, points[3].y});
        }

        float area() const
        {
            return width() * height();
        }

        // Uses the two first points to calculate the "width".
        float width() const
        {
            float x = points[1].x - points[0].x;
            float y = points[1].y - points[0].y;
            return sqrt(x * x + y * y);
        }

        // Uses the second and third points to calculate the "height".
        float height() const
        {
            float x = points[2].x - points[1].x;
            float y = points[2].y - points[1].y;
            return sqrt(x * x + y * y);
        }

        Point& operator[](size_t idx)
        {
            assert(idx < size());
            return points[idx];
        }

        Point operator[](size_t idx) const
        {
            assert(idx < size());
            return points[idx];
        }

    private:
        Point points[4]; // The points are ordered in a clockwise manner.
    };

    typedef TightBoundingBox TBB;

    template <size_t N>
    class Bezier
    {
    public:
        template <size_t M>
        struct Split
        {
            Split(const Point *l, const Point *r)
                : left(l, M+1)
                , right(r, M+1)
            {}

            Bezier<M> left;
            Bezier<M> right;
        };

    public:
        Bezier()
        {
            for (size_t i = 0; i < N+1; i++)
                mControlPoints[i].set(0, 0);
        }

        Bezier(const std::vector<Point>& controlPoints)
        {
            assert(controlPoints.size() == size()); // The Bezier curve must be initialized with the expected number og points
            for (size_t i = 0; i < controlPoints.size(); i++)
                mControlPoints[i] = Point(controlPoints[i]);
        }

        Bezier(const Point *points, size_t size)
        {
            assert(size == N+1);
            for (size_t i = 0; i < size; i++)
                mControlPoints[i] = points[i];
        }

        Bezier(const Bezier<N>& other)
        {
            for (size_t i = 0; i < other.size(); i++)
                mControlPoints[i] = Point(other[i]);
        }

        // The order of the bezier curve.
        size_t order() const
        {
            return N;
        }

        // Number of control points.
        size_t size() const
        {
            return N + 1;
        }

        Bezier<N-1> derivative() const
        {
            assert(N != 0);

            // Note: derivative weights/control points are not actual control points.
            std::vector<Point> derivativeWeights(N);
            for (size_t i = 0; i < N; i++)
                derivativeWeights[i].set(Point((mControlPoints[i+1] - mControlPoints[i]) * N));

            return Bezier<N-1>(derivativeWeights);
        }

    public:
        float valueAt(float t, size_t axis) const
        {
            assert(axis < Vec2::size); // Currently only support 2D
            float sum = 0;
            for (size_t n = 0; n < N+1; n++)
            {
                sum += binomialCoefficients[n] * polynomialCoefficients[n].valueAt(t) * mControlPoints[n][axis];
            }
            return sum;
        }

        Point valueAt(float t) const
        {
            Point p;
            for (size_t i = 0; i < Point::size; i++)
            {
                p[i] = (float) valueAt(t, i);
            }
            return p;
        }

        Tangent tangentAt(float t, bool normalize = true) const
        {
            Point p;
            Bezier<N-1> derivative = this->derivative();
            p.set(derivative.valueAt(t));
            if (normalize)
                p.normalize();
            return p;
        }

        Normal normalAt(float t, bool normalize = true) const
        {
            Point tangent = tangentAt(t, normalize);
            return Normal(-tangent.y, tangent.x, normalize);
        }

        void translate(const Vec2& distance)
        {
            for (size_t i = 0; i < N+1; i++)
            {
                mControlPoints[i].translate(distance);
            }
        }

        void translate(float dx, float dy)
        {
            for (size_t i = 0; i < N+1; i++)
            {
                mControlPoints[i].translate(dx, dy);
            }
        }

        void rotate(float angle, Vec2 pivot = Vec2(0, 0))
        {
            for (size_t i = 0; i < N+1; i++)
            {
                mControlPoints[i].rotate(angle, pivot);
            }
        }

        // Note: This is a brute force length calculation. If more precision is needed,
        // use something like https://pomax.github.io/bezierinfo/#arclength
        float length(size_t intervals = 100) const
        {
            float length = 0.0f;

            if (intervals > 0)
            {
                float t = 0.0f;
                const float dt = 1.0f / (float)intervals;

                Point p1 = valueAt(t);
                Point p2;

                for (size_t i = 0; i < intervals; i++)
                {
                    p2 = valueAt(t + dt);
                    float x = p2.x - p1.x;
                    float y = p2.y - p1.y;
                    length += sqrt(x * x + y * y);
                    p1.set(p2);
                    t += dt;
                }
            }

            return length;
        }

        Split<N> split(float t) const
        {
            Point l[N+1];
            Point r[N+1];
            l[0] = mControlPoints[0];
            r[0] = mControlPoints[N];

            std::array<Point, N+1> prev = mControlPoints;
            std::array<Point, N+1> curr;

            // de Casteljau: https://pomax.github.io/bezierinfo/#splitting
            int subs = 0;
            while (subs < N)
            {
                for (size_t i = 0; i < N - subs; i++)
                {
                    curr[i].x = (1.0f - t) * prev[i].x + t * prev[i + 1].x;
                    curr[i].y = (1.0f - t) * prev[i].y + t * prev[i + 1].y;
                    if (i == 0)
                        l[subs+1].set(curr[i]);
                    if (i == (N - subs - 1))
                        r[subs+1].set(curr[i]);
                }
                std::swap(prev, curr);
                subs++;
            }

            return Split<N>(l, r);
        }

        Split<N> split() const
        {
            return split(0.5f);
        }

        float archMidPoint(const float epsilon = 0.001f, const size_t maxDepth = 100) const
        {
            float t = 0.5f;
            float s = 0.5f; // Binary search split value

            size_t iter = 0;
            while (iter < maxDepth)
            {
                auto split = this->split(t);
                float low  = split.left.length();
                float high = split.right.length();
                float diff = low - high;

                if (std::abs(diff) <= epsilon)
                {
                    break;
                }

                s *= 0.5f;
                t += (diff > 0 ? -1 : 1) * s;
                iter++;
            }

            return t;
        }

        ExtremeValues derivativeZero(size_t intervals = BEZIER_DEFAULT_INTERVALS,
                                     float epsilon = BEZIER_FUZZY_EPSILON,
                                     size_t maxIterations = BEZIER_DEFAULT_MAX_ITERATIONS) const
        {
            switch (N)
            {
                case 1:
                    return derivativeZero1();
                case 2:
                    return derivativeZero2();
                case 3:
//                    return derivativeZero3();
                    return newtonRhapson(intervals, epsilon, maxIterations);
                default:
                    return newtonRhapson(intervals, epsilon, maxIterations);
            }
        }

        ExtremePoints extremePoints() const
        {
            ExtremeValues xVals = derivativeZero();
            xVals.add(0.0f, 0);
            xVals.add(1.0f, 0);

            ExtremePoints xPoints;
            for (size_t i = 0; i < xVals.size(); i++)
                xPoints.add(valueAt(xVals[i].t));

            return xPoints;
        }

        AxisAlignedBoundingBox aabb() const
        {
            return AxisAlignedBoundingBox(extremePoints());
        }

        AxisAlignedBoundingBox aabb(const ExtremePoints& xPoints) const
        {
            return AxisAlignedBoundingBox(xPoints);
        }

        TightBoundingBox tbb() const
        {
            Bezier<N> bezier = *this;

            // Translate last control point (highest order) to origo.
            Vec2 translation(-bezier[N]);
            bezier.translate(translation);

            // Rotate bezier to align the first control point (lowest order) with the x-axis
            float angle = -bezier[0].angle();
            bezier.rotate(angle);

            return TightBoundingBox(bezier.extremePoints(), translation, angle);
        }

    public:
        Point& operator [](size_t idx)
        {
            assert(idx < size());
            return mControlPoints[idx];
        }

        Point operator [](size_t idx) const
        {
            assert(idx < size());
            return mControlPoints[idx];
        }

    private:
        ExtremeValues derivativeZero1() const
        {
            assert(N == 1);
            return ExtremeValues();
        }

        ExtremeValues derivativeZero2() const
        {
            assert(N == 2);
            ExtremeValues xVals;
            Point roots = (mControlPoints[0] - mControlPoints[1]) / (mControlPoints[0] - mControlPoints[1] * 2 + mControlPoints[2]);
            if (Math::isWithinZeroAndOne(roots[0]))
                xVals.add(roots[0], 0);
            if (Math::isWithinZeroAndOne(roots[1]))
                xVals.add(roots[1], 1);
            return xVals;
        }

        ExtremeValues derivativeZero3() const
        {
            // Note: NOT IMPLMENTED YET
            assert(N == 3);
            return ExtremeValues();
        }

        ExtremeValues newtonRhapson(size_t intervals = BEZIER_DEFAULT_INTERVALS,
                                    float epsilon = BEZIER_FUZZY_EPSILON,
                                    size_t maxIterations = BEZIER_DEFAULT_MAX_ITERATIONS) const
        {
            assert(N >= 2);
            ExtremeValues xVals;
            const float dt = 1.0f / (float) intervals;
            const float absEpsilon = fabs(epsilon);
            const Bezier<N-1> db = derivative();
            const Bezier<N-2> ddb = db.derivative();

            for (size_t i = 0; i < Point::size; i++)
            {
                float t = 0;

                while(t <= 1.0)
                {
                    float zeroVal = t;
                    size_t current_iter = 0;

                    while (current_iter < maxIterations)
                    {
                        float dbVal = db.valueAt(zeroVal, i);
                        float ddbVal = ddb.valueAt(zeroVal, i);
                        float nextZeroVal = zeroVal - (dbVal / ddbVal);

                        if (fabs(nextZeroVal - zeroVal) < absEpsilon)
                        {
                            if (Math::isWithinZeroAndOne(nextZeroVal))
                            {
                                xVals.add(nextZeroVal, i);
                                break;
                            }
                        }

                        zeroVal = nextZeroVal;
                        current_iter++;
                    }

                    t += dt;
                }
            }

            return xVals;
        }

    public:
        static const BinomialCoefficients<N> binomialCoefficients;
        static const PolynomialCoefficients<N> polynomialCoefficients;

    private:
        std::array<Point, N+1> mControlPoints;
    };

    template<size_t N>
    const BinomialCoefficients<N> Bezier<N>::binomialCoefficients = BinomialCoefficients<N>();

    template<size_t N>
    const PolynomialCoefficients<N> Bezier<N>::polynomialCoefficients = PolynomialCoefficients<N>();

} // namespace Bezier
