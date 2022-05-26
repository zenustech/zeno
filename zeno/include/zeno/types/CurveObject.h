#pragma once


#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>
#include <algorithm>
#include <cassert>
#include <vector>
#include <tuple>
#include <cmath>
#include <map>


namespace zeno {

struct _CurveDataDetails {
    static float ratio(float from, float to, float t) {
        return (t - from) / (to - from);
    }

    static float lerp(float from, float to, float t) {
        return from + (to - from) * t;
    }

    static vec2f lerp(vec2f from, vec2f to, float t) {
        return {lerp(from[0], to[0], t), lerp(from[1], to[1], t)};
    }

    static vec2f bezier(vec2f p1, vec2f p2, vec2f h1, vec2f h2, float t) {
        vec2f a = lerp(p1, h1, t);
        vec2f b = lerp(h1, h2, t);
        vec2f c = lerp(h2, p2, t);
        vec2f d = lerp(a, b, t);
        vec2f e = lerp(b, c, t);
        vec2f f = lerp(d, e, t);
        return f;
    }

    static float eval_bezier_value(vec2f p1, vec2f p2, vec2f h1, vec2f h2, float x) {
        float lower = 0;
        float upper = 1;
        float t = (lower + upper) / 2;
        vec2f np = bezier(p1, p2, h1, h2, t);
        int left_calc_count = 100;
        while (std::abs(np[0] - x) > 0.00001f && left_calc_count > 0) {
            if (x < np[0]) {
                upper = t;
            } else {
                lower = t;
            }
            t = (lower + upper) / 2;
            np = bezier(p1, p2, h1, h2, t);
            left_calc_count -= 1;
        }
        return np[1];
    }
};

struct CurveData : private _CurveDataDetails {
    enum PointType {
        kBezier = 0,
        kLinear,
        kConstant,
    };

    enum CycleType {
        kClamp = 0,
        kCycle,
        kMirror,
    };

    struct ControlPoint {
        float v{0};
        PointType cp_type{PointType::kConstant};
        vec2f left_handler{0, 0};
        vec2f right_handler{0, 0};
    };

    struct Range {
        float xFrom{0};
        float xTo{0};
        float yFrom{0};
        float yTo{0};
    };

    std::vector<float> cpbases;
    std::vector<ControlPoint> cpoints;
    Range rg;
    CycleType cycleType{CycleType::kClamp};

    void addPoint(float f, float v, PointType cp_type, vec2f left_handler, vec2f right_handler) {
        cpbases.push_back(f);
        cpoints.push_back({v, cp_type, left_handler, right_handler});
    }

    float eval(float cf) const {
        assert(!cpoints.empty());
        assert(cpbases.size() == cpoints.size());
        auto moreit = std::lower_bound(cpbases.begin(), cpbases.end(), cf);
        if (cycleType != CycleType::kClamp) {
            cf -= cpbases.front();
            cf = std::fmod(cf - cpbases.front(), cpbases.back() - cpbases.front());
            if (cycleType == CycleType::kCycle)
                cf = cpbases.front() + cf;
            else  // CycleType::kMirror
                cf = cpbases.back() - cf;
        }
        if (moreit == cpbases.end())
            return cpbases.back();
        else if (moreit == cpbases.begin())
            return cpbases.front();
        auto lessit = std::prev(moreit);
        ControlPoint p = cpoints[lessit - cpbases.begin()];
        ControlPoint n = cpoints[moreit - cpbases.begin()];
        float pf = *lessit;
        float nf = *moreit;
        float t = (cf - pf) / (nf - pf);
        if (p.cp_type == PointType::kBezier) {
            float x_scale = nf - pf;
            float y_scale = n.v - p.v;
            return eval_bezier_value(
                vec2f(pf, p.v),
                vec2f(nf, n.v),
                vec2f(pf, p.v) + p.right_handler,
                vec2f(nf, n.v) + n.left_handler,
                cf);
        } else if (p.cp_type == PointType::kLinear) {
            return lerp(p.v, n.v, t);
        } else {  // PointType::kConstant
            return p.v;
        }
    }
};

struct CurveObject : IObjectClone<CurveObject> {
    std::map<std::string, CurveData> keys;

    auto getEvaluator(std::string const &key) const {
        return [&data = keys.at(key)] (float x) {
            return data.eval(x);
        };
    }

    template <class ...Ts>
    void addPoint(std::string const &key, Ts ...ts) {
        return keys[key].addPoint(ts...);
    }

    float eval(std::string const &key, float x) const {
        return getEvaluator(key)(x);
    }

    float eval(float x) const {
        return eval("x", x);
    }

    vec2f eval(vec2f v) const {
        auto &[x, y] = v;
        return {eval("x", x), eval("y", y)};
    }

    vec3f eval(vec3f v) const {
        auto &[x, y, z] = v;
        return {eval("x", x), eval("y", y), eval("z", z)};
    }

    vec4f eval(vec4f v) const {
        auto &[x, y, z, w] = v;
        return {eval("x", x), eval("y", y), eval("z", z), eval("w", w)};
    }
};

}
