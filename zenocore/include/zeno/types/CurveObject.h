#pragma once


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

    enum HANDLE_TYPE
    {
        HDL_FREE,
        HDL_ALIGNED,
        HDL_VECTOR,
        HDL_ASYM
    };

    struct ControlPoint {
        float v{0};
        PointType cp_type{PointType::kConstant};
        vec2f left_handler{0, 0};
        vec2f right_handler{0, 0};
        HANDLE_TYPE controlType = HDL_VECTOR;
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
    bool visible = false;
    bool timeline = true;

    void addPoint(float f, float v, PointType cp_type, vec2f left_handler, vec2f right_handler, HANDLE_TYPE hdl_type) {
        cpbases.push_back(f);
        cpoints.push_back({v, cp_type, left_handler, right_handler, hdl_type});
    }

    void updateRange(Range const &newRg) {
        for (auto &x: cpbases) {
            x = mix(newRg.xFrom, newRg.xTo, unmix(rg.xFrom, rg.xTo, x));
        }
        for (auto &cp: cpoints) {
            auto app = [&] (auto &y) {
                y = mix(newRg.yFrom, newRg.yTo, unmix(rg.yFrom, rg.yTo, y));
            };
            app(cp.v);
            app(cp.left_handler[0]);
            app(cp.left_handler[1]);
            app(cp.right_handler[0]);
            app(cp.right_handler[1]);
        }
        rg = newRg;
    }

    float eval(float cf) const {
        assert(!cpoints.empty());
        assert(cpbases.size() == cpoints.size());
        assert(cpbases.front() <= cpbases.back());
        if (cycleType != CycleType::kClamp) {
            auto delta = cpbases.back() - cpbases.front();
            if (cycleType == CycleType::kMirror) {
                int cd;
                if (delta != 0) {
                    cd = int(std::floor((cf - cpbases.front()) / delta)) & 1;
                    cf = std::fmod(cf - cpbases.front(), delta);
                } else {
                    cd = 0;
                    cf = 0;
                }
                if (cd != 0) {
                    if (cf < 0) {
                        cf = -cf;
                    } else {
                        cf = delta - cf;
                    }
                }
                if (cf < 0)
                    cf = cpbases.back() + cf;
                else
                    cf = cpbases.front() + cf;
            } else {
                if (delta != 0)
                    cf = std::fmod(cf - cpbases.front(), delta);
                else
                    cf = 0;
                if (cf < 0)
                    cf = cpbases.back() + cf;
                else
                    cf = cpbases.front() + cf;
            }
        }
        auto moreit = std::lower_bound(cpbases.begin(), cpbases.end(), cf);
        if (moreit == cpbases.end())
            return cpoints.back().v;
        else if (moreit == cpbases.begin())
            return cpoints.front().v;
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

    bool operator==(const CurveData& other) const {
        if (other.cpbases.size() != cpbases.size() || other.cpoints.size() != cpoints.size())
            return false;
        if (other.cycleType != cycleType || other.visible != visible || other.timeline != timeline)
            return false;
        for (int i = 0; i < other.cpbases.size(); i++) {
            if (other.cpbases[i] != cpbases[i])
                return false;
        }
        for (int i = 0; i < other.cpoints.size(); i++) {
            if (other.cpoints[i].v != cpoints[i].v || other.cpoints[i].cp_type != cpoints[i].cp_type ||
                other.cpoints[i].controlType != cpoints[i].controlType || 
                other.cpoints[i].left_handler[0] != cpoints[i].left_handler[0] || other.cpoints[i].left_handler[1] != cpoints[i].left_handler[1] ||
                other.cpoints[i].right_handler[0] != cpoints[i].right_handler[0] || other.cpoints[i].right_handler[1] != cpoints[i].right_handler[1]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const CurveData& rhs) const {
        return !operator==(rhs);
    }
};

struct CurvesData {
    std::map<std::string, CurveData> keys;

    bool empty() const {
        return keys.empty();
    }

    size_t size() const {
        return keys.size();
    }

    bool contains(const std::string& key) const {
        return keys.find(key) != keys.end();
    }

    auto getEvaluator(std::string const& key) const {
        return [&data = keys.at(key)](float x) {
            return data.eval(x);
        };
    }

    CurveData operator[](std::string const& key) {
        if (!contains(key)) return CurveData();
        return keys[key];
    }

    bool operator==(const CurvesData& other) {
        if (other.keys.size() != keys.size())
            return false;
        for (auto& [otherKey, otherCurve]: other.keys) {
            const auto& iter = keys.find(otherKey);
            if (iter == keys.end()) {
                return false;
            }else if (!(iter->second == otherCurve)){
                return false;
            }
        }
        return true;
    }

    template <class ...Ts>
    void addPoint(std::string const& key, Ts ...ts) {
        return keys[key].addPoint(ts...);
    }

    float eval(std::string const& key, float x) const {
        return keys.at(key).eval(x);
    }

    float eval(float x) const {
        return eval("x", x);
    }

    vec2f eval(vec2f v) const {
        auto& [x, y] = v;
        return { eval("x", x), eval("y", y) };
    }

    vec3f eval(vec3f v) const {
        auto& [x, y, z] = v;
        return { eval("x", x), eval("y", y), eval("z", z) };
    }

    vec4f eval(vec4f v) const {
        auto& [x, y, z, w] = v;
        return { eval("x", x), eval("y", y), eval("z", z), eval("w", w) };
    }
};

struct BCurveObject {
    std::vector<zeno::vec3f> points;
    float precision = 0.0;
    std::vector<zeno::vec3f> bPoints;
    std::string sampleTag = "";
    std::string SampleAttr = "";

};

}
