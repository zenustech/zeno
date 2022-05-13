#pragma once


#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>
#include <vector>
#include <tuple>
#include <cmath>


namespace zeno {

struct CurveObject : zeno::IObject {
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

    float eval_value(float x) const {
        if (x <= 0) {
            return 0;
        } else if (x >= 1) {
            return 1;
        }
        int i = -1;
        for (vec2f const &p: points) {
            if (p[0] < x) {
                i += 1;
            }
        }
        vec2f p1 = points[i];
        vec2f p2 = points[i + 1];
        vec2f h1 = std::get<1>(handlers[i]) + p1;
        vec2f h2 = std::get<0>(handlers[i+1]) + p2;
        return eval_bezier_value(p1, p2, h1, h2, x);
    }

    std::vector<zeno::vec2f> points;
    std::vector<std::tuple<zeno::vec2f, zeno::vec2f>> handlers;
    float input_min;
    float input_max;
    float output_min;
    float output_max;

    // correctness modify: make bezier to be function
    std::vector<std::tuple<vec2f, vec2f>> correct_handlers() {
        std::vector<std::tuple<vec2f, vec2f>> handlers = this->handlers;
        for (int i = 0; i < points.size(); i++) {
            vec2f cur_p = points[i];
            if (i != 0) {
                vec2f hp = std::get<0>(handlers[i]);
                vec2f prev_p = points[i-1];
                if (std::abs(hp[0]) > std::abs(cur_p[0] - prev_p[0])) {
                    float s = std::abs(cur_p[0] - prev_p[0]) / std::abs(hp[0]);
                    handlers[i] = {hp * s, std::get<1>(handlers[i])};
                }

            }
            if (i != points.size() - 1) {
                vec2f hn = std::get<1>(handlers[i]);
                vec2f next_p = points[i+1];
                if (std::abs(hn[0]) > std::abs(cur_p[0] - next_p[0])) {
                    float s = std::abs(cur_p[0] - next_p[0]) / std::abs(hn[0]);
                    handlers[i] = {std::get<0>(handlers[i]), hn * s};
                }
            }
        }
        return handlers;
    }

    float eval(float x) const {
        x = ratio(input_min, input_max, x);
        float y = eval_value(x);
        return lerp(output_min, output_max, y);
    }
};

}
