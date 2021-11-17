#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <sstream>
#include <iostream>
#include <cmath>

namespace zeno {
    using zeno::vec2f;
    static float epsilon = 0.00001;

    static float ratio(float from, float to, float t) {
        return (t - from) / (to - from);
    }
    static float lerp(float from, float to, float t) {
        return from + (to - from) * t;
    }

    static vec2f lerp(vec2f from, vec2f to, float t) {
        return vec2f(
            lerp(from[0], to[0], t),
            lerp(from[1], to[1], t)
        );
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
        while (std::abs(np[0] - x) > epsilon && left_calc_count > 0) {
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

    static float eval_value(const std::vector<vec2f>& points, const std::vector<std::tuple<vec2f, vec2f>>& handlers, float x) {
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
        vec2f h1 = std::get<1>(handlers[i]);
        vec2f h2 = std::get<0>(handlers[i+1]);
        return eval_bezier_value(p1, p2, h1, h2, x);
    }

    struct CurvemapObject : zeno::IObject {
        std::vector<zeno::vec2f> points;
        std::vector<std::tuple<zeno::vec2f, zeno::vec2f>> handlers;
        float input_min;
        float input_max;
        float output_min;
        float output_max;
        float eval(float x) {
            x = ratio(input_min, input_max, x);
            float y = eval_value(points, handlers, x);
            return lerp(output_min, output_max, y);
        }
    };
    struct MakeCurvemap : zeno::INode {
        virtual void apply() override {
            auto curvemap = std::make_shared<zeno::CurvemapObject>();
            auto _points = get_param<std::string>("_POINTS");
            std::stringstream ss(_points);
            int count;
            ss >> count;
            for (int i = 0; i < count; i++) {
                float x = 0;
                float y = 0;
                ss >> x >> y;
                curvemap->points.emplace_back(x, y);
            }
            auto _handlers = get_param<std::string>("_HANDLERS");
            std::stringstream hss(_handlers);
            for (int i = 0; i < count; i++) {
                float x0 = 0;
                float y0 = 0;
                float x1 = 0;
                float y1 = 0;
                hss >> x0 >> y0 >> x1 >> y1;
                curvemap->handlers.emplace_back(
                    zeno::vec2f(x0, y0),
                    zeno::vec2f(x1, y1)
                );
            }
            curvemap->input_min = get_param<float>("input_min");
            curvemap->input_max = get_param<float>("input_max");
            curvemap->output_min = get_param<float>("output_min");
            curvemap->output_max = get_param<float>("output_max");
            set_output("curvemap", std::move(curvemap));
        }
    };
    ZENDEFNODE(
        MakeCurvemap,
        {
            // inputs
            {
            },
            // outpus
            {
                "curvemap",
            },
            // params
            {
                {
                    "float",
                    "input_min",
                    "0",
                },
                {
                    "float",
                    "input_max",
                    "1",
                },
                {
                    "float",
                    "output_min",
                    "0",
                },
                {
                    "float",
                    "output_max",
                    "1",
                },
            },
            // category
            {
                "numeric",
            }
        }
    );
    struct Curvemap : zeno::INode {
        virtual void apply() override {
            auto curvemap = get_input<zeno::CurvemapObject>("curvemap");
            auto input = get_input<zeno::NumericObject>("value");
            auto res = std::make_shared<zeno::NumericObject>(input->value);

            std::visit([&r = res->value, &curvemap](const auto &src) {
                using T = std::remove_cv_t<std::remove_reference_t<decltype(src)>>;

                T tmp{};
                if constexpr (std::is_arithmetic_v<T>) {
                    tmp = curvemap->eval(src);
                } else {
                    for (int i = 0; i != src.size(); ++i) {
                        tmp[i] = curvemap->eval(src[i]);
                    }
                }
                r = tmp;
            }, input->value);

            set_output("res", std::move(res));
        }
    };
    ZENDEFNODE(
        Curvemap,
        {
            // inputs
            {
                "curvemap",
                "value",
            },
            // outpus
            {
                "res",
            },
            // params
            {
            },
            // category
            {
                "numeric",
            }
        }
    );
}