#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/extra/GlobalState.h>
#include <sstream>
#include <iostream>
#include <cmath>
#include <map>

namespace zeno { // credits by zhouhang
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

        // correctness modify: make bezier to be function
        std::vector<std::tuple<vec2f, vec2f>> correct_handlers() {
            std::vector<std::tuple<vec2f, vec2f>> _handlers = handlers;
            for (int i = 0; i < points.size(); i++) {
                vec2f cur_p = points[i];
                if (i != 0) {
                    vec2f hp = std::get<0>(handlers[i]);
                    vec2f prev_p = points[i-1];
                    if (std::abs(hp[0]) > std::abs(cur_p[0] - prev_p[0])) {
                        float s = std::abs(cur_p[0] - prev_p[0]) / std::abs(hp[0]);
                        handlers[i] = std::make_tuple(hp * s, std::get<1>(handlers[i]));
                    }

                }
                if (i != points.size() - 1) {
                    vec2f hn = std::get<1>(handlers[i]);
                    vec2f next_p = points[i+1];
                    if (std::abs(hn[0]) > std::abs(cur_p[0] - next_p[0])) {
                        float s = std::abs(cur_p[0] - next_p[0]) / std::abs(hn[0]);
                        handlers[i] = std::make_tuple(std::get<0>(handlers[i]), hn * s);
                    }
                }
            }
            return _handlers;
        }

        std::vector<std::tuple<vec2f, vec2f>> handersToPoints(std::vector<std::tuple<vec2f, vec2f>> _handlers) const {
            std::vector<std::tuple<vec2f, vec2f>> ps;
            for (int i = 0; i < points.size(); i++) {
                ps.emplace_back(
                    points[i] + std::get<0>(_handlers[i]),
                    points[i] + std::get<1>(_handlers[i])
                );
            }
            return ps;
        }

        std::vector<std::tuple<vec2f, vec2f>> m_handlersToPoints;

        void prepare() {
            m_handlersToPoints = handersToPoints(correct_handlers());
        }

        float eval(float x) const {
            x = ratio(input_min, input_max, x);
            float y = eval_value(points, m_handlersToPoints, x);
            return lerp(output_min, output_max, y);
        }

        template <class T>
        T generic_eval(T const &src) const {
            if constexpr (std::is_arithmetic_v<T>) {
                return eval(src);
            } else {
                T tmp{};
                for (int i = 0; i != src.size(); ++i) {
                    tmp[i] = eval(src[i]);
                }
                return tmp;
            }
        }

        template <class T>
        T generic_eval(T const &src, int sourceX, int sourceY, int sourceZ) const {
            if constexpr (std::is_arithmetic_v<T>) {
                return eval(src);
            } else {
                T tmp{
                    sourceX == -1 ? src[0] : eval(src[sourceX]),
                    sourceY == -1 ? src[1] : eval(src[sourceY]),
                    sourceZ == -1 ? src[2] : eval(src[sourceZ]),
                };
                return tmp;
            }
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
            curvemap->prepare();
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
                "deprecated"
            }
        }
    );
    struct Curvemap : zeno::INode {
        virtual void apply() override {
            auto curvemap = get_input<zeno::CurvemapObject>("curvemap");
            auto input = get_input<zeno::NumericObject>("value");
            auto res = std::make_shared<zeno::NumericObject>(input->value);

            std::visit([&](const auto &src) {
                res->value = curvemap->generic_eval(src);
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
                "deprecated",
            }
        }
    );
    struct PrimitiveCurvemap : zeno::INode {
        virtual void apply() override {
            auto curvemap = get_input<zeno::CurvemapObject>("curvemap");
            auto prim = get_input<zeno::PrimitiveObject>("prim");
            auto attrName = get_input<zeno::StringObject>("attrName")->get();
            auto sourceX = get_input<zeno::NumericObject>("sourceX")->get<int>();
            auto sourceY = get_input<zeno::NumericObject>("sourceY")->get<int>();
            auto sourceZ = get_input<zeno::NumericObject>("sourceZ")->get<int>();

            prim->attr_visit(attrName, [&] (auto &attr) {
                using T = std::decay_t<decltype(attr[0])>;
                if (std::is_arithmetic_v<T> || (sourceX == 0 && sourceY == 1 && sourceZ == 2)) {
#pragma omp parallel for
                    for (intptr_t i = 0; i < attr.size(); ++i) {
                        attr[i] = curvemap->generic_eval(attr[i]);
                    }
                } else {
#pragma omp parallel for
                    for (intptr_t i = 0; i < attr.size(); ++i) {
                        attr[i] = curvemap->generic_eval(attr[i], sourceX, sourceY, sourceZ);
                    }
                }
            });

            set_output("prim", std::move(prim));
        }
    };
ZENDEFNODE(PrimitiveCurvemap, {
    {
    {"PrimitiveObject", "prim"},
    "curvemap",
    {"string", "attrName", "pos"},
    {"int", "sourceX", "0"},
    {"int", "sourceY", "1"},
    {"int", "sourceZ", "2"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

    struct ControlPoint {
        int f;
        float v;
        std::string cp_type;
        zeno::vec2f left_handler;
        zeno::vec2f right_handler;
    };


    static zeno::vec4f query_cur_value(
        std::map<std::string, std::vector<ControlPoint>> keyframes_table,
        int cur_frame
    ) {
        zeno::vec4f v;
        std::array<std::string, 4> channel_names = {
            "x",
            "y",
            "z",
            "w",
        };
        for (size_t i = 0; i < 4; i++) {
            auto& keyframes = keyframes_table[channel_names[i]];
            std::vector<int> less;
            std::vector<int> more;
            for (auto it = keyframes.begin(); it != keyframes.end(); it++) {
                if (it->f <= cur_frame) {
                    less.push_back(it->f);
                } else {
                    more.push_back(it->f);
                }
            }
            if (less.size() == 0) {
                v[i] = 0;
            } else if (more.size() == 0 || less.back() == cur_frame) {
                v[i] = keyframes[less.size() -1].v;
            } else {
                ControlPoint p = keyframes[less.size() -1];
                ControlPoint n = keyframes[less.size()];
                float t = (float)(cur_frame - less.back()) / (more.front() - less.back());
                if (p.cp_type == "constant") {
                    v[i] = p.v;
                } else if (p.cp_type == "straight") {
                    v[i] = lerp(p.v, n.v, t);
                } else {
                    float x_scale = n.f - p.f;
                    float y_scale = n.v - p.v;

                    v[i] = eval_bezier_value(
                        zeno::vec2f(p.f, p.v),
                        zeno::vec2f(n.f, n.v),
                        zeno::vec2f(p.f, p.v) + p.right_handler,
                        zeno::vec2f(n.f, n.v) + n.left_handler,
                        (float)cur_frame
                    );

                }
            }
        }
        return v;

    }
    struct DynamicNumber : zeno::INode {
        virtual void apply() override {
            std::map<std::string, std::vector<ControlPoint>> keyframes_table;
            auto _tmp = get_param<std::string>("_TMP");
            auto _points = get_param<std::string>("_CONTROL_POINTS");
            zeno::vec4f v;
            if (_tmp.size() > 0) {
                std::stringstream ss(_tmp);
                float x = 0;
                float y = 0;
                float z = 0;
                float w = 0;
                ss >> x >> y >> z >> w;
                v = zeno::vec4f(x, y, z, w);

            } else {
                auto type = get_param<std::string>("type");
                std::stringstream ss(_points);
                int channel_count;
                ss >> channel_count;
                int max_frame = 0;
                for (int i = 0; i < channel_count; i++) {
                    std::string channel_name;
                    ss >> channel_name;
                    int frame_count = 0;
                    ss >> frame_count;
                    keyframes_table[channel_name] = std::vector<ControlPoint>();
                    for (int j = 0; j < frame_count; j++) {
                        ControlPoint cp;
                        ss >> cp.f;
                        ss >> cp.v;
                        ss >> cp.cp_type;
                        ss >> cp.left_handler[0];
                        ss >> cp.left_handler[1];
                        ss >> cp.right_handler[0];
                        ss >> cp.right_handler[1];
                        keyframes_table[channel_name].emplace_back(cp);
                        max_frame = cp.f > max_frame? cp.f : max_frame;
                    }
                }
                int cur_frame = has_input("frame") ? get_input<NumericObject>("frame")->get<int>() : getGlobalState()->frameid;
                if (cur_frame > max_frame) {
                    if (type == "zero") {
                        v = zeno::vec4f(0, 0, 0, 0);
                    } else if (type == "cycle") {
                        cur_frame = cur_frame % max_frame;
                        v = query_cur_value(keyframes_table, cur_frame);
                    } else {
                        v = query_cur_value(keyframes_table, cur_frame);
                    }
                } else {
                    v = query_cur_value(keyframes_table, cur_frame);
                }
            }

            set_output("x", std::make_shared<zeno::NumericObject>(v[0]));
            set_output("y", std::make_shared<zeno::NumericObject>(v[1]));
            set_output("z", std::make_shared<zeno::NumericObject>(v[2]));
            set_output("w", std::make_shared<zeno::NumericObject>(v[3]));
            set_output("vec3", std::make_shared<zeno::NumericObject>(zeno::vec3f(v[0], v[1], v[2])));
        }
    };
    ZENDEFNODE(
        DynamicNumber,
        {
            // inputs
            {
                "frame"
            },
            // outpus
            {
                "vec3",
                "x",
                "y",
                "z",
                "w",
            },
            // params
            {
                {
                    "enum 100 10 1 0.1 0.01 0.001",
                    "speed",
                    "1",
                },
                {
                    "floatslider",
                    "x",
                    "0",
                },
                {
                    "floatslider",
                    "y",
                    "0",
                },
                {
                    "floatslider",
                    "z",
                    "0",
                },
                {
                    "floatslider",
                    "w",
                    "0",
                },
                {
                    "enum clamp zero cycle",
                    "type",
                    "clamp",
                },
            },
            // category
            {
                "deprecated",
            }
        }
    );
}
