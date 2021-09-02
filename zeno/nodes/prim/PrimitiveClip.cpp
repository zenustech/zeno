#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <unordered_map>

namespace zeno {
    struct PrimitiveClip : zeno::INode {
        static zeno::vec3f line_plane_intersection(const zeno::vec3f& plane_point, const zeno::vec3f& plane_normal, const zeno::vec3f& line_point, const zeno::vec3f& line_direction) {
            float t = (dot(plane_normal, plane_point) - dot(plane_normal, line_point)) / dot(plane_normal, line_direction);
            return line_point + line_direction * t;
        }

        struct hash_pair {
            template <class T1, class T2>
            size_t operator()(const std::pair<T1, T2>& p) const {
                auto hash1 = std::hash<T1>{}(p.first);
                auto hash2 = std::hash<T2>{}(p.second);
                return hash1 ^ hash2;
            }
        };

        template<typename T>
        static void append_element(const T& ref_element, std::vector<T>& element_arr, 
                std::unordered_map<int32_t, int32_t>& point_map) {
            T new_element;
            for (size_t i = 0; i < new_element.size(); ++i) {
                new_element[i] = point_map[ref_element[i]];
            }
            element_arr.emplace_back(new_element);
        }

        static void clip_points(PrimitiveObject* outprim, const PrimitiveObject* refprim,
                std::unordered_map<int32_t, int32_t>& point_map,
                const std::vector<bool>& is_above_arr) {
            for (size_t i = 0; i < refprim->points.size(); ++i) {
                const int32_t ref_point = refprim->points[i];
                if (!is_above_arr[ref_point]) {
                    outprim->points.emplace_back(point_map[ref_point]);
                }
            }
        }

        static void clip_lines(PrimitiveObject* outprim, const PrimitiveObject* refprim,
                std::vector<zeno::vec3f>& new_pos_attr,
                const std::vector<zeno::vec3f>& ref_pos_attr,
                std::unordered_map<int32_t, int32_t>& point_map,
                const std::vector<bool>& is_above_arr,
                const zeno::vec3f& origin,
                const zeno::vec3f& direction) {
            for (size_t line_idx = 0; line_idx < refprim->lines.size(); ++line_idx) {
                const zeno::vec2i& line_points = refprim->lines[line_idx];
                if (!is_above_arr[line_points[0]] && !is_above_arr[line_points[1]]) {
                    append_element<zeno::vec2i>(line_points, outprim->lines, point_map);
                }
                else if (is_above_arr[line_points[0]] ^ is_above_arr[line_points[1]]) {
                    const size_t above_idx = is_above_arr[line_points[0]] ? 0 : 1;
                    const int32_t above_point = line_points[above_idx];
                    const int32_t below_point = line_points[1 - above_idx];

                    const zeno::vec3f& pos1 = ref_pos_attr[below_point];
                    const zeno::vec3f& pos2 = ref_pos_attr[above_point];
                    const zeno::vec3f new_pos = line_plane_intersection(origin, direction, pos1, normalize(pos2 - pos1));
                    const int32_t new_point1 = new_pos_attr.size();
                    const int32_t new_point2 = point_map[below_point];
                    new_pos_attr.emplace_back(new_pos);
                    outprim->lines.emplace_back(new_point1, new_point2);
                }
            }
        }

        static void clip_tris(PrimitiveObject* outprim, const PrimitiveObject* refprim, 
                std::vector<zeno::vec3f>& new_pos_attr, 
                const std::vector<zeno::vec3f>& ref_pos_attr,
                std::unordered_map<int32_t, int32_t>& point_map,
                const std::vector<bool>& is_above_arr,
                const zeno::vec3f& origin,
                const zeno::vec3f& direction) {
            std::unordered_map<std::pair<int32_t, int32_t>, int32_t, hash_pair> edge_point_map;
            for (size_t tri_idx = 0; tri_idx < refprim->tris.size(); ++tri_idx) {
                const zeno::vec3i& tri_points = refprim->tris[tri_idx];

                std::vector<size_t> above_points;
                std::vector<size_t> below_points;
                bool is_continuous = true;
                int32_t last_above = -1;
                int32_t last_below = -1;

                for (size_t i = 0; i < 3; ++i) {
                    if (is_above_arr[tri_points[i]]) {
                        above_points.push_back(tri_points[i]);
                        if (last_above >= 0) {
                            if (last_above != (i - 1)) {
                                is_continuous = 0;
                            }
                        }
                        else {
                            last_above = i;
                        }
                    }
                    else {
                        below_points.push_back(tri_points[i]);
                        if (last_below >= 0) {
                            if (last_below != (i - 1)) {
                                is_continuous = 0;
                            }
                        }
                        else {
                            last_below = i;
                        }
                    }
                }

                if (above_points.size() == 1) {
                    const zeno::vec3f& pos = ref_pos_attr[above_points[0]];
                    const zeno::vec3f& pos1 = ref_pos_attr[below_points[0]];
                    const zeno::vec3f& pos2 = ref_pos_attr[below_points[1]];

                    const std::pair<int32_t, int32_t> edge1(below_points[0], above_points[0]);
                    const std::pair<int32_t, int32_t> edge2(below_points[1], above_points[0]);

                    int32_t new_point1 = -1;
                    int32_t new_point2 = -1;
                    const int32_t below_point1 = point_map[below_points[0]];
                    const int32_t below_point2 = point_map[below_points[1]];

                    auto edge_it1 = edge_point_map.find(edge1);
                    if (edge_it1 == edge_point_map.end()) {
                        const zeno::vec3f p1 = line_plane_intersection(origin, direction, pos1, normalize(pos - pos1));
                        new_point1 = new_pos_attr.size();
                        new_pos_attr.emplace_back(p1);
                        edge_point_map[edge1] = new_point1;
                    }
                    else {
                        new_point1 = edge_it1->second;
                    }

                    auto edge_it2 = edge_point_map.find(edge2);
                    if (edge_it2 == edge_point_map.end()) {
                        const zeno::vec3f p2 = line_plane_intersection(origin, direction, pos2, normalize(pos - pos2));
                        new_point2 = new_pos_attr.size();
                        new_pos_attr.emplace_back(p2);
                        edge_point_map[edge2] = new_point2;
                    }
                    else {
                        new_point2 = edge_it2->second;
                    }

                    if (is_continuous) {
                        outprim->tris.emplace_back(below_point1, below_point2, new_point2);
                        outprim->tris.emplace_back(new_point2, new_point1, below_point1);
                    }
                    else {
                        outprim->tris.emplace_back(below_point2, below_point1, new_point2);
                        outprim->tris.emplace_back(new_point1, new_point2, below_point1);
                    }
                }
                else if (above_points.size() == 2) {
                    const zeno::vec3f& pos = ref_pos_attr[below_points[0]];
                    const zeno::vec3f& pos1 = ref_pos_attr[above_points[0]];
                    const zeno::vec3f& pos2 = ref_pos_attr[above_points[1]];

                    const std::pair<int32_t, int32_t> edge1(below_points[0], above_points[0]);
                    const std::pair<int32_t, int32_t> edge2(below_points[0], above_points[1]);

                    int32_t new_point1 = -1;
                    int32_t new_point2 = -1;
                    const int32_t below_point = point_map[below_points[0]];

                    auto edge_it1 = edge_point_map.find(edge1);
                    if (edge_it1 == edge_point_map.end()) {
                        const zeno::vec3f new_pos = line_plane_intersection(origin, direction, pos, normalize(pos1 - pos));
                        new_point1 = new_pos_attr.size();
                        new_pos_attr.emplace_back(new_pos);
                        edge_point_map[edge1] = new_point1;
                    }
                    else {
                        new_point1 = edge_it1->second;
                    }

                    auto edge_it2 = edge_point_map.find(edge2);
                    if (edge_it2 == edge_point_map.end()) {
                        const zeno::vec3f new_pos = line_plane_intersection(origin, direction, pos, normalize(pos2 - pos));
                        new_point2 = new_pos_attr.size();
                        new_pos_attr.emplace_back(new_pos);
                        edge_point_map[edge2] = new_point2;
                    }
                    else {
                        new_point2 = edge_it2->second;
                    }

                    if (is_continuous) {
                        outprim->tris.emplace_back(below_point, new_point1, new_point2);
                    }
                    else {
                        outprim->tris.emplace_back(below_point, new_point2, new_point1);
                    }
                }
                else if (above_points.size() == 0) {
                    append_element<zeno::vec3i>(tri_points, outprim->tris, point_map);
                }
            }

        }

        virtual void apply() override {
            zeno::vec3f origin = { 0,0,0 };
            zeno::vec3f direction = { 0,1,0 };
            float distance = 0.0f;
            auto reverse = get_param<bool>("reverse");
            if (has_input("origin"))
                origin = get_input<zeno::NumericObject>("origin")->get<zeno::vec3f>();
            if (has_input("direction"))
                direction = get_input<zeno::NumericObject>("direction")->get<zeno::vec3f>();
            if (has_input("distance"))
                distance = get_input<zeno::NumericObject>("distance")->get<float>();
            if (lengthSquared(direction) < 0.000001f) {
                set_output("outPrim", get_input("prim"));
                return;
            }
            direction = reverse ? -normalize(direction) : normalize(direction);
            origin += direction * distance;

            auto refprim = get_input<PrimitiveObject>("prim");
            auto& ref_pos_attr = refprim->attr<zeno::vec3f>("pos");

            auto outprim = std::make_unique<PrimitiveObject>();
            std::vector<zeno::vec3f> new_pos_attr;
            std::unordered_map<int32_t, int32_t> point_map;
            
            std::vector<bool> is_above_arr(ref_pos_attr.size());
            for (int32_t i = 0; i < ref_pos_attr.size(); ++i)
            {
                is_above_arr[i] = dot(ref_pos_attr[i] - origin, direction) > 0;
                if (!is_above_arr[i]) {
                    point_map[i] = new_pos_attr.size();
                    new_pos_attr.emplace_back(ref_pos_attr[i]);
                }
            }

            clip_points(outprim.get(), refprim.get(),
                point_map,
                is_above_arr);

            clip_lines(outprim.get(), refprim.get(),
                new_pos_attr,
                ref_pos_attr,
                point_map,
                is_above_arr,
                origin,
                direction);

            clip_tris(outprim.get(), refprim.get(),
                new_pos_attr,
                ref_pos_attr,
                point_map,
                is_above_arr,
                origin,
                direction);

            outprim->attr<zeno::vec3f>("pos") = new_pos_attr;
            outprim->resize(new_pos_attr.size());
            set_output("outPrim", std::move(outprim));
        }
    };

ZENDEFNODE(PrimitiveClip, {
    {{"PrimitiveObject", "prim"}, {"vec3f", "origin", "0,0,0"}, {"vec3f", "direction", "0,0,1"}, {"float", "distance", "0"}},
    {{"PrimitiveObject", "outPrim"}},
    {{"bool", "reverse", "0"}},
    {"primitive"},
    });
}
