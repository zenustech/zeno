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

        virtual void apply() override {
            zeno::vec3f origin = { 0,0,0 };
            zeno::vec3f direction = { 0,1,0 };
            if (has_input("origin"))
                origin = get_input<zeno::NumericObject>("origin")->get<zeno::vec3f>();
            if (has_input("direction"))
                direction = get_input<zeno::NumericObject>("direction")->get<zeno::vec3f>();
            if (lengthSquared(direction) < 0.000001f) {
                set_output("outPrim", get_input("prim"));
                return;
            }
            direction = normalize(direction);

            auto prim = get_input<PrimitiveObject>("prim");
            auto& pos_attr = prim->attr<zeno::vec3f>("pos");

            auto outprim = std::make_unique<PrimitiveObject>();
            std::vector<zeno::vec3f> new_pos_attr;
            std::unordered_map<int32_t, int32_t> point_map;
            std::unordered_map<std::pair<int32_t, int32_t>, int32_t, hash_pair> edge_point_map;

            for (size_t tri_idx = 0; tri_idx < prim->tris.size(); ++tri_idx) {
                zeno::vec3i& vertices = prim->tris[tri_idx];
                
                std::vector<size_t> above_points;
                std::vector<size_t> below_points;
                bool is_continuous = true;
                int32_t last_above = -1;
                int32_t last_below = -1;

                for (size_t i = 0; i < 3; ++i) {
                    const zeno::vec3f& pos = pos_attr[vertices[i]];

                    if (dot(pos - origin, direction) > 0) {
                        above_points.push_back(vertices[i]);
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
                        below_points.push_back(vertices[i]);
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
                    const zeno::vec3f& pos = pos_attr[above_points[0]];
                    const zeno::vec3f& pos1 = pos_attr[below_points[0]];
                    const zeno::vec3f& pos2 = pos_attr[below_points[1]];

                    const std::pair<int32_t, int32_t> edge1(below_points[0], above_points[0]);
                    const std::pair<int32_t, int32_t> edge2(below_points[1], above_points[0]);

                    int32_t new_pt1 = -1;
                    int32_t new_pt2 = - 1;
                    int32_t below_pt1 = -1;
                    int32_t below_pt2 = -1;

                    auto edge_it1 = edge_point_map.find(edge1);
                    if (edge_it1 == edge_point_map.end()) {
                        const zeno::vec3f p1 = line_plane_intersection(origin, direction, pos1, normalize(pos - pos1));
                        new_pt1 = new_pos_attr.size();
                        new_pos_attr.emplace_back(p1);
                        edge_point_map[edge1] = new_pt1;
                    }
                    else {
                        new_pt1 = edge_it1->second;
                    }

                    auto edge_it2 = edge_point_map.find(edge2);
                    if (edge_it2 == edge_point_map.end()) {
                        const zeno::vec3f p2 = line_plane_intersection(origin, direction, pos2, normalize(pos - pos2));
                        new_pt2 = new_pos_attr.size();
                        new_pos_attr.emplace_back(p2);
                        edge_point_map[edge2] = new_pt2;
                    }
                    else {
                        new_pt2 = edge_it2->second;
                    }

                    auto it1 = point_map.find(below_points[0]);
                    if (it1 == point_map.end()) {
                        below_pt1 = new_pos_attr.size();
                        new_pos_attr.emplace_back(pos1);
                        point_map[below_points[0]] = below_pt1;
                    }
                    else {
                        below_pt1 = it1->second;
                    }

                    auto it2 = point_map.find(below_points[1]);
                    if (it2 == point_map.end()) {
                        below_pt2 = new_pos_attr.size();
                        new_pos_attr.emplace_back(pos2);
                        point_map[below_points[1]] = below_pt2;
                    }
                    else {
                        below_pt2 = it2->second;
                    }

                    if (is_continuous) {
                        outprim->tris.emplace_back(below_pt1, below_pt2, new_pt2);
                        outprim->tris.emplace_back(new_pt2, new_pt1, below_pt1);
                    }
                    else {
                        outprim->tris.emplace_back(below_pt2, below_pt1, new_pt2);
                        outprim->tris.emplace_back(new_pt1, new_pt2, below_pt1);
                    }
                }
                else if (above_points.size() == 2) {
                    const zeno::vec3f& pos = pos_attr[below_points[0]];
                    const zeno::vec3f& pos1 = pos_attr[above_points[0]];
                    const zeno::vec3f& pos2 = pos_attr[above_points[1]];

                    const std::pair<int32_t, int32_t> edge1(below_points[0], above_points[0]);
                    const std::pair<int32_t, int32_t> edge2(below_points[0], above_points[1]);

                    int32_t new_pt1 = -1;
                    int32_t new_pt2 = -1;
                    int32_t below_pt = -1;

                    auto edge_it1 = edge_point_map.find(edge1);
                    if (edge_it1 == edge_point_map.end()) {
                        const zeno::vec3f p1 = line_plane_intersection(origin, direction, pos, normalize(pos1 - pos));
                        new_pt1 = new_pos_attr.size();
                        new_pos_attr.emplace_back(p1);
                        edge_point_map[edge1] = new_pt1;
                    }
                    else {
                        new_pt1 = edge_it1->second;
                    }

                    auto edge_it2 = edge_point_map.find(edge2);
                    if (edge_it2 == edge_point_map.end()) {
                        const zeno::vec3f p2 = line_plane_intersection(origin, direction, pos, normalize(pos2 - pos));
                        new_pt2 = new_pos_attr.size();
                        new_pos_attr.emplace_back(p2);
                        edge_point_map[edge2] = new_pt2;
                    }
                    else {
                        new_pt2 = edge_it2->second;
                    }

                    auto it = point_map.find(below_points[0]);
                    if (it == point_map.end()) {
                        below_pt = new_pos_attr.size();
                        new_pos_attr.emplace_back(pos);
                        point_map[below_points[0]] = below_pt;
                    }
                    else {
                        below_pt = it->second;
                    }

                    if (is_continuous) {
                        outprim->tris.emplace_back(below_pt, new_pt1, new_pt2);
                    }
                    else {
                        outprim->tris.emplace_back(below_pt, new_pt2, new_pt1);
                    }
                }
                else if (above_points.size() == 0)
                {
                    zeno::vec3i new_tri;
                    for(size_t i = 0; i < 3; ++i) {
                        auto it = point_map.find(vertices[i]);
                        if (it == point_map.end()) {
                            new_tri[i] = new_pos_attr.size();
                            new_pos_attr.emplace_back(pos_attr[vertices[i]]);
                            point_map[vertices[i]] = new_tri[i];
                        }
                        else {
                            new_tri[i] = it->second;
                        }
                    }
                    outprim->tris.emplace_back(new_tri);
                }
            }

            outprim->m_attrs["pos"] = new_pos_attr;
            outprim->m_size = new_pos_attr.size();
            set_output("outPrim", std::move(outprim));
        }
    };

ZENDEFNODE(PrimitiveClip, {
    {"prim", "origin", "direction"},
    {"outPrim"},
    {},
    {"primitive"},
    });

}