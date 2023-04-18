#include <cctype>
#include <filesystem>
#include <sstream>
#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

#include "./SurfaceMesh.h"
#include "./algorithms/SurfaceRemeshing.h"

#include "zensim/container/Bht.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zeno {

struct UniformRemeshing : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto iterations = get_input2<int>("iterations");
        float edge_length = edge_length = get_input2<float>("edge_length");
        auto line_pick_tag = get_input<zeno::StringObject>("line_pick_tag")->get();
        auto &pos = prim->attr<vec3f>("pos");

        // init line_pick attribute
        if (!prim->lines.has_attr(line_pick_tag)) {
            prim->lines.add_attr<int>(line_pick_tag, 0);
        }
        auto &elocked = prim->lines.attr<int>(line_pick_tag);

        // init v_duplicate attribute
        auto &vduplicate = prim->verts.add_attr<int>("v_duplicate", 0);
        int vert_size = prim->verts.size();
        for (int i = 0; i < vert_size; ++i) {
            vduplicate[i] = i;
        }

        // if there exist marked lines
        std::set<std::pair<int, int>> marked_lines{};
#if 0
        if (has_input("marked_lines")) {
            const auto &markedLines = get_input<PrimitiveObject>("marked_lines")->lines.values;
            for (vec2i line : markedLines) {
                marked_lines.insert(std::make_pair(line[0], line[1]));
            }
        }
        auto &lines = prim->lines;
        lines.clear();
        elocked.clear();
#elif 0
        /// DEBUG, all perimeter lines preserved
        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();
        bht<int, 2, int> tab{prim->tris.size()};
        tab.reset(pol, true);
        pol(prim->tris.values, [tab = proxy<space>(tab)](auto tri) mutable {
            for (int d = 0; d != 3; ++d) {
                auto a = tri[d];
                auto b = tri[(d + 1) % 3];
                tab.insert(zs::vec<int, 2>{a, b});
            }
        });
        auto ne = tab.size();
        std::vector<int> marks(ne);
        pol(zip(range(ne), tab._activeKeys), [&marks, tab = proxy<space>(tab)](int no, auto line) mutable {
            if (tab.query(zs::vec<int, 2>{line[1], line[0]}) < 0) {
                marks[no] = 1;
            }
        });
        for (int i = 0; i != ne; ++i)
            if (marks[i]) {
                auto line = tab._activeKeys[i];
                marked_lines.insert(std::make_pair(line[0], line[1]));
            }
        auto &lines = prim->lines;
        lines.clear();
        elocked.clear();
#else
        auto &lines = prim->lines;
        if (lines.size() > 0) {
            int siz = lines.size();
            for (int i = 0; i < siz; ++i) {
                if (elocked[i] == 1) {
                    marked_lines.insert(std::make_pair(lines[i][0], lines[i][1]));
                }
            }
            lines.clear();
            elocked.clear();
        }
#endif

        // handle non-manifold edges
        std::map<std::pair<int, int>, int> lines_map{};
        int line_size = 0;
        for (auto &it : prim->tris) {
            std::vector<bool> new_vert(3, false);
            std::vector<bool> new_line(3, false);
            std::vector<bool> marked(3, false);
            auto edge = std::make_pair(it[0], it[1]);
            new_line[0] = (lines_map.count(edge) > 0);
            marked[0] = (marked_lines.count(edge) > 0 || marked_lines.count(std::make_pair(it[1], it[0])) > 0);
            if (new_line[0]) {
                new_vert[0] = new_vert[1] = true;
                elocked[lines_map[edge]] = 1;
            }
            edge = std::make_pair(it[1], it[2]);
            new_line[1] = (lines_map.count(edge) > 0);
            marked[1] = (marked_lines.count(edge) > 0 || marked_lines.count(std::make_pair(it[2], it[1])) > 0);
            if (new_line[1]) {
                new_vert[1] = new_vert[2] = true;
                elocked[lines_map[edge]] = 1;
            }
            edge = std::make_pair(it[2], it[0]);
            new_line[2] = (lines_map.count(edge) > 0);
            marked[2] = (marked_lines.count(edge) > 0 || marked_lines.count(std::make_pair(it[0], it[2])) > 0);
            if (new_line[2]) {
                new_vert[2] = new_vert[0] = true;
                elocked[lines_map[edge]] = 1;
            }

            for (int j = 0; j < 3; ++j) {
                if (new_vert[j]) {
                    pos.push_back(pos[it[j]]);
                    vduplicate.push_back(it[j]);
                    it[j] = vert_size;
                    ++vert_size;
                }
            }

            for (int j = 0; j < 3; ++j) {
                int line_id;
                bool flag = false;
                if (lines_map.count(std::make_pair(it[j], it[(j + 1) % 3])) > 0) {
                    line_id = lines_map[std::make_pair(it[j], it[(j + 1) % 3])];
                } else if (lines_map.count(std::make_pair(it[(j + 1) % 3], it[j])) > 0) {
                    line_id = lines_map[std::make_pair(it[(j + 1) % 3], it[j])];
                } else {
                    line_id = line_size;
                    flag = true;
                    ++line_size;
                }
                lines_map[std::make_pair(it[j], it[(j + 1) % 3])] = line_id;
                if (flag) {
                    lines.push_back(vec2i(it[j], it[(j + 1) % 3]));
                    if (new_line[j] || marked[j]) {
                        elocked.push_back(1);
                    } else {
                        elocked.push_back(0);
                    }
                }
            }
        }

        auto mesh = new zeno::pmp::SurfaceMesh(prim, line_pick_tag);

        if (edge_length < 1e-10) {
            // If no edge length input,
            // take the average of all edges as default.
            auto &lines = prim->lines;
            for (auto eit : lines)
                edge_length += length(pos[eit[0]] - pos[eit[1]]);
            edge_length /= (double)lines.size();
            zeno::log_info("default edge_length: {}", edge_length);
        }
        zeno::pmp::SurfaceRemeshing(mesh, line_pick_tag).uniform_remeshing(edge_length, iterations);

        // delete duplicate vertices
        int tri_size = prim->tris.size();
        for (int i = 0; i < tri_size; ++i) {
            for (int j = 0; j < 3; ++j) {
                int v = prim->tris[i][j];
                if (vduplicate[v] != v) {
                    prim->tris[i][j] = vduplicate[v];
                }
            }
        }
        line_size = prim->lines.size();
        for (int i = 0; i < line_size; ++i) {
            for (int j = 0; j < 2; ++j) {
                int v = prim->lines[i][j];
                if (vduplicate[v] != v) {
                    prim->lines[i][j] = vduplicate[v];
                }
            }
        }

        // delete v_duplicate at last
        prim->verts.erase_attr("v_duplicate");

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(UniformRemeshing)
({
    {{"prim"},
     {"int", "iterations", "10"},
     {"float", "edge_length", "0"},
     {"string", "line_pick_tag", "line_selected"},
     {"marked_lines"}},
    {"prim"},
    {},
    {"primitive"},
});

struct AdaptiveRemeshing : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto iterations = get_input2<int>("iterations");
        float max_length = get_input2<float>("max_length");
        float min_length = get_input2<float>("min_length");
        float approximation_tolerance = get_input2<float>("approximation_tolerance");
        auto line_pick_tag = get_input<zeno::StringObject>("line_pick_tag")->get();
        auto &pos = prim->attr<vec3f>("pos");

        // init line_pick attribute
        if (!prim->lines.has_attr(line_pick_tag)) {
            prim->lines.add_attr<int>(line_pick_tag, 0);
        }
        auto &elocked = prim->lines.attr<int>(line_pick_tag);

        // init v_duplicate attribute
        auto &vduplicate = prim->verts.add_attr<int>("v_duplicate", 0);
        int vert_size = prim->verts.size();
        for (int i = 0; i < vert_size; ++i) {
            vduplicate[i] = i;
        }

        // if there exist marked lines
        auto &lines = prim->lines;
        std::set<std::pair<int, int>> marked_lines{};
        if (lines.size() > 0) {
            int siz = lines.size();
            for (int i = 0; i < siz; ++i) {
                if (elocked[i] == 1) {
                    marked_lines.insert(std::make_pair(lines[i][0], lines[i][1]));
                }
            }
            lines.clear();
            elocked.clear();
        }

        // handle non-manifold edges
        std::map<std::pair<int, int>, int> lines_map{};
        int line_size = 0;
        for (auto &it : prim->tris) {
            std::vector<bool> new_vert(3, false);
            std::vector<bool> new_line(3, false);
            std::vector<bool> marked(3, false);
            auto edge = std::make_pair(it[0], it[1]);
            new_line[0] = (lines_map.count(edge) > 0);
            marked[0] = (marked_lines.count(edge) > 0 || marked_lines.count(std::make_pair(it[1], it[0])) > 0);
            if (new_line[0]) {
                new_vert[0] = new_vert[1] = true;
                elocked[lines_map[edge]] = 1;
            }
            edge = std::make_pair(it[1], it[2]);
            new_line[1] = (lines_map.count(edge) > 0);
            marked[1] = (marked_lines.count(edge) > 0 || marked_lines.count(std::make_pair(it[2], it[1])) > 0);
            if (new_line[1]) {
                new_vert[1] = new_vert[2] = true;
                elocked[lines_map[edge]] = 1;
            }
            edge = std::make_pair(it[2], it[0]);
            new_line[2] = (lines_map.count(edge) > 0);
            marked[2] = (marked_lines.count(edge) > 0 || marked_lines.count(std::make_pair(it[0], it[2])) > 0);
            if (new_line[2]) {
                new_vert[2] = new_vert[0] = true;
                elocked[lines_map[edge]] = 1;
            }

            for (int j = 0; j < 3; ++j) {
                if (new_vert[j]) {
                    pos.push_back(pos[it[j]]);
                    vduplicate.push_back(it[j]);
                    it[j] = vert_size;
                    ++vert_size;
                }
            }

            for (int j = 0; j < 3; ++j) {
                int line_id;
                bool flag = false;
                if (lines_map.count(std::make_pair(it[j], it[(j + 1) % 3])) > 0) {
                    line_id = lines_map[std::make_pair(it[j], it[(j + 1) % 3])];
                } else if (lines_map.count(std::make_pair(it[(j + 1) % 3], it[j])) > 0) {
                    line_id = lines_map[std::make_pair(it[(j + 1) % 3], it[j])];
                } else {
                    line_id = line_size;
                    flag = true;
                    ++line_size;
                }
                lines_map[std::make_pair(it[j], it[(j + 1) % 3])] = line_id;
                if (flag) {
                    lines.push_back(vec2i(it[j], it[(j + 1) % 3]));
                    if (new_line[j] || marked[j]) {
                        elocked.push_back(1);
                    } else {
                        elocked.push_back(0);
                    }
                }
            }
        }

        auto mesh = new zeno::pmp::SurfaceMesh(prim, line_pick_tag);
        auto bb = mesh->bounds().size();
        if (max_length < 1e-10) {
            max_length = 0.0500 * bb;
            zeno::log_info("default max_length: {}", max_length);
        }
        if (min_length < 1e-10) {
            min_length = 0.0010 * bb;
            zeno::log_info("default min_length: {}", min_length);
        }
        if (approximation_tolerance < 1e-10) {
            approximation_tolerance = 0.0005 * bb;
            zeno::log_info("default approximation_tolerance: {}", approximation_tolerance);
        }
        zeno::pmp::SurfaceRemeshing(mesh, line_pick_tag)
            .adaptive_remeshing(min_length, max_length, approximation_tolerance, iterations);

        // delete duplicate vertices
        int tri_size = prim->tris.size();
        for (int i = 0; i < tri_size; ++i) {
            for (int j = 0; j < 3; ++j) {
                int v = prim->tris[i][j];
                if (vduplicate[v] != v) {
                    prim->tris[i][j] = vduplicate[v];
                }
            }
        }
        line_size = prim->lines.size();
        for (int i = 0; i < line_size; ++i) {
            for (int j = 0; j < 2; ++j) {
                int v = prim->lines[i][j];
                if (vduplicate[v] != v) {
                    prim->lines[i][j] = vduplicate[v];
                }
            }
        }

        // delete v_duplicate at last
        prim->verts.erase_attr("v_duplicate");

        set_output("prim", std::move(prim));
    }
};

ZENO_DEFNODE(AdaptiveRemeshing)
({
    {{"prim"},
     {"int", "iterations", "10"},
     {"float", "max_length", "0"},
     {"float", "min_length", "0"},
     {"float", "approximation_tolerance", "0"},
     {"string", "line_pick_tag", "line_selected"}},
    {"prim"},
    {},
    {"primitive"},
});
} // namespace zeno
