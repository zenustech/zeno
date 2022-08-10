#include <zeno/para/parallel_for.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/extra/TempNode.h>
#include <zeno/core/INode.h>
#include <zeno/zeno.h>
#include <numeric>
#include <set>

namespace zeno {
namespace {

struct PrimExtrude : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto maskAttr = get_input2<std::string>("maskAttr");
        auto extrude = get_input2<float>("extrude");
        auto inset = get_input2<float>("inset");
        auto offset = get_input2<vec3f>("offset");
        auto bridgeMaskAttrO = get_input2<std::string>("bridgeMaskAttrO");
        auto sourceMaskAttrO = get_input2<std::string>("sourceMaskAttrO");
        auto autoFlipFace = get_input2<bool>("autoFlipFace");
        auto autoFindEdges = get_input2<bool>("autoFindEdges");
        auto averagedExtrude = get_input2<bool>("averagedExtrude");

        auto prim2 = std::make_shared<PrimitiveObject>(*prim);
        bool flipNewFace = autoFlipFace && extrude < 0;
        bool flipOldFace = autoFlipFace && extrude > 0;

        if (autoFindEdges && !maskAttr.empty()) {
            AttrVector<vec2i> oldlines = std::move(prim2->lines);
            primWireframe(prim2.get(), false);
            prim2->edges = std::move(prim2->lines);
            prim2->lines = std::move(oldlines);
        }

        std::vector<int> oldinds;
        if (!maskAttr.empty()) {
            std::string tmpOldindAttr = "%%extrude1";
            primFilterVerts(prim2.get(), maskAttr, 0, true, tmpOldindAttr);
            oldinds.swap(prim2->verts.attr<int>(tmpOldindAttr));
            prim2->verts.erase_attr(tmpOldindAttr);
        }

        //{
            //std::vector<int> wirelinesrevamp;
            //linesrevamp.reserve(wirelines.size());
            //for (int i = 0; i < wirelines.size(); i++) {
                //auto &line = wirelines[i];
                //if (mock(line[0]) && mock(line[1]))
                    //wirelinesrevamp.emplace_back(i);
            //}
            //for (int i = 0; i < wirelinesrevamp.size(); i++) {
                //wirelines[i] = wirelines[wirelinesrevamp[i]];
            //}
            //wirelines.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto &arr) {
                //revamp_vector(arr, wirelinesrevamp);
            //});
            //wirelines.resize(linesrevamp.size());
        //}

        std::vector<vec3f> p2norms;
        if (extrude != 0 || inset != 0) {
            std::string tmpNormAttr = "%%extrude2";
            primCalcNormal(prim2.get(), 1.0f, tmpNormAttr);
            p2norms = std::move(prim2->verts.attr<vec3f>(tmpNormAttr));
            prim2->verts.erase_attr(tmpNormAttr);
        }

        std::vector<vec3f> p2inset;
        if (inset != 0) {//havebug
            p2inset.resize(prim2->verts.size());
            //std::string tmpInsetAttr = "%%extrude3";
            //primCalcInsetDir(prim2.get(), 1.0f, tmpInsetAttr);
            auto &out = p2inset;
            for (size_t i = 0; i < prim->tris.size(); i++) {
                auto ind = prim->tris[i];
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto c = prim->verts[ind[2]];
                auto &oa = out[ind[0]];
                auto &ob = out[ind[1]];
                auto &oc = out[ind[2]];
                oa += normalizeSafe(b + c - a - a);
                ob += normalizeSafe(a + c - b - b);
                oc += normalizeSafe(a + b - c - c);
            }
            for (size_t i = 0; i < prim->quads.size(); i++) {
                auto ind = prim->quads[i];
                auto a = prim->verts[ind[0]];
                auto b = prim->verts[ind[1]];
                auto c = prim->verts[ind[2]];
                auto d = prim->verts[ind[3]];
                auto &oa = out[ind[0]];
                auto &ob = out[ind[1]];
                auto &oc = out[ind[2]];
                auto &od = out[ind[3]];
                oa += normalizeSafe(b + c + d - a - a - a);
                ob += normalizeSafe(a + c + d - b - b - b);
                oc += normalizeSafe(a + b + d - c - c - c);
                od += normalizeSafe(a + b + c - d - d - d);
            }
            for (size_t i = 0; i < prim->polys.size(); i++) {
                auto [start, len] = prim->polys[i];
                for (int j = start; j < start + len; j++) {
                    auto curr = prim->verts[prim->loops[j]];
                    vec3f accum = -(len - 1) * curr;
                    for (int k = start; k < start + len; k++) {
                        if (k == j) continue;
                        accum += prim->verts[prim->loops[k]];
                    }
                    out[prim->loops[j]] += normalizeSafe(accum);
                }
            }
            for (size_t i = 0; i < out.size(); i++) {
                auto insd = out[i];
                auto norm = p2norms[i];
                insd -= dot(insd, norm) * norm;
                out[i] = normalizeSafe(insd);
            }
            //p2inset = std::move(prim2->verts.attr<vec3f>(tmpInsetAttr));
            //prim2->verts.erase_attr(tmpInsetAttr);

            if (!(extrude != 0))
                p2norms.clear();
        }

        if (extrude != 0 && averagedExtrude) {
            auto avgdir = std::reduce(p2norms.begin(), p2norms.end());
            avgdir = normalizeSafe(avgdir);
            offset += extrude * avgdir;
            extrude = 0;
            p2norms.clear();
        }

        if (flipNewFace) {
            primFlipFaces(prim2.get());
        }
        if (flipOldFace) {
            primFlipFaces(prim.get());
        }

        struct segment_less {
            bool operator()(vec2i const &a, vec2i const &b) const {
                return std::make_pair(std::min(a[0], a[1]), std::max(a[0], a[1]))
                    < std::make_pair(std::min(b[0], b[1]), std::max(b[0], b[1]));
            }
        };
        std::map<vec2i, bool, segment_less> segments;
        auto append = [&] (int i, int j) {
            auto [it, succ] = segments.emplace(vec2i(i, j), false);
            if (!succ)
                it->second = true;
        };
        for (auto const &ind: prim2->lines) {
            append(ind[0], ind[1]);
        }
        for (auto const &ind: prim2->tris) {
            append(ind[0], ind[1]);
            append(ind[1], ind[2]);
            append(ind[2], ind[0]);
        }
        for (auto const &ind: prim2->quads) {
            append(ind[0], ind[1]);
            append(ind[1], ind[2]);
            append(ind[2], ind[3]);
            append(ind[3], ind[0]);
        }
        for (auto const &[start, len]: prim2->polys) {
            if (len < 2)
                continue;
            for (int i = start + 1; i < start + len; i++) {
                append(prim2->loops[i - 1], prim2->loops[i]);
            }
            append(prim2->loops[start + len - 1], prim2->loops[start]);
        }
        for (auto const &ind: prim2->edges) {
            segments.emplace(vec2i(ind[0], ind[1]), false); // if fail then just let it fail
        }

        //if (avgoffset != 0) {
            //auto &pos = prim2->verts;
            //vec3f avg(0);
            //for (auto const &ind: prim2->tris) {
                //avg += cross(pos[ind[1]] - pos[ind[0]], pos[ind[2]] - pos[ind[0]]);
            //}
            //for (auto const &ind: prim2->quads) {
                //avg += cross(pos[ind[1]] - pos[ind[0]], pos[ind[2]] - pos[ind[0]]);
                //avg += cross(pos[ind[2]] - pos[ind[1]], pos[ind[3]] - pos[ind[1]]);
                //avg += cross(pos[ind[3]] - pos[ind[2]], pos[ind[0]] - pos[ind[2]]);
                //avg += cross(pos[ind[0]] - pos[ind[3]], pos[ind[1]] - pos[ind[3]]);
            //}
            //for (auto const &[start, len]: prim2->polys) {
                //auto ind = [loops = prim2->loops.data(), start = start, len = len] (int t) -> int {
                    //if (t >= len) t -= len;
                    //return loops[start + t];
                //};
                //for (int j = 0; j < len; ++j) {
                    //avg += cross(pos[ind(j + 1)] - pos[ind(j)], pos[ind(j + 2)] - pos[ind(j)]);
                //}
            //}
            //offset += avgoffset * normalizeSafe(avg);
        //}

        //auto tmpBoundTagAttr = "%%extrude1";
        //primMarkBoundaryEdges(prim2.get(), tmpBoundTagAttr);
        std::vector<vec2i> bounds;
        for (auto const &[edge, hasdup]: segments) {
            if (!hasdup)
                bounds.push_back(edge);
        }

        int p1size = prim->verts.size();
        int p2size = prim2->verts.size();
        *prim = std::move(*primMerge({prim.get(), prim2.get()}, sourceMaskAttrO));

        bool hasFiltered = !maskAttr.empty();
        for (auto const &edge: bounds) {
            vec2i l2 = edge + p1size;
            vec2i l1 = hasFiltered ? vec2i(oldinds[edge[0]], oldinds[edge[1]]) : edge;
            //if (flipMidFace)
                //std::swap(l1, l2);
            vec4i quad(l1[0], l1[1], l2[1], l2[0]);
            prim->quads.push_back(quad);
        }

        if (extrude != 0 && inset != 0) {
            for (int i = 0; i < p2size; i++) {
                prim->verts[i + p1size] += p2norms[i] * extrude + p2inset[i] * inset + offset;
            }
        } else if (extrude != 0) {
            for (int i = 0; i < p2size; i++) {
                prim->verts[i + p1size] += p2norms[i] * extrude + offset;
            }
        } else if (inset != 0) {
            for (int i = 0; i < p2size; i++) {
                prim->verts[i + p1size] += p2inset[i] * inset + offset;
            }
        } else if (offset[0] != 0 || offset[1] != 0 || offset[2] != 0) {
            for (int i = 0; i < p2size; i++) {
                prim->verts[i + p1size] += offset;
            }
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimExtrude, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "maskAttr", ""},
    {"float", "extrude", "0.1"},
    {"float", "inset", "0"},
    {"vec3f", "offset", "0,0,0"},
    {"string", "bridgeMaskAttrO", ""},
    {"string", "sourceMaskAttrO", ""},
    {"bool", "autoFlipFace", "1"},
    {"bool", "autoFindEdges", "1"},
    {"bool", "averagedExtrude", "1"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
