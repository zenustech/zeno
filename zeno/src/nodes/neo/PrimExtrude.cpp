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

        auto prim2 = std::make_shared<PrimitiveObject>(*prim);
        std::vector<int> oldinds;
        std::vector<vec3f> p2norms;
        if (!maskAttr.empty()) {
            std::string tmpOldindAttr = "%%extrude1";
            primFilterVerts(prim2.get(), maskAttr, 0, true, tmpOldindAttr);
            oldinds.swap(prim2->verts.attr<int>(tmpOldindAttr));
            prim2->verts.erase_attr(tmpOldindAttr);
        }
        if (extrude != 0) {
            std::string tmpNormAttr = "%%extrude2";
            primCalcNormal(prim2.get(), 1.0f, tmpNormAttr);
            p2norms.swap(prim2->verts.attr<vec3f>(tmpNormAttr));
            prim2->verts.erase_attr(tmpNormAttr);
        }
        bool flipNewFace = autoFlipFace && extrude < 0;
        bool flipOldFace = autoFlipFace && extrude > 0;
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

        if (extrude != 0) {
            for (int i = 0; i < p2size; i++) {
                prim->verts[i + p1size] += p2norms[i] * extrude + offset;
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
