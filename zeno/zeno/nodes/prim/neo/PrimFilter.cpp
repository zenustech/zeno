#include "zeno/types/StringObject.h"
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>

namespace zeno {

ZENO_API void primFilter(PrimitiveObject *prim, std::string tagAttr, int tagValue, bool isInversed) {
    std::vector<int> revamp;
    revamp.reserve(prim->size());
    auto const &tagArr = prim->attr<int>(tagAttr);
    if (!isInversed) {
        for (int i = 0; i < prim->size(); i++) {
            if (tagArr[i] == tagValue)
                revamp.emplace_back(i);
        }
    } else {
        for (int i = 0; i < prim->size(); i++) {
            if (tagArr[i] != tagValue)
                revamp.emplace_back(i);
        }
    }

    prim->foreach_attr([&] (auto const &key, auto &arr) {
        for (int i = 0; i < revamp.size(); i++) {
            arr[i] = arr[revamp[i]];
        }
    });
    auto old_prim_size = prim->size();
    prim->resize(revamp.size());

    if ((0
            || prim->tris.size()
            || prim->quads.size()
            || prim->lines.size()
            || prim->polys.size()
            || prim->points.size()
         )) {

        std::vector<int> unrevamp(old_prim_size, -1);
        for (int i = 0; i < revamp.size(); i++) {
            unrevamp[revamp[i]] = i;
        }
        auto mock = [&] (int &x) -> bool {
            int loc = unrevamp[x];
            if (loc == -1)
                return false;
            x = loc;
            return true;
        };

        if (prim->tris.size()) {
            std::vector<int> trisrevamp;
            trisrevamp.reserve(prim->tris.size());
            for (int i = 0; i < prim->tris.size(); i++) {
                auto &tri = prim->tris[i];
                if (mock(tri[0]) && mock(tri[1]) && mock(tri[2]))
                    trisrevamp.emplace_back(i);
            }
            for (int i = 0; i < trisrevamp.size(); i++) {
                prim->tris[i] = prim->tris[trisrevamp[i]];
            }
            prim->tris.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < trisrevamp.size(); i++) {
                    arr[i] = arr[trisrevamp[i]];
                }
            });
            prim->tris.resize(trisrevamp.size());
        }

        if (prim->quads.size()) {
            std::vector<int> quadsrevamp;
            quadsrevamp.reserve(prim->quads.size());
            for (int i = 0; i < prim->quads.size(); i++) {
                auto &quad = prim->quads[i];
                if (mock(quad[0]) && mock(quad[1]) && mock(quad[2]) && mock(quad[3]))
                    quadsrevamp.emplace_back(i);
            }
            for (int i = 0; i < quadsrevamp.size(); i++) {
                prim->quads[i] = prim->quads[quadsrevamp[i]];
            }
            prim->quads.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < quadsrevamp.size(); i++) {
                    arr[i] = arr[quadsrevamp[i]];
                }
            });
            prim->quads.resize(quadsrevamp.size());
        }

        if (prim->lines.size()) {
            std::vector<int> linesrevamp;
            linesrevamp.reserve(prim->lines.size());
            for (int i = 0; i < prim->lines.size(); i++) {
                auto &line = prim->lines[i];
                if (mock(line[0]) && mock(line[1]))
                    linesrevamp.emplace_back(i);
            }
            for (int i = 0; i < linesrevamp.size(); i++) {
                prim->lines[i] = prim->lines[linesrevamp[i]];
            }
            prim->lines.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < linesrevamp.size(); i++) {
                    arr[i] = arr[linesrevamp[i]];
                }
            });
            prim->lines.resize(linesrevamp.size());
        }

        if (prim->polys.size()) {
            std::vector<int> polysrevamp;
            polysrevamp.reserve(prim->polys.size());
            for (int i = 0; i < prim->polys.size(); i++) {
                auto &poly = prim->polys[i];
                bool succ = [&] {
                    for (int p = poly.first; p < poly.first + poly.second; p++)
                        if (!mock(prim->loops[p]))
                            return false;
                    return true;
                }();
                if (succ)
                    polysrevamp.emplace_back(i);
            }
            for (int i = 0; i < polysrevamp.size(); i++) {
                prim->polys[i] = prim->polys[polysrevamp[i]];
            }
            prim->polys.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < polysrevamp.size(); i++) {
                    arr[i] = arr[polysrevamp[i]];
                }
            });
            prim->polys.resize(polysrevamp.size());
        }

        if (prim->points.size()) {
            std::vector<int> pointsrevamp;
            pointsrevamp.reserve(prim->points.size());
            for (int i = 0; i < prim->points.size(); i++) {
                auto &point = prim->points[i];
                if (mock(point))
                    pointsrevamp.emplace_back(i);
            }
            for (int i = 0; i < pointsrevamp.size(); i++) {
                prim->points[i] = prim->points[pointsrevamp[i]];
            }
            prim->points.foreach_attr([&] (auto const &key, auto &arr) {
                for (int i = 0; i < pointsrevamp.size(); i++) {
                    arr[i] = arr[pointsrevamp[i]];
                }
            });
            prim->points.resize(pointsrevamp.size());
        }

    }
}

struct PrimFilter : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto tagAttr = get_input<StringObject>("tagAttr")->get();
    auto tagValue = get_input<NumericObject>("tagValue")->get<int>();
    auto isInversed = get_input<NumericObject>("isInversed")->get<bool>();
    primFilter(prim.get(), tagAttr, tagValue, isInversed);
    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimFilter, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "tagAttr", "tag"},
    {"int", "tagValue", "1"},
    {"bool", "isInversed", "0"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
