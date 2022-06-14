#include <algorithm>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace zeno {

ZENO_API std::shared_ptr<PrimitiveObject> primitive_merge(std::shared_ptr<zeno::ListObject> list, std::string tagAttr) {
    auto outprim = std::make_shared<PrimitiveObject>();

    size_t len = 0;
    size_t poly_len = 0;

    //
#if defined(_OPENMP)
    size_t nTotalVerts{0}, nTotalPts{0}, nTotalLines{0}, nTotalTris{0}, nTotalQuads{0}, nTotalLoops{0}, nTotalPolys{0};
    size_t nCurPts{0}, nCurLines{0}, nCurTris{0}, nCurQuads{0}, nCurLoops{0}, nCurPolys{0};
#endif
    //

    for (auto const &prim: list->get<PrimitiveObject>()) {
#if defined(_OPENMP)
        nTotalVerts += prim->verts.size();
        nTotalPts += prim->points.size();
        nTotalLines += prim->lines.size();
        nTotalTris += prim->tris.size();
        nTotalQuads += prim->quads.size();
        nTotalLoops += prim->loops.size();
        nTotalPolys += prim->polys.size();
#endif
        prim->foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            outprim->add_attr<T>(key);
        });
    }
#if defined(_OPENMP)
    // leave verts out, since these are 'std::insert'ed
    outprim->verts.resize(nTotalVerts);
    outprim->points.resize(nTotalPts);
    outprim->lines.resize(nTotalLines);
    outprim->tris.resize(nTotalTris);
    outprim->quads.resize(nTotalQuads);
    outprim->loops.resize(nTotalLoops);
    outprim->polys.resize(nTotalPolys);
#endif

    int tagcounter = 0;
    if (!tagAttr.empty()) {
        outprim->add_attr<int>(tagAttr);
    }

    for (auto const &prim: list->get<PrimitiveObject>()) {
        //const auto base = outprim->size();
        prim->foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            //fix pyb
            auto &outarr = outprim->attr<T>(key);

#if defined(_OPENMP)
            static_assert(std::is_trivially_copyable_v<T>, "if T is not trivially copyable, then perf is damaged.");
            // since attr is stored in std::vector, its iterator must be LegacyContiguousIterator.
            std::memcpy(outarr.data() + len, arr.data(), sizeof(T) * arr.size());
            // std::copy(std::begin(arr), std::end(arr), std::begin(outarr) + len);
#else
            outarr.insert(outarr.end(), std::begin(arr), std::end(arr));
#endif
            //for (auto const &val: arr) outarr.push_back(val);
            //end fix pyb
        });
#if defined(_MSC_VER) && defined(_OPENMP)
#define omp_size_t intptr_t
#else
#define omp_size_t size_t
#endif
        if (!tagAttr.empty()) {
            auto &tagArr = outprim->attr<int>(tagAttr);
#if defined(_OPENMP)
            for (std::size_t i = 0; i < prim->size(); i++) {
                tagArr[len + i] = tagcounter;
            }
#else
            for (std::size_t i = 0; i < prim->size(); i++) {
                tagArr.push_back(tagcounter);
            }
#endif
        }
#if defined(_OPENMP)
        auto concat = [&](auto &dst, const auto &src, size_t &offset) {
#pragma omp parallel for
            for (omp_size_t i = 0; i < src.size(); ++i) {
                dst[offset + i] = src[i] + len;
            }
            offset += src.size();
        };
        // insertion
        concat(outprim->points, prim->points, nCurPts);
        concat(outprim->lines, prim->lines, nCurLines);
        concat(outprim->tris, prim->tris, nCurTris);
        concat(outprim->quads, prim->quads, nCurQuads);
        // exception: poly
#pragma omp parallel for
        for (omp_size_t i = 0; i < prim->polys.size(); ++i) {
            const auto &poly = prim->polys[i];
            outprim->polys[nCurPolys + i] = std::make_pair(poly.first + nCurLoops, poly.second);
        }
        nCurPolys += prim->polys.size();
        // update nCurLoops after poly update!
        concat(outprim->loops, prim->loops, nCurLoops);
#else
        for (auto const &idx: prim->points) {
            outprim->points.push_back(idx + len);
        }
        for (auto const &idx: prim->lines) {
            outprim->lines.push_back(idx + len);
        }
        for (auto const &idx: prim->tris) {
            outprim->tris.push_back(idx + len);
        }
        for (auto const &idx: prim->quads) {
            outprim->quads.push_back(idx + len);
        }
        for (auto const &idx: prim->loops) {
            outprim->loops.push_back(idx + len);
        }
        size_t sub_poly_len = 0;
        for (auto const &poly: prim->polys) {
            sub_poly_len = std::max(sub_poly_len, (size_t)(poly.first + poly.second));
            outprim->polys.emplace_back(poly.first + poly_len, poly.second);
        }
        poly_len += sub_poly_len;
#endif
        len += prim->size();
        //fix pyb
#if defined(_OPENMP)
#else
        outprim->resize(len);
#endif
    }

    return outprim;
}

struct PrimitiveMerge : zeno::INode {
  virtual void apply() override {
    auto list = get_input<ListObject>("listPrim");
    if(!has_input("dst")){
        auto outprim = primitive_merge(list);
        set_output("prim", std::move(outprim));
    }
    else
    { // dage, weishenme buyong Assign jiedian ne?
        auto dst = get_input<PrimitiveObject>("dst");
        auto outprim = primitive_merge(list);
        *dst = *outprim;
        set_output("prim", std::move(dst));
    }
  }
};

ZENDEFNODE(PrimitiveMerge, {
    {"listPrim", "dst"},
    {"prim"},
    {},
    {"deprecated"},
});


}
