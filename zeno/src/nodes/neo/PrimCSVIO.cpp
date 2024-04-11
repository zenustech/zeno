#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/fileio.h>
#include <fstream>

namespace zeno {
namespace {

static void dump(int const &v, std::ostream &fout) {
    fout << v;
}

static void dump(float const &v, std::ostream &fout) {
    fout << v;
}

template <size_t N, class T>
static void dump(vec<N, T> const &v, std::ostream &fout) {
    fout << v[0];
    for (int i = 1; i < N; i++)
        fout << ' ' << v[i];
}

template <class T>
void dump_csv(AttrVector<T> avec, std::ostream &fout) {
    fout << "pos";
    avec.template foreach_attr<AttrAcceptAll>([&] (auto const &key, auto &arr) {
        fout << ',' << key;
    });
    fout << '\n';
    for (int i = 0; i < avec.size(); i++) {
        dump(avec[i], fout);
        avec.template foreach_attr<AttrAcceptAll>([&] (auto const &key, auto &arr) {
            fout << ',';
            dump(arr[i], fout);
        });
        fout << '\n';
    }
}

struct WritePrimToCSV : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto path = get_input<StringObject>("path")->get();
        path = create_directories_when_write_file(path);
        std::ofstream fout(path);
        auto memb = invoker_variant(array_index(
                {"verts", "points", "lines", "tris", "quads", "loops", "polys"},
                get_input2<std::string>("type")),
            &PrimitiveObject::verts,
            &PrimitiveObject::points,
            &PrimitiveObject::lines,
            &PrimitiveObject::tris,
            &PrimitiveObject::quads,
            &PrimitiveObject::loops,
            &PrimitiveObject::polys);
        std::visit([&] (auto const &memb) {
            dump_csv(memb(*prim), fout);
        }, memb);
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(WritePrimToCSV,
        { /* inputs: */ {
        {"primitive", "prim"},
        {"writepath", "path"},
        {"enum verts points lines tris quads loops polys", "type", "verts"},
        }, /* outputs: */ {
        {"primitive", "prim"},
        }, /* params: */ {
        }, /* category: */ {
        "primitive",
        }});

}
}
