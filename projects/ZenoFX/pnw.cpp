#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include "dbg_printf.h"

namespace {

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
    int which = 0;
};

struct HashGrid : zeno::IObject {
    float inv_dx;
    float radius;
    float radius_sqr;
    float radius_sqr_min;
    std::vector<zeno::vec3f> const &refpos;

    std::vector<std::vector<int>> table;

//#define DILEI
#define XUBEN

    int hash(int x, int y, int z) {
#ifdef XUBEN
        return x + y * gridRes[0] + z * gridRes[0] * gridRes[1];
#else
        return ((73856093 * x) ^ (19349663 * y) ^ (83492791 * z)) % table.size();
#endif
    }

#ifdef XUBEN
    zeno::vec3f pMin, pMax;
    zeno::vec3i gridRes;
#endif

    HashGrid(std::vector<zeno::vec3f> const &refpos_,
            float radius_, float radius_min)
        : refpos(refpos_) {

        radius = radius_;
        radius_sqr = radius * radius;
        radius_sqr_min = radius_min < 0.f ? -1.f : radius_min * radius_min;
#ifdef DILEI
        inv_dx = 0.5f / radius;
#else
        inv_dx = 1.0f / radius;
#endif

#ifdef XUBEN
        pMin = refpos[0];
        pMax = refpos[0];
        for (int i = 1; i < refpos.size(); i++) {
            auto coor = refpos[i];
            pMin = zeno::min(pMin, coor);
            pMax = zeno::max(pMax, coor);
        }
        pMin -= radius;
        pMax += radius;
        gridRes = zeno::toint(zeno::floor((pMax - pMin) * inv_dx)) + 1;

        dbg_printf("grid res: %dx%dx%d\n", gridRes[0], gridRes[1], gridRes[2]);
        table.clear();
        table.resize(gridRes[0] * gridRes[1] * gridRes[2]);
#else
        int table_size = refpos.size() / 8;
        dbg_printf("table size: %d\n", table_size);
        table.clear();
        table.resize(table_size);
#endif

        for (int i = 0; i < refpos.size(); i++) {
#ifdef XUBEN
            auto coor = zeno::toint(zeno::floor((refpos[i] - pMin) * inv_dx));
#else
            auto coor = zeno::toint(zeno::floor(refpos[i] * inv_dx));
#endif
            auto key = hash(coor[0], coor[1], coor[2]);
            table[key].push_back(i);
        }
    }

    template <class F>
    void iter_neighbors(zeno::vec3f const &pos, F const &f) {
#ifdef XUBEN
#ifdef DILEI
        auto coor = zeno::toint(zeno::floor((pos - pMin) * inv_dx - 0.5f));
#else
        auto coor = zeno::toint(zeno::floor((pos - pMin) * inv_dx));
#endif
#else
#ifdef DILEI
        auto coor = zeno::toint(zeno::floor(pos * inv_dx - 0.5f));
#else
        auto coor = zeno::toint(zeno::floor(pos * inv_dx));
#endif
#endif
#ifdef DILEI
        for (int dz = 0; dz < 2; dz++) {
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
#else
        for (int dz = -1; dz < 2; dz++) {
            for (int dy = -1; dy < 2; dy++) {
                for (int dx = -1; dx < 2; dx++) {
#endif
                    int key = hash(coor[0] + dx, coor[1] + dy, coor[2] + dz);
                    for (int pid: table[key]) {
                        auto dist = refpos[pid] - pos;
                        auto dis2 = zeno::dot(dist, dist);
                        if (dis2 <= radius_sqr && dis2 > radius_sqr_min) {
                            f(pid);
                        }
                    }
                }
            }
        }
    }
};

static void vectors_wrangle
    ( zfx::x64::Executable *exec
    , std::vector<Buffer> const &chs
    , std::vector<Buffer> const &chs2
    , std::vector<zeno::vec3f> const &pos
    , HashGrid *hashgrid
    ) {
    if (chs.size() == 0)
        return;

    #pragma omp parallel for
    for (int i = 0; i < pos.size(); i++) {
        auto ctx = exec->make_context();
        for (int k = 0; k < chs.size(); k++) {
            if (!chs[k].which)
                ctx.channel(k)[0] = chs[k].base[chs[k].stride * i];
        }
        hashgrid->iter_neighbors(pos[i], [&] (int pid) {
            for (int k = 0; k < chs.size(); k++) {
                if (chs[k].which)
                    ctx.channel(k)[0] = chs2[k].base[chs2[k].stride * pid];
            }
            ctx.execute();
        });
        for (int k = 0; k < chs.size(); k++) {
            if (!chs[k].which)
                chs[k].base[chs[k].stride * i] = ctx.channel(k)[0];
        }
    }
}

struct ParticlesBuildHashGrid : zeno::INode {
    virtual void apply() override {
        auto primNei = get_input<zeno::PrimitiveObject>("primNei");
        float radius = get_input<zeno::NumericObject>("radius")->get<float>();
        float radiusMin = has_input("radiusMin") ?
            get_input<zeno::NumericObject>("radiusMin")->get<float>() : -1.f;
        auto hashgrid = std::make_shared<HashGrid>(
                primNei->attr<zeno::vec3f>("pos"), radius, radiusMin);
        set_output("hashGrid", std::move(hashgrid));
    }
};

ZENDEFNODE(ParticlesBuildHashGrid, {
    {{"PrimitiveObject", "primNei"}, {"numeric:float", "radius"}, {"numeric:float", "radiusMin"}},
    {{"hashgrid", "hashGrid"}},
    {},
    {"zenofx"},
});

struct ParticlesNeighborWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto primNei = get_input<zeno::PrimitiveObject>("primNei");
        auto hashgrid = get_input<HashGrid>("hashGrid");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        zfx::Options opts(zfx::Options::for_x64);
        opts.detect_new_symbols = true;
        prim->foreach_attr([&] (auto const &key, auto const &attr) {
            int dim = ([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            })(attr);
            dbg_printf("define symbol: @%s dim %d\n", key.c_str(), dim);
            opts.define_symbol('@' + key, dim);
        });
        primNei->foreach_attr([&] (auto const &key, auto const &attr) {
            int dim = ([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            })(attr);
            dbg_printf("define symbol: @@%s dim %d\n", key.c_str(), dim);
            opts.define_symbol("@@" + key, dim);
        });

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, obj]: params->lut) {
            auto key = '$' + key_;
            if (auto o = zeno::silent_any_cast<zeno::NumericValue>(obj); o.has_value()) {
                auto par = o.value();
                auto dim = std::visit([&] (auto const &v) {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<T, zeno::vec3f>) {
                        parvals.push_back(v[0]);
                        parvals.push_back(v[1]);
                        parvals.push_back(v[2]);
                        parnames.emplace_back(key, 0);
                        parnames.emplace_back(key, 1);
                        parnames.emplace_back(key, 2);
                        return 3;
                    } else if constexpr (std::is_same_v<T, float>) {
                        parvals.push_back(v);
                        parnames.emplace_back(key, 0);
                        return 1;
                    } else {
                        printf("invalid parameter type encountered: `%s`\n",
                                typeid(T).name());
                        return 0;
                    }
                }, par);
                dbg_printf("define param: %s dim %d\n", key.c_str(), dim);
                opts.define_param(key, dim);
            }
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        for (auto const &[name, dim]: prog->newsyms) {
            dbg_printf("auto-defined new attribute: %s with dim %d\n",
                    name.c_str(), dim);
            assert(name[0] == '@');
            if (name[1] == '@') {
                dbg_printf("ERROR: cannot define new attribute %s on primNei\n",
                        name.c_str());
            }
            auto key = name.substr(1);
            if (dim == 3) {
                prim->add_attr<zeno::vec3f>(key);
            } else if (dim == 1) {
                prim->add_attr<float>(key);
            } else {
                dbg_printf("ERROR: bad attribute dimension for primitive: %d\n",
                    dim);
                abort();
            }
        }

        for (int i = 0; i < prog->params.size(); i++) {
            auto [name, dimid] = prog->params[i];
            dbg_printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '$');
            auto it = std::find(parnames.begin(),
                parnames.end(), std::pair{name, dimid});
            auto value = parvals.at(it - parnames.begin());
            dbg_printf("(valued %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }

        std::vector<Buffer> chs(prog->symbols.size());
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            dbg_printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            Buffer iob;
            zeno::PrimitiveObject *primPtr;
            if (name[1] == '@') {
                name = name.substr(2);
                primPtr = primNei.get();
                iob.which = 1;
            } else {
                name = name.substr(1);
                primPtr = prim.get();
                iob.which = 0;
            }
            prim->attr_visit(name, [&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            });
            chs[i] = iob;
        }
        std::vector<Buffer> chs2(prog->symbols.size());
        for (int i = 0; i < chs2.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            dbg_printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '@');
            Buffer iob;
            zeno::PrimitiveObject *primPtr;
            if (name[1] == '@') {
                name = name.substr(2);
                primPtr = primNei.get();
                iob.which = 1;
            } else {
                name = name.substr(1);
                primPtr = prim.get();
                iob.which = 0;
            }
            primNei->attr_visit(name, [&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            });
            chs2[i] = iob;
        }

        vectors_wrangle(exec, chs, chs2, prim->attr<zeno::vec3f>("pos"),
                hashgrid.get());

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesNeighborWrangle, {
    {{"PrimitiveObject", "prim"}, {"PrimitiveObject", "primNei"}, {"HashGrid", "hashGrid"},
     {"string", "zfxCode"}, {"DictObject:NumericObject", "params"}},
    {{"PrimitiveObject", "prim"}},
    {},
    {"zenofx"},
});

}
