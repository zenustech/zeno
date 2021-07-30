#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/DictObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>

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
    zeno::vec3i GridRes;
    zeno::vec3f pMin;
    zeno::vec3f pMax;
    std::vector<zeno::vec3f> const &refpos;

    using CoordType = std::tuple<int, int, int>;
    std::vector<std::vector<int>> table;

    int hash(int x, int y, int z) {
        return x + y * GridRes[0] + z * GridRes[0] * GridRes[1];
    }

    HashGrid(std::vector<zeno::vec3f> const &refpos_,
            float radius_, float radius_min)
        : refpos(refpos_) {
        for(int i=0;i<table.size();++i)
            table[i].clear();

        radius = radius_;
        radius_sqr = radius * radius;
        radius_sqr_min = radius_min < 0.f ? -1.f : radius_min * radius_min;
        inv_dx = 1.0f / radius;
        pMin = zeno::vec3f(10000000,10000000,1000000);
        pMax = zeno::vec3f(-1000000,-1000000,-1000000);
        // find pmin and pmax
        for(int i = 0; i < refpos.size(); i++){
            auto coor = refpos[i];
            for(int j=0;j<3;++j)
            {
                if(pMin[j] > coor[j])
                    pMin[j] = coor[j];
                if(pMax[j] < coor[j])
                    pMax[j] = coor[j];
            }
        }
        pMin += zeno::vec3f(-radius, -radius, -radius);
        pMax += zeno::vec3f(radius, radius, radius);
        GridRes = zeno::floor((pMax - pMin) * inv_dx) + zeno::vec3i(1,1,1);
        table.resize(GridRes[0] * GridRes[1] * GridRes[2]);
        // printf("hash table size is %d, particle num is %d\n", table.size(), refpos.size());
        for (int i = 0; i < refpos.size(); i++) {
            auto coor = zeno::toint(zeno::floor((refpos[i]-pMin) * inv_dx));
            auto key = hash(coor[0], coor[1], coor[2]);
            table[key].push_back(i);
        }
    }

    template <class F>
    void iter_neighbors(zeno::vec3f const &pos, F const &f) {
        auto coor = zeno::toint(zeno::floor((pos - pMin) * inv_dx));
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int key = hash(coor[0] + dx, coor[1] + dy, coor[2] + dz);
                    if(key < 0 || key >= table.size())
                        continue;
                    for (int pid: table[key]) {
                        auto dist = refpos[pid] - pos;
                        auto dis2 = zeno::dot(dist, dist);
                        if (dis2 <= radius_sqr && dis2 >= radius_sqr_min) {
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
                    ctx.channel(k)[0] = chs[k].base[chs[k].stride * pid];
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
            get_input<zeno::NumericObject>("radiusMin")->get<float>() : 0.f;
        auto hashgrid = std::make_shared<HashGrid>(
                primNei->attr<zeno::vec3f>("pos"), radius, radiusMin);
        set_output("hashGrid", std::move(hashgrid));
    }
};

ZENDEFNODE(ParticlesBuildHashGrid, {
    {{"primitive", "primNei"}, {"numeric:float", "radius"}, {"numeric:float", "radiusMin"}},
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
        for (auto const &[key, attr]: prim->m_attrs) {
            int dim = std::visit([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            }, attr);
            printf("define symbol: @%s dim %d\n", key.c_str(), dim);
            opts.define_symbol('@' + key, dim);
        }
        for (auto const &[key, attr]: primNei->m_attrs) {
            int dim = std::visit([] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) return 3;
                else if constexpr (std::is_same_v<T, float>) return 1;
                else return 0;
            }, attr);
            printf("define symbol: @@%s dim %d\n", key.c_str(), dim);
            opts.define_symbol("@@" + key, dim);
        }

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, obj]: params->lut) {
            auto key = '$' + key_;
            auto par = dynamic_cast<zeno::NumericObject *>(obj.get());
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
                } else return 0;
            }, par->value);
            printf("define param: %s dim %d\n", key.c_str(), dim);
            opts.define_param(key, dim);
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        for (auto const &[name, dim]: prog->newsyms) {
            printf("auto-defined new attribute: %s with dim %d\n",
                    name.c_str(), dim);
            assert(name[0] == '@');
            if (name[1] == '@') {
                printf("ERROR: cannot define new attribute %s on primNei\n",
                        name.c_str());
            }
            auto key = name.substr(1);
            if (dim == 3) {
                prim->add_attr<zeno::vec3f>(key);
            } else if (dim == 1) {
                prim->add_attr<float>(key);
            } else {
                printf("ERROR: bad attribute dimension for primitive: %d\n",
                    dim);
                abort();
            }
        }

        for (int i = 0; i < prog->params.size(); i++) {
            auto [name, dimid] = prog->params[i];
            printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
            assert(name[0] == '$');
            auto it = std::find(parnames.begin(),
                parnames.end(), std::pair{name, dimid});
            auto value = parvals.at(it - parnames.begin());
            printf("(valued %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }

        std::vector<Buffer> chs(prog->symbols.size());
        for (int i = 0; i < chs.size(); i++) {
            auto [name, dimid] = prog->symbols[i];
            printf("channel %d: %s.%d\n", i, name.c_str(), dimid);
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
            auto const &attr = primPtr->attr(name);
            std::visit([&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            }, attr);
            chs[i] = iob;
        }

        vectors_wrangle(exec, chs, prim->attr<zeno::vec3f>("pos"),
                hashgrid.get());

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesNeighborWrangle, {
    {{"primitive", "prim"}, {"primitive", "primNei"}, {"hashgrid", "hashGrid"},
     {"string", "zfxCode"}, {"dict:numeric", "params"}},
    {{"primitive", "prim"}},
    {},
    {"zenofx"},
});

}
