#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/DictObject.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include <tuple>
#include <unordered_map>

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
    int which = 0;
};

struct HashGrid {
    float inv_radius;
    float radius_squared;
    zeno::vec3f *pos_data;

    using CoordType = std::tuple<int, int, int>;
    std::array<std::vector<int>, 1024> table;
    
    int hash(int x, int y, int z) {
        return ((73856093 * x) ^ (19349663 * y) ^ (83492791 * z)) % table.size();
    }

    void build(std::vector<zeno::vec3f> const &pos, float radius) {
        for (auto &ent: table) {
            ent.clear();
        }

        radius_squared = radius * radius;
        inv_radius = 1.f / radius;
        pos_data = const_cast<zeno::vec3f *>(pos.data());

        for (int i = 0; i < pos.size(); i++) {
            auto coor = zeno::toint(zeno::floor(pos[i] * inv_radius));
            auto key = hash(coor[0], coor[1], coor[2]);
            table[key].push_back(i);
        }
    }

    template <class F>
    void iter_neighbors(zeno::vec3f const &pos, F const &f) {
        auto coor = zeno::toint(zeno::floor(pos * inv_radius));
        auto key = hash(coor[0], coor[1], coor[2]);
        for (auto pid: table[key]) {
            auto dist = pos_data[pid] - pos;
            auto dis2 = zeno::dot(dist, dist);
            if (dis2 <= radius_squared) {
                f(pid);
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
                    ctx.channel(k)[0] = chs[k].base[chs[k].stride * i];
            }
            ctx.execute();
        });
        for (int k = 0; k < chs.size(); k++) {
            if (!chs[k].which)
                chs[k].base[chs[k].stride * i] = ctx.channel(k)[0];
        }
    }
}

struct ParticlesNeighborWrangle : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto primNei = get_input<zeno::PrimitiveObject>("primNei");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();
        auto radius = get_input<zeno::NumericObject>("radius")->get<float>();

        zfx::Options opts(zfx::Options::for_x64);
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

        std::vector<float> pars(prog->params.size());
        for (int i = 0; i < pars.size(); i++) {
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
            if (name[0] == '@') {
                name = name.substr(2);
                primPtr = primNei.get();
                iob.which = 0;
            } else {
                name = name.substr(1);
                primPtr = prim.get();
                iob.which = 1;
            }
            auto const &attr = primPtr->attr(name);
            std::visit([&, dimid_ = dimid] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid_;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            }, attr);
            chs[i] = iob;
        }

        std::unique_ptr<HashGrid> hashgrid;
        hashgrid->build(primNei->attr<zeno::vec3f>("pos"), radius);
        vectors_wrangle(exec, chs, prim->attr<zeno::vec3f>("pos"), hashgrid.get());

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(ParticlesNeighborWrangle, {
    {"prim", "primNei", "zfxCode", "params", "radius"},
    {"prim"},
    {},
    {"zenofx"},
});
