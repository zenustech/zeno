#include <zeno/zeno.h>
#include <zeno/oldzfx.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/ListObject.h>
#include <cassert>

#define HASH_MAX 1024

struct HashEntry {
    std::vector<int> pid;
};

struct HashTable {
    HashEntry entries[HASH_MAX];
};

static int hash3i(zeno::vec3i const &v) {
    return (v[0] * 985211 + v[1] * 54321 + v[2] * 3141592 + 142857) % HASH_MAX;
}

struct Buffer {
    float *base = nullptr;
    size_t count = 0;
    size_t stride = 0;
    int which = 0;
};

template <class F>
static void vectors_neighbor_wrangle
    ( zfx::Program const *prog
    , std::vector<zeno::vec3f> const &pos
    , std::vector<Buffer> const &chs
    , std::vector<float> const &pars
    , size_t size1, size_t size2
    , F const &get_neighbor
    ) {
    if (chs.size() == 0)
        return;
    #pragma omp parallel for
    for (int i1 = 0; i1 < size1; i1++) {
        zfx::Context ctx;
        for (int j = 0; j < pars.size(); j++) {
            ctx.regtable[j] = pars[j];
        }
        for (int j = 0; j < chs.size(); j++) {
            if (chs[j].which == 0)
                ctx.memtable[j] = chs[j].base + chs[j].stride * i1;
        }
        for (int i2: get_neighbor(pos[i1])) {
            for (int j = 0; j < chs.size(); j++) {
                if (chs[j].which == 1)
                    ctx.memtable[j] = chs[j].base + chs[j].stride * i2;
            }
            prog->execute(&ctx);
        }
    }
}

struct ParticlesNeighborWrangle : zeno::INode {
    virtual void apply() override {
        auto prim1 = get_input<zeno::PrimitiveObject>("prim1");
        auto prim2 = get_input<zeno::PrimitiveObject>("prim2");
        auto radius = get_input<zeno::NumericObject>("radius")->get<float>();

        auto code = get_input<zeno::StringObject>("zfxCode")->get();
        std::ostringstream oss;
        for (auto const &[key, attr]: prim1->m_attrs) {
            oss << "define ";
            std::visit([&oss] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) oss << "f3";
                else if constexpr (std::is_same_v<T, float>) oss << "f1";
                else oss << "unknown";
            }, attr);
            oss << " @" << key << '\n';
        }
        for (auto const &[key, attr]: prim2->m_attrs) {
            oss << "define ";
            std::visit([&oss] (auto const &v) {
                using T = std::decay_t<decltype(v[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) oss << "f3";
                else if constexpr (std::is_same_v<T, float>) oss << "f1";
                else oss << "unknown";
            }, attr);
            oss << " @" << key << ":j" << '\n';
        }

        auto params = get_input<zeno::ListObject>("params");
        std::vector<float> pars;
        std::vector<std::string> parnames;
        for (int i = 0; i < params->arr.size(); i++) {
            auto const &obj = params->arr[i];
            std::ostringstream keyss; keyss << "arg" << i;
            auto key = keyss.str();
            auto par = dynamic_cast<zeno::NumericObject *>(obj.get());
            oss << "define ";
            std::visit([&] (auto const &v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) {
                    oss << "f3";
                    pars.push_back(v[0]);
                    pars.push_back(v[1]);
                    pars.push_back(v[2]);
                    parnames.push_back(key + ".0");
                    parnames.push_back(key + ".1");
                    parnames.push_back(key + ".2");
                } else if constexpr (std::is_same_v<T, float>) {
                    oss << "f1";
                    pars.push_back(v);
                    parnames.push_back(key + ".0");
                } else oss << "unknown";
            }, par->value);
            oss << " " << key << '\n';
        }
        for (auto const &par: parnames) {
            oss << "parname " << par << '\n';
        }

        code = oss.str() + code;
        auto prog = zfx::compile_program(code);

        std::vector<Buffer> chs(prog->channels.size());
        for (int i = 0; i < chs.size(); i++) {
            auto channe = zfx::split_str(prog->channels[i], '.');
            auto chan = zfx::split_str(channe[0], ':');
            if (chan.size() == 1) {
                chan.push_back("i");
            }
            assert(chan.size() == 2);
            int dimid = 0;
            std::stringstream(channe[1]) >> dimid;
            Buffer iob;
            auto const &attr = prim1->attr(chan[0]);
            std::visit([&] (auto const &arr) {
                iob.base = (float *)arr.data() + dimid;
                iob.count = arr.size();
                iob.stride = sizeof(arr[0]) / sizeof(float);
            }, attr);
            iob.which = chan[1][0] - 'i';
            chs[i] = iob;
        }
        auto const &p1pos = prim1->attr<zeno::vec3f>("pos");
        auto const &p2pos = prim2->attr<zeno::vec3f>("pos");

        auto pmin = p2pos[0], pmax = p2pos[0];
        for (int i = 1; i < p2pos.size(); i++) {
            pmin = zeno::min(pmin, p2pos[i]);
            pmax = zeno::max(pmax, p2pos[i]);
        }
        //printf("pmin = %f %f %f\n", pmin[0], pmin[1], pmin[2]);
        //printf("pmax = %f %f %f\n", pmax[0], pmax[1], pmax[2]);

        auto psize = pmax - pmin;
        auto pinvscale = std::max(psize[0], std::max(psize[1], psize[2]));
        int nres = (int)(0.5f * pinvscale / radius);
        printf("[pnw] hash grid resolution: %d\n", nres);
        auto pscale = nres / pinvscale;
        auto hash3d = [=](zeno::vec3f const &p) {
            auto q = (p - pmin) * pscale;
            return hash3i(zeno::clamp(zeno::vec3i(q), 0, nres));
        };

        HashTable ht;
        for (int i = 0; i < p2pos.size(); i++) {
            auto m = hash3d(p2pos[i]);
            ht.entries[m].pid.push_back(i);
        }

        vectors_neighbor_wrangle(prog, p1pos,
            chs, pars, prim1->size(), prim2->size(), [&](zeno::vec3f const &p) {
                std::vector<int> res;
#if 1  // modify this to 0 to enjoy ultra-slow brute-force neighbor
                std::set<int> ms;
                for (int d = -1; d < 2; d++) for (int e = -1; e < 2; e++)
                for (int f = -1; f < 2; f++)
                    ms.insert(hash3d(p + radius * zeno::vec3f(d, e, f)));
                for (int m: ms) for (int i: ht.entries[m].pid) {
#else
                for (int i = 0; i < p2pos.size(); i++) {
#endif
                    if (zeno::length(p2pos[i] - p) <= radius) {
                        res.push_back(i);
                    }
                }
                return res;
            });

        set_output("prim1", std::move(prim1));
    }
};

ZENDEFNODE(ParticlesNeighborWrangle, {
    {"prim1", "prim2", "radius", "zfxCode", "params"},
    {"prim1"},
    {},
    {"zenofx"},
});
