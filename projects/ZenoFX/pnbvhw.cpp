#include "LinearBvh.h"
#include "dbg_printf.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/zeno.h>
#include <zfx/x64.h>
#include <zfx/zfx.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

struct Buffer {
  float *base = nullptr;
  size_t count = 0;
  size_t stride = 0;
  int which = 0;
};

static void bvh_vectors_wrangle(zfx::x64::Executable *exec,
                                std::vector<Buffer> const &chs,
                                std::vector<Buffer> const &chs2,
                                std::vector<zeno::vec3f> const &pos,
                                zeno::LBvh *lbvh) {
  if (chs.size() == 0)
    return;

#pragma omp parallel for
  for (int i = 0; i < pos.size(); i++) {
    auto ctx = exec->make_context();
    for (int k = 0; k < chs.size(); k++) {
      if (!chs[k].which)
        ctx.channel(k)[0] = chs[k].base[chs[k].stride * i];
    }
    lbvh->iter_neighbors(pos[i], [&](int pid) {
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

struct ParticlesBuildBvh : zeno::INode {
  virtual void apply() override {
    auto primNei = get_input<zeno::PrimitiveObject>("primNei");
    float radius = get_input<zeno::NumericObject>("radius")->get<float>();
    float radiusMin =
        has_input("radiusMin")
            ? get_input<zeno::NumericObject>("radiusMin")->get<float>()
            : -1.f;
    auto lbvh = std::make_shared<zeno::LBvh>(
        primNei, radius, zeno::LBvh::element_c<zeno::LBvh::element_e::point>);
    set_output("lbvh", std::move(lbvh));
  }
};

ZENDEFNODE(ParticlesBuildBvh, {
                                  {{"PrimitiveObject", "primNei"},
                                   {"numeric:float", "radius"},
                                   {"numeric:float", "radiusMin"}},
                                  {{"LBvh", "lbvh"}},
                                  {},
                                  {"zenofx"},
                              });

struct BuildPrimitiveBvh : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<zeno::PrimitiveObject>("prim");
    float thickness =
        has_input("thickness")
            ? get_input<zeno::NumericObject>("thickness")->get<float>()
            : 0.f;
    auto primType = get_param<std::string>("prim_type");
    if (primType == "auto") {
      auto lbvh = std::make_shared<zeno::LBvh>(prim, thickness);
      set_output("lbvh", std::move(lbvh));
    } else if (primType == "point") {
      auto lbvh = std::make_shared<zeno::LBvh>(
          prim, thickness, zeno::LBvh::element_c<zeno::LBvh::element_e::point>);
      set_output("lbvh", std::move(lbvh));
    } else if (primType == "line") {
      auto lbvh = std::make_shared<zeno::LBvh>(
          prim, thickness, zeno::LBvh::element_c<zeno::LBvh::element_e::line>);
      set_output("lbvh", std::move(lbvh));
    } else if (primType == "tri") {
      auto lbvh = std::make_shared<zeno::LBvh>(
          prim, thickness, zeno::LBvh::element_c<zeno::LBvh::element_e::tri>);
      set_output("lbvh", std::move(lbvh));
    } else if (primType == "quad") {
      auto lbvh = std::make_shared<zeno::LBvh>(
          prim, thickness, zeno::LBvh::element_c<zeno::LBvh::element_e::tet>);
      set_output("lbvh", std::move(lbvh));
    }
  }
};

ZENDEFNODE(BuildPrimitiveBvh,
           {
               {{"PrimitiveObject", "prim"}, {"numeric:float", "thickness"}},
               {{"LBvh", "lbvh"}},
               {{"enum auto point line tri quad", "prim_type", "auto"}},
               {"zenofx"},
           });

struct RefitPrimitiveBvh : zeno::INode {
  virtual void apply() override {
    auto lbvh = get_input<zeno::LBvh>("lbvh");
    lbvh->refit();
    set_output("lbvh", std::move(lbvh));
  }
};

ZENDEFNODE(RefitPrimitiveBvh, {
                                  {{"LBvh", "lbvh"}},
                                  {{"LBvh", "lbvh"}},
                                  {},
                                  {"zenofx"},
                              });

struct QueryNearestPrimitive : zeno::INode {
  struct KVPair {
    zeno::vec3f w;
    float dist;
    int pid;
    bool operator<(const KVPair &o) const noexcept { return dist < o.dist; }
  };
  virtual void apply() override {
    using namespace zeno;

    auto lbvh = get_input<LBvh>("lbvh");
    auto line = std::make_shared<PrimitiveObject>();

    using Ti = typename LBvh::Ti;
    Ti pid = 0;
    Ti bvhId = -1;
    float dist = std::numeric_limits<float>::max();
    zeno::vec3f w{0.f, 0.f, 0.f};
    if (has_input<PrimitiveObject>("prim")) {
      auto prim = get_input<PrimitiveObject>("prim");

      auto idTag = get_input2<std::string>("idTag");
      auto distTag = get_input2<std::string>("distTag");
      auto weightTag = get_input2<std::string>("weightTag");

      auto &bvhids = prim->add_attr<float>(idTag);
      auto &dists = prim->add_attr<float>(distTag);
      auto &ws = prim->add_attr<zeno::vec3f>(weightTag);

      std::vector<KVPair> kvs(prim->size());
      std::vector<Ti> ids(prim->size(), -1);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (Ti i = 0; i < prim->size(); ++i) {
        kvs[i].dist = std::numeric_limits<float>::max();
        kvs[i].pid = i;
        kvs[i].w = lbvh->find_nearest(prim->verts[i], ids[i], kvs[i].dist);
        // record info as attribs
        bvhids[i] = ids[i];
        dists[i] = kvs[i].dist;
        ws[i] = kvs[i].w;
      }

      KVPair mi{zeno::vec3f{0.f, 0.f, 0.f}, std::numeric_limits<float>::max(), -1};
// ref:
// https://stackoverflow.com/questions/28258590/using-openmp-to-get-the-index-of-minimum-element-parallelly
#ifndef _MSC_VER
#if defined(_OPENMP)
#pragma omp declare reduction(minimum:KVPair                                   \
                              : omp_out = omp_in < omp_out ? omp_in : omp_out) \
    initializer(omp_priv = KVPair{zeno::vec3f{0.f, 0.f, 0.f}, std::numeric_limits <float>::max(), -1})
#pragma omp parallel for reduction(minimum : mi)
#endif
#endif
      for (Ti i = 0; i < kvs.size(); ++i) {
        if (kvs[i].dist < mi.dist)
          mi = kvs[i];
      }
      pid = mi.pid;
      dist = mi.dist;
      w = mi.w;
      bvhId = ids[pid];
      line->verts.push_back(prim->verts[pid]);
#if 0
      fmt::print("done nearest reduction. dist: {}, bvh[{}] (of {})-prim[{}]"
                 "(of {})\n",
                 dist, bvhId, lbvh->getNumLeaves(), pid, prim->size());
#endif
    } else if (has_input<NumericObject>("prim")) {
      auto p = get_input<NumericObject>("prim")->get<vec3f>();
      w = lbvh->find_nearest(p, bvhId, dist);
      line->verts.push_back(p);
    } else
      throw std::runtime_error("unknown primitive kind (only supports "
                               "PrimitiveObject and NumericObject::vec3f).");

    line->verts.push_back(lbvh->retrievePrimitiveCenter(bvhId, w));
    line->lines.push_back({0, 1});

    set_output("primid", std::make_shared<NumericObject>(pid));
    set_output("bvh_primid", std::make_shared<NumericObject>(bvhId));
    set_output("dist", std::make_shared<NumericObject>(dist));
    set_output("bvh_prim", lbvh->retrievePrimitive(bvhId));
    set_output("segment", std::move(line));
  }
};

ZENDEFNODE(QueryNearestPrimitive, {
                                      {{"prim"}, {"LBvh", "lbvh"},
                                      {"string", "idTag", "bvh_id"},
                                      {"string", "distTag", "bvh_dist"},
                                      {"string", "weightTag", "bvh_ws"}
                                      },
                                      {{"NumericObject", "primid"},
                                       {"NumericObject", "bvh_primid"},
                                       {"NumericObject", "dist"},
                                       {"PrimitiveObject", "bvh_prim"},
                                       {"PrimitiveObject", "segment"}},
                                      {},
                                      {"zenofx"},
                                  });

struct ParticlesNeighborBvhWrangle : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<zeno::PrimitiveObject>("prim");
    auto primNei = get_input<zeno::PrimitiveObject>("primNei");
    auto lbvh = get_input<zeno::LBvh>("lbvh");
    auto code = get_input<zeno::StringObject>("zfxCode")->get();

    zfx::Options opts(zfx::Options::for_x64);
    opts.detect_new_symbols = true;
    prim->foreach_attr([&](auto const &key, auto const &attr) {
      int dim = ([](auto const &v) {
        using T = std::decay_t<decltype(v[0])>;
        if constexpr (std::is_same_v<T, zeno::vec3f>)
          return 3;
        else if constexpr (std::is_same_v<T, float>)
          return 1;
        else
          return 0;
      })(attr);
      dbg_printf("define symbol: @%s dim %d\n", key.c_str(), dim);
      opts.define_symbol('@' + key, dim);
    });
    primNei->foreach_attr([&](auto const &key, auto const &attr) {
      int dim = ([](auto const &v) {
        using T = std::decay_t<decltype(v[0])>;
        if constexpr (std::is_same_v<T, zeno::vec3f>)
          return 3;
        else if constexpr (std::is_same_v<T, float>)
          return 1;
        else
          return 0;
      })(attr);
      dbg_printf("define symbol: @@%s dim %d\n", key.c_str(), dim);
      opts.define_symbol("@@" + key, dim);
    });

    auto params = has_input("params") ? get_input<zeno::DictObject>("params")
                                      : std::make_shared<zeno::DictObject>();
    std::vector<float> parvals;
    std::vector<std::pair<std::string, int>> parnames;
    for (auto const &[key_, obj] : params->lut) {
      auto key = '$' + key_;
      if (auto o = zeno::silent_any_cast<zeno::NumericValue>(obj);
          o.has_value()) {
        auto par = o.value();
        auto dim = std::visit(
            [&](auto const &v) {
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
            },
            par);
        dbg_printf("define param: %s dim %d\n", key.c_str(), dim);
        opts.define_param(key, dim);
      }
    }

    auto prog = compiler.compile(code, opts);
    auto exec = assembler.assemble(prog->assembly);

    for (auto const &[name, dim] : prog->newsyms) {
      dbg_printf("auto-defined new attribute: %s with dim %d\n", name.c_str(),
                 dim);
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
        dbg_printf("ERROR: bad attribute dimension for primitive: %d\n", dim);
        abort();
      }
    }

    for (int i = 0; i < prog->params.size(); i++) {
      auto [name, dimid] = prog->params[i];
      dbg_printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
      assert(name[0] == '$');
      auto it =
          std::find(parnames.begin(), parnames.end(), std::pair{name, dimid});
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
      prim->attr_visit(name, [&, dimid_ = dimid](auto const &arr) {
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
      primNei->attr_visit(name, [&, dimid_ = dimid](auto const &arr) {
        iob.base = (float *)arr.data() + dimid_;
        iob.count = arr.size();
        iob.stride = sizeof(arr[0]) / sizeof(float);
      });
      chs2[i] = iob;
    }

    bvh_vectors_wrangle(exec, chs, chs2, prim->attr<zeno::vec3f>("pos"),
                        lbvh.get());

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(ParticlesNeighborBvhWrangle,
           {
               {{"PrimitiveObject", "prim"},
                {"PrimitiveObject", "primNei"},
                {"LBvh", "lbvh"},
                {"string", "zfxCode"},
                {"DictObject:NumericObject", "params"}},
               {{"PrimitiveObject", "prim"}},
               {},
               {"zenofx"},
           });

} // namespace
