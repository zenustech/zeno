#include "LinearBvh.h"
#include <limits>
#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Graph.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include "dbg_printf.h"
#include <cmath>
#include <atomic>
#include <algorithm>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace zeno {

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

struct Buffer {
  float *base = nullptr;
  size_t count = 0;
  size_t stride = 0;
  int which = 0;
};

static void sorted_bvh_vectors_wrangle(zfx::x64::Executable *exec,
                                std::vector<Buffer> const &chs,
                                std::vector<Buffer> const &chs2,
                                std::vector<zeno::vec3f> const &pos,
                                std::vector<zeno::vec3f> const &opos,
                                bool isBox, float radius2, int upper,
                                zeno::LBvh *lbvh) {
  if (chs.size() == 0)
    return;

  if (upper < 0)
    upper = std::numeric_limits<int>::max();

#pragma omp parallel for
  for (int i = 0; i < pos.size(); i++) {
    using pair = std::pair<float, int>;
    std::vector<pair> neighbors;
    auto ctx = exec->make_context();
    for (int k = 0; k < chs.size(); k++) {
      if (!chs[k].which)
        ctx.channel(k)[0] = chs[k].base[chs[k].stride * i];
    }
    /// count
    lbvh->iter_neighbors(pos[i], [&](int pid) {
      auto dist2 = lengthSquared(pos[i] - opos[pid]);
      if (!isBox)
        if (dist2 > radius2)
          return;
      neighbors.push_back(std::make_pair(dist2, pid));
    });
    std::sort(std::begin(neighbors), std::end(neighbors));
    int id = 0;
    for (const auto &neighbor : neighbors) {
      if (id++ >= upper) break;
      for (int k = 0; k < chs.size(); k++) {
        if (chs[k].which)
          ctx.channel(k)[0] = chs2[k].base[chs2[k].stride * neighbor.second];
      }
      ctx.execute();
    }
    for (int k = 0; k < chs.size(); k++) {
      if (!chs[k].which)
        chs[k].base[chs[k].stride * i] = ctx.channel(k)[0];
    }
  }
}

static void bvh_vectors_wrangle(zfx::x64::Executable *exec,
                                std::vector<Buffer> const &chs,
                                std::vector<Buffer> const &chs2,
                                std::vector<zeno::vec3f> const &pos,
                                std::vector<zeno::vec3f> const &opos,
                                bool isBox, float radius2,
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
      if (!isBox)
        if (lengthSquared(pos[i] - opos[pid]) > radius2)
          return;
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
                                   {"float", "radius"},
                                   {"float", "radiusMin"}},
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
               {{"PrimitiveObject", "prim"}, {"float", "thickness", "0"}},
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
      auto closestPointTag = get_input2<std::string>("closestPointTag");

      auto &bvhids = prim->add_attr<float>(idTag);
      auto &dists = prim->add_attr<float>(distTag);
      auto &ws = prim->add_attr<zeno::vec3f>(weightTag);
      auto &closestPoints = prim->add_attr<zeno::vec3f>(closestPointTag);

      std::vector<KVPair> kvs(prim->size());
      std::vector<Ti> ids(prim->size(), -1);
#if defined(_OPENMP)
#pragma omp parallel for schedule(guided, 4)
#endif
      for (Ti i = 0; i < prim->size(); ++i) {
        kvs[i].dist = std::numeric_limits<float>::max();
        kvs[i].pid = i;
        kvs[i].w = lbvh->find_nearest(prim->verts[i], ids[i], kvs[i].dist);
        // record info as attribs
        bvhids[i] = ids[i];
        dists[i] = kvs[i].dist;
        ws[i] = kvs[i].w;
        closestPoints[i] = lbvh->retrievePrimitiveCenter(ids[i], kvs[i].w);
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
                                      {"string", "closestPointTag", "cp"},
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

        // BEGIN张心欣快乐自动加@IND
        if (auto pos = code.find("@IND"); pos != code.npos && (code.size() <= pos + 4 || !(isalnum(code[pos + 4]) || strchr("_@$", code[pos + 4]))) && (pos == 0 || !(isalnum(code[pos - 1]) || strchr("_@$", code[pos - 1])))) {
            auto &indatt = prim->verts.add_attr<float>("IND");
            for (size_t i = 0; i < indatt.size(); i++) indatt[i] = float(i);
        }
        if (auto pos = code.find("@@IND"); pos != code.npos && (code.size() <= pos + 4 || !(isalnum(code[pos + 4]) || strchr("_@$", code[pos + 4]))) && (pos == 0 || !(isalnum(code[pos - 1]) || strchr("_@$", code[pos - 1])))) {
            auto &indatt = primNei->verts.add_attr<float>("IND");
            for (size_t i = 0; i < indatt.size(); i++) indatt[i] = float(i);
        }
        // END张心欣快乐自动加@IND

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
    {
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        auto const &gs = *this->getGlobalState();
        params->lut["PI"] = objectFromLiterial((float)(std::atan(1.f) * 4));
        params->lut["F"] = objectFromLiterial((float)gs.frameid);
        params->lut["DT"] = objectFromLiterial(gs.frame_time);
        params->lut["T"] = objectFromLiterial(gs.frame_time * gs.frameid + gs.frame_time_elapsed);
        // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
        for (auto const &[key, ref]: getThisGraph()->portalIns) {
            if (auto i = code.find('$' + key); i != std::string::npos) {
                i = i + key.size() + 1;
                if (code.size() <= i || !std::isalnum(code[i])) {
                    if (params->lut.count(key)) continue;
                    dbg_printf("ref portal %s\n", key.c_str());
                    auto res = getThisGraph()->callTempNode("PortalOut",
                          {{"name:", objectFromLiterial(key)}}).at("port");
                    params->lut[key] = std::move(res);
                }
            }
        }
        // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
        // BEGIN伺候心欣伺候懒得extract出变量了
        std::vector<std::string> keys;
        for (auto const &[key, val]: params->lut) {
            keys.push_back(key);
        }
        for (auto const &key: keys) {
            if (!dynamic_cast<zeno::NumericObject*>(params->lut.at(key).get())) {
                dbg_printf("ignored non-numeric %s\n", key.c_str());
                params->lut.erase(key);
            }
        }
        // END伺候心欣伺候懒得extract出变量了
        }
    std::vector<float> parvals;
    std::vector<std::pair<std::string, int>> parnames;
    for (auto const &[key_, par]: params->getLiterial<zeno::NumericValue>()) {
            auto key = '$' + key_;
                auto dim = std::visit([&] (auto const &v) {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_convertible_v<T, zeno::vec3f>) {
                        parvals.push_back(v[0]);
                        parvals.push_back(v[1]);
                        parvals.push_back(v[2]);
                        parnames.emplace_back(key, 0);
                        parnames.emplace_back(key, 1);
                        parnames.emplace_back(key, 2);
                        return 3;
                    } else if constexpr (std::is_convertible_v<T, float>) {
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
            //auto par = zeno::safe_any_cast<zeno::NumericValue>(obj);
            
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
                        primNei->attr<zeno::vec3f>("pos"), get_input2<bool>("is_box"),
                        lbvh.get()->thickness * lbvh.get()->thickness, lbvh.get());

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(ParticlesNeighborBvhWrangle,
           {
               {{"PrimitiveObject", "prim"},
                {"PrimitiveObject", "primNei"},
                {"LBvh", "lbvh"},
                {"bool", "is_box", "1"},
                {"string", "zfxCode"},
                {"DictObject:NumericObject", "params"}},
               {{"PrimitiveObject", "prim"}},
               {},
               {"zenofx"},
           });

struct ParticlesNeighborBvhWrangleSorted : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<zeno::PrimitiveObject>("prim");
    auto primNei = get_input<zeno::PrimitiveObject>("primNei");
    auto lbvh = get_input<zeno::LBvh>("lbvh");
    auto code = get_input<zeno::StringObject>("zfxCode")->get();

        // BEGIN张心欣快乐自动加@IND
        if (auto pos = code.find("@IND"); pos != code.npos && (code.size() <= pos + 4 || !(isalnum(code[pos + 4]) || strchr("_@$", code[pos + 4]))) && (pos == 0 || !(isalnum(code[pos - 1]) || strchr("_@$", code[pos - 1])))) {
            auto &indatt = prim->verts.add_attr<float>("IND");
            for (size_t i = 0; i < indatt.size(); i++) indatt[i] = float(i);
        }
        if (auto pos = code.find("@@IND"); pos != code.npos && (code.size() <= pos + 4 || !(isalnum(code[pos + 4]) || strchr("_@$", code[pos + 4]))) && (pos == 0 || !(isalnum(code[pos - 1]) || strchr("_@$", code[pos - 1])))) {
            auto &indatt = primNei->verts.add_attr<float>("IND");
            for (size_t i = 0; i < indatt.size(); i++) indatt[i] = float(i);
        }
        // END张心欣快乐自动加@IND

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
    {
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        auto const &gs = *this->getGlobalState();
        params->lut["PI"] = objectFromLiterial((float)(std::atan(1.f) * 4));
        params->lut["F"] = objectFromLiterial((float)gs.frameid);
        params->lut["DT"] = objectFromLiterial(gs.frame_time);
        params->lut["T"] = objectFromLiterial(gs.frame_time * gs.frameid + gs.frame_time_elapsed);
        // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
        for (auto const &[key, ref]: getThisGraph()->portalIns) {
            if (auto i = code.find('$' + key); i != std::string::npos) {
                i = i + key.size() + 1;
                if (code.size() <= i || !std::isalnum(code[i])) {
                    if (params->lut.count(key)) continue;
                    dbg_printf("ref portal %s\n", key.c_str());
                    auto res = getThisGraph()->callTempNode("PortalOut",
                          {{"name:", objectFromLiterial(key)}}).at("port");
                    params->lut[key] = std::move(res);
                }
            }
        }
        // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
        // BEGIN伺候心欣伺候懒得extract出变量了
        std::vector<std::string> keys;
        for (auto const &[key, val]: params->lut) {
            keys.push_back(key);
        }
        for (auto const &key: keys) {
            if (!dynamic_cast<zeno::NumericObject*>(params->lut.at(key).get())) {
                dbg_printf("ignored non-numeric %s\n", key.c_str());
                params->lut.erase(key);
            }
        }
        // END伺候心欣伺候懒得extract出变量了
        }
    std::vector<float> parvals;
    std::vector<std::pair<std::string, int>> parnames;
    for (auto const &[key_, par]: params->getLiterial<zeno::NumericValue>()) {
            auto key = '$' + key_;
                auto dim = std::visit([&] (auto const &v) {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_convertible_v<T, zeno::vec3f>) {
                        parvals.push_back(v[0]);
                        parvals.push_back(v[1]);
                        parvals.push_back(v[2]);
                        parnames.emplace_back(key, 0);
                        parnames.emplace_back(key, 1);
                        parnames.emplace_back(key, 2);
                        return 3;
                    } else if constexpr (std::is_convertible_v<T, float>) {
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
            //auto par = zeno::safe_any_cast<zeno::NumericValue>(obj);
            
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

    sorted_bvh_vectors_wrangle(exec, chs, chs2, prim->attr<zeno::vec3f>("pos"),
                        primNei->attr<zeno::vec3f>("pos"), get_input2<bool>("is_box"),
                        lbvh.get()->thickness * lbvh.get()->thickness, get_input2<int>("limit"), lbvh.get());

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(ParticlesNeighborBvhWrangleSorted,
           {
               {{"PrimitiveObject", "prim"},
                {"PrimitiveObject", "primNei"},
                {"LBvh", "lbvh"},
                {"bool", "is_box", "1"},
                {"int", "limit", "-1"},
                {"string", "zfxCode"},
                {"DictObject:NumericObject", "params"}},
               {{"PrimitiveObject", "prim"}},
               {},
               {"zenofx"},
           });

} // namespace
