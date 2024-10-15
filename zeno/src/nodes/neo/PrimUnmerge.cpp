#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/para/parallel_reduce.h>
#include <zeno/para/parallel_for.h>
#include <zeno/types/UserData.h>
#include "zeno/utils/log.h"

namespace zeno {

ZENO_API std::vector<std::shared_ptr<PrimitiveObject>> primUnmergeVerts(PrimitiveObject *prim, std::string tagAttr) {
    if (!prim->verts.size()) return {};

    auto const &tagArr = prim->verts.attr<int>(tagAttr);
    int tagMax = parallel_reduce_max(tagArr.begin(), tagArr.end()) + 1;

    std::vector<std::shared_ptr<PrimitiveObject>> primList(tagMax);
    for (int tag = 0; tag < tagMax; tag++) {
        primList[tag] = std::make_shared<PrimitiveObject>();
    }

#if 1
    std::vector<std::vector<int>> aux_arrays;
    aux_arrays.resize(tagMax);
    for (int tag = 0; tag < tagMax; tag++) {
      aux_arrays[tag].resize(0);
    }
    //auto const &tagArr = prim->verts.attr<int>(tagAttr);
    for (int i = 0; i < prim->size(); i++) {
      aux_arrays[tagArr[i]].emplace_back(i);
    }
    for (int tag = 0; tag < tagMax; tag++) {
        primList[tag]->assign(prim);
        primFilterVerts(primList[tag].get(), tagAttr, tag, false, {}, "verts", aux_arrays[tag].data(), aux_arrays[tag].size(),true);
    }

#else
    std::vector<std::vector<int>> vert_revamp(tagMax);
    std::vector<int> vert_unrevamp(prim->verts.size());

    for (size_t i = 0; i < prim->verts.size(); i++) {
        int tag = tagArr[i];
        vert_revamp[tag].push_back(i);
        vert_unrevamp[i] = tag;
    }

    for (int tag = 0; tag < tagMax; tag++) {
        auto &revamp = vert_revamp[tag];
        auto const &outprim = primList[tag];

        outprim->verts.resize(revamp.size());
        parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
            outprim->verts[i] = prim->verts[revamp[i]];
        });
        prim->verts.foreach_attr([&] (auto const &key, auto const &inarr) {
            using T = std::decay_t<decltype(inarr[0])>;
            auto &outarr = outprim->verts.add_attr<T>(key);
            parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                outarr[i] = inarr[revamp[i]];
            });
        });
    }

    std::vector<std::vector<int>> face_revamp;

    auto mock = [&] (auto getter) {
        auto &prim_tris = getter(prim);
        if (prim_tris.size()) {
            face_revamp.clear();
            face_revamp.resize(tagMax);
            using T = std::decay_t<decltype(prim_tris[0])>;

            for (size_t i = 0; i < prim_tris.size(); i++) {
                auto ind = reinterpret_cast<decay_vec_t<T> const *>(&prim_tris[i]);
                int tag = vert_unrevamp[ind[0]];
                bool bad = false;
                for (int j = 1; j < is_vec_n<T>; j++) {
                    int new_tag = vert_unrevamp[ind[j]];
                    if (tag != new_tag) {
                        bad = true;
                        break;
                    }
                }
                if (!bad) face_revamp[tag].push_back(i);
            }

            for (int tag = 0; tag < tagMax; tag++) {
                auto &revamp = face_revamp[tag];
                auto &v_revamp = vert_revamp[tag];
                auto *outprim = primList[tag].get();
                auto &outprim_tris = getter(outprim);

                outprim_tris.resize(revamp.size());
                parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                    auto ind = reinterpret_cast<decay_vec_t<T> const *>(&prim_tris[revamp[i]]);
                    auto outind = reinterpret_cast<decay_vec_t<T> *>(&outprim_tris[i]);
                    for (int j = 0; j < is_vec_n<T>; j++) {
                        outind[j] = v_revamp[ind[j]];
                    }
                });

                prim_tris.foreach_attr([&] (auto const &key, auto const &inarr) {
                    using T = std::decay_t<decltype(inarr[0])>;
                    auto &outarr = outprim_tris.template add_attr<T>(key);
                    parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                        outarr[i] = inarr[revamp[i]];
                    });
                });
            }
        }
    };
    mock([] (auto &&p) -> auto & { return p->points; });
    mock([] (auto &&p) -> auto & { return p->lines; });
    mock([] (auto &&p) -> auto & { return p->tris; });
    mock([] (auto &&p) -> auto & { return p->quads; });

    if (prim->polys.size()) {
        face_revamp.clear();
        face_revamp.resize(tagMax);

        for (size_t i = 0; i < prim->polys.size(); i++) {
            auto &[base, len] = prim->polys[i];
            if (len <= 0) continue;
            int tag = vert_unrevamp[prim->loops[base]];
            bool bad = false;
            for (int j = base + 1; j < base + len; i++) {
                int new_tag = vert_unrevamp[prim->loops[j]];
                if (tag != new_tag) {
                    bad = true;
                    break;
                }
            }
            if (!bad) face_revamp[tag].push_back(i);
        }

        for (int tag = 0; tag < tagMax; tag++) {
            auto &revamp = face_revamp[tag];
            auto &v_revamp = vert_revamp[tag];
            auto *outprim = primList[tag].get();

            outprim->polys.resize(revamp.size());
            for (size_t i = 0; i < revamp.size(); i++) {
                auto const &[base, len] = prim->polys[revamp[i]];
                int new_base = outprim->loops.size();
                for (int j = base; j < base + len; j++) {
                    outprim->loops.push_back(prim->loops[j]);
                }
                outprim->polys[i] = {new_base, len};
            }

            prim->polys.foreach_attr([&] (auto const &key, auto const &inarr) {
                using T = std::decay_t<decltype(inarr[0])>;
                auto &outarr = outprim->polys.add_attr<T>(key);
                parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                    outarr[i] = inarr[revamp[i]];
                });
            });
        }
    }
#endif

    return primList;
}

std::set<int> get_attr_on_faces(PrimitiveObject *prim, std::string tagAttr, bool skip_negative_number) {
    std::set<int> set;
    if (prim->tris.size()) {
        auto &attr = prim->tris.attr<int>(tagAttr);
        for (auto i = 0; i < prim->tris.size(); i++) {
            if (skip_negative_number && attr[i] < 0) {
                continue;
            }
            set.insert(attr[i]);
        }
    }
    if (prim->polys.size()) {
        auto &attr = prim->polys.attr<int>(tagAttr);
        for (auto i = 0; i < prim->polys.size(); i++) {
            if (skip_negative_number && attr[i] < 0) {
                continue;
            }
            set.insert(attr[i]);
        }
    }
    return set;
}

void remap_attr_on_faces(PrimitiveObject *prim, std::string tagAttr, std::map<int, int> mapping) {
    if (prim->tris.size()) {
        auto &attr = prim->tris.attr<int>(tagAttr);
        for (auto i = 0; i < prim->tris.size(); i++) {
            if (mapping.count(attr[i])) {
                attr[i] = mapping[attr[i]];
            }
        }
    }
    if (prim->polys.size()) {
        auto &attr = prim->polys.attr<int>(tagAttr);
        for (auto i = 0; i < prim->polys.size(); i++) {
            if (mapping.count(attr[i])) {
                attr[i] = mapping[attr[i]];
            }
        }
    }
}

ZENO_API std::vector<std::shared_ptr<PrimitiveObject>> primUnmergeFaces(PrimitiveObject *prim, std::string tagAttr) {
    if (!prim->verts.size()) return {};

    if (prim->tris.size() > 0 && prim->polys.size() > 0) {
        primPolygonate(prim, true);
    }

    std::vector<std::shared_ptr<PrimitiveObject>> list;

    std::map<int, std::vector<int>> mapping;
    if (prim->tris.size() > 0) {
        auto &attr = prim->tris.attr<int>(tagAttr);
        for (auto i = 0; i < prim->tris.size(); i++) {
            if (mapping.count(attr[i]) == 0) {
                mapping[attr[i]] = {};
            }
            mapping[attr[i]].push_back(i);
        }
        for (auto &[key, val]: mapping) {
            auto new_prim = std::dynamic_pointer_cast<PrimitiveObject>(prim->clone());
            new_prim->tris.resize(val.size());
            for (auto i = 0; i < val.size(); i++) {
                new_prim->tris[i] = prim->tris[val[i]];
            }
            new_prim->tris.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto &attr = prim->tris.attr<T>(key);
                for (auto i = 0; i < arr.size(); i++) {
                    arr[i] = attr[val[i]];
                }
            });
            list.push_back(new_prim);
        }
    }
    else if (prim->polys.size() > 0) {
        auto &attr = prim->polys.attr<int>(tagAttr);
        for (auto i = 0; i < prim->polys.size(); i++) {
            if (mapping.count(attr[i]) == 0) {
                mapping[attr[i]] = {};
            }
            mapping[attr[i]].push_back(i);
        }
        for (auto &[key, val]: mapping) {
            auto new_prim = std::dynamic_pointer_cast<PrimitiveObject>(prim->clone());
            new_prim->polys.resize(val.size());
            for (auto i = 0; i < val.size(); i++) {
                new_prim->polys[i] = prim->polys[val[i]];
            }
            new_prim->polys.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto &attr = prim->polys.attr<T>(key);
                for (auto i = 0; i < arr.size(); i++) {
                    arr[i] = attr[val[i]];
                }
            });
            list.push_back(new_prim);
        }
    }
    for (auto i = 0; i < list.size(); i++) {
        primKillDeadVerts(list[i].get());
        // remove unused abcpath
        {
            auto abcpath_set = get_attr_on_faces(list[i].get(), "abcpath", true);
            std::map<int, int> mapping;
            std::vector<std::string> abcpaths;
            for (auto &k: abcpath_set) {
                mapping[k] = abcpaths.size();
                abcpaths.push_back(list[i]->userData().get2<std::string>(format("abcpath_{}", k)));
            }
            remap_attr_on_faces(list[i].get(), "abcpath", mapping);
            auto old_abcpath_count = list[i]->userData().get2<int>("abcpath_count", 0);
            for (int j = 0; j < old_abcpath_count; j++) {
                list[i]->userData().del(format("abcpath_{}", j));
            }

            for (int j = 0; j < abcpaths.size(); j++) {
                list[i]->userData().set2(format("abcpath_{}", j), abcpaths[j]);
            }
            list[i]->userData().set2("abcpath_count", int(abcpath_set.size()));
        }
        // remove unused faceset
        {
            auto abcpath_set = get_attr_on_faces(list[i].get(), "faceset", true);
            std::map<int, int> mapping;
            std::vector<std::string> abcpaths;
            for (auto &k: abcpath_set) {
                mapping[k] = abcpaths.size();
                abcpaths.push_back(list[i]->userData().get2<std::string>(format("faceset_{}", k)));
            }
            remap_attr_on_faces(list[i].get(), "faceset", mapping);
            auto old_abcpath_count = list[i]->userData().get2<int>("faceset_count", 0);
            for (int j = 0; j < old_abcpath_count; j++) {
                list[i]->userData().del(format("faceset_{}", j));
            }

            for (int j = 0; j < abcpaths.size(); j++) {
                list[i]->userData().set2(format("faceset_{}", j), abcpaths[j]);
            }
            list[i]->userData().set2("faceset_count", int(abcpath_set.size()));
        }
    }
    return list;
}

namespace {

struct PrimUnmerge : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        auto method = get_input<StringObject>("method")->get();

        if (get_input2<bool>("preSimplify")) {
            primSimplifyTag(prim.get(), tagAttr);
        }
        std::vector<std::shared_ptr<PrimitiveObject>> primList;
        if (method == "verts") {
            primList = primUnmergeVerts(prim.get(), tagAttr);
        }
        else {
            primList = primUnmergeFaces(prim.get(), tagAttr);
        }

        auto listPrim = std::make_shared<ListObject>();
        for (auto &primPtr: primList) {
            listPrim->arr.push_back(std::move(primPtr));
        }
        set_output("listPrim", std::move(listPrim));
    }
};

ZENDEFNODE(PrimUnmerge, {
    {
        {"primitive", "prim"},
        {"string", "tagAttr", "tag"},
        {"bool", "preSimplify", "0"},
        {"enum verts faces", "method", "verts"},
    },
    {
        {"list", "listPrim"},
    },
    {
    },
    {"primitive"},
});

void cleanMesh(std::shared_ptr<zeno::PrimitiveObject> prim,
               std::vector<zeno::vec3f> &verts,
               std::vector<zeno::vec3f> &nrm,
               std::vector<zeno::vec3f> &clr,
               std::vector<zeno::vec3f> &tang,
               std::vector<zeno::vec3f> &uv,
               std::vector<zeno::vec3i> &idxBuffer)
{
  //first pass, scan the prim to see if verts require duplication
  std::vector<std::vector<zeno::vec3f>> vert_uv;
  std::vector<std::vector<zeno::vec2i>> idx_mapping;
  vert_uv.resize(prim->verts.size());
  idx_mapping.resize(prim->verts.size());
  int count = 0;
  for(int i=0;i<prim->tris.size();i++)
  {
    //so far, all value has already averaged on verts, except uv
    zeno::vec3i idx = prim->tris[i];
    for(int j=0;j<3;j++)
    {
      std::string uv_name;
      uv_name = "uv" + std::to_string(j);
      auto vid = idx[j];
      if(vert_uv[vid].size()==0)
      {
        vert_uv[vid].push_back(prim->tris.attr<zeno::vec3f>(uv_name)[i]);
        //idx_mapping[vid].push_back(zeno::vec2i(vid,count));
        //count++;
      }
      else
      {
        zeno::vec3f uv = prim->tris.attr<zeno::vec3f>(uv_name)[i];
        bool have = false;
        for(int k=0;k<vert_uv[vid].size();k++)
        {
          auto & tester = vert_uv[vid][k];
          if(tester[0] == uv[0] && tester[1] == uv[1] && tester[2] == uv[2] )
          {
            have = true;
          }
        }
        if(have == false)
        {
          //need a push_back
          vert_uv[vid].push_back(prim->tris.attr<zeno::vec3f>(uv_name)[i]);
          //idx_mapping[vid].push_back(zeno::vec2i(vid,count));
          //count++;
        }
      }
    }
  }
  count = 0;
  for(int i=0;i<vert_uv.size();i++) {
    for(int j=0;j<vert_uv[i].size();j++) {
      idx_mapping[i].push_back(zeno::vec2i(i, count));
      count++;
    }
  }
  //first pass done

  // [old_idx, new_idx ] = idx_mapping[vid][k] tells index mapping of old and new vert

  //run a pass to assemble new data
  verts.resize(0);
  nrm.resize(0);
  clr.resize(0);
  uv.resize(0);
  tang.resize(0);
  verts.reserve(count);
  nrm.reserve(count);
  clr.reserve(count);
  uv.reserve(count);
  tang.reserve(count);
  for(int i=0;i<vert_uv.size();i++)
  {
    for(int j=0;j<vert_uv[i].size();j++)
    {
      auto vid = idx_mapping[i][j][0];
      auto uvt = vert_uv[i][j];
      auto v  = prim->verts[vid];
      auto n  = prim->verts.attr<zeno::vec3f>("nrm")[vid];
      auto c  = prim->verts.attr<zeno::vec3f>("clr")[vid];
      auto t  = prim->verts.attr<zeno::vec3f>("atang")[vid];
      verts.push_back(v);
      nrm.push_back(n);
      clr.push_back(c);
      tang.push_back(t);
      uv.push_back(uvt);
    }
  }

  idxBuffer.resize(prim->tris.size());
  //third pass: assemble new idx map
  for(int i=0;i<prim->tris.size();i++)
  {
    zeno::vec3i idx = prim->tris[i];
    for(int j=0;j<3;j++) {

      auto old_vid = idx[j];
      if(idx_mapping[old_vid].size()==1)
      {
        idxBuffer[i][j] = idx_mapping[old_vid][0][1];
      }
      else
      {
        std::string uv_name = "uv" + std::to_string(j);
        auto &tuv = prim->tris.attr<zeno::vec3f>(uv_name)[i];
        for(int k=0;k<vert_uv[old_vid].size();k++)
        {
          auto &vuv = vert_uv[old_vid][k];
          if(vuv[0] == tuv[0] && vuv[1] == tuv[1] && vuv[2] == tuv[2])
          {
            idxBuffer[i][j] = idx_mapping[old_vid][k][1];
          }
        }
      }
    }
  }
}
void computeVertexTangent(zeno::PrimitiveObject *prim)
{
  auto &atang = prim->add_attr<zeno::vec3f>("atang");
  auto &tang = prim->tris.attr<zeno::vec3f>("tang");
  atang.assign(atang.size(), zeno::vec3f(0));
  const auto &pos = prim->attr<zeno::vec3f>("pos");
  for(size_t i=0;i<prim->tris.size();++i)
  {

    auto vidx = prim->tris[i];
    zeno::vec3f v0 = pos[vidx[0]];
    zeno::vec3f v1 = pos[vidx[1]];
    zeno::vec3f v2 = pos[vidx[2]];
    auto e1 = v1-v0, e2=v2-v0;
    float area = zeno::length(zeno::cross(e1, e2)) * 0.5;
    atang[vidx[0]] += area * tang[i];
    atang[vidx[1]] += area * tang[i];
    atang[vidx[2]] += area * tang[i];
  }
#pragma omp parallel for
  for(auto i=0;i<atang.size();i++)
  {
    atang[i] = atang[i]/(length(atang[i])+1e-6);

  }
}
void computeTrianglesTangent(zeno::PrimitiveObject *prim)
{
  const auto &tris = prim->tris;
  const auto &pos = prim->attr<zeno::vec3f>("pos");
  auto const &nrm = prim->add_attr<zeno::vec3f>("nrm");
  auto &tang = prim->tris.add_attr<zeno::vec3f>("tang");
  bool has_uv = tris.has_attr("uv0")&&tris.has_attr("uv1")&&tris.has_attr("uv2");
  //printf("!!has_uv = %d\n", has_uv);
  if(has_uv) {
    const auto &uv0data = tris.attr<zeno::vec3f>("uv0");
    const auto &uv1data = tris.attr<zeno::vec3f>("uv1");
    const auto &uv2data = tris.attr<zeno::vec3f>("uv2");
#pragma omp parallel for
    for (auto i = 0; i < prim->tris.size(); ++i) {
      const auto &pos0 = pos[tris[i][0]];
      const auto &pos1 = pos[tris[i][1]];
      const auto &pos2 = pos[tris[i][2]];
      zeno::vec3f uv0;
      zeno::vec3f uv1;
      zeno::vec3f uv2;

      uv0 = uv0data[i];
      uv1 = uv1data[i];
      uv2 = uv2data[i];

      auto edge0 = pos1 - pos0;
      auto edge1 = pos2 - pos0;
      auto deltaUV0 = uv1 - uv0;
      auto deltaUV1 = uv2 - uv0;

      auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[0] * deltaUV0[1] + 1e-5);

      zeno::vec3f tangent;
      tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
      tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
      tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
      //printf("tangent:%f %f %f\n", tangent[0], tangent[1], tangent[2]);
      //zeno::log_info("tangent {} {} {}",tangent[0], tangent[1], tangent[2]);
      auto tanlen = zeno::length(tangent);
      tangent *(1.f / (tanlen + 1e-8));
      /*if (std::abs(tanlen) < 1e-8) {//fix by BATE
          zeno::vec3f n = nrm[tris[i][0]], unused;
          zeno::pixarONB(n, tang[i], unused);//TODO calc this in shader?
      } else {
          tang[i] = tangent * (1.f / tanlen);
      }*/
      tang[i] = tangent;
    }
  } else {
    const auto &uvarray = prim->attr<zeno::vec3f>("uv");
#pragma omp parallel for
    for (auto i = 0; i < prim->tris.size(); ++i) {
      const auto &pos0 = pos[tris[i][0]];
      const auto &pos1 = pos[tris[i][1]];
      const auto &pos2 = pos[tris[i][2]];
      zeno::vec3f uv0;
      zeno::vec3f uv1;
      zeno::vec3f uv2;

      uv0 = uvarray[tris[i][0]];
      uv1 = uvarray[tris[i][1]];
      uv2 = uvarray[tris[i][2]];

      auto edge0 = pos1 - pos0;
      auto edge1 = pos2 - pos0;
      auto deltaUV0 = uv1 - uv0;
      auto deltaUV1 = uv2 - uv0;

      auto f = 1.0f / (deltaUV0[0] * deltaUV1[1] - deltaUV1[0] * deltaUV0[1] + 1e-5);

      zeno::vec3f tangent;
      tangent[0] = f * (deltaUV1[1] * edge0[0] - deltaUV0[1] * edge1[0]);
      tangent[1] = f * (deltaUV1[1] * edge0[1] - deltaUV0[1] * edge1[1]);
      tangent[2] = f * (deltaUV1[1] * edge0[2] - deltaUV0[1] * edge1[2]);
      //printf("tangent:%f %f %f\n", tangent[0], tangent[1], tangent[2]);
      //zeno::log_info("tangent {} {} {}",tangent[0], tangent[1], tangent[2]);
      auto tanlen = zeno::length(tangent);
      tangent *(1.f / (tanlen + 1e-8));
      /*if (std::abs(tanlen) < 1e-8) {//fix by BATE
          zeno::vec3f n = nrm[tris[i][0]], unused;
          zeno::pixarONB(n, tang[i], unused);//TODO calc this in shader?
          } else {
          tang[i] = tangent * (1.f / tanlen);
          }*/
      tang[i] = tangent;
    }
  }
}
struct primClean : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    std::vector<zeno::vec3f> verts;
    std::vector<zeno::vec3f> nrm;
    std::vector<zeno::vec3f> clr;
    std::vector<zeno::vec3f> tang;
    std::vector<zeno::vec3f> uv;
    std::vector<zeno::vec3i> idxBuffer;
    computeTrianglesTangent(prim.get());
    computeVertexTangent(prim.get());
    cleanMesh(prim, verts, nrm, clr, tang, uv, idxBuffer);
    auto oPrim = std::make_shared<zeno::PrimitiveObject>();
    oPrim->verts.resize(verts.size());
    oPrim->add_attr<zeno::vec3f>("nrm");
    oPrim->add_attr<zeno::vec3f>("clr");
    oPrim->add_attr<zeno::vec3f>("uv");
    oPrim->add_attr<zeno::vec3f>("atang");
    oPrim->tris.resize(idxBuffer.size());


    oPrim->verts.attr<zeno::vec3f>("pos") = verts;
    oPrim->verts.attr<zeno::vec3f>("nrm") = nrm;
    oPrim->verts.attr<zeno::vec3f>("clr") = clr;
    oPrim->verts.attr<zeno::vec3f>("uv") = uv;
    oPrim->verts.attr<zeno::vec3f>("atang") = tang;


    oPrim->tris = idxBuffer;


    set_output("prim", std::move(oPrim));
  }
};

ZENDEFNODE(primClean, {
                            {
                                {"primitive", "prim"}
                            },
                            {
                                {"primitive", "prim"},
                            },
                            {
                            },
                            {"primitive"},
                        });


}
}
