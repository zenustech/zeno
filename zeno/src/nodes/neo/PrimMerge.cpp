#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUV.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include <unordered_set>

namespace zeno {
ZENO_API void prim_set_abcpath(PrimitiveObject* prim, std::string path_name) {
    int faceset_count = prim->userData().get2<int>("abcpath_count",0);
    for (auto j = 0; j < faceset_count; j++) {
        prim->userData().del(zeno::format("abcpath_{}", j));
    }
    prim->userData().set2("abcpath_count", 1);
    prim->userData().set2("abcpath_0", path_name);

    if (prim->tris.size() > 0) {
        prim->tris.add_attr<int>("abcpath").assign(prim->tris.size(),0);
    }
    if (prim->quads.size() > 0) {
        prim->quads.add_attr<int>("abcpath").assign(prim->quads.size(),0);
    }
    if (prim->polys.size() > 0) {
        prim->polys.add_attr<int>("abcpath").assign(prim->polys.size(),0);
    }
}

ZENO_API void prim_copy_faceset_to_matid(PrimitiveObject* prim) {
    auto &ud = prim->userData();
    auto faceset_count = ud.get2<int>("faceset_count", 0);
    ud.set2("matNum", faceset_count);
    for (auto i = 0; i < faceset_count; i++) {
        auto value = ud.get2<std::string>(format("faceset_{}", i));
        ud.set2(format("Material_{}", i), value);
    }

    if (prim->tris.attr_is<int>("faceset")) {
        prim->tris.attr_visit<AttrAcceptAll>("faceset", [&] (auto const &attarr) {
            using T = std::decay_t<decltype(attarr[0])>;
            auto &targetAttr = prim->tris.template add_attr<T>("matid");
            std::copy(attarr.begin(), attarr.end(), targetAttr.begin());
        });
    }
    if (prim->polys.attr_is<int>("faceset")) {
        prim->polys.attr_visit<AttrAcceptAll>("faceset", [&] (auto const &attarr) {
            using T = std::decay_t<decltype(attarr[0])>;
            auto &targetAttr = prim->polys.template add_attr<T>("matid");
            std::copy(attarr.begin(), attarr.end(), targetAttr.begin());
        });
    }
}

ZENO_API void prim_set_faceset(PrimitiveObject* prim, std::string faceset_name) {
    int faceset_count = prim->userData().get2<int>("faceset_count",0);
    for (auto j = 0; j < faceset_count; j++) {
        prim->userData().del(zeno::format("faceset_{}", j));
    }
    prim->userData().set2("faceset_count", 1);
    prim->userData().set2("faceset_0", faceset_name);

    if (prim->tris.size() > 0) {
        prim->tris.add_attr<int>("faceset").assign(prim->tris.size(),0);
    }
    if (prim->quads.size() > 0) {
        prim->quads.add_attr<int>("faceset").assign(prim->quads.size(),0);
    }
    if (prim->polys.size() > 0) {
        prim->polys.add_attr<int>("faceset").assign(prim->polys.size(),0);
    }
}
static void set_special_attr_remap(PrimitiveObject *p, std::string attr_name, std::unordered_map<std::string, int> &facesetNameMap) {
    int faceset_count = p->userData().get2<int>(attr_name + "_count", 0);
    {
        std::unordered_map<int, int> cur_faceset_index_map;
        for (auto i = 0; i < faceset_count; i++) {
            auto path = p->userData().get2<std::string>(format("{}_{}", attr_name, i));
            if (facesetNameMap.count(path) == 0) {
                int new_index = facesetNameMap.size();
                facesetNameMap[path] = new_index;
            }
            cur_faceset_index_map[i] = facesetNameMap[path];
        }

        for (int i = 0; i < p->tris.size(); i++) {
            p->tris.attr<int>(attr_name)[i] = cur_faceset_index_map[p->tris.attr<int>(attr_name)[i]];
        }
        for (int i = 0; i < p->quads.size(); i++) {
            p->quads.attr<int>(attr_name)[i] = cur_faceset_index_map[p->quads.attr<int>(attr_name)[i]];
        }
        for (int i = 0; i < p->polys.size(); i++) {
            p->polys.attr<int>(attr_name)[i] = cur_faceset_index_map[p->polys.attr<int>(attr_name)[i]];
        }
    }
}
ZENO_API std::shared_ptr<zeno::PrimitiveObject> primMergeWithFacesetMatid(std::vector<zeno::PrimitiveObject *> const &primList,
                                                          std::string const &tagAttr, bool tag_on_vert, bool tag_on_face) {
    std::vector<std::string> matNameList(0);
    std::unordered_map<std::string, int> facesetNameMap;
    std::unordered_map<std::string, int> abcpathNameMap;
    for (auto p : primList) {
        //if p has material
        int matNum = p->userData().get2<int>("matNum", 0);
        if (matNum > 0) {
            //for p's tris, quads...
            //    tris("matid")[i] += matNameList.size();
            for (int i = 0; i < p->tris.size(); i++) {
                if (p->tris.attr<int>("matid")[i] != -1) {
                    p->tris.attr<int>("matid")[i] += matNameList.size();
                }
            }
            for (int i = 0; i < p->quads.size(); i++) {
                if (p->quads.attr<int>("matid")[i] != -1) {
                    p->quads.attr<int>("matid")[i] += matNameList.size();
                }
            }
            for (int i = 0; i < p->polys.size(); i++) {
                if (p->polys.attr<int>("matid")[i] != -1) {
                    p->polys.attr<int>("matid")[i] += matNameList.size();
                }
            }
            //for p's materials
            //    add them to material list
            for (int i = 0; i < matNum; i++) {
                auto matIdx = "Material_" + to_string(i);
                auto matName = p->userData().get2<std::string>(matIdx, "Default");
                matNameList.emplace_back(matName);
            }
        } else {
            //for p's tris, quads...
            //    tris("matid")[] = -1;
            if (p->tris.size() > 0) {
                p->tris.add_attr<int>("matid");
                p->tris.attr<int>("matid").assign(p->tris.size(), -1);
            }
            if (p->quads.size() > 0) {
                p->quads.add_attr<int>("matid");
                p->quads.attr<int>("matid").assign(p->quads.size(), -1);
            }
            if (p->polys.size() > 0) {
                p->polys.add_attr<int>("matid");
                p->polys.attr<int>("matid").assign(p->polys.size(), -1);
            }
        }
        int faceset_count = p->userData().get2<int>("faceset_count", 0);
        if (faceset_count == 0) {
            prim_set_faceset(p, "defFS");
            faceset_count = 1;
        }
        set_special_attr_remap(p, "faceset", facesetNameMap);
        int path_count = p->userData().get2<int>("abcpath_count", 0);
        if (path_count == 0) {
            prim_set_abcpath(p, "/ABC/unassigned");
            path_count = 1;
        }
        set_special_attr_remap(p, "abcpath", abcpathNameMap);
    }
    auto outprim = primMerge(primList, tagAttr, tag_on_vert, tag_on_face);

    for (auto &p : primList) {
        outprim->userData().merge(p->userData());
    }

    if (matNameList.size() > 0) {
        //add matNames to userData
        int i = 0;
        for (auto name : matNameList) {
            auto matIdx = "Material_" + to_string(i);
            outprim->userData().setLiterial(matIdx, name);
            i++;
        }
    }
    int oMatNum = matNameList.size();
    outprim->userData().set2("matNum", oMatNum);
    {
        for (auto const &[k, v]: facesetNameMap) {
            outprim->userData().set2(format("faceset_{}", v), k);
        }
        int faceset_count = facesetNameMap.size();
        outprim->userData().set2("faceset_count", faceset_count);
    }
    {
        for (auto const &[k, v]: abcpathNameMap) {
            outprim->userData().set2(format("abcpath_{}", v), k);
        }
        int abcpath_count = abcpathNameMap.size();
        outprim->userData().set2("abcpath_count", abcpath_count);
    }
    return outprim;
}
ZENO_API std::shared_ptr<zeno::PrimitiveObject>
primMerge(std::vector<zeno::PrimitiveObject *> const &inputList, std::string const &tagAttr, bool tag_on_vert,
          bool tag_on_face) {
    auto primList = inputList;
    std::vector<std::shared_ptr<PrimitiveObject>> polygonated;
    //zeno::log_critical("asdfjhl {}", primList.size());
    //throw;
    int poly_flag = 0;
    for (auto &p : primList) {
        if (p->polys.size()) {
            poly_flag += 1;
        }
    }
    // check if mix polys and tris
    if (0 < poly_flag && poly_flag < primList.size()) {
        for (auto &p : primList) {
            if (p->polys.size() == 0) {
                polygonated.push_back(std::make_shared<PrimitiveObject>(*p));
                p = polygonated.back().get();
                primPolygonate(p, true);
            }
        }
    }

    auto outprim = std::make_shared<PrimitiveObject>();

    if (primList.size()) {
        std::vector<size_t> bases(primList.size() + 1);
        std::vector<size_t> pointbases(primList.size() + 1);
        std::vector<size_t> linebases(primList.size() + 1);
        std::vector<size_t> tribases(primList.size() + 1);
        std::vector<size_t> quadbases(primList.size() + 1);
        std::vector<size_t> loopbases(primList.size() + 1);
        std::vector<size_t> uvbases(primList.size() + 1);
        std::vector<size_t> polybases(primList.size() + 1);
        size_t total = 0;
        size_t pointtotal = 0;
        size_t linetotal = 0;
        size_t tritotal = 0;
        size_t quadtotal = 0;
        size_t looptotal = 0;
        size_t uvtotal = 0;
        size_t polytotal = 0;
        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto prim = primList[primIdx];
            /// @note promote pure vert prim to point-based prim
            if (!(prim->points.size() || prim->lines.size() || prim->tris.size() || prim->quads.size() ||
                  prim->polys.size())) {
                auto nverts = prim->verts.size();
                prim->points.resize(nverts);
                parallel_for(nverts, [&points = prim->points.values](size_t i) { points[i] = i; });
            }
            ///
            total += prim->verts.size();
            pointtotal += prim->points.size();
            linetotal += prim->lines.size();
            tritotal += prim->tris.size();
            quadtotal += prim->quads.size();
            looptotal += prim->loops.size();
            uvtotal += prim->uvs.size();
            polytotal += prim->polys.size();
            bases[primIdx + 1] = total;
            pointbases[primIdx + 1] = pointtotal;
            linebases[primIdx + 1] = linetotal;
            tribases[primIdx + 1] = tritotal;
            quadbases[primIdx + 1] = quadtotal;
            loopbases[primIdx + 1] = looptotal;
            uvbases[primIdx + 1] = uvtotal;
            polybases[primIdx + 1] = polytotal;
        }
        outprim->verts.resize(total);
        outprim->points.resize(pointtotal);
        outprim->lines.resize(linetotal);
        outprim->tris.resize(tritotal);
        outprim->quads.resize(quadtotal);
        outprim->loops.resize(looptotal);
        outprim->uvs.resize(uvtotal);
        outprim->polys.resize(polytotal);

        if (tagAttr.size()) {
            if (tag_on_vert) {
                outprim->verts.add_attr<int>(tagAttr);
            }
            if (tag_on_face) {
                outprim->tris.add_attr<int>(tagAttr);
                outprim->polys.add_attr<int>(tagAttr);
            }
        }
        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto const &prim = primList[primIdx];
            prim->verts.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->verts.add_attr<T>(key);
            });
            prim->points.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->points.add_attr<T>(key);
            });
            prim->lines.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->lines.add_attr<T>(key);
            });
            prim->tris.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->tris.add_attr<T>(key);
            });
            prim->quads.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->quads.add_attr<T>(key);
            });
            prim->loops.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->loops.add_attr<T>(key);
            });
            prim->uvs.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->uvs.add_attr<T>(key);
            });
            prim->polys.foreach_attr<AttrAcceptAll>([&](auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->polys.add_attr<T>(key);
            });
        }

        parallel_for(primList.size(), [&](size_t primIdx) {
            auto prim = primList[primIdx];
            auto base = bases[primIdx];
            auto core = [&](auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->verts.values;
                    } else {
                        return outprim->verts.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->verts.size());
                for (size_t i = 0; i < n; i++) {
                    outarr[base + i] = arr[i];
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->verts.values;
                    size_t n = std::min(arr.size(), prim->verts.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                } else {
                    auto &outarr = outprim->verts.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->verts.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->verts.values);
            prim->verts.foreach_attr<AttrAcceptAll>(core);
            if (tag_on_vert && tagAttr.size()) {
                auto &outarr = outprim->verts.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->verts.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });

        parallel_for(primList.size(), [&](size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = pointbases[primIdx];
            auto core = [&](auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->points.values;
                    } else {
                        return outprim->points.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->points.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->points.values;
                    size_t n = std::min(arr.size(), prim->points.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->points.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->points.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->points.values);
            prim->points.foreach_attr<AttrAcceptAll>(core);
//            if (tagAttr.size()) {
//                auto &outarr = outprim->points.attr<int>(tagAttr);
//                for (size_t i = 0; i < prim->points.size(); i++) {
//                    outarr[base + i] = primIdx;
//                }
//            }
        });

        parallel_for(primList.size(), [&](size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = linebases[primIdx];
            auto core = [&](auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->lines.values;
                    } else {
                        return outprim->lines.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->lines.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->lines.values;
                    size_t n = std::min(arr.size(), prim->lines.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->lines.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->lines.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->lines.values);
            prim->lines.foreach_attr<AttrAcceptAll>(core);
//            if (tagAttr.size()) {
//                auto &outarr = outprim->lines.attr<int>(tagAttr);
//                for (size_t i = 0; i < prim->lines.size(); i++) {
//                    outarr[base + i] = primIdx;
//                }
//            }
        });

        parallel_for(primList.size(), [&](size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = tribases[primIdx];
            auto core = [&](auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->tris.values;
                    } else {
                        return outprim->tris.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->tris.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->tris.values;
                    size_t n = std::min(arr.size(), prim->tris.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->tris.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->tris.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->tris.values);
            prim->tris.foreach_attr<AttrAcceptAll>(core);
            if (tag_on_face && tagAttr.size()) {
                auto &outarr = outprim->tris.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->tris.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });

        parallel_for(primList.size(), [&](size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = quadbases[primIdx];
            auto core = [&](auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->quads.values;
                    } else {
                        return outprim->quads.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->quads.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->quads.values;
                    size_t n = std::min(arr.size(), prim->quads.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->quads.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->quads.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->quads.values);
            prim->quads.foreach_attr<AttrAcceptAll>(core);
//            if (tagAttr.size()) {
//                auto &outarr = outprim->quads.attr<int>(tagAttr);
//                for (size_t i = 0; i < prim->quads.size(); i++) {
//                    outarr[base + i] = primIdx;
//                }
//            }
        });

        parallel_for(primList.size(), [&](size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = loopbases[primIdx];
            auto core = [&](auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->loops.values;
                    } else {
                        return outprim->loops.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->loops.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->loops.values;
                    size_t n = std::min(arr.size(), prim->loops.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                    return;
                } else if constexpr (std::is_same_v<T, int>) {
                    if (key == "uvs") {
                        auto &outarr = outprim->loops.attr<int>("uvs");
                        size_t n = std::min(arr.size(), prim->loops.size());
                        size_t offset = uvbases[primIdx];
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = arr[i] + offset;
                        }
                        return;
                    }
                }
                if constexpr (!std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->loops.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->loops.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->loops.values);
            prim->loops.foreach_attr<AttrAcceptAll>(core);
//            if (tagAttr.size()) {
//                auto &outarr = outprim->loops.attr<int>(tagAttr);
//                for (size_t i = 0; i < prim->loops.size(); i++) {
//                    outarr[base + i] = primIdx;
//                }
//            }
        });

        parallel_for(primList.size(), [&](size_t primIdx) {
            auto prim = primList[primIdx];
            auto base = uvbases[primIdx];
            auto core = [&](auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->uvs.values;
                    size_t n = std::min(arr.size(), prim->uvs.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                } else {
                    auto &outarr = outprim->uvs.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->uvs.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
            };
            core(std::true_type{}, prim->uvs.values);
            prim->uvs.foreach_attr<AttrAcceptAll>(core);
//            if (tagAttr.size()) {
//                auto &outarr = outprim->uvs.attr<int>(tagAttr);
//                for (size_t i = 0; i < prim->uvs.size(); i++) {
//                    outarr[base + i] = primIdx;
//                }
//            }
        });

        parallel_for(primList.size(), [&](size_t primIdx) {
            auto prim = primList[primIdx];
            auto lbase = loopbases[primIdx];
            auto base = polybases[primIdx];
            auto core = [&](auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->polys.values;
                    } else {
                        return outprim->polys.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->polys.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = {arr[i].first + lbase, arr[i].second};
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->polys.values;
                    size_t n = std::min(arr.size(), prim->polys.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = {arr[i][0] + (int)lbase, arr[i][1]};
                    }
                } else {
                    auto &outarr = outprim->polys.add_attr<T>(key);
                    size_t n = std::min(arr.size(), prim->polys.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->polys.values);
            prim->polys.foreach_attr<AttrAcceptAll>(core);
            if (tag_on_face && tagAttr.size()) {
                auto &outarr = outprim->polys.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->polys.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });
    }

    primMergeUVs(*outprim, primList);
    return outprim;
}

ZENO_API void primMergeUVs(PrimitiveObject &out, const std::vector<PrimitiveObject *> &inputs) {
    size_t sets = 0;
    std::vector<std::vector<PrimitiveUVSet>> inputSets;
    for (const auto p : inputs) {
        sets = std::max(sets, p->append_uvs.size());
        inputSets.push_back(primitiveUVSets(*p));
    }
    out.append_uvs.resize(sets);
    for (size_t set = 1; set <= sets; ++set) {
        auto &palette = out.append_uvs[set - 1];
        size_t count = 0;
        for (const auto p : inputs) {
            if (const auto src = primitiveUVPalette(*p, set)) {
                count += src->size();
                src->foreach_attr<AttrAcceptAll>([&](const auto &key, const auto &values) {
                    using T = std::decay_t<decltype(values[0])>;
                    palette.add_attr<T>(key);
                });
            }
        }
        if (count > size_t(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Merged UV palette is too large");
        }
        palette.resize(count);
        std::vector<int> *loops = nullptr;
        std::vector<vec3i> *triangles = nullptr;
        for (const auto p : inputs) {
            if (primitiveUVLoopIndices(*p, set)) {
                loops = &out.loops.add_attr<int>(primitiveUVLoopName(set));
            }
            if (p->tris.attr_is<vec3i>(primitiveUVTriangleName(set)) ||
                p->tris.attr_is<int>("uv" + std::to_string(set) + "0")) {
                triangles = &out.tris.add_attr<vec3i>(primitiveUVTriangleName(set));
            }
        }
        if (loops) {
            std::fill(loops->begin(), loops->end(), -1);
        }
        if (triangles) {
            std::fill(triangles->begin(), triangles->end(), vec3i(-1));
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (const auto &uv : inputSets[i]) {
                if (uv.domain == "point" && uv.palette == int(set) &&
                    inputs[i]->verts.attr_is<int>(uv.source)) {
                    auto &ids = out.verts.add_attr<int>(uv.source);
                    std::fill(ids.begin(), ids.end(), -1);
                }
            }
        }
        size_t valueBase = 0, loopBase = 0, triBase = 0, pointBase = 0;
        for (size_t mesh = 0; mesh < inputs.size(); ++mesh) {
            const auto p = inputs[mesh];
            const auto src = primitiveUVPalette(*p, set);
            if (src) {
                std::copy(src->begin(), src->end(), palette.begin() + valueBase);
                src->foreach_attr<AttrAcceptAll>([&](const auto &key, const auto &values) {
                    using T = std::decay_t<decltype(values[0])>;
                    std::copy(values.begin(), values.end(), palette.attr<T>(key).begin() + valueBase);
                });
            }
            if (const auto ids = primitiveUVLoopIndices(*p, set); ids && loops) {
                for (size_t i = 0; i < p->loops.size(); ++i) {
                    (*loops)[loopBase + i] = !src || ids->at(i) < 0 || size_t(ids->at(i)) >= src->size()
                                                 ? -1
                                                 : int(valueBase) + ids->at(i);
                }
            }
            const auto packed = primitiveUVTriangleName(set);
            const auto old = "uv" + std::to_string(set);
            if (triangles) {
                for (size_t t = 0; t < p->tris.size(); ++t) {
                    for (int k = 0; k < 3; ++k) {
                        int index = -1;
                        if (p->tris.attr_is<vec3i>(packed)) {
                            index = p->tris.attr<vec3i>(packed)[t][k];
                        } else if (p->tris.attr_is<int>(old + std::to_string(k))) {
                            index = p->tris.attr<int>(old + std::to_string(k))[t];
                        }
                        (*triangles)[triBase + t][k] =
                            !src || index < 0 || size_t(index) >= src->size() ? -1 : int(valueBase) + index;
                    }
                }
            }
            for (const auto &uv : inputSets[mesh]) {
                if (uv.domain == "point" && uv.palette == int(set) && p->verts.attr_is<int>(uv.source)) {
                    auto &dst = out.verts.attr<int>(uv.source);
                    const auto &ids = p->verts.attr<int>(uv.source);
                    for (size_t v = 0; v < p->verts.size(); ++v) {
                        dst[pointBase + v] = !src || ids.at(v) < 0 || size_t(ids.at(v)) >= src->size()
                                                 ? -1
                                                 : int(valueBase) + ids.at(v);
                    }
                }
            }
            valueBase += src ? src->size() : 0;
            loopBase += p->loops.size();
            triBase += p->tris.size();
            pointBase += p->verts.size();
        }
        // Keep old cached index spellings consistent when they are present.
        const auto oldLoop = "uv" + std::to_string(set) + "s";
        if (loops && out.loops.attr_is<int>(oldLoop)) {
            out.loops.attr<int>(oldLoop) = *loops;
        }
        for (int k = 0; k < 3; ++k) {
            const auto old = "uv" + std::to_string(set) + std::to_string(k);
            if (triangles && out.tris.attr_is<int>(old)) {
                for (size_t t = 0; t < out.tris.size(); ++t) {
                    out.tris.attr<int>(old)[t] = (*triangles)[t][k];
                }
            }
        }
    }
    std::map<unsigned, PrimitiveUVSet> descriptors;
    for (const auto &list : inputSets) {
        for (const auto &set : list) {
            auto [it, inserted] = descriptors.emplace(set.index, set);
            auto &dst = it->second;
            for (const auto &name : set.aliases) {
                if (name != dst.name &&
                    std::find(dst.aliases.begin(), dst.aliases.end(), name) == dst.aliases.end()) {
                    dst.aliases.push_back(name);
                }
            }
            if (set.name != dst.name &&
                std::find(dst.aliases.begin(), dst.aliases.end(), set.name) == dst.aliases.end()) {
                dst.aliases.push_back(set.name);
            }
        }
    }
    for (auto &[index, dst] : descriptors) {
        bool compatible = true;
        for (const auto &list : inputSets) {
            for (const auto &set : list) {
                if (set.index == index) {
                    compatible &= set.domain == dst.domain && set.palette == dst.palette &&
                                  ((set.domain == "corner" && set.palette >= 0) || set.source == dst.source);
                }
            }
        }
        if (compatible) {
            continue;
        }
        // Bake only heterogeneous domains to corners; indexed sets stay indexed.
        const auto source = "__merged_uv_" + std::to_string(index);
        auto &loops = out.loops.add_attr<vec2f>(source);
        for (int k = 0; k < 3; ++k) {
            out.tris.add_attr<vec2f>(primitiveCornerName(source, k));
        }
        size_t loopBase = 0, triBase = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const PrimitiveUVSet *set = nullptr;
            for (const auto &candidate : inputSets[i]) {
                if (candidate.index == index) {
                    set = &candidate;
                }
            }
            const auto p = inputs[i];
            for (size_t l = 0; l < p->loops.size(); ++l) {
                vec2f value{};
                if (set && !primitiveUVLoopValue(*p, *set, l, value)) {
                    value = {};
                }
                loops[loopBase + l] = value;
            }
            for (int k = 0; k < 3; ++k) {
                for (size_t t = 0; t < p->tris.size(); ++t) {
                    vec2f value{};
                    if (set && !primitiveUVValue(*p, *set, t, k, value)) {
                        value = {};
                    }
                    out.tris.attr<vec2f>(primitiveCornerName(source, k))[triBase + t] = value;
                }
            }
            loopBase += p->loops.size();
            triBase += p->tris.size();
        }
        dst.source = source;
        dst.domain = "corner";
        dst.palette = -1;
    }
    if (!descriptors.empty()) {
        std::vector<PrimitiveUVSet> result;
        for (const auto &[index, set] : descriptors) {
            result.push_back(set);
        }
        setPrimitiveUVSets(out, result);
    }
}

namespace {

struct PrimMerge : INode {
    virtual void apply() override {
        auto primList = get_input<ListObject>("listPrim")->getRaw<PrimitiveObject>();
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        //initialize
        bool tag_on_vert = false;
        bool tag_on_face = false;
        auto tag_scope = get_input2<std::string>("tag_scope");
        if (tag_scope == "vert") {
            tag_on_vert = true;
        }
        else if (tag_scope == "face") {
            tag_on_face = true;
        }
        else {
            tag_on_vert = true;
            tag_on_face = true;
        }

        auto outprim = primMergeWithFacesetMatid(primList, tagAttr, tag_on_vert, tag_on_face);

        //auto outprim = std::make_shared<PrimitiveObject>(*primList[0]);
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(PrimMerge, {
  {
      {"list", "listPrim"},
      {"string", "tagAttr", ""},
      {"enum vert face vert_face", "tag_scope", "vert"},
  },
  {
      {"primitive", "prim"},
  },
  {},
  {"primitive"},
});

} // namespace
} // namespace zeno
