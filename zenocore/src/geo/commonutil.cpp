#include <zeno/geo/commonutil.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/interfaceutil.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/fileio.h>
#include <zeno/para/parallel_scan.h>
#include <zeno/types/UserData.h>
#include <stdexcept>
#include <filesystem>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include <tinygltf/stb_image.h>
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#include "zeno/utils/string.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/stb_image_write.h>


namespace zeno
{
    template <class T>
    static void revamp_vector(std::vector<T>& arr, std::vector<int> const& revamp) {
        std::vector<T> newarr(arr.size());
        for (int i = 0; i < revamp.size(); i++) {
            newarr[i] = arr[revamp[i]];
        }
        std::swap(arr, newarr);
    }

    template <typename DstT, typename SrcT> constexpr auto reinterpret_bits(SrcT &&val) {
        using Src = std::remove_cv_t<std::remove_reference_t<SrcT>>;
        using Dst = std::remove_cv_t<std::remove_reference_t<DstT>>;
        static_assert(sizeof(Src) == sizeof(Dst),
                      "Source Type and Destination Type must be of the same size");
        static_assert(std::is_trivially_copyable_v<Src> && std::is_trivially_copyable_v<Dst>,
                      "Both types should be trivially copyable.");
        static_assert(std::alignment_of_v<Src> % std::alignment_of_v<Dst> == 0,
                      "The original type should at least have an alignment as strict.");
        Dst dst{};
        std::memcpy(&dst, const_cast<const Src *>(&val), sizeof(Dst));
        return dst;
    }

    static void set_special_attr_remap(PrimitiveObject* p, std::string attr_name, std::unordered_map<std::string, int>& facesetNameMap) {
        UserData* pUserData = dynamic_cast<UserData*>(p->userData());
        int faceset_count = pUserData->get2<int>(attr_name + "_count", 0);
        {
            std::unordered_map<int, int> cur_faceset_index_map;
            for (auto i = 0; i < faceset_count; i++) {
                auto path = pUserData->get2<std::string>(format("{}_{}", attr_name, i));
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

    void primTriangulate(PrimitiveObject* prim, bool with_uv, bool has_lines, bool with_attr) {
        if (prim->polys.size() == 0) {
            return;
        }
        boolean_switch(has_lines, [&] (auto has_lines) {
            std::vector<std::conditional_t<has_lines.value, vec2i, int>> scansum(prim->polys.size());
            auto redsum = parallel_exclusive_scan_sum(prim->polys.begin(), prim->polys.end(),
                                           scansum.begin(), [&] (auto &ind) {
                                               if constexpr (has_lines.value) {
                                                   return vec2i(ind[1] >= 3 ? ind[1] - 2 : 0, ind[1] == 2 ? 1 : 0);
                                               } else {
                                                   return ind[1] >= 3 ? ind[1] - 2 : 0;
                                               }
                                           });
            std::vector<int> mapping;
            int tribase = prim->tris.size();
            int linebase = prim->lines.size();
            if constexpr (has_lines.value) {
                prim->tris.resize(tribase + redsum[0]);
                mapping.resize(tribase + redsum[0]);
                prim->lines.resize(linebase + redsum[1]);
            } else {
                prim->tris.resize(tribase + redsum);
                mapping.resize(tribase + redsum);
            }

            if (!(prim->loops.has_attr("uvs") && prim->uvs.size() > 0) || !with_uv) {
                parallel_for(prim->polys.size(), [&] (size_t i) {
                    auto [start, len] = prim->polys[i];

                    if (len >= 3) {
                        int scanbase;
                        if constexpr (has_lines.value) {
                            scanbase = scansum[i][0] + tribase;
                        } else {
                            scanbase = scansum[i] + tribase;
                        }
                        prim->tris[scanbase] = vec3i(
                                prim->loops[start],
                                prim->loops[start + 1],
                                prim->loops[start + 2]);
                        mapping[scanbase] = i;
                        scanbase++;
                        for (int j = 3; j < len; j++) {
                            prim->tris[scanbase] = vec3i(
                                    prim->loops[start],
                                    prim->loops[start + j - 1],
                                    prim->loops[start + j]);
                            mapping[scanbase] = i;
                            scanbase++;
                        }
                    }
                    if constexpr (has_lines.value) {
                        if (len == 2) {
                            int scanbase = scansum[i][1] + linebase;
                            prim->lines[scanbase] = vec2i(
                                prim->loops[start],
                                prim->loops[start + 1]);
                        }
                    }
                });

            } else {
                auto &loop_uv = prim->loops.attr<int>("uvs");
                auto &uvs = prim->uvs;
                auto &uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
                auto &uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
                auto &uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");

                parallel_for(prim->polys.size(), [&] (size_t i) {
                    auto [start, len] = prim->polys[i];

                    if (len >= 3) {
                        int scanbase;
                        if constexpr (has_lines.value) {
                            scanbase = scansum[i][0] + tribase;
                        } else {
                            scanbase = scansum[i] + tribase;
                        }
                        uv0[scanbase] = {uvs[loop_uv[start]][0], uvs[loop_uv[start]][1], 0};
                        uv1[scanbase] = {uvs[loop_uv[start + 1]][0], uvs[loop_uv[start + 1]][1], 0};
                        uv2[scanbase] = {uvs[loop_uv[start + 2]][0], uvs[loop_uv[start + 2]][1], 0};
                        prim->tris[scanbase] = vec3i(
                                prim->loops[start],
                                prim->loops[start + 1],
                                prim->loops[start + 2]);
                        mapping[scanbase] = i;
                        scanbase++;
                        for (int j = 3; j < len; j++) {
                            uv0[scanbase] = {uvs[loop_uv[start]][0], uvs[loop_uv[start]][1], 0};
                            uv1[scanbase] = {uvs[loop_uv[start + j - 1]][0], uvs[loop_uv[start + j - 1]][1], 0};
                            uv2[scanbase] = {uvs[loop_uv[start + j]][0], uvs[loop_uv[start + j]][1], 0};
                            prim->tris[scanbase] = vec3i(
                                    prim->loops[start],
                                    prim->loops[start + j - 1],
                                    prim->loops[start + j]);
                            mapping[scanbase] = i;
                            scanbase++;
                        }
                    }
                    if constexpr (has_lines.value) {
                        if (len == 2) {
                            int scanbase = scansum[i][1] + linebase;
                            prim->lines[scanbase] = vec2i(
                                prim->loops[start],
                                prim->loops[start + 1]);
                        }
                    }
                });

            }
            if (with_attr) {
                prim->polys.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                  using T = std::decay_t<decltype(arr[0])>;
                  auto &attr = prim->tris.add_attr<T>(key);
                  for (auto i = tribase; i < attr.size(); i++) {
                      attr[i] = arr[mapping[i]];
                  }
                });
            }
            prim->loops.clear_with_attr();
            prim->polys.clear_with_attr();
            prim->uvs.clear_with_attr();
        });
    }

    void primTriangulateQuads(PrimitiveObject* prim) {
        if (prim->quads.size() == 0) {
            return;
        }
        auto base = prim->tris.size();
        prim->tris.resize(base + prim->quads.size() * 2);
        bool hasmat = prim->quads.has_attr("matid");
        if (hasmat == false)
        {
            prim->quads.add_attr<int>("matid");
            prim->quads.attr<int>("matid").assign(prim->quads.size(), -1);
        }

        if (prim->tris.has_attr("matid")) {
            prim->tris.attr<int>("matid").resize(base + prim->quads.size() * 2);
        }
        else {
            prim->tris.add_attr<int>("matid");
        }


        for (size_t i = 0; i < prim->quads.size(); i++) {
            auto quad = prim->quads[i];
            prim->tris[base + i * 2 + 0] = vec3f(quad[0], quad[1], quad[2]);
            prim->tris[base + i * 2 + 1] = vec3f(quad[0], quad[2], quad[3]);
            if (hasmat) {
                prim->tris.attr<int>("matid")[base + i * 2 + 0] = prim->quads.attr<int>("matid")[i];
                prim->tris.attr<int>("matid")[base + i * 2 + 1] = prim->quads.attr<int>("matid")[i];
            }
            else
            {
                prim->tris.attr<int>("matid")[base + i * 2 + 0] = -1;
                prim->tris.attr<int>("matid")[base + i * 2 + 1] = -1;
            }
        }
        prim->quads.clear();
    }

    zeno::Vector<std::unique_ptr<zeno::PrimitiveObject>> get_prims_from_list(zeno::ListObject* spList) {
        zeno::Vector<std::unique_ptr<zeno::PrimitiveObject>> vec;
        for (auto obj : spList->get()) {
            auto prim = dynamic_cast<zeno::PrimitiveObject*>(obj);
            if (prim)
                vec.push_back(safe_uniqueptr_cast<PrimitiveObject>(prim->clone()));
        }
        return vec;
    }

    void primKillDeadUVs(PrimitiveObject* prim) {
        if (!(prim->loops.size() > 0 && prim->loops.attr_is<int>("uvs"))) {
            return;
        }
        auto& uvs = prim->loops.attr<int>("uvs");
        std::set<int> reached;
        for (auto const& [start, len] : prim->polys) {
            for (int i = start; i < start + len; i++) {
                reached.insert(uvs[i]);
            }
        }
        std::vector<int> revamp(reached.begin(), reached.end());
        auto old_prim_size = prim->uvs.size();
        prim->uvs.forall_attr<AttrAcceptAll>([&](auto const& key, auto& arr) {
            revamp_vector(arr, revamp);
            });
        prim->uvs.resize(revamp.size());

        std::vector<int> unrevamp(old_prim_size);
        for (int i = 0; i < revamp.size(); i++) {
            unrevamp[revamp[i]] = i;
        }

        auto mock = [&](int& ind) {
            ind = unrevamp[ind];
        };
        for (auto const& [start, len] : prim->polys) {
            for (int i = start; i < start + len; i++) {
                mock(uvs[i]);
            }
        }
    }

    void primKillDeadLoops(PrimitiveObject* prim) {
        if (prim->loops.size() == 0) {
            return;
        }
        std::set<int> reached;
        for (auto const& [start, len] : prim->polys) {
            for (int i = start; i < start + len; i++) {
                reached.insert(i);
            }
        }
        std::vector<int> revamp(reached.begin(), reached.end());
        auto old_prim_size = prim->loops.size();
        prim->loops.forall_attr<AttrAcceptAll>([&](auto const& key, auto& arr) {
            revamp_vector(arr, revamp);
            });
        prim->loops.resize(revamp.size());

        std::vector<int> unrevamp(old_prim_size);
        for (int i = 0; i < revamp.size(); i++) {
            unrevamp[revamp[i]] = i;
        }
        int count = 0;
        for (auto& [start, len] : prim->polys) {
            start = count;
            count += len;
        }
    }

    void remap_attr_on_faces(PrimitiveObject* prim, std::string tagAttr, std::map<int, int> mapping) {
        if (prim->tris.size()) {
            auto& attr = prim->tris.attr<int>(tagAttr);
            for (auto i = 0; i < prim->tris.size(); i++) {
                if (mapping.count(attr[i])) {
                    attr[i] = mapping[attr[i]];
                }
            }
        }
        if (prim->polys.size()) {
            auto& attr = prim->polys.attr<int>(tagAttr);
            for (auto i = 0; i < prim->polys.size(); i++) {
                if (mapping.count(attr[i])) {
                    attr[i] = mapping[attr[i]];
                }
            }
        }
    }

    std::set<int> get_attr_on_faces(PrimitiveObject* prim, std::string tagAttr, bool skip_negative_number) {
        std::set<int> set;
        if (prim->tris.size()) {
            auto& attr = prim->tris.attr<int>(tagAttr);
            for (auto i = 0; i < prim->tris.size(); i++) {
                if (skip_negative_number && attr[i] < 0) {
                    continue;
                }
                set.insert(attr[i]);
            }
        }
        if (prim->polys.size()) {
            auto& attr = prim->polys.attr<int>(tagAttr);
            for (auto i = 0; i < prim->polys.size(); i++) {
                if (skip_negative_number && attr[i] < 0) {
                    continue;
                }
                set.insert(attr[i]);
            }
        }
        return set;
    }

    void prim_set_faceset(PrimitiveObject* prim, String faceset_name) {
        IUserData* pUserData = prim->userData();
        int faceset_count = pUserData->get_int("faceset_count", 0);
        for (auto j = 0; j < faceset_count; j++) {
            pUserData->del(stdString2zs(zeno::format("faceset_{}", j)));
        }
        pUserData->set_int("faceset_count", 1);
        pUserData->set_string("faceset_0", faceset_name);

        if (prim->tris.size() > 0) {
            prim->tris.add_attr<int>("faceset").assign(prim->tris.size(), 0);
        }
        if (prim->quads.size() > 0) {
            prim->quads.add_attr<int>("faceset").assign(prim->quads.size(), 0);
        }
        if (prim->polys.size() > 0) {
            prim->polys.add_attr<int>("faceset").assign(prim->polys.size(), 0);
        }
    }

    void prim_set_abcpath(PrimitiveObject* prim, String path_name) {
        auto pUserData = prim->userData();
        int faceset_count = pUserData->get_int("abcpath_count", 0);
        for (auto j = 0; j < faceset_count; j++) {
            pUserData->del(stdString2zs(zeno::format("abcpath_{}", j)));
        }
        pUserData->set_int("abcpath_count", 1);
        pUserData->set_string("abcpath_0", path_name);

        if (prim->tris.size() > 0) {
            prim->tris.add_attr<int>("abcpath").assign(prim->tris.size(), 0);
        }
        if (prim->quads.size() > 0) {
            prim->quads.add_attr<int>("abcpath").assign(prim->quads.size(), 0);
        }
        if (prim->polys.size() > 0) {
            prim->polys.add_attr<int>("abcpath").assign(prim->polys.size(), 0);
        }
    }

    void prim_copy_faceset_to_matid(PrimitiveObject* prim) {
        auto ud = prim->userData();
        auto faceset_count = ud->get_int("faceset_count", 0);
        ud->set_int("matNum", faceset_count);
        for (auto i = 0; i < faceset_count; i++) {
            auto value = ud->get_string(stdString2zs(format("faceset_{}", i)));
            ud->set_string(stdString2zs(format("Material_{}", i)), value);
        }

        if (prim->tris.attr_is<int>("faceset")) {
            prim->tris.attr_visit<AttrAcceptAll>("faceset", [&](auto const& attarr) {
                using T = std::decay_t<decltype(attarr[0])>;
                auto& targetAttr = prim->tris.template add_attr<T>("matid");
                std::copy(attarr.begin(), attarr.end(), targetAttr.begin());
                });
        }
        if (prim->polys.attr_is<int>("faceset")) {
            prim->polys.attr_visit<AttrAcceptAll>("faceset", [&](auto const& attarr) {
                using T = std::decay_t<decltype(attarr[0])>;
                auto& targetAttr = prim->polys.template add_attr<T>("matid");
                std::copy(attarr.begin(), attarr.end(), targetAttr.begin());
                });
        }
    }

    std::unique_ptr<zeno::PrimitiveObject> primMergeWithFacesetMatid(Vector<zeno::PrimitiveObject*> const& primList, String const& tagAttr, bool tag_on_vert, bool tag_on_face)
    {
        std::vector<std::string> matNameList(0);
        std::unordered_map<std::string, int> facesetNameMap;
        std::unordered_map<std::string, int> abcpathNameMap;
        for (auto p : primList) {
            //if p has material
            int matNum = p->userData()->get_int("matNum", 0);
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
                    auto matName = p->userData()->get_string(stdString2zs(matIdx), "Default");
                    matNameList.emplace_back(zsString2Std(matName));
                }
            }
            else {
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
            int faceset_count = p->userData()->get_int("faceset_count", 0);
            if (faceset_count == 0) {
                prim_set_faceset(p, "defFS");
                faceset_count = 1;
            }
            set_special_attr_remap(p, "faceset", facesetNameMap);
            int path_count = p->userData()->get_int("abcpath_count", 0);
            if (path_count == 0) {
                prim_set_abcpath(p, "/ABC/unassigned");
                path_count = 1;
            }
            set_special_attr_remap(p, "abcpath", abcpathNameMap);
        }
        auto outprim = PrimMerge(primList, tagAttr, tag_on_vert, tag_on_face);
        auto pUserData = dynamic_cast<UserData*>(outprim->userData());
        for (auto& p : primList) {
            pUserData->merge(*dynamic_cast<UserData*>(p->userData()));
        }

        if (matNameList.size() > 0) {
            //add matNames to userData
            int i = 0;
            for (auto name : matNameList) {
                auto matIdx = "Material_" + to_string(i);
                pUserData->setLiterial(matIdx, name);
                i++;
            }
        }
        int oMatNum = matNameList.size();
        pUserData->set2("matNum", oMatNum);
        {
            for (auto const& [k, v] : facesetNameMap) {
                pUserData->set2(format("faceset_{}", v), k);
            }
            int faceset_count = facesetNameMap.size();
            pUserData->set2("faceset_count", faceset_count);
        }
        {
            for (auto const& [k, v] : abcpathNameMap) {
                pUserData->set2(format("abcpath_{}", v), k);
            }
            int abcpath_count = abcpathNameMap.size();
            pUserData->set2("abcpath_count", abcpath_count);
        }
        return outprim;
    }

    void primKillDeadVerts(PrimitiveObject* prim) {
        std::set<int> reached(prim->points.begin(), prim->points.end());
        for (auto const& ind : prim->lines) {
            reached.insert(ind[0]);
            reached.insert(ind[1]);
        }
        for (auto const& ind : prim->tris) {
            reached.insert(ind[0]);
            reached.insert(ind[1]);
            reached.insert(ind[2]);
        }
        for (auto const& ind : prim->quads) {
            reached.insert(ind[0]);
            reached.insert(ind[1]);
            reached.insert(ind[2]);
            reached.insert(ind[3]);
        }
        for (auto const& [start, len] : prim->polys) {
            for (int i = start; i < start + len; i++) {
                reached.insert(prim->loops[i]);
            }
        }
        std::vector<int> revamp(reached.begin(), reached.end());
        auto old_prim_size = prim->verts.size();
        prim->verts.forall_attr<AttrAcceptAll>([&](auto const& key, auto& arr) {
            revamp_vector(arr, revamp);
            });
        prim->verts.resize(revamp.size());

        std::vector<int> unrevamp(old_prim_size);
        for (int i = 0; i < revamp.size(); i++) {
            unrevamp[revamp[i]] = i;
        }

        auto mock = [&](int& ind) {
            ind = unrevamp[ind];
        };
        for (auto& ind : prim->points) {
            mock(ind);
        }
        for (auto& ind : prim->lines) {
            mock(ind[0]);
            mock(ind[1]);
        }
        for (auto& ind : prim->tris) {
            mock(ind[0]);
            mock(ind[1]);
            mock(ind[2]);
        }
        for (auto& ind : prim->quads) {
            mock(ind[0]);
            mock(ind[1]);
            mock(ind[2]);
            mock(ind[3]);
        }
        for (auto const& [start, len] : prim->polys) {
            for (int i = start; i < start + len; i++) {
                mock(prim->loops[i]);
            }
        }
        primKillDeadUVs(prim);
        primKillDeadLoops(prim);
    }

    zeno::Vector<std::unique_ptr<PrimitiveObject>> PrimUnmergeFaces(PrimitiveObject* prim, String tagAttr) {
        if (!prim->verts.size()) return {};

        if (prim->tris.size() > 0 && prim->polys.size() > 0) {
            primPolygonate(prim, true);
        }

        std::string sTagAttr = zsString2Std(tagAttr);

        Vector<std::unique_ptr<PrimitiveObject>> list;

        std::map<int, std::vector<int>> mapping;
        if (prim->tris.size() > 0) {
            auto& attr = prim->tris.attr<int>(sTagAttr);
            for (auto i = 0; i < prim->tris.size(); i++) {
                if (mapping.count(attr[i]) == 0) {
                    mapping[attr[i]] = {};
                }
                mapping[attr[i]].push_back(i);
            }
            for (auto& [key, val] : mapping) {
                auto new_prim = safe_uniqueptr_cast<PrimitiveObject>(prim->clone());
                new_prim->tris.resize(val.size());
                for (auto i = 0; i < val.size(); i++) {
                    new_prim->tris[i] = prim->tris[val[i]];
                }
                new_prim->tris.foreach_attr<AttrAcceptAll>([&](auto const& key, auto& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    auto& attr = prim->tris.attr<T>(key);
                    for (auto i = 0; i < arr.size(); i++) {
                        arr[i] = attr[val[i]];
                    }
                    });
                list.push_back(std::move(new_prim));
            }
        }
        else if (prim->polys.size() > 0) {
            auto& attr = prim->polys.attr<int>(sTagAttr);
            for (auto i = 0; i < prim->polys.size(); i++) {
                if (mapping.count(attr[i]) == 0) {
                    mapping[attr[i]] = {};
                }
                mapping[attr[i]].push_back(i);
            }
            for (auto& [key, val] : mapping) {
                auto new_prim = safe_uniqueptr_cast<PrimitiveObject>(prim->clone());
                new_prim->polys.resize(val.size());
                for (auto i = 0; i < val.size(); i++) {
                    new_prim->polys[i] = prim->polys[val[i]];
                }
                new_prim->polys.foreach_attr<AttrAcceptAll>([&](auto const& key, auto& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    auto& attr = prim->polys.attr<T>(key);
                    for (auto i = 0; i < arr.size(); i++) {
                        arr[i] = attr[val[i]];
                    }
                    });
                list.push_back(std::move(new_prim));
            }
        }
        for (auto i = 0; i < list.size(); i++) {
            primKillDeadVerts(list[i].get());
            auto pUserData = dynamic_cast<UserData*>(list[i]->userData());
            // remove unused abcpath
            {
                auto abcpath_set = get_attr_on_faces(list[i].get(), "abcpath", true);
                std::map<int, int> mapping;
                std::vector<std::string> abcpaths;
                for (auto& k : abcpath_set) {
                    mapping[k] = abcpaths.size();
                    abcpaths.push_back(zsString2Std(pUserData->get_string(stdString2zs(format("abcpath_{}", k)))));
                }
                remap_attr_on_faces(list[i].get(), "abcpath", mapping);
                auto old_abcpath_count = pUserData->get_int("abcpath_count", 0);
                for (int j = 0; j < old_abcpath_count; j++) {
                    pUserData->del(stdString2zs(format("abcpath_{}", j)));
                }

                for (int j = 0; j < abcpaths.size(); j++) {
                    pUserData->set2(format("abcpath_{}", j), abcpaths[j]);
                }
                pUserData->set2("abcpath_count", int(abcpath_set.size()));
            }
            // remove unused faceset
            {
                auto abcpath_set = get_attr_on_faces(list[i].get(), "faceset", true);
                std::map<int, int> mapping;
                std::vector<std::string> abcpaths;
                for (auto& k : abcpath_set) {
                    mapping[k] = abcpaths.size();
                    abcpaths.push_back(zsString2Std(pUserData->get_string(stdString2zs(format("faceset_{}", k)))));
                }
                remap_attr_on_faces(list[i].get(), "faceset", mapping);
                int old_abcpath_count = pUserData->get_int("faceset_count", 0);
                for (int j = 0; j < old_abcpath_count; j++) {
                    pUserData->del(stdString2zs(format("faceset_{}", j)));
                }

                for (int j = 0; j < abcpaths.size(); j++) {
                    pUserData->set2(format("faceset_{}", j), abcpaths[j]);
                }
                pUserData->set2("faceset_count", int(abcpath_set.size()));
            }
        }
        return list;
    }

    void primFlipFaces(PrimitiveObject* prim, bool only_face) {
        if (!only_face && prim->lines.size())
            parallel_for_each(prim->lines.begin(), prim->lines.end(), [&](auto& line) {
            std::swap(line[1], line[0]);
                });
        if (prim->tris.size()) {
            parallel_for_each(prim->tris.begin(), prim->tris.end(), [&](auto& tri) {
                std::swap(tri[2], tri[0]);
                });
            if (prim->tris.attr_is<vec3f>("uv0")) {
                auto& uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
                auto& uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");
                for (auto i = 0; i < prim->tris.size(); i++) {
                    std::swap(uv0[i], uv2[i]);
                }
            }
        }
        if (prim->quads.size())
            parallel_for_each(prim->quads.begin(), prim->quads.end(), [&](auto& quad) {
            std::swap(quad[3], quad[0]);
            std::swap(quad[2], quad[1]);
                });
        if (prim->polys.size()) {
            prim->loops.forall_attr<AttrAcceptAll>([&](auto const& key, auto& arr) {
                parallel_for_each(prim->polys.begin(), prim->polys.end(), [&](auto const& poly) {
                    auto const& [start, len] = poly;
                    for (int i = 0; i < (len >> 1); i++) {
                        std::swap(arr[start + i], arr[start + len - 1 - i]);
                    }
                    });
                });
        }
    }

    void primCalcNormal(zeno::PrimitiveObject* prim, float flip, zeno::String nrmAttr)
    {
        const std::string sNrmAttr = zsString2Std(nrmAttr);
        auto& nrm = prim->add_attr<zeno::vec3f>(sNrmAttr);
        auto& pos = prim->verts.values;

#if defined(_OPENMP) && defined(__GNUG__)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < nrm.size(); i++) {
            nrm[i] = zeno::vec3f(0);
        }

#if defined(_OPENMP) && defined(__GNUG__)
        auto atomicCas = [](int* dest, int expected, int desired) {
            __atomic_compare_exchange_n(const_cast<int volatile*>(dest), &expected,
                desired, false, __ATOMIC_ACQ_REL,
                __ATOMIC_RELAXED);
            return expected;
        };
        auto atomicFloatAdd = [&](float* dst, float val) {
            static_assert(sizeof(float) == sizeof(int), "sizeof float != sizeof int");
            int oldVal = reinterpret_bits<int>(*dst);
            int newVal = reinterpret_bits<int>(reinterpret_bits<float>(oldVal) + val), readVal{};
            while ((readVal = atomicCas((int*)dst, oldVal, newVal)) != oldVal) {
                oldVal = readVal;
                newVal = reinterpret_bits<int>(reinterpret_bits<float>(readVal) + val);
            }
            return reinterpret_bits<float>(oldVal);
        };
#endif

#if defined(_OPENMP) && defined(__GNUG__)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto ind = prim->tris[i];
            auto n = cross(pos[ind[1]] - pos[ind[0]], pos[ind[2]] - pos[ind[0]]);

#if defined(_OPENMP) && defined(__GNUG__)
            for (int j = 0; j != 3; ++j) {
                auto& n_i = nrm[ind[j]];
                for (int d = 0; d != 3; ++d)
                    atomicFloatAdd(&n_i[d], n[d]);
            }
#else
            nrm[ind[0]] += n;
            nrm[ind[1]] += n;
            nrm[ind[2]] += n;
#endif
        }

#if defined(_OPENMP) && defined(__GNUG__)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < prim->quads.size(); i++) {
            auto ind = prim->quads[i];
            std::array<vec3f, 4> ns = {
                cross(pos[ind[1]] - pos[ind[0]], pos[ind[2]] - pos[ind[0]]),
                cross(pos[ind[2]] - pos[ind[1]], pos[ind[3]] - pos[ind[1]]),
                cross(pos[ind[3]] - pos[ind[2]], pos[ind[0]] - pos[ind[2]]),
                cross(pos[ind[0]] - pos[ind[3]], pos[ind[1]] - pos[ind[3]]),
            };

#if defined(_OPENMP) && defined(__GNUG__)
            for (int j = 0; j != 4; ++j) {
                auto& n_i = nrm[ind[j]];
                for (int d = 0; d != 3; ++d)
                    atomicFloatAdd(&n_i[d], ns[j][d]);
            }
#else
            for (int j = 0; j != 4; ++j) {
                nrm[ind[j]] += ns[j];
            }
#endif
        }

#if defined(_OPENMP) && defined(__GNUG__)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < prim->polys.size(); i++) {
            auto [beg, len] = prim->polys[i];

            auto ind = [loops = prim->loops.data(), beg = beg, len = len](int t) -> int {
                if (t >= len) t -= len;
                return loops[beg + t];
            };
            for (int j = 0; j < len; ++j) {
                auto nsj = cross(pos[ind(j + 1)] - pos[ind(j)], pos[ind(j + 2)] - pos[ind(j)]);
#if defined(_OPENMP) && defined(__GNUG__)
                auto& n_i = nrm[ind(j)];
                for (int d = 0; d != 3; ++d)
                    atomicFloatAdd(&n_i[d], nsj[d]);
#else
                nrm[ind(j)] += nsj;
#endif
            }
        }

#if defined(_OPENMP) && defined(__GNUG__)
#pragma omp parallel for
#endif
        for (size_t i = 0; i < nrm.size(); i++) {
            nrm[i] = flip * normalizeSafe(nrm[i]);
        }
    }

    void primWireframe(PrimitiveObject* prim, bool removeFaces, bool toEdges) {
        struct segment_less {
            bool operator()(vec2i const& a, vec2i const& b) const {
                return std::make_pair(std::min(a[0], a[1]), std::max(a[0], a[1]))
                    < std::make_pair(std::min(b[0], b[1]), std::max(b[0], b[1]));
            }
        };
        std::set<vec2i, segment_less> segments;
        auto append = [&](int i, int j) {
            segments.emplace(i, j);
        };
        for (auto const& ind : prim->lines) {
            append(ind[0], ind[1]);
        }
        for (auto const& ind : prim->tris) {
            append(ind[0], ind[1]);
            append(ind[1], ind[2]);
            append(ind[2], ind[0]);
        }
        for (auto const& ind : prim->quads) {
            append(ind[0], ind[1]);
            append(ind[1], ind[2]);
            append(ind[2], ind[3]);
            append(ind[3], ind[0]);
        }
        for (auto const& [start, len] : prim->polys) {
            if (len < 2)
                continue;
            for (int i = start + 1; i < start + len; i++) {
                append(prim->loops[i - 1], prim->loops[i]);
            }
            append(prim->loops[start + len - 1], prim->loops[start]);
        }
        //if (isAccumulate) {
            //for (auto const &ind: prim->lines) {
                //segments.erase(vec2i(ind[0], ind[1]));
            //}
            //prim->lines.values.insert(prim->lines.values.end(), segments.begin(), segments.end());
            //prim->lines.update();
        //} else {
        if (toEdges) {
            prim->edges.attrs.clear();
            prim->edges.values.assign(segments.begin(), segments.end());
            prim->edges.update();
        }
        else {
            prim->lines.attrs.clear();
            prim->lines.values.assign(segments.begin(), segments.end());
            prim->lines.update();
        }
        //}
        if (removeFaces) {
            prim->tris.clear();
            prim->quads.clear();
            prim->loops.clear();
            prim->polys.clear();
        }
    }

    void primSampleTexture(
        PrimitiveObject* prim,
        const zeno::String& srcChannel,
        const zeno::String& srcSource,
        const zeno::String& dstChannel,
        PrimitiveObject* img,
        const zeno::String& wrap,
        zeno::Vec3f borderColor,
        float remapMin,
        float remapMax
    ) {

    }

    void primSampleHeatmap(
        PrimitiveObject* prim,
        const zeno::String& srcChannel,
        const zeno::String& dstChannel,
        HeatmapObject* heatmap,
        float remapMin,
        float remapMax
    ) {
        auto& clr = prim->add_attr<zeno::vec3f>(zsString2Std(dstChannel));
        auto& src = prim->attr<float>(zsString2Std(srcChannel));
#pragma omp parallel for //ideally this could be done in opengl
        for (int i = 0; i < src.size(); i++) {
            auto x = (src[i] - remapMin) / (remapMax - remapMin);
            clr[i] = heatmap->interp(x);
        }
    }

    void primPolygonate(PrimitiveObject* prim, bool with_uv) {
        prim->loops.reserve(prim->loops.size() + prim->tris.size() * 3 +
            prim->quads.size() * 4 + prim->lines.size() * 2 +
            prim->points.size());
        prim->polys.reserve(prim->polys.size() + prim->tris.size() +
            prim->quads.size() + prim->lines.size() +
            prim->points.size());

        int old_loop_base = prim->loops.size();
        int polynum = prim->polys.size();
        if (prim->tris.size()) {
            int base = prim->loops.size();
            for (int i = 0; i < prim->tris.size(); i++) {
                auto const& ind = prim->tris[i];
                prim->loops.push_back(ind[0]);
                prim->loops.push_back(ind[1]);
                prim->loops.push_back(ind[2]);
                prim->polys.push_back({ base + i * 3, 3 });
            }
            prim->polys.update();

            prim->tris.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                if (key == "uv0" || key == "uv1" || key == "uv2") {
                    return;
                }
                using T = std::decay_t<decltype(arr[0])>;
                auto& newarr = prim->polys.add_attr<T>(key);
                for (auto i = 0; i < arr.size(); i++) {
                    newarr[polynum + i] = arr[i];
                }
                });
        }

        polynum = prim->polys.size();
        if (prim->quads.size()) {
            int base = prim->loops.size();
            for (int i = 0; i < prim->quads.size(); i++) {
                auto const& ind = prim->quads[i];
                prim->loops.push_back(ind[0]);
                prim->loops.push_back(ind[1]);
                prim->loops.push_back(ind[2]);
                prim->loops.push_back(ind[3]);
                prim->polys.push_back({ base + i * 4, 4 });
            }
            prim->polys.update();

            prim->quads.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                if (key == "uv0" || key == "uv1" || key == "uv2" || key == "uv3") {
                    return;
                }
                using T = std::decay_t<decltype(arr[0])>;
                auto& newarr = prim->polys.add_attr<T>(key);
                for (auto i = 0; i < arr.size(); i++) {
                    newarr[polynum + i] = arr[i];
                }
                });
        }

        polynum = prim->polys.size();
        if (prim->lines.size()) {
            int base = prim->loops.size();
            for (int i = 0; i < prim->lines.size(); i++) {
                auto const& ind = prim->lines[i];
                prim->loops.push_back(ind[0]);
                prim->loops.push_back(ind[1]);
                prim->polys.push_back({ base + i * 2, 2 });
            }
            prim->polys.update();

            prim->lines.foreach_attr([&](auto const& key, auto const& arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto& newarr = prim->polys.add_attr<T>(key);
                for (auto i = 0; i < arr.size(); i++) {
                    newarr[polynum + i] = arr[i];
                }
                });
        }

        polynum = prim->polys.size();
        if (prim->points.size()) {
            int base = prim->loops.size();
            for (int i = 0; i < prim->points.size(); i++) {
                auto ind = prim->points[i];
                prim->loops.push_back(ind);
                prim->polys.push_back({ base + i, 1 });
            }
            prim->polys.update();

            prim->points.foreach_attr([&](auto const& key, auto const& arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto& newarr = prim->polys.add_attr<T>(key);
                for (auto i = 0; i < arr.size(); i++) {
                    newarr[polynum + i] = arr[i];
                }
                });
        }

        prim->loops.update();
        prim->polys.update();

        if (!(!prim->tris.has_attr("uv0") || !prim->tris.has_attr("uv1") ||
            !prim->tris.has_attr("uv2") || !with_uv)) {
            auto old_uvs_base = prim->uvs.size();
            prim->loops.add_attr<int>("uvs");
            auto& uv0 = prim->tris.attr<zeno::vec3f>("uv0");
            auto& uv1 = prim->tris.attr<zeno::vec3f>("uv1");
            auto& uv2 = prim->tris.attr<zeno::vec3f>("uv2");
            for (int i = 0; i < prim->tris.size(); i++) {
                prim->loops.attr<int>("uvs")[old_loop_base + i * 3 + 0] = old_uvs_base + i * 3 + 0;
                prim->loops.attr<int>("uvs")[old_loop_base + i * 3 + 1] = old_uvs_base + i * 3 + 1;
                prim->loops.attr<int>("uvs")[old_loop_base + i * 3 + 2] = old_uvs_base + i * 3 + 2;
                prim->uvs.emplace_back(uv0[i][0], uv0[i][1]);
                prim->uvs.emplace_back(uv1[i][0], uv1[i][1]);
                prim->uvs.emplace_back(uv2[i][0], uv2[i][1]);
            }
            // remove duplicate uv index
            {
                std::map<std::tuple<float, float>, int> mapping;
                auto& loopsuv = prim->loops.attr<int>("uvs");
                for (auto i = 0; i < prim->loops.size(); i++) {
                    vec2f uv = prim->uvs[loopsuv[i]];
                    if (mapping.count({ uv[0], uv[1] }) == false) {
                        auto index = mapping.size();
                        mapping[{uv[0], uv[1]}] = index;
                    }
                    loopsuv[i] = mapping[{uv[0], uv[1]}];
                }
                prim->uvs.resize(mapping.size());
                for (auto const& [uv, index] : mapping) {
                    prim->uvs[index] = { std::get<0>(uv), std::get<1>(uv) };
                }
            }
        }

        prim->tris.clear_with_attr();
        prim->quads.clear_with_attr();
        prim->lines.clear_with_attr();
        prim->points.clear_with_attr();
    }

    std::unique_ptr<zeno::PrimitiveObject> PrimMerge(Vector<zeno::PrimitiveObject*> const& primList, String const& tagAttr, bool tag_on_vert, bool tag_on_face) {
        //zeno::log_critical("asdfjhl {}", primList.size());
    //throw;
        std::string stagAttr = zsString2Std(tagAttr);

        int poly_flag = 0;
        for (auto& p : primList) {
            if (p->polys.size()) {
                poly_flag += 1;
            }
        }
        // check if mix polys and tris
        if (0 < poly_flag && poly_flag < primList.size()) {
            for (auto& p : primList) {
                if (p->polys.size() == 0) {
                    primPolygonate(p, true);
                }
            }
        }

        auto outprim = std::make_unique<PrimitiveObject>();

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

            if (stagAttr.size()) {
                if (tag_on_vert) {
                    outprim->verts.add_attr<int>(stagAttr);
                }
                if (tag_on_face) {
                    outprim->tris.add_attr<int>(stagAttr);
                    outprim->polys.add_attr<int>(stagAttr);
                }
            }
            for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
                auto const& prim = primList[primIdx];
                prim->verts.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    outprim->verts.add_attr<T>(key);
                    });
                prim->points.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    outprim->points.add_attr<T>(key);
                    });
                prim->lines.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    outprim->lines.add_attr<T>(key);
                    });
                prim->tris.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    outprim->tris.add_attr<T>(key);
                    });
                prim->quads.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    outprim->quads.add_attr<T>(key);
                    });
                prim->loops.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    outprim->loops.add_attr<T>(key);
                    });
                prim->uvs.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    outprim->uvs.add_attr<T>(key);
                    });
                prim->polys.foreach_attr<AttrAcceptAll>([&](auto const& key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    outprim->polys.add_attr<T>(key);
                    });
            }

            parallel_for(primList.size(), [&](size_t primIdx) {
                auto prim = primList[primIdx];
                auto base = bases[primIdx];
                auto core = [&](auto key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
#if 0
                    auto& outarr = [&]() -> auto& {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            return outprim->verts.values;
                        }
                        else {
                            return outprim->verts.attr<T>(key);
                        }
                    }();
                    size_t n = std::min(arr.size(), prim->verts.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
#else
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        auto& outarr = outprim->verts.values;
                        size_t n = std::min(arr.size(), prim->verts.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = arr[i];
                        }
                    }
                    else {
                        auto& outarr = outprim->verts.attr<T>(key);
                        size_t n = std::min(arr.size(), prim->verts.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = arr[i];
                        }
                    }
#endif
                };
                core(std::true_type{}, prim->verts.values);
                prim->verts.foreach_attr<AttrAcceptAll>(core);
                if (tag_on_vert && stagAttr.size()) {
                    auto& outarr = outprim->verts.attr<int>(stagAttr);
                    for (size_t i = 0; i < prim->verts.size(); i++) {
                        outarr[base + i] = primIdx;
                    }
                }
                });

            parallel_for(primList.size(), [&](size_t primIdx) {
                auto prim = primList[primIdx];
                auto vbase = bases[primIdx];
                auto base = pointbases[primIdx];
                auto core = [&](auto key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
#if 0
                    auto& outarr = [&]() -> auto& {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            return outprim->points.values;
                        }
                        else {
                            return outprim->points.attr<T>(key);
                        }
                    }();
                    size_t n = std::min(arr.size(), prim->points.size());
                    for (size_t i = 0; i < n; i++) {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            outarr[base + i] = vbase + arr[i];
                        }
                        else {
                            outarr[base + i] = arr[i];
                        }
                    }
#else
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        auto& outarr = outprim->points.values;
                        size_t n = std::min(arr.size(), prim->points.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = vbase + arr[i];
                        }
                    }
                    else {
                        auto& outarr = outprim->points.attr<T>(key);
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
                auto core = [&](auto key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
#if 0
                    auto& outarr = [&]() -> auto& {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            return outprim->lines.values;
                        }
                        else {
                            return outprim->lines.attr<T>(key);
                        }
                    }();
                    size_t n = std::min(arr.size(), prim->lines.size());
                    for (size_t i = 0; i < n; i++) {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            outarr[base + i] = vbase + arr[i];
                        }
                        else {
                            outarr[base + i] = arr[i];
                        }
                    }
#else
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        auto& outarr = outprim->lines.values;
                        size_t n = std::min(arr.size(), prim->lines.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = vbase + arr[i];
                        }
                    }
                    else {
                        auto& outarr = outprim->lines.attr<T>(key);
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
                auto core = [&](auto key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
#if 0
                    auto& outarr = [&]() -> auto& {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            return outprim->tris.values;
                        }
                        else {
                            return outprim->tris.attr<T>(key);
                        }
                    }();
                    size_t n = std::min(arr.size(), prim->tris.size());
                    for (size_t i = 0; i < n; i++) {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            outarr[base + i] = vbase + arr[i];
                        }
                        else {
                            outarr[base + i] = arr[i];
                        }
                    }
#else
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        auto& outarr = outprim->tris.values;
                        size_t n = std::min(arr.size(), prim->tris.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = vbase + arr[i];
                        }
                    }
                    else {
                        auto& outarr = outprim->tris.attr<T>(key);
                        size_t n = std::min(arr.size(), prim->tris.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = arr[i];
                        }
                    }
#endif
                };
                core(std::true_type{}, prim->tris.values);
                prim->tris.foreach_attr<AttrAcceptAll>(core);
                if (tag_on_face && stagAttr.size()) {
                    auto& outarr = outprim->tris.attr<int>(stagAttr);
                    for (size_t i = 0; i < prim->tris.size(); i++) {
                        outarr[base + i] = primIdx;
                    }
                }
                });

            parallel_for(primList.size(), [&](size_t primIdx) {
                auto prim = primList[primIdx];
                auto vbase = bases[primIdx];
                auto base = quadbases[primIdx];
                auto core = [&](auto key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
#if 0
                    auto& outarr = [&]() -> auto& {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            return outprim->quads.values;
                        }
                        else {
                            return outprim->quads.attr<T>(key);
                        }
                    }();
                    size_t n = std::min(arr.size(), prim->quads.size());
                    for (size_t i = 0; i < n; i++) {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            outarr[base + i] = vbase + arr[i];
                        }
                        else {
                            outarr[base + i] = arr[i];
                        }
                    }
#else
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        auto& outarr = outprim->quads.values;
                        size_t n = std::min(arr.size(), prim->quads.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = vbase + arr[i];
                        }
                    }
                    else {
                        auto& outarr = outprim->quads.attr<T>(key);
                        size_t n = std::min(arr.size(), prim->quads.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = arr[i];
                        }
                    }
#endif
                };
                core(std::true_type{}, prim->quads.values);
                prim->quads.foreach_attr<AttrAcceptAll>(core);
                //            if (stagAttr.size()) {
                //                auto &outarr = outprim->quads.attr<int>(stagAttr);
                //                for (size_t i = 0; i < prim->quads.size(); i++) {
                //                    outarr[base + i] = primIdx;
                //                }
                //            }
                });

            parallel_for(primList.size(), [&](size_t primIdx) {
                auto prim = primList[primIdx];
                auto vbase = bases[primIdx];
                auto base = loopbases[primIdx];
                auto core = [&](auto key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
#if 0
                    auto& outarr = [&]() -> auto& {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            return outprim->loops.values;
                        }
                        else {
                            return outprim->loops.attr<T>(key);
                        }
                    }();
                    size_t n = std::min(arr.size(), prim->loops.size());
                    for (size_t i = 0; i < n; i++) {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            outarr[base + i] = vbase + arr[i];
                        }
                        else {
                            outarr[base + i] = arr[i];
                        }
                    }
#else
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        auto& outarr = outprim->loops.values;
                        size_t n = std::min(arr.size(), prim->loops.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = vbase + arr[i];
                        }
                        return;
                    }
                    else if constexpr (std::is_same_v<T, int>) {
                        if (key == "uvs") {
                            auto& outarr = outprim->loops.attr<int>("uvs");
                            size_t n = std::min(arr.size(), prim->loops.size());
                            size_t offset = uvbases[primIdx];
                            for (size_t i = 0; i < n; i++) {
                                outarr[base + i] = arr[i] + offset;
                            }
                            return;
                        }
                    }
                    if constexpr (!std::is_same_v<decltype(key), std::true_type>) {
                        auto& outarr = outprim->loops.attr<T>(key);
                        size_t n = std::min(arr.size(), prim->loops.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = arr[i];
                        }
                    }
#endif
                };
                core(std::true_type{}, prim->loops.values);
                prim->loops.foreach_attr<AttrAcceptAll>(core);
                //            if (stagAttr.size()) {
                //                auto &outarr = outprim->loops.attr<int>(stagAttr);
                //                for (size_t i = 0; i < prim->loops.size(); i++) {
                //                    outarr[base + i] = primIdx;
                //                }
                //            }
                });

            parallel_for(primList.size(), [&](size_t primIdx) {
                auto prim = primList[primIdx];
                auto base = uvbases[primIdx];
                auto core = [&](auto key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        auto& outarr = outprim->uvs.values;
                        size_t n = std::min(arr.size(), prim->uvs.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = arr[i];
                        }
                    }
                    else {
                        auto& outarr = outprim->uvs.attr<T>(key);
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
                auto core = [&](auto key, auto const& arr) {
                    using T = std::decay_t<decltype(arr[0])>;
#if 0
                    auto& outarr = [&]() -> auto& {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            return outprim->polys.values;
                        }
                        else {
                            return outprim->polys.attr<T>(key);
                        }
                    }();
                    size_t n = std::min(arr.size(), prim->polys.size());
                    for (size_t i = 0; i < n; i++) {
                        if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                            outarr[base + i] = { arr[i].first + lbase, arr[i].second };
                        }
                        else {
                            outarr[base + i] = arr[i];
                        }
                    }
#else
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        auto& outarr = outprim->polys.values;
                        size_t n = std::min(arr.size(), prim->polys.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = { arr[i][0] + (int)lbase, arr[i][1] };
                        }
                    }
                    else {
                        auto& outarr = outprim->polys.add_attr<T>(key);
                        size_t n = std::min(arr.size(), prim->polys.size());
                        for (size_t i = 0; i < n; i++) {
                            outarr[base + i] = arr[i];
                        }
                    }
#endif
                };
                core(std::true_type{}, prim->polys.values);
                prim->polys.foreach_attr<AttrAcceptAll>(core);
                if (tag_on_face && stagAttr.size()) {
                    auto& outarr = outprim->polys.attr<int>(stagAttr);
                    for (size_t i = 0; i < prim->polys.size(); i++) {
                        outarr[base + i] = primIdx;
                    }
                }
            });
        }

        return outprim;
    }

    std::unique_ptr<PrimitiveObject> readExrFile(zeno::String const& path) {
        int nx, ny, nc = 4;
        float* rgba;
        const char* err;
        std::string native_path = std::filesystem::u8path(zsString2Std(path)).string();
        int ret = LoadEXR(&rgba, &nx, &ny, native_path.c_str(), &err);
        if (ret != 0) {
            zeno::log_error("load exr: {}", err);
            throw std::runtime_error(zeno::format("load exr: {}", err));
        }
        nx = std::max(nx, 1);
        ny = std::max(ny, 1);
        for (auto i = 0; i < ny / 2; i++) {
            for (auto x = 0; x < nx * 4; x++) {
                auto index1 = i * (nx * 4) + x;
                auto index2 = (ny - 1 - i) * (nx * 4) + x;
                std::swap(rgba[index1], rgba[index2]);
            }
        }

        auto img = std::make_unique<PrimitiveObject>();
        img->verts.resize(nx * ny);

        auto& alpha = img->verts.add_attr<float>("alpha");
        for (int i = 0; i < nx * ny; i++) {
            img->verts[i] = { rgba[i * 4 + 0], rgba[i * 4 + 1], rgba[i * 4 + 2] };
            alpha[i] = rgba[i * 4 + 3];
        }
        //
        img->userData()->set_int("isImage", 1);
        img->userData()->set_int("w", nx);
        img->userData()->set_int("h", ny);
        return img;
    }

    std::unique_ptr<PrimitiveObject> readImageFile(zeno::String const& path) {
        int w, h, n;
        stbi_set_flip_vertically_on_load(true);
        std::string native_path = std::filesystem::u8path(zsString2Std(path)).string();
        float* data = stbi_loadf(native_path.c_str(), &w, &h, &n, 0);
        if (!data) {
            throw zeno::Exception("cannot open image file at path: " + native_path);
        }
        scope_exit delData = [=] { stbi_image_free(data); };
        auto img = std::make_unique<PrimitiveObject>();
        img->verts.resize(w * h);
        if (n == 3) {
            std::memcpy(img->verts.data(), data, w * h * n * sizeof(float));
        }
        else if (n == 4) {
            auto& alpha = img->verts.add_attr<float>("alpha");
            for (int i = 0; i < w * h; i++) {
                img->verts[i] = { data[i * 4 + 0], data[i * 4 + 1], data[i * 4 + 2] };
                alpha[i] = data[i * 4 + 3];
            }
        }
        else if (n == 2) {
            for (int i = 0; i < w * h; i++) {
                img->verts[i] = { data[i * 2 + 0], data[i * 2 + 1], 0 };
            }
        }
        else if (n == 1) {
            for (int i = 0; i < w * h; i++) {
                img->verts[i] = vec3f(data[i]);
            }
        }
        else {
            throw zeno::Exception("too much number of channels");
        }
        img->userData()->set_int("isImage", 1);
        img->userData()->set_int("w", w);
        img->userData()->set_int("h", h);
        return img;
    }

    std::unique_ptr<PrimitiveObject> readPFMFile(zeno::String const& path) {
        int nx = 0;
        int ny = 0;
        std::ifstream file(zsString2Std(path), std::ios::binary);
        std::string format;
        file >> format;
        file >> nx >> ny;
        float scale = 0;
        file >> scale;
        file.ignore(1);

        auto img = std::make_unique<PrimitiveObject>();
        int size = nx * ny;
        img->resize(size);
        file.read(reinterpret_cast<char*>(img->verts.data()), sizeof(vec3f) * nx * ny);

        img->userData()->set_int("isImage", 1);
        img->userData()->set_int("w", nx);
        img->userData()->set_int("h", ny);
        return img;
    }

    void write_pfm(const std::string& path, int w, int h, vec3f* rgb) {
        std::string header = zeno::format("PF\n{} {}\n-1.0\n", w, h);
        std::vector<char> data(header.size() + w * h * sizeof(vec3f));
        memcpy(data.data(), header.data(), header.size());
        memcpy(data.data() + header.size(), rgb, w * h * sizeof(vec3f));
        file_put_binary(data, path);
    }

    void write_pfm(const zeno::String& path, PrimitiveObject* image) {
        auto ud = image->userData();
        int w = ud->get_int("w");
        int h = ud->get_int("h");
        write_pfm(zsString2Std(path), w, h, image->verts->data());
    }

    void write_jpg(const zeno::String& path, PrimitiveObject* image) {
        int w = image->userData()->get_int("w");
        int h = image->userData()->get_int("h");
        std::vector<uint8_t> colors;
        for (auto i = 0; i < w * h; i++) {
            auto rgb = zeno::pow(image->verts[i], 1.0f / 2.2f);
            int r = zeno::clamp(int(rgb[0] * 255.99), 0, 255);
            int g = zeno::clamp(int(rgb[1] * 255.99), 0, 255);
            int b = zeno::clamp(int(rgb[2] * 255.99), 0, 255);
            colors.push_back(r);
            colors.push_back(g);
            colors.push_back(b);
        }
        stbi_flip_vertically_on_write(1);
        stbi_write_jpg(path.c_str(), w, h, 3, colors.data(), 100);
    }

}