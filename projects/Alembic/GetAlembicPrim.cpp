#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/geo/commonutil.h>
#include <glm/glm.hpp>
#include <zeno/types/ListObject.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/IGeometryObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/NumericObject.h>
#include <zeno/extra/GlobalState.h>
#include "ABCCommon.h"
#include "ABCTree.h"
#include "zeno/utils/string.h"
#include <queue>
#include <utility>

namespace zeno {
struct JsonObject : IObjectClone<JsonObject> {
    Json json;
};

int count_alembic_prims(zeno::ABCTree* abctree) {
    int count = 0;
    abctree->visitPrims([&] (auto const &p) {
        count++;
    });
    return count;
}

struct CountAlembicPrims : INode {
    virtual void apply() override {
        auto abctree = safe_dynamic_cast<ABCTree>(get_input("abctree"));
        int count = count_alembic_prims(abctree);
        set_output_int("count", count);
    }
};

ZENDEFNODE(CountAlembicPrims, {
    {{gParamType_ABCTree, "abctree"}},
    {{gParamType_Int, "count"}},
    {},
    {"alembic"},
});

std::unique_ptr<PrimitiveObject> get_alembic_prim(zeno::ABCTree* abctree, int index) {
    std::unique_ptr<PrimitiveObject> prim;
    abctree->visitPrims([&] (auto const &p) {
        if (index == 0) {
            prim = safe_uniqueptr_cast<PrimitiveObject>(p->clone());
            return false;
        }
        index--;
        return true;
    });
    if (!prim) {
        throw Exception("index out of range in abctree");
    }
    return prim;
}

int get_alembic_prim_index(zeno::ABCTree* abctree, std::string name) {
    int index = 0;
    abctree->visitPrims([&] (auto const &p) {
        auto ud = p->userData();
        auto _abc_path = zsString2Std(ud->get_string("abcpath_0", ""));
        if (_abc_path == name) {
            return false;
        }
        else {
            index++;
            return true;
        }
    });
    return index;
}
void dfs_abctree(
    ABCTree* root,
    int parent_index,
    std::vector<ABCTree*>& linear_abctrees,
    std::vector<int>& linear_abctree_parent
) {
    int self_index = linear_abctrees.size();
    linear_abctrees.push_back(root);
    linear_abctree_parent.push_back(parent_index);
    for (auto const &ch: root->children) {
        dfs_abctree(ch.get(), self_index, linear_abctrees, linear_abctree_parent);
    }
}

std::unique_ptr<PrimitiveObject> get_xformed_prim(zeno::ABCTree* abctree, int index) {
    std::vector<ABCTree*> linear_abctrees;
    std::vector<int> linear_abctree_parent;
    dfs_abctree(abctree, -1, linear_abctrees, linear_abctree_parent);
    std::unique_ptr<PrimitiveObject> prim;
    std::vector<Alembic::Abc::M44d> transforms;
    for (auto i = 0; i < linear_abctrees.size(); i++) {
        auto const& abc_node = linear_abctrees[i];
        int parent_index = linear_abctree_parent[i];
        if (parent_index >= 0) {
            transforms.push_back(abc_node->xform * transforms[parent_index]);
        } else {
            transforms.push_back(abc_node->xform);
        }
        if (abc_node->prim) {
            if (index == 0) {
                prim = safe_uniqueptr_cast<PrimitiveObject>(abc_node->prim->clone());
                auto& mat = transforms.back();
                for (auto& p: prim->verts) {
                    auto pos = Imath::V4d(p[0], p[1], p[2], 1) * mat;
                    p = zeno::vec3f((float)pos.x, (float)pos.y, (float)pos.z);
                }
            }
            index--;
        }
    }
    return std::move(prim);
}

std::unique_ptr<zeno::ListObject> get_xformed_prims(zeno::ABCTree* abctree) {
    auto prims = std::make_unique<zeno::ListObject>();
    std::vector<ABCTree*> linear_abctrees;
    std::vector<int> linear_abctree_parent;
    dfs_abctree(abctree, -1, linear_abctrees, linear_abctree_parent);
    std::vector<Alembic::Abc::M44d> transforms;
    for (auto i = 0; i < linear_abctrees.size(); i++) {
        auto const& abc_node = linear_abctrees[i];
        int parent_index = linear_abctree_parent[i];
        if (parent_index >= 0) {
            transforms.push_back(abc_node->xform * transforms[parent_index]);
        } else {
            transforms.push_back(abc_node->xform);
        }
        if (abc_node->prim) {
            auto prim = safe_uniqueptr_cast<PrimitiveObject>(abc_node->prim->clone());
            auto& mat = transforms.back();
            for (auto& p: prim->verts) {
                auto pos = Imath::V4d(p[0], p[1], p[2], 1) * mat;
                p = zeno::vec3f((float)pos.x, (float)pos.y, (float)pos.z);
            }
            prims->push_back(std::move(prim));
        }
    }
    return prims;
}
struct GetAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = safe_dynamic_cast<ABCTree>(get_input("abctree"));
        int index = get_input2_int("index");
        bool use_xform = get_input2_bool("use_xform");
        std::unique_ptr<PrimitiveObject> prim;
        if (get_input2_bool("use_name")) {
            index = get_alembic_prim_index(abctree, std::string(get_input2_string("name").c_str()));
        }
        if (use_xform) {
            prim = get_xformed_prim(abctree, index);
        } else {
            prim = get_alembic_prim(abctree, index);
        }
        if (get_input2_bool("flipFrontBack")) {
            primFlipFaces(prim.get(), true);
        }
        if (get_input2_bool("triangulate")) {
            zeno::primTriangulate(prim.get());
        }
        auto geom = create_GeometryObject(prim.get());
        set_output("prim", std::move(geom));
    }
};

ZENDEFNODE(GetAlembicPrim, {
    {
        {gParamType_Bool, "flipFrontBack", "1"},
        {gParamType_ABCTree, "abctree"},
        {gParamType_Int, "index", "0"},
        {gParamType_Bool, "use_xform", "0"},
        {gParamType_Bool, "triangulate", "0"},
        {gParamType_Bool, "use_name", "0"},
        {gParamType_String, "name", ""},
    },
    {{gParamType_Geometry, "prim"}},
    {},
    {"alembic"},
});

struct AllAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = safe_dynamic_cast<ABCTree>(get_input("abctree"));
        auto prims = std::make_unique<zeno::ListObject>();
        int use_xform = get_input2_int("use_xform");
        if (use_xform) {
            prims = get_xformed_prims(abctree);
        } else {
            abctree->visitPrims([&] (auto const &p) {
                prims->push_back(p->clone());
            });
        }

        Vector<zeno::PrimitiveObject*> primlst;
        for (auto spobj : prims->get()) {
            primlst.push_back(dynamic_cast<PrimitiveObject*>(spobj));
        }

        auto outprim = zeno::PrimMerge(primlst);
        if (get_input2_bool("flipFrontBack")) {
            primFlipFaces(outprim.get(), true);
        }
        if (get_input2_int("triangulate") == 1) {
            zeno::primTriangulate(outprim.get());
        }
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(AllAlembicPrim, {
    {
        {gParamType_Bool, "flipFrontBack", "1"},
        {gParamType_ABCTree, "abctree"},
        {gParamType_Bool, "use_xform", "0"},
        {gParamType_Bool, "triangulate", "0"},
    },
    {{gParamType_Primitive, "prim"}},
    {},
    {"alembic"},
});

struct AlembicPrimList : INode {
    virtual void apply() override {
        auto abctree = safe_dynamic_cast<ABCTree>(get_input("abctree"));
        auto prims = std::make_shared<zeno::ListObject>();
        bool use_xform = get_input2_bool("use_xform");
        if (use_xform) {
            prims = get_xformed_prims(abctree);
        } else {
            abctree->visitPrims([&] (auto const &p) {
                prims->push_back(p->clone());
            });
        }

        bool bSplitByFaceset = get_input2_bool("splitByFaceset");
        std::unique_ptr<ListObject> new_prims;
        if (bSplitByFaceset) {
            new_prims = std::make_unique<zeno::ListObject>();
        }
        else {
            new_prims = safe_uniqueptr_cast<zeno::ListObject>(prims->clone());
        }
        
        std::vector<zany>& arr = new_prims->m_impl->m_objects;
        if (get_input2_bool("splitByFaceset")) {
            for (auto prim: prims->get()) {
                auto list = abc_split_by_name(dynamic_cast<PrimitiveObject*>(prim), false);
                auto listarr = list->get();
                arr.insert(arr.end(), listarr.begin(), listarr.end());
            }
        }

        auto pathInclude = zeno::split_str(zsString2Std(get_input2_string("pathInclude")), {' ', '\n'});
        auto pathExclude = zeno::split_str(zsString2Std(get_input2_string("pathExclude")), {' ', '\n'});
        auto facesetInclude = zeno::split_str(zsString2Std(get_input2_string("facesetInclude")), {' ', '\n'});
        auto facesetExclude = zeno::split_str(zsString2Std(get_input2_string("facesetExclude")), {' ', '\n'});
        for (auto it = arr.begin(); it != arr.end();) {
            auto np = safe_dynamic_cast<PrimitiveObject>((*it).get());
            auto abc_path = zsString2Std(np->userData()->get_string("abcpath_0"));
            bool contain = false;
            if (pathInclude.empty()) {
                contain = true;
            }
            else {
                for (const auto & p: pathInclude) {
                    if (starts_with(abc_path, p)) {
                        contain = true;
                    }
                }
            }
            if (contain) {
                for (const auto & p: pathExclude) {
                    if (starts_with(abc_path, p)) {
                        contain = false;
                    }
                }
            }
            if (contain && np->userData()->has("faceset_0")) {
                auto faceset = zsString2Std(np->userData()->get_string("faceset_0"));
                contain = false;
                if (facesetInclude.empty()) {
                    contain = true;
                }
                else {
                    for (const auto & p: facesetInclude) {
                        if (starts_with(faceset, p)) {
                            contain = true;
                        }
                    }
                }
                if (contain) {
                    for (const auto & p: facesetExclude) {
                        if (starts_with(faceset, p)) {
                            contain = false;
                        }
                    }
                }
            }
            if (contain) {
                ++it;
            } else {
                it = arr.erase(it);
            }
        }
        auto new_prims2 = create_ListObject();
        for (auto& prim : arr) {
            auto _prim = safe_dynamic_cast<PrimitiveObject>(prim.get());
            if (get_input2_bool("flipFrontBack")) {
                primFlipFaces(_prim, true);
            }
            if (get_input2_bool("splitByFaceset") && get_input2_bool("killDeadVerts")) {
                primKillDeadVerts(_prim);
            }
            if (get_input2_bool("triangulate")) {
                zeno::primTriangulate(_prim);
            }
            auto abcpath_0 = zsString2Std(_prim->userData()->get_string("abcpath_0"));
            abcpath_0 += "/mesh";
            _prim->userData()->set_string("abcpath_0", stdString2zs(abcpath_0));
            new_prims2->push_back(prim->clone());
        }

        auto new_geoms = create_ListObject();
        for (auto obj : new_prims2->get()) {
            auto prim = static_cast<PrimitiveObject*>(obj);
            new_geoms->push_back(create_GeometryObject(prim));
        }
        set_output("geoms", std::move(new_geoms));
    }
};

ZENDEFNODE(AlembicPrimList, {
    {
        {gParamType_Bool, "flipFrontBack", "1"},
        {gParamType_ABCTree, "abctree"},
        {gParamType_Bool, "use_xform", "0"},
        {gParamType_Bool, "triangulate", "0"},
        {gParamType_Bool, "splitByFaceset", "0"},
        {gParamType_Bool, "killDeadVerts", "1"},
        {gParamType_String, "pathInclude", ""},
        {gParamType_String, "pathExclude", ""},
        {gParamType_String, "facesetInclude", ""},
        {gParamType_String, "facesetExclude", ""},
    },
    {{gParamType_List, "geoms"}},
    {},
    {"alembic"},
});

struct AlembicSceneInfo : INode {
    virtual void apply() override {
        auto abctree = safe_dynamic_cast<ABCTree>(get_input("abctree"));
        auto json_obj = std::make_unique<JsonObject>();
        json_obj->json = abctree->get_scene_info();
        set_output("json", std::move(json_obj));
    }
};

ZENDEFNODE(AlembicSceneInfo, {
    {
        {gParamType_ABCTree, "abctree"},
    },
    {
        {gParamType_JsonObject, "json"},
    },
    {},
    {"Alembic"},
});

struct GetAlembicCamera : INode {
    virtual void apply() override {
        auto abctree = safe_dynamic_cast<ABCTree>(get_input("abctree"));
        std::queue<std::pair<Alembic::Abc::v12::M44d, ABCTree*>> q;
        q.emplace(Alembic::Abc::v12::M44d(), abctree);
        Alembic::Abc::v12::M44d mat;
        std::optional<CameraInfo> cam_info;
        while (q.size() > 0) {
            auto [m, t] = q.front();
            q.pop();
            if (t->camera_info) {
                mat = m;
                cam_info = *(t->camera_info);
                break;
            }
            for (auto& ch: t->children) {
                q.emplace(t->xform * m, ch.get());
            }
        }
        if (!cam_info.has_value()) {
            log_error("Not found camera!");
        }

        auto pos = Imath::V4d(0, 0, 0, 1) * mat;
        auto up = Imath::V4d(0, 1, 0, 0) * mat;
        auto right = Imath::V4d(1, 0, 0, 0) * mat;

        float focal_length = cam_info.value().focal_length;

        set_output_vec3f("pos", { (float)pos.x, (float)pos.y, (float)pos.z });

        auto _up = zeno::normalize(zeno::vec3f((float)up.x, (float)up.y, (float)up.z));
        auto _right = zeno::normalize(zeno::vec3f((float)right.x, (float)right.y, (float)right.z));
        auto view = zeno::cross(_up, _right);

        set_output_vec3f("up", toAbiVec3f(_up));//  set_output2("up", _up);
        set_output_vec3f("right", toAbiVec3f(_right));
        set_output_vec3f("view", toAbiVec3f(view));

        set_output_float("focal_length", focal_length);
        set_output_float("near", (float)cam_info.value()._near);
        set_output_float("far", (float)cam_info.value()._far);
        set_output_float("horizontalAperture", (float)cam_info->horizontalAperture);
        set_output_float("verticalAperture", (float)cam_info->verticalAperture);

        float m_nx = get_input2_float("nx");
        float m_ny = get_input2_float("ny");
        float m_ha = (float)cam_info->horizontalAperture;
        float m_va = (float)cam_info->verticalAperture;
        float c_aspect = m_ha/m_va;
        float u_aspect = m_nx/m_ny;
        float fov_y = glm::degrees(2.0f * std::atan(m_va/(u_aspect/c_aspect) / (2.0f * focal_length)));
        set_output_float("fov_y", fov_y);
    }
};

ZENDEFNODE(GetAlembicCamera, {
    {
        {"ABCTree", "abctree"},
        {gParamType_Int, "nx", "1920"},
        {gParamType_Int, "ny", "1080"},
    },
    {
        {gParamType_Vec3f, "pos"},
        {gParamType_Vec3f, "up"},
        {gParamType_Vec3f, "view"},
        {gParamType_Vec3f, "right"},
        {gParamType_Float,"fov_y"},
        {gParamType_Float,"focal_length"},
        {gParamType_Float,"horizontalAperture"},
        {gParamType_Float,"verticalAperture"},
        {gParamType_Float,"near"},
        {gParamType_Float,"far"},
    },
    {},
    {"alembic"},
});

struct ImportAlembicPrim : INode {
    Alembic::Abc::v12::IArchive archive;
    std::string usedPath;
    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_input2_int("frameid");
        } else {
            frameid = GetFrameId();
        }
        auto abctree = std::make_unique<ABCTree>();
        {
            auto path = zsString2Std(get_input2_string("path"));
            bool read_done = archive.valid() && (path == usedPath);
            if (!read_done) {
                archive = readABC(path);
                usedPath = path;
            }
            double start, _end;
            GetArchiveStartAndEndTime(archive, start, _end);
            TimeAndSamplesMap timeMap;
            Alembic::Util::uint32_t numSamplings = archive.getNumTimeSamplings();
            for (Alembic::Util::uint32_t s = 0; s < numSamplings; ++s)             {
                timeMap.add(archive.getTimeSampling(s),
                            archive.getMaxNumSamplesForTimeSamplingIndex(s));
            }
            auto obj = archive.getTop();
            bool read_face_set = get_input2_bool("read_face_set");
            bool outOfRangeAsEmpty = get_input2_bool("outOfRangeAsEmpty");
            traverseABC(obj, *abctree, frameid, read_done, read_face_set, "", timeMap, ObjectVisibility::kVisibilityDeferred, false, outOfRangeAsEmpty, 0);
        }
        bool use_xform = get_input2_bool("use_xform");
        auto index = get_input2_int("index");
        std::unique_ptr<PrimitiveObject> outprim;
        if (index == -1) {
            auto prims = std::make_unique<zeno::ListObject>();
            if (use_xform) {
                prims = get_xformed_prims(abctree.get());
            } else {
                abctree->visitPrims([&] (auto const &p) {
                    prims->push_back(p->clone());
                });
            }

            Vector<zeno::PrimitiveObject*> primlst;
            for (auto spobj : prims->get()) {
                primlst.push_back(dynamic_cast<PrimitiveObject*>(spobj));
            }
            outprim = zeno::PrimMerge(primlst);
        }
        else {
            if (use_xform) {
                outprim = get_xformed_prim(abctree.get(), index);
            } else {
                outprim = get_alembic_prim(abctree.get(), index);
            }
        }
        primFlipFaces(outprim.get(), true);
        if (get_input2_bool("triangulate")) {
            zeno::primTriangulate(outprim.get());
        }
        outprim->userData()->set_int("_abc_prim_count", count_alembic_prims(abctree.get()));
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(ImportAlembicPrim, {
    {
        {gParamType_String, "path", "", zeno::Socket_Primitve, zeno::ReadPathEdit},
        {gParamType_Int, "frameid"},
        {gParamType_Int, "index", "-1"},
        {gParamType_Bool, "use_xform", "0"},
        {gParamType_Bool, "triangulate", "0"},
        {gParamType_Bool, "read_face_set", "0"},
        {gParamType_Bool, "outOfRangeAsEmpty", "0"},
    },
    {
        {gParamType_Primitive, "prim"},
    },
    {},
    {"alembic"},
});

} // namespace zeno

