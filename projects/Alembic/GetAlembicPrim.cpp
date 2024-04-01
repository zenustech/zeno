#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <glm/glm.hpp>
#include <zeno/types/ListObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/extra/GlobalState.h>
#include "ABCCommon.h"
#include "ABCTree.h"
#include "zeno/utils/string.h"
#include <queue>
#include <utility>

namespace zeno {

int count_alembic_prims(std::shared_ptr<zeno::ABCTree> abctree) {
    int count = 0;
    abctree->visitPrims([&] (auto const &p) {
        count++;
    });
    return count;
}

struct CountAlembicPrims : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        std::shared_ptr<PrimitiveObject> prim;
        int count = count_alembic_prims(abctree);
        set_output("count", std::make_shared<NumericObject>(count));
    }
};

ZENDEFNODE(CountAlembicPrims, {
    {{"ABCTree", "abctree"}},
    {{"int", "count"}},
    {},
    {"alembic"},
});

std::shared_ptr<PrimitiveObject> get_alembic_prim(std::shared_ptr<zeno::ABCTree> abctree, int index) {
    std::shared_ptr<PrimitiveObject> prim;
    abctree->visitPrims([&] (auto const &p) {
        if (index == 0) {
            prim = p;
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

int get_alembic_prim_index(std::shared_ptr<zeno::ABCTree> abctree, std::string name) {
    int index = 0;
    abctree->visitPrims([&] (auto const &p) {
        auto &ud = p->userData();
        auto _abc_path = ud.template get2<std::string>("abcpath_0", "");
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
    std::shared_ptr<ABCTree> root,
    int parent_index,
    std::vector<std::shared_ptr<ABCTree>>& linear_abctrees,
    std::vector<int>& linear_abctree_parent
) {
    int self_index = linear_abctrees.size();
    linear_abctrees.push_back(root);
    linear_abctree_parent.push_back(parent_index);
    for (auto const &ch: root->children) {
        dfs_abctree(ch, self_index, linear_abctrees, linear_abctree_parent);
    }
}

std::shared_ptr<PrimitiveObject> get_xformed_prim(std::shared_ptr<zeno::ABCTree> abctree, int index) {
    std::vector<std::shared_ptr<ABCTree>> linear_abctrees;
    std::vector<int> linear_abctree_parent;
    dfs_abctree(abctree, -1, linear_abctrees, linear_abctree_parent);
    std::shared_ptr<PrimitiveObject> prim;
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
                prim = std::static_pointer_cast<PrimitiveObject>(abc_node->prim->clone());
                auto& mat = transforms.back();
                for (auto& p: prim->verts) {
                    auto pos = Imath::V4d(p[0], p[1], p[2], 1) * mat;
                    p = zeno::vec3f((float)pos.x, (float)pos.y, (float)pos.z);
                }
            }
            index--;
        }
    }
    return prim;
}

std::shared_ptr<zeno::ListObject>
get_xformed_prims(
    std::shared_ptr<zeno::ABCTree> abctree
) {
    auto prims = std::make_shared<zeno::ListObject>();
    std::vector<std::shared_ptr<ABCTree>> linear_abctrees;
    std::vector<int> linear_abctree_parent;
    dfs_abctree(abctree, -1, linear_abctrees, linear_abctree_parent);
    std::shared_ptr<PrimitiveObject> prim;
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
            prim = std::static_pointer_cast<PrimitiveObject>(abc_node->prim->clone());
            auto& mat = transforms.back();
            for (auto& p: prim->verts) {
                auto pos = Imath::V4d(p[0], p[1], p[2], 1) * mat;
                p = zeno::vec3f((float)pos.x, (float)pos.y, (float)pos.z);
            }
            prims->arr.push_back(prim);
        }
    }
    return prims;
}
struct GetAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        int index = get_input<NumericObject>("index")->get<int>();
        int use_xform = get_input<NumericObject>("use_xform")->get<int>();
        std::shared_ptr<PrimitiveObject> prim;
        if (get_input2<bool>("use_name")) {
            index = get_alembic_prim_index(abctree, get_input2<std::string>("name"));
        }
        if (use_xform) {
            prim = get_xformed_prim(abctree, index);
        } else {
            prim = get_alembic_prim(abctree, index);
        }
        if (get_input2<bool>("flipFrontBack")) {
            primFlipFaces(prim.get());
        }
        if (get_input2<bool>("triangulate")) {
            zeno::primTriangulate(prim.get());
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(GetAlembicPrim, {
    {
        {"bool", "flipFrontBack", "1"},
        {"ABCTree", "abctree"},
        {"int", "index", "0"},
        {"bool", "use_xform", "0"},
        {"bool", "triangulate", "0"},
        {"bool", "use_name", "0"},
        {"string", "name", ""},
    },
    {{"PrimitiveObject", "prim"}},
    {},
    {"alembic"},
});

struct AllAlembicPrim : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        auto prims = std::make_shared<zeno::ListObject>();
        int use_xform = get_input<NumericObject>("use_xform")->get<int>();
        if (use_xform) {
            prims = get_xformed_prims(abctree);
        } else {
            abctree->visitPrims([&] (auto const &p) {
                auto np = std::static_pointer_cast<PrimitiveObject>(p->clone());
                prims->arr.push_back(np);
            });
        }
        auto outprim = zeno::primMerge(prims->getRaw<PrimitiveObject>());
        if (get_input2<bool>("flipFrontBack")) {
            primFlipFaces(outprim.get());
        }
        if (get_input2<int>("triangulate") == 1) {
            zeno::primTriangulate(outprim.get());
        }
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(AllAlembicPrim, {
    {
        {"bool", "flipFrontBack", "1"},
        {"ABCTree", "abctree"},
        {"bool", "use_xform", "0"},
        {"bool", "triangulate", "0"},
    },
    {{"PrimitiveObject", "prim"}},
    {},
    {"alembic"},
});

struct AlembicPrimList : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        auto prims = std::make_shared<zeno::ListObject>();
        int use_xform = get_input2<int>("use_xform");
        if (use_xform) {
            prims = get_xformed_prims(abctree);
        } else {
            abctree->visitPrims([&] (auto const &p) {
                auto np = std::static_pointer_cast<PrimitiveObject>(p->clone());
                prims->arr.push_back(np);
            });
        }
        auto new_prims = std::make_shared<zeno::ListObject>();
        if (get_input2<bool>("splitByFaceset")) {
            for (auto &prim: prims->arr) {
                auto list = abc_split_by_name(std::dynamic_pointer_cast<PrimitiveObject>(prim), false);
                new_prims->arr.insert(new_prims->arr.end(), list->arr.begin(), list->arr.end());
            }
        }
        else {
            new_prims = std::dynamic_pointer_cast<zeno::ListObject>(prims->clone());
        }
        auto pathInclude = zeno::split_str(get_input2<std::string>("pathInclude"), {' ', '\n'});
        auto pathExclude = zeno::split_str(get_input2<std::string>("pathExclude"), {' ', '\n'});
        auto facesetInclude = zeno::split_str(get_input2<std::string>("facesetInclude"), {' ', '\n'});
        auto facesetExclude = zeno::split_str(get_input2<std::string>("facesetExclude"), {' ', '\n'});
        for (auto it = new_prims->arr.begin(); it != new_prims->arr.end();) {
            auto np = std::dynamic_pointer_cast<PrimitiveObject>(*it);
            auto abc_path = np->userData().template get2<std::string>("abcpath_0");
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
            if (contain && np->userData().template has<std::string>("faceset_0")) {
                auto faceset = np->userData().template get2<std::string>("faceset_0");
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
                it = new_prims->arr.erase(it);
            }
        }
        for (auto &prim: new_prims->arr) {
            auto _prim = std::dynamic_pointer_cast<PrimitiveObject>(prim);
            if (get_input2<bool>("flipFrontBack")) {
                primFlipFaces(_prim.get());
            }
            if (get_input2<bool>("splitByFaceset") && get_input2<bool>("killDeadVerts")) {
                primKillDeadVerts(_prim.get());
            }
            if (get_input2<bool>("triangulate")) {
                zeno::primTriangulate(_prim.get());
            }
        }
        set_output("prims", std::move(new_prims));
    }
};

ZENDEFNODE(AlembicPrimList, {
    {
        {"bool", "flipFrontBack", "1"},
        {"ABCTree", "abctree"},
        {"bool", "use_xform", "0"},
        {"bool", "triangulate", "0"},
        {"bool", "splitByFaceset", "0"},
        {"bool", "killDeadVerts", "1"},
        {"string", "pathInclude", ""},
        {"string", "pathExclude", ""},
        {"string", "facesetInclude", ""},
        {"string", "facesetExclude", ""},
    },
    {"prims"},
    {},
    {"alembic"},
});

struct GetAlembicCamera : INode {
    virtual void apply() override {
        auto abctree = get_input<ABCTree>("abctree");
        std::queue<std::pair<Alembic::Abc::v12::M44d, std::shared_ptr<ABCTree>>> q;
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
            for (auto ch: t->children) {
                q.emplace(t->xform * m, ch);
            }
        }
        if (!cam_info.has_value()) {
            log_error("Not found camera!");
        }

        auto pos = Imath::V4d(0, 0, 0, 1) * mat;
        auto up = Imath::V4d(0, 1, 0, 0) * mat;
        auto right = Imath::V4d(1, 0, 0, 0) * mat;

        float focal_length = cam_info.value().focal_length;

        set_output("pos", std::make_shared<NumericObject>(zeno::vec3f((float)pos.x, (float)pos.y, (float)pos.z)));

        auto _up = zeno::normalize(zeno::vec3f((float)up.x, (float)up.y, (float)up.z));
        auto _right = zeno::normalize(zeno::vec3f((float)right.x, (float)right.y, (float)right.z));
        auto view = zeno::cross(_up, _right);
        set_output2("up", _up);
        set_output2("right", _right);
        set_output2("view", view);

        set_output("focal_length", std::make_shared<NumericObject>(focal_length));
        set_output("near", std::make_shared<NumericObject>((float)cam_info.value()._near));
        set_output("far", std::make_shared<NumericObject>((float)cam_info.value()._far));
        set_output("horizontalAperture", std::make_shared<NumericObject>((float)cam_info->horizontalAperture));
        set_output("verticalAperture", std::make_shared<NumericObject>((float)cam_info->verticalAperture));
        auto m_nx = get_input2<float>("nx");
        auto m_ny = get_input2<float>("ny");
        float m_ha = (float)cam_info->horizontalAperture;
        float m_va = (float)cam_info->verticalAperture;
        float c_aspect = m_ha/m_va;
        float u_aspect = m_nx/m_ny;
        float fov_y = glm::degrees(2.0f * std::atan(m_va/(u_aspect/c_aspect) / (2.0f * focal_length)));
        set_output("fov_y", std::make_shared<NumericObject>(fov_y));
    }
};

ZENDEFNODE(GetAlembicCamera, {
    {
        {"ABCTree", "abctree"},
        {"int", "nx", "1920"},
        {"int", "ny", "1080"},
    },
    {
        "pos",
        "up",
        "view",
        "right",
        "fov_y",
        "focal_length",
        "horizontalAperture",
        "verticalAperture",
        "near",
        "far",
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
            frameid = get_input2<int>("frameid");
        } else {
            frameid = getGlobalState()->frameid;
        }
        auto abctree = std::make_shared<ABCTree>();
        {
            auto path = get_input2<std::string>("path");
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
            bool read_face_set = get_input2<bool>("read_face_set");
            bool outOfRangeAsEmpty = get_input2<bool>("outOfRangeAsEmpty");
            traverseABC(obj, *abctree, frameid, read_done, read_face_set, "", timeMap, ObjectVisibility::kVisibilityDeferred, outOfRangeAsEmpty);
        }
        bool use_xform = get_input2<bool>("use_xform");
        auto index = get_input2<int>("index");
        std::shared_ptr<PrimitiveObject> outprim;
        if (index == -1) {
            auto prims = std::make_shared<zeno::ListObject>();
            if (use_xform) {
                prims = get_xformed_prims(abctree);
            } else {
                abctree->visitPrims([&] (auto const &p) {
                    auto np = std::static_pointer_cast<PrimitiveObject>(p->clone());
                    prims->arr.push_back(np);
                });
            }
            outprim = zeno::primMerge(prims->getRaw<PrimitiveObject>());
        }
        else {
            if (use_xform) {
                outprim = get_xformed_prim(abctree, index);
            } else {
                outprim = get_alembic_prim(abctree, index);
            }
        }
        primFlipFaces(outprim.get());
        if (get_input2<bool>("triangulate")) {
            zeno::primTriangulate(outprim.get());
        }
        outprim->userData().set2("_abc_prim_count", count_alembic_prims(abctree));
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(ImportAlembicPrim, {
    {
        {"readpath", "path"},
        {"frameid"},
        {"int", "index", "-1"},
        {"bool", "use_xform", "0"},
        {"bool", "triangulate", "0"},
        {"bool", "read_face_set", "0"},
        {"bool", "outOfRangeAsEmpty", "0"},
    },
    {
        "prim",
    },
    {},
    {"alembic"},
});

} // namespace zeno
