#include <cctype>
#include <zeno/zeno.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/prim_ops.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <spdlog/spdlog.h>

namespace zeno{
struct StringToMesh : zeno::INode {
    virtual void apply() override {
        auto alphaset = std::make_shared<zeno::ListObject>();
        for (auto i = 33; i <= 126; i++) {
            auto path = fmt::format("assets/ascii/{:03}.obj", i);
            auto prim = std::make_shared<zeno::PrimitiveObject>();
            auto &pos = prim->verts;
            auto &uv = prim->verts.add_attr<zeno::vec3f>("uv");
            auto &norm = prim->verts.add_attr<zeno::vec3f>("nrm");
            auto &tris = prim->tris;
            std::vector<zeno::vec3i> triuv;
            read_obj_file(pos, uv, norm, tris, triuv, path.c_str());
            if (!triuv.empty())
            {
                auto &uv0 = prim->tris.add_attr<vec3f>("uv0");
                auto &uv1 = prim->tris.add_attr<vec3f>("uv1");
                auto &uv2 = prim->tris.add_attr<vec3f>("uv2");
                const auto triuvSize = triuv.size();
                uv0.reserve(triuvSize);
                uv1.reserve(triuvSize);
                uv2.reserve(triuvSize);
                for (size_t i = 0; i < triuvSize; ++i)
                {
                    const auto &index = triuv[i];
                    uv0[i] = index[0] != -1 ? uv[index[0]] : vec3f{0.0f, 0.0f, 0.0f};
                    uv1[i] = index[1] != -1 ? uv[index[1]] : vec3f{0.0f, 0.0f, 0.0f};
                    uv2[i] = index[2] != -1 ? uv[index[2]] : vec3f{0.0f, 0.0f, 0.0f};
                }
            }
            prim->resize(pos.size());
            alphaset->arr.push_back(prim);
        }
        auto spacing = get_input("spacing")->as<zeno::NumericObject>()->get<float>();
        auto list = std::make_shared<zeno::ListObject>();
        auto list2 = std::make_shared<zeno::ListObject>();
        auto zello = get_input("string")->as<zeno::StringObject>()->get();
        std::vector<char> chars(zello.c_str(), zello.c_str() + zello.size() + 1u);
        for(int i=0;i<chars.size();i++)
            std::cout<<chars[i]<<std::endl;
        int count = 0;
        for (auto c:chars) {
            if (33 <= c && c <= 126) {
                int idx = c - 33;
                auto vec = zeno::IObject::make<zeno::NumericObject>();
                vec->set<zeno::vec3f>(zeno::vec3f((float)count * spacing, 0.0f,0.0f));
                //auto p = zeno::IObject::make<PrimitiveObject>();
                auto const &obj = smart_any_cast<std::shared_ptr<IObject>>(alphaset->arr[idx]);
                auto p = obj->clone();
                //p->copy(dynamic_cast<PrimitiveObject *>(obj.get()));
                list->arr.push_back(std::move(p));
                list2->arr.push_back(std::move(vec));
            }
            count++;
        }
        set_output("StringMeshList", std::move(list));
        set_output("SpacingList", std::move(list2));
    }
};

ZENDEFNODE(StringToMesh, {
    {"string", "AZmesh" ,"spacing"},
    {"StringMeshList", "SpacingList"},
    {},
    {"math"},
});
}
