#include <cctype>
#include <zeno/zeno.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <vector>
#include <cstring>
#include <iostream>
namespace zeno{
struct StringToMesh : zeno::INode {
    virtual void apply() override {
        auto spacing = get_input("spacing")->as<zeno::NumericObject>()->get<float>();
        auto list = std::make_shared<zeno::ListObject>();
        auto list2 = std::make_shared<zeno::ListObject>();
        auto AZ = get_input("AZmesh")->as<zeno::ListObject>();
        auto zello = get_input("string")->as<zeno::StringObject>()->get();
        std::vector<char> chars(zello.c_str(), zello.c_str() + zello.size() + 1u);
        for(int i=0;i<chars.size();i++)
            std::cout<<chars[i]<<std::endl;
        int count = 0;
        for (auto c:chars) {
            if((c>='a'&&c<='z') ||(c>='A'&&c<='Z')){
                char cu = toupper(c);
                int idx = cu - 'A';
                std::cout<<cu<<idx<<std::endl;
                auto vec = zeno::IObject::make<zeno::NumericObject>();
                vec->set<zeno::vec3f>(zeno::vec3f((float)count * spacing, 0.0f,0.0f));
                //auto p = zeno::IObject::make<PrimitiveObject>();
                auto const &obj = smart_any_cast<std::shared_ptr<IObject>>(AZ->arr[idx]);
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
