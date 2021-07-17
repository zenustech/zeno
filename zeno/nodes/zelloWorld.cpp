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
        float total_spacing = 0;
        for (auto c:chars) {
            if((c>='a'&&c<='z') ||(c>='A'&&c<='Z')){
                char cu = toupper(c);
                total_spacing += cu=='I'?0.5*spacing:spacing;
                int idx = cu - 'A';
                std::cout<<cu<<idx<<std::endl;
                auto vec = zeno::IObject::make<zeno::NumericObject>();
                vec->set<zeno::vec3f>(zeno::vec3f(total_spacing, 0.0f,0.0f));
                auto const &obj = AZ->arr[idx];
                auto p = obj->clone();
                list->arr.push_back(std::move(p));
                list2->arr.push_back(std::move(vec));
                total_spacing += cu=='I'?0.5*spacing:spacing;
            }
            total_spacing += spacing;
        }
        set_output("StringMeshList", std::move(list));
        set_output("SpacingList", std::move(list2));
    }
};

ZENDEFNODE(StringToMesh, {
    {"string", "AZmesh" ,"spacing"},
    {"StringMeshList", "SpacingList"},
    {},
    {"zelloWorld"},
});
}