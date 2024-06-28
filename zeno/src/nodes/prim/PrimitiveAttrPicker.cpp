#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <iostream>

#include <sstream>

namespace zeno {

struct PrimitiveAttrPicker : zeno::INode {
    virtual void apply() override {
        // parse selected elements string
        auto selected = get_param<std::string>("selected");
        std::vector<zany> selected_indices_numeric;
        auto prim = get_input<PrimitiveObject>("prim");
        auto size_of_prim = prim->size();
        if (!selected.empty()) {
            std::vector<int> selected_indices;
            std::stringstream ss;
            auto get_split = [&ss, &selected_indices, &selected_indices_numeric]() {
                int i;
                ss >> i;
                selected_indices.push_back(i);
                auto idx = std::make_shared<NumericObject>(i);
                selected_indices_numeric.push_back(idx);
                ss.clear();
            };
            for (auto c : selected) {
                if (c == ',')
                    get_split();
                else
                    ss << c;
            }
            get_split();

            // set new attr
            auto new_attr = get_input<StringObject>("newAttr");
            if (!new_attr->get().empty()) {
                auto new_value = get_input2<float>("attrVal");
                auto &attr = prim->add_attr<float>(new_attr->get());
                for (const auto& idx : selected_indices) {
                    if(idx >= size_of_prim) {
                        std::cout << "selected idx overflow\t" << idx << "\t" << size_of_prim << std::endl;
                        throw std::runtime_error("selected idx overflow");
                    }
                    attr[idx] = new_value;
                }
            }
        }
        auto list = std::make_shared<ListObject>(selected_indices_numeric);

        set_output("list", std::move(list));
        set_output("outPrim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveAttrPicker, {
    // inputs
    {
    {"PrimitiveObject", "prim", "", zeno::Socket_ReadOnly},
    {"enum point line triangle", "mode", "point"},
    {"string", "newAttr", ""},
    {"float", "attrVal", ""},
    },
    // outputs
    {
    {"PrimitiveObject", "outPrim"},
    {"list"}
    },
    // params
    {{"string", "selected", ""}},
    // category
    {"primitive"}
});
}