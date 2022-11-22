#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>

#include <sstream>

namespace zeno {

struct PrimitiveAttrPicker : zeno::INode {
    virtual void apply() override {
        // parse selected elements string
        auto selected = get_param<std::string>("selected");
        std::vector<zany> selected_indices_numeric;
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
            if (c == ',') get_split();
            else ss << c;
        }
        get_split();
        auto list = std::make_shared<ListObject>(selected_indices_numeric);

        // set new attr
        auto new_attr = get_input<StringObject>("newAttr");
        if (!new_attr->get().empty()) {
            auto prim = get_input<PrimitiveObject>("prim");
            auto new_value = get_input2<int>("attrVal");
            auto &attr = prim->add_attr<int>(new_attr->get());
            for (const auto& idx : selected_indices)
                attr[idx] = new_value;
        }

        set_output("list", list);
    }
};

ZENDEFNODE(PrimitiveAttrPicker, {
    // inputs
    {
    {"PrimitiveObject", "prim"},
    {"enum point line triangle", "mode", "point"},
    {"string", "newAttr", ""},
    {"int", "attrVal", ""},
    },
    // outputs
    {{"list"}},
    // params
    {{"string", "selected", ""}},
    // category
    {"primitive"}
});
}