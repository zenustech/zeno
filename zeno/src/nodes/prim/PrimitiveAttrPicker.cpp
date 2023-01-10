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

        set_output("list", std::move(list));
    }
};

ZENDEFNODE(PrimitiveAttrPicker, {
    // inputs
    {
    {"PrimitiveObject", "prim"},
    {"enum point line triangle", "mode", "point"},
    },
    // outputs
    {
    {"list"},
    },
    // params
    {{"string", "selected", ""}},
    // category
    {"primitive"}
});
}