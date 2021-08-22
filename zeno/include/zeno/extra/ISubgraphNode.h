#pragma once

#include <zeno/zeno.h>

namespace zeno {

struct ISubgraphNode : zeno::INode {
    virtual std::string subgraph_name() = 0;

    virtual void apply() override {
        auto name = subgraph_name();

        auto subg = safe_at(graph->scene->graphs, name, "subgraph");
        assert(subg->scene == graph->scene);

#ifdef ZENO_VISUALIZATION
        // VIEW subnodes only if subgraph is VIEW'ed
        subg->isViewed = has_option("VIEW");
#endif

        for (auto const &[key, obj]: inputs) {
            subg->setGraphInput2(key, obj);
        }
        subg->applyGraph();

        for (auto &[key, obj]: subg->subOutputs) {
#ifdef ZENO_VISUALIZATION
            if (subg->isViewed && !subg->hasAnyView) {
                auto path = zeno::Visualization::exportPath();
                if (auto p = zeno::silent_any_cast<
                        std::shared_ptr<zeno::IObject>>(obj); p.has_value()) {
                    p.value()->dumpfile(path);
                }
                subg->hasAnyView = true;
            }
#endif
            set_output2(key, std::move(obj));
        }

        subg->subInputs.clear();
        subg->subOutputs.clear();
    }
};

}
