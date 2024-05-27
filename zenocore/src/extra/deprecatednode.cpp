#include <zeno/zeno.h>
#include <zeno/core/CoreParam.h>
#include <zeno/utils/log.h>


namespace zeno {
    namespace {

        struct DeprecatedNode : zeno::INode {
            virtual void apply() override {

            }

            std::vector<std::shared_ptr<CoreParam>> get_input_params() const override {
                std::vector<std::shared_ptr<CoreParam>> params;
                for (auto param : m_input_names) {
                    auto it = m_inputs.find(param);
                    if (it == m_inputs.end()) {
                        zeno::log_warn("unknown param {}", param);
                        continue;
                    }
                    params.push_back(it->second);
                }
                return params;
            }

            std::vector<std::shared_ptr<CoreParam>> get_output_params() const override {
                std::vector<std::shared_ptr<CoreParam>> params;
                for (auto param : m_output_names) {
                    auto it = m_outputs.find(param);
                    if (it == m_outputs.end()) {
                        zeno::log_warn("unknown param {}", param);
                        continue;
                    }
                    params.push_back(it->second);
                }
                return params;
            }

            void initParams(const NodeData& dat) override
            {
                for (const ParamPrimitive& param : dat.inputs)
                {
                    std::shared_ptr<CoreParam> sparam = std::make_shared<CoreParam>();
                    sparam->name = param.name;
                    sparam->isLegacy = true;
                    sparam->defl = param.defl;
                    sparam->type = param.type;
                    sparam->m_wpNode = shared_from_this();
                    add_input_param(sparam);
                    m_input_names.push_back(param.name);
                }
                for (const ParamPrimitive& param : dat.outputs)
                {
                    std::shared_ptr<CoreParam> sparam = std::make_shared<CoreParam>();
                    sparam->name = param.name;
                    sparam->isLegacy = true;
                    sparam->defl = param.defl;
                    sparam->type = param.type;
                    sparam->m_wpNode = shared_from_this();
                    add_output_param(sparam);
                    m_output_names.push_back(param.name);
                }
            }

            std::vector<std::string> m_input_names;
            std::vector<std::string> m_output_names;
        };

        ZENDEFNODE(DeprecatedNode, {
            {},
            {},
            {},
            {"subgraph"}
        });

    }
}