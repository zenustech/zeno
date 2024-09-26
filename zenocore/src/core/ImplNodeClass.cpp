#include <zeno/zeno.h>
#include <zeno/core/INodeClass.h>
#include <regex>
#include <zeno/utils/helper.h>

namespace zeno {

    static void initCoreParams(std::shared_ptr<INode> spNode, CustomUI customui)
    {
        //init all params, and set defl value
        for (const ParamObject& param : customui.inputObjs)
        {
            spNode->add_input_obj_param(param);
        }
        for (const ParamPrimitive& param : customUiToParams(customui.inputPrims))
        {
            spNode->add_input_prim_param(param);
        }
        for (const ParamPrimitive& param : customui.outputPrims)
        {
            spNode->add_output_prim_param(param);
        }
        for (const ParamObject& param : customui.outputObjs)
        {
            spNode->add_output_obj_param(param);
        }
        //根据customui上的约束信息调整所有控件的可见可用情况
        spNode->checkParamsConstrain();
    }


    ImplNodeClass::ImplNodeClass(std::shared_ptr<INode>(*ctor)(), CustomUI const& customui, std::string const& name)
        : INodeClass(customui, name), ctor(ctor) {}

    std::shared_ptr<INode> ImplNodeClass::new_instance(std::shared_ptr<Graph> pGraph, std::string const& name) {
        std::shared_ptr<INode> spNode = ctor();
        spNode->initUuid(pGraph, classname);
        spNode->set_name(name);
        initCoreParams(spNode, m_customui);
        return spNode;
    }

}
