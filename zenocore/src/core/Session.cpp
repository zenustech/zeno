#include <zeno/core/Session.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Assets.h>
#include <zeno/core/ObjectManager.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GlobalError.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/types/UserData.h>
#include <zeno/core/Graph.h>
#include <zeno/core/INode.h>
#include <zeno/core/CoreParam.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/string.h>
#include <zeno/utils/helper.h>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/GraphException.h>
#include <zeno/core/ReferManager.h>
#include <zeno/core/GlobalVariable.h>

#include <reflect/core.hpp>
#include <reflect/type.hpp>
#include <reflect/metadata.hpp>
#include <reflect/registry.hpp>
#include <reflect/container/object_proxy>
#include <reflect/container/any>
#include <reflect/container/arraylist>
#include <zeno/core/reflectdef.h>
#include "zeno_types/reflect/reflection.generated.hpp"


using namespace zeno::reflect;
using namespace zeno::types;

namespace zeno {

namespace {

struct ImplNodeClass : INodeClass {
    std::shared_ptr<INode>(*ctor)();

    ImplNodeClass(std::shared_ptr<INode>(*ctor)(), CustomUI const& customui, std::string const& name)
        : INodeClass(customui, name), ctor(ctor) {}

    virtual std::shared_ptr<INode> new_instance(std::shared_ptr<Graph> pGraph, std::string const &name) override {
        std::shared_ptr<INode> spNode = ctor();
        spNode->initUuid(pGraph, classname);
        spNode->set_name(name);

        //init all params, and set defl value
        for (const ParamObject& param : m_customui.inputObjs)
        {
            spNode->add_input_obj_param(param);
        }
        for (const ParamPrimitive& param : customUiToParams(m_customui.inputPrims))
        {
            spNode->add_input_prim_param(param);
        }
        for (const ParamPrimitive& param : m_customui.outputPrims)
        {
            spNode->add_output_prim_param(param);
        }
        for (const ParamObject& param : m_customui.outputObjs)
        {
            spNode->add_output_obj_param(param);
        }
        return spNode;
    }
};

struct ReflectNodeClass : INodeClass {
    std::function<std::shared_ptr<INode>()> ctor;
    zeno::reflect::TypeBase* typebase;

    ReflectNodeClass(std::function<std::shared_ptr<INode>()> ctor, std::string const& nodecls, zeno::reflect::TypeBase* pTypeBase)
        : INodeClass(CustomUI(), nodecls)
        , ctor(ctor)
        , typebase(pTypeBase)
    {
    }

    //整理customui层级
    void adjustCustomUiStructure(std::shared_ptr<INode> spNode,
        std::map<std::string, ParamPrimitive>& inputPrims, std::map<std::string, ParamObject>& inputObjs,
        std::map<std::string, ParamPrimitive>& outputPrims, std::map<std::string, ParamObject>& outputObjs) {
        for (IMemberField* field : typebase->get_member_fields()) {
            zeno::reflect::TypeHandle fieldType = field->get_field_type();
            if (fieldType == zeno::reflect::get_type<ReflectCustomUI>()) {
                zeno::reflect::Any defl = field->get_field_value(spNode.get());
                if (defl.has_value()) {
                    ReflectCustomUI reflectCustomUi = zeno::reflect::any_cast<ReflectCustomUI>(defl);
                    ParamTab tab;
                    tab.name = reflectCustomUi.inputPrims.name;
                    for (auto& reflectgroup : reflectCustomUi.inputPrims.groups) {
                        ParamGroup group;
                        group.name = reflectgroup.name;
                        for (auto& param : reflectgroup.params) {
                            if (inputPrims.find(param.mapTo) != inputPrims.end()) { //按照ReflectCustomUI的信息更新ParamPrimitive并放入对应的group
                                inputPrims[param.mapTo].name = param.dispName;
                                inputPrims[param.mapTo].defl = param.defl;
                                group.params.push_back(std::move(inputPrims[param.mapTo]));
                                inputPrims.erase(param.mapTo);
                            }
                        }
                        tab.groups.push_back(std::move(group));
                    }
                    m_customui.inputPrims.tabs.push_back(std::move(tab));
                    for (auto& reflectInputObj : reflectCustomUi.inputObjs.objs) {
                        if (inputObjs.find(reflectInputObj.mapTo) != inputObjs.end()) {
                            inputObjs[reflectInputObj.mapTo].name = reflectInputObj.dispName;
                            inputObjs[reflectInputObj.mapTo].socketType = reflectInputObj.type;
                            m_customui.inputObjs.push_back(std::move(inputObjs[reflectInputObj.mapTo]));
                            inputObjs.erase(reflectInputObj.mapTo);
                        }
                    }
                    for (auto& reflectOutputObj : reflectCustomUi.outputObjs.objs) {
                        if (outputObjs.find(reflectOutputObj.mapTo) != outputObjs.end()) {
                            outputObjs[reflectOutputObj.mapTo].name = reflectOutputObj.dispName;
                            outputObjs[reflectOutputObj.mapTo].socketType = reflectOutputObj.type;
                            m_customui.outputObjs.push_back(std::move(outputObjs[reflectOutputObj.mapTo]));
                            outputObjs.erase(reflectOutputObj.mapTo);
                        }else if (reflectOutputObj.mapTo.empty()) {
                            if (outputObjs.find("result") != outputObjs.end()) {    //空串mapping到返回值,返回值名为"result"
                                outputObjs["result"].name = reflectOutputObj.dispName;
                                outputObjs["result"].socketType = reflectOutputObj.type;
                                m_customui.outputObjs.push_back(std::move(outputObjs["result"]));
                                outputObjs.erase("result");
                            } else {    //apply返回值为void但ReflectCustomUI定义了空串，加入一个默认IObject输出
                                ParamObject outputObj;
                                outputObj.name = "result";
                                outputObj.bInput = false;
                                outputObj.socketType = Socket_Output;
                                outputObj.type = gParamType_sharedIObject;
                            }
                        }
                    }
                    for (auto& reflectOutputPrim : reflectCustomUi.outputPrims.params) {
                        if (outputPrims.find(reflectOutputPrim.mapTo) != outputPrims.end()) {
                            outputPrims[reflectOutputPrim.mapTo].name = reflectOutputPrim.dispName;
                            outputPrims[reflectOutputPrim.mapTo].defl = reflectOutputPrim.defl;
                            m_customui.outputPrims.push_back(std::move(outputPrims[reflectOutputPrim.mapTo]));
                            outputPrims.erase(reflectOutputPrim.mapTo);
                        }
                    }
                    //若有剩余(是成员变量或apply中有,但ReflectCustomUI没有的参数)，再将剩余加入
                    if (m_customui.inputPrims.tabs.empty()) {
                        zeno::ParamTab tab;
                        tab.name = "Tab1";
                        zeno::ParamGroup group;
                        group.name = "Group1";
                        tab.groups.emplace_back(group);
                        m_customui.inputPrims.tabs.emplace_back(tab);
                    } else if (m_customui.inputPrims.tabs[0].groups.empty()) {
                        zeno::ParamGroup group;
                        group.name = "Group1";
                        m_customui.inputPrims.tabs[0].groups.emplace_back(group);
                    }
                    for (auto& [name, primParam] : inputPrims)
                        m_customui.inputPrims.tabs[0].groups[0].params.push_back(std::move(primParam));
                    for (auto& [name, objParam] : inputObjs)
                        m_customui.inputObjs.push_back(std::move(objParam));
                    for (auto& [name, primParam] : outputPrims)
                        m_customui.outputPrims.push_back(std::move(primParam));
                    for (auto& [name, objParam] : outputObjs)
                        m_customui.outputObjs.push_back(std::move(objParam));
                    return;
                }
            }
        }
        //如果没有定义ReflectCustomUI类型成员变量，使用默认
        if (m_customui.inputPrims.tabs.empty())
        {
            zeno::ParamTab tab;
            tab.name = "Tab1";
            zeno::ParamGroup group;
            group.name = "Group1";
            tab.groups.emplace_back(group);
            m_customui.inputPrims.tabs.emplace_back(tab);
        }
        if (!m_customui.inputPrims.tabs.empty() && !m_customui.inputPrims.tabs[0].groups.empty()) {
            m_customui.inputPrims.tabs[0].groups[0].params.clear();
        }
        for (auto& [name, primParam] : inputPrims)
            m_customui.inputPrims.tabs[0].groups[0].params.push_back(std::move(primParam));
        for (auto& [name, objParam] : inputObjs)
            m_customui.inputObjs.push_back(std::move(objParam));
        for (auto& [name, primParam] : outputPrims)
            m_customui.outputPrims.push_back(std::move(primParam));
        for (auto& [name, objParam] : outputObjs)
            m_customui.outputObjs.push_back(std::move(objParam));
    }
        
    std::shared_ptr<INode> new_instance(std::shared_ptr<Graph> pGraph, std::string const& name) override {

        std::shared_ptr<INode> spNode = ctor();
        spNode->initUuid(pGraph, classname);
        spNode->set_name(name);

        m_customui.inputObjs.clear();
        m_customui.outputPrims.clear();
        m_customui.outputObjs.clear();

        std::set<std::string> reg_inputobjs, reg_inputprims, reg_outputobjs, reg_outputprims;
        std::map<std::string, ParamPrimitive> inputPrims;
        std::map<std::string, ParamPrimitive> outputPrims;
        std::map<std::string, ParamObject> inputObjs;
        std::map<std::string, ParamObject> outputObjs;

        //先遍历所有成员，收集所有参数，目前假定所有成员变量都作为节点的参数存在，后续看情况可以指定
        for (IMemberField* field : typebase->get_member_fields()) {
            // 找到我们要的
            std::string field_name(field->get_name().c_str());
            std::string param_name;
            if (const zeno::reflect::IRawMetadata* metadata = field->get_metadata()) {

                //name:
                if (const zeno::reflect::IMetadataValue* value = metadata->get_value("DisplayName")) {
                    param_name = value->as_string();
                }
                else {
                    param_name = field_name;
                }
                //TODO: 名称合法性判断

                //根据类型判断一下是object还是primitive
                zeno::reflect::TypeHandle fieldType = field->get_field_type();
                ParamType type = fieldType.type_hash();
                const RTTITypeInfo& typeInfo = ReflectionRegistry::get().getRttiMap()->get(type);
                assert(typeInfo.hash_code());
                std::string rttiname(typeInfo.name());
                //后续会有更好的判断方式，原理都是一样：把rtti拿出来
                bool bObject = rttiname.find("std::shared_ptr") != rttiname.npos;

                //role:
                NodeDataGroup role = Role_InputObject;
                if (const zeno::reflect::IMetadataValue* value = metadata->get_value("Role")) {
                    int _role = value->as_int();
                    if (_role < Role_InputObject || _role > Role_OutputPrimitive) {
                        throw makeError<UnimplError>("parsing error when parsing reflected node.");
                    }
                    role = static_cast<NodeDataGroup>(_role);
                }
                else {
                    //没有指定role，一律都是按input处理，是否为obj根据类型做判断
                    role = bObject ? Role_InputObject : Role_InputPrimitive;
                }

                if (role == Role_InputObject)
                {
                    if (reg_inputobjs.find(param_name) != reg_inputobjs.end()) {
                        //因为是定义在PROPERTY上，所以理论上可以重复写
                        throw makeError<UnimplError>("repeated name on input objs");
                    }

                    //观察有无定义socket属性
                    SocketType socketProp = Socket_Owning;
                    if (const zeno::reflect::IMetadataValue* value = metadata->get_value("Socket")) {
                        int _role = value->as_int();
                        if (_role < NoSocket || _role > Socket_WildCard) {
                            throw makeError<UnimplError>("parsing error when parsing reflected node.");
                        }
                        socketProp = (SocketType)_role;
                    }

                    //TODO: wilecard

                    ParamObject inputObj;
                    inputObj.name = param_name;
                    inputObj.type = type;
                    inputObj.socketType = socketProp;

                    inputObjs.insert({ field_name,inputObj });
                    reg_inputobjs.insert(param_name);
                }
                else if (role == Role_OutputObject)
                {
                    if (reg_outputobjs.find(param_name) != reg_outputobjs.end()) {
                        //因为是定义在PROPERTY上，所以理论上可以重复写
                        throw makeError<UnimplError>("repeated name on input objs");
                    }
                    ParamObject outputObj;
                    outputObj.name = param_name;
                    outputObj.type = type;
                    outputObj.socketType = Socket_Output;

                    outputObjs.insert({ field_name,outputObj });
                    reg_outputobjs.insert(param_name);
                }
                else if (role == Role_InputPrimitive)
                {
                    //defl value
                    zeno::reflect::Any defl = field->get_field_value(spNode.get());
                    zeno::reflect::Any controlProps;
                    ParamPrimitive prim;

                    ParamControl ctrl = getDefaultControl(type);
                    //control:
                    if (const zeno::reflect::IMetadataValue* value = metadata->get_value("Control")) {
                        ctrl = (ParamControl)value->as_int();
                        if (ctrl == Slider || ctrl == SpinBox || ctrl == SpinBoxSlider || ctrl == DoubleSpinBox)
                        {
                            if (const zeno::reflect::IMetadataValue* value = metadata->get_value("range")) {
                                if (value->is_list() && value->list_length() > 0)
                                {
                                    if (value->list_get_item(0)->is_int()) {
                                        std::vector<int> vec;
                                        for (int i = 0; i < value->list_length(); i++) {
                                            auto pItem = value->list_get_item(i);
                                            assert(pItem->is_int());
                                            vec.push_back(pItem->as_int());
                                        }
                                        if (vec.size() == 2) {
                                            controlProps = vec;
                                        }
                                    }
                                    else if (value->list_get_item(0)->is_float()) {
                                        std::vector<float> vec;
                                        for (int i = 0; i < value->list_length(); i++) {
                                            auto pItem = value->list_get_item(i);
                                            assert(pItem->is_float());
                                            vec.push_back(pItem->as_float());
                                        }
                                        if (vec.size() == 2) {
                                            controlProps = vec;
                                        }
                                    }
                                }
                            }
                        }
                        else if (ctrl == Combobox)
                        {
                            //comobox items:
                            if (const zeno::reflect::IMetadataValue* value = metadata->get_value("ComboBoxItems")) {
                                assert(value->is_list());
                                std::vector<std::string> items;
                                for (int i = 0; i < value->list_length(); i++) {
                                    items.push_back(value->list_get_item(i)->as_string());
                                }
                                controlProps = items;
                            }
                        }
                    }
                    prim.name = param_name;
                    prim.type = type;
                    prim.bInput = true;
                    prim.bVisible = true;
                    prim.control = ctrl;
                    prim.ctrlProps = controlProps;
                    prim.defl = defl;
                    prim.socketType = Socket_Primitve;
                    prim.tooltip;
                    prim.wildCardGroup;

                    //缓存在inputrims，后面再移动到正确层级
                    inputPrims.insert({ field_name, prim});
                }
                else if (role == Role_OutputPrimitive)
                {
                    if (reg_outputprims.find(param_name) != reg_outputprims.end()) {
                        //因为是定义在PROPERTY上，所以理论上可以重复写
                        throw makeError<UnimplError>("repeated name on output prims");
                    }

                    ParamPrimitive prim;
                    prim.name = param_name;
                    prim.bInput = false;
                    prim.bVisible = true;
                    prim.control = NullControl;
                    prim.socketType = Socket_Primitve;
                    prim.tooltip;
                    prim.wildCardGroup;

                    outputPrims.insert({ field_name, prim });
                    reg_outputprims.insert(param_name);
                }
            }
        }

        //通过寻找apply函数上的参数和返回值，为节点添加参数，不过ZenoReflect还没支持参数名称的反射，只有类型信息
        for (IMemberFunction* func : typebase->get_member_functions())
        {
            const auto& funcname = func->get_name();
            if (funcname != "apply") {
                continue;
            }
            const RTTITypeInfo& ret_type = func->get_return_rtti();
            ParamType type = ret_type.get_decayed_hash() == 0 ? ret_type.hash_code() : ret_type.get_decayed_hash();
            bool isConstPtr = false;
            bool isObject = zeno::isObjectType(ret_type, isConstPtr);
            if (type != Param_Null)
            {
                //存在返回类型，说明有输出，需要分配一个输出参数
                int idx = 1;
                std::string param_name = "result";
                if (isObject) {
                    while (reg_outputobjs.find(param_name) != reg_outputobjs.end()) {
                        param_name = "result" + std::to_string(idx++);
                    }
                    ParamObject outputObj;
                    outputObj.name = param_name;
                    outputObj.bInput = false;
                    outputObj.socketType = Socket_Output;
                    outputObj.type = type;

                    outputObjs.insert({ param_name, outputObj });
                    reg_outputobjs.insert(param_name);
                }
                else {
                    while (reg_outputprims.find(param_name) != reg_outputprims.end()) {
                        param_name = "result" + std::to_string(idx++);
                    }
                    ParamPrimitive outPrim;
                    outPrim.name = param_name;
                    outPrim.bInput = false;
                    outPrim.socketType = Socket_Primitve;
                    outPrim.type = type;
                    outPrim.bVisible = false;
                    outPrim.wildCardGroup;

                    outputPrims.insert({ param_name, outPrim });
                    reg_outputprims.insert(param_name);
                }
            }

            const ArrayList<RTTITypeInfo>& params = func->get_params();
            const auto& param_names = func->get_params_name();
            assert(params.size() == param_names.size());
            for (int idxParam = 0; idxParam < params.size(); idxParam++)
            {
                const RTTITypeInfo& param_type = params[idxParam];
                isObject = zeno::isObjectType(param_type, isConstPtr);

                std::string const& param_name(param_names[idxParam].c_str());
                type = param_type.get_decayed_hash() == 0 ? param_type.hash_code() : param_type.get_decayed_hash();
                if (param_name.empty()) {
                    //空白参数不考虑
                    continue;
                }
                if (!param_type.has_flags(TF_IsConst) && param_type.has_flags(TF_IsLValueRef)) {
                    //引用返回当作是输出处理
                    if (isObject)
                    {
                        if (reg_outputobjs.find(param_name) == reg_outputobjs.end()) {
                            ParamObject outputObj;
                            outputObj.name = param_name;
                            outputObj.bInput = false;
                            outputObj.socketType = Socket_Output;
                            outputObj.type = type;

                            outputObjs.insert({ param_name, outputObj });
                            reg_outputobjs.insert(param_name);
                        }
                    }
                    else {
                        if (reg_outputprims.find(param_name) == reg_outputprims.end()) {
                            ParamPrimitive prim;
                            prim.name = param_name;
                            prim.bInput = false;
                            prim.bVisible = false;
                            prim.control = NullControl;
                            prim.socketType = Socket_Primitve;
                            prim.type = type;
                            prim.defl = func->init_param_default_value(idxParam);
                            prim.tooltip;
                            prim.wildCardGroup;
                            m_customui.outputPrims.emplace_back(prim);

                            outputPrims.insert({ param_name, prim });
                            reg_outputprims.insert(param_name);
                        }
                    }
                }
                else {
                    //观察是否为shared_ptr<IObject>
                    if (isObject)
                    {
                        if (reg_inputobjs.find(param_name) != reg_inputobjs.end()) {
                            //同名情况，说明成员变量定义了一个相同名字的参数，很罕见，但可以直接跳过
                        }
                        else {
                            ParamObject inObj;
                            inObj.name = param_name;
                            inObj.bInput = true;
                            if (isConstPtr)
                                inObj.socketType = Socket_ReadOnly;
                            else
                                inObj.socketType = Socket_Owning;   //默认还是owning
                            inObj.type = type;

                            inputObjs.insert({ param_name, inObj });
                            reg_inputobjs.insert(param_name);
                        }
                    }
                    else {
                        if (reg_inputprims.find(param_name) == reg_inputprims.end()) {
                            ParamPrimitive inPrim;
                            inPrim.name = param_name;
                            inPrim.bInput = true;
                            inPrim.socketType = Socket_Primitve;
                            inPrim.type = type;

                            //检查函数是否带有默认参数
                            const Any& deflVal = func->get_param_default_value(idxParam);
                            if (deflVal.has_value()) {
                                inPrim.defl = deflVal;
                            }
                            else {
                                inPrim.defl = func->init_param_default_value(idxParam);
                            }

                            inPrim.control = getDefaultControl(type);
                            inPrim.bVisible = false;
                            inPrim.wildCardGroup;

                            //缓存在inputrims，后面再移动到正确层级
                            inputPrims.insert({ param_name, inPrim});
                            reg_inputprims.insert(param_name);
                        }
                    }
                }
            }
        }

        adjustCustomUiStructure(spNode, inputPrims, inputObjs, outputPrims, outputObjs);

        //init all params, and set defl value
        for (const ParamObject& param : m_customui.inputObjs)
        {
            spNode->add_input_obj_param(param);
        }
        for (const ParamPrimitive& param : customUiToParams(m_customui.inputPrims))
        {
            spNode->add_input_prim_param(param);
        }
        for (const ParamPrimitive& param : m_customui.outputPrims)
        {
            spNode->add_output_prim_param(param);
        }
        for (const ParamObject& param : m_customui.outputObjs)
        {
            spNode->add_output_obj_param(param);
        }
        return spNode;
    }
};

}

ZENO_API Session::Session()
    : globalState(std::make_unique<GlobalState>())
    , globalComm(std::make_unique<GlobalComm>())
    , globalError(std::make_unique<GlobalError>())
    , eventCallbacks(std::make_unique<EventCallbacks>())
    , m_userData(std::make_unique<UserData>())
    , mainGraph(std::make_shared<Graph>("main"))
    , assets(std::make_shared<AssetsMgr>())
    , objsMan(std::make_unique<ObjectManager>())
    , referManager(std::make_unique<ReferManager>())
    , globalVariableManager(std::make_unique<GlobalVariableManager>())
{
    initReflectNodes();
    //initNodeCates();  //should init after all static initialization finished.
}

ZENO_API Session::~Session() = default;


static CustomUI descToCustomui(const Descriptor& desc) {
    //兼容以前写的各种ZENDEFINE
    CustomUI ui;

    ui.nickname = desc.displayName;
    ui.iconResPath = desc.iconResPath;
    ui.doc = desc.doc;
    if (!desc.categories.empty())
        ui.category = desc.categories[0];   //很多cate都只有一个

    ParamGroup default;
    for (const SocketDescriptor& param_desc : desc.inputs) {
        ParamType type = zeno::convertToType(param_desc.type, param_desc.name);
        if (isPrimitiveType(type)) {
            //如果是数值类型，就添加到组里
            ParamPrimitive param;
            param.name = param_desc.name;
            param.type = type;
            param.defl = zeno::str2any(param_desc.defl, param.type);
            if (param_desc.socketType != zeno::NoSocket)
                param.socketType = param_desc.socketType;
            if (param_desc.control != NullControl)
                param.control = param_desc.control;
            if (starts_with(param_desc.type, "enum ")) {
                //compatible with old case of combobox items.
                param.type = zeno::types::gParamType_String;
                param.control = Combobox;
                std::vector<std::string> items = split_str(param_desc.type, ' ');
                if (!items.empty()) {
                    items.erase(items.begin());
                    param.ctrlProps = items;
                }
            }
            if (param.type != Param_Null && param.control == NullControl)
                param.control = getDefaultControl(param.type);
            param.tooltip = param_desc.doc;
            param.prop = Socket_Normal;
            param.wildCardGroup = param_desc.wildCard;
            param.bVisible = false;
            default.params.emplace_back(std::move(param));
        }
        else
        {
            //其他一律认为是对象（Zeno目前的类型管理非常混乱，有些类型值是空字符串，但绝大多数是对象类型
            ParamObject param;
            param.name = param_desc.name;
            param.type = type;
            if (param_desc.socketType != zeno::NoSocket)
                param.socketType = param_desc.socketType;
            param.bInput = true;
            param.wildCardGroup = param_desc.wildCard;
            ui.inputObjs.emplace_back(std::move(param));
        }
    }
    for (const ParamDescriptor& param_desc : desc.params) {
        ParamPrimitive param;
        param.name = param_desc.name;
        param.type = zeno::convertToType(param_desc.type, param.name);
        param.defl = zeno::str2any(param_desc.defl, param.type);
        param.socketType = NoSocket;
        //其他控件估计是根据类型推断的。
        if (starts_with(param_desc.type, "enum ")) {
            //compatible with old case of combobox items.
            param.type = zeno::types::gParamType_String;
            param.control = Combobox;
            std::vector<std::string> items = split_str(param_desc.type, ' ');
            if (!items.empty()) {
                items.erase(items.begin());
                param.ctrlProps = items;
            }
        }
        if (param.type != Param_Null && param.control == NullControl)
            param.control = getDefaultControl(param.type);
        param.tooltip = param_desc.doc;
        param.bVisible = false;
        default.params.emplace_back(std::move(param));
    }
    for (const SocketDescriptor& param_desc : desc.outputs) {
        ParamType type = zeno::convertToType(param_desc.type, param_desc.name);
        if (isPrimitiveType(type)) {
            //如果是数值类型，就添加到组里
            ParamPrimitive param;
            param.name = param_desc.name;
            param.type = type;
            param.defl = zeno::str2any(param_desc.defl, param.type);
            if (param_desc.socketType != zeno::NoSocket)
                param.socketType = param_desc.socketType;
            param.control = NullControl;
            param.tooltip = param_desc.doc;
            param.prop = Socket_Normal;
            param.wildCardGroup = param_desc.wildCard;
            param.bVisible = false;
            ui.outputPrims.emplace_back(std::move(param));
        }
        else
        {
            //其他一律认为是对象（Zeno目前的类型管理非常混乱，有些类型值是空字符串，但绝大多数是对象类型
            ParamObject param;
            param.name = param_desc.name;
            param.type = type;
            if (param_desc.socketType != zeno::NoSocket)
                param.socketType = param_desc.socketType;
            param.socketType = Socket_Output;
            param.bInput = false;
            param.prop = Socket_Normal;
            param.wildCardGroup = param_desc.wildCard;
            ui.outputObjs.emplace_back(std::move(param));
        }
    }
    ParamTab tab;
    tab.groups.emplace_back(std::move(default));
    ui.inputPrims.tabs.emplace_back(std::move(tab));
    return ui;
}

ZENO_API void Session::defNodeClass(std::shared_ptr<INode>(*ctor)(), std::string const &clsname, Descriptor const &desc) {
    if (nodeClasses.find(clsname) != nodeClasses.end()) {
        log_error("node class redefined: `{}`\n", clsname);
    }

    if (clsname == "PrimitiveTransform") {
        int j;
        j = 0;
    }

    CustomUI ui = descToCustomui(desc);
    auto cls = std::make_unique<ImplNodeClass>(ctor, ui, clsname);
    nodeClasses.emplace(clsname, std::move(cls));
}

ZENO_API void Session::defNodeClass2(std::shared_ptr<INode>(*ctor)(), std::string const& nodecls, CustomUI const& customui) {
    if (nodeClasses.find(nodecls) != nodeClasses.end()) {
        log_error("node class redefined: `{}`\n", nodecls);
    }
    CustomUI ui = customui;
    initControlsByType(ui);
    auto cls = std::make_unique<ImplNodeClass>(ctor, ui, nodecls);
    nodeClasses.emplace(nodecls, std::move(cls));
}

ZENO_API void Session::defNodeReflectClass(std::function<std::shared_ptr<INode>()> ctor, zeno::reflect::TypeBase* pTypeBase)
{
    assert(pTypeBase);
    const zeno::reflect::ReflectedTypeInfo& info = pTypeBase->get_info();
    auto& nodecls = std::string(info.qualified_name.c_str());
    //有些name反射出来可能带有命名空间比如zeno::XXX
    int idx = nodecls.find_last_of(':');
    if (idx != std::string::npos) {
        nodecls = nodecls.substr(idx + 1);
    }

    if (nodeClasses.find(nodecls) != nodeClasses.end()) {
        log_error("node class redefined: `{}`\n", nodecls);
    }
    auto cls = std::make_unique<ReflectNodeClass>(ctor, nodecls, pTypeBase);
    //TODO: From metadata
    cls->m_customui.category = "reflect";
    nodeClasses.emplace(nodecls, std::move(cls));
}

ZENO_API INodeClass::INodeClass(CustomUI const &customui, std::string const& classname)
    : m_customui(customui)
    , classname(classname)
{
}

ZENO_API INodeClass::~INodeClass() = default;

ZENO_API std::shared_ptr<Graph> Session::createGraph(const std::string& name) {
    auto graph = std::make_shared<Graph>(name);
    return graph;
}

ZENO_API void Session::resetMainGraph() {
    mainGraph.reset();
    mainGraph = std::make_shared<Graph>("main");
    referManager.reset();
    referManager = std::make_unique<ReferManager>();
    globalVariableManager.reset();
    globalVariableManager = std::make_unique<GlobalVariableManager>();
}

ZENO_API void Session::setApiLevelEnable(bool bEnable)
{
    m_bApiLevelEnable = bEnable;
}

ZENO_API void Session::beginApiCall()
{
    if (!m_bApiLevelEnable || m_bDisableRunning) return;
    m_apiLevel++;
}

ZENO_API void Session::endApiCall()
{
    if (!m_bApiLevelEnable || m_bDisableRunning) return;
    m_apiLevel--;
    if (m_apiLevel == 0) {
        if (m_bAutoRun) {
            if (m_callbackRunTrigger) {
                m_callbackRunTrigger();
            }
            else {
                run();
            }
        }
    }
}

ZENO_API void Session::setDisableRunning(bool bOn)
{
    m_bDisableRunning = bOn;
}

ZENO_API void Session::registerRunTrigger(std::function<void()> func)
{
    m_callbackRunTrigger = func;
}

ZENO_API void Session::registerNodeCallback(F_NodeStatus func)
{
    m_funcNodeStatus = func;
}

void Session::reportNodeStatus(const ObjPath& path, bool bDirty, NodeRunStatus status)
{
    if (m_funcNodeStatus) {
        m_funcNodeStatus(path, bDirty, status);
    }
}

ZENO_API int Session::registerObjId(const std::string& objprefix)
{
    int objid = objsMan->registerObjId(objprefix);
    return objid;
}

ZENO_API void Session::switchToFrame(int frameid)
{
    CORE_API_BATCH
    mainGraph->markDirtyWhenFrameChanged();
    globalState->updateFrameId(frameid);
}

ZENO_API void Session::interrupt() {
    m_bInterrupted = true;
}

ZENO_API bool Session::is_interrupted() const {
    return m_bInterrupted;
}

ZENO_API bool Session::run() {
    if (m_bDisableRunning)
        return false;

    m_bInterrupted = false;
    globalState->set_working(true);

    zeno::log_info("Session::run()");

    objsMan->beforeRun();
    zeno::scope_exit sp([&]() { objsMan->afterRun(); });

    globalError->clearState();

    //本次运行清除m_objects中上一次运行时被标记移除的obj，不能立刻清除因为视窗export_loading_objs时，需要在m_objects中查找被删除的obj
    objsMan->clearLastUnregisterObjs();
    //对之前删除节点时记录的obj，对应的所有其他关联节点，都标脏
    objsMan->remove_attach_node_by_removing_objs();

    zeno::GraphException::catched([&] {
        mainGraph->runGraph();
        }, *globalError);
    if (globalError->failed()) {
        zeno::log_error("");
    }

    return true;
}

ZENO_API void Session::set_auto_run(bool bOn) {
    m_bAutoRun = bOn;
}

ZENO_API bool Session::is_auto_run() const {
    return m_bAutoRun;
}

ZENO_API void Session::set_Rerun()
{
    mainGraph->markDirtyAll();
    objsMan->clear();
}

void Session::initNodeCates() {
    for (auto const& [key, cls] : nodeClasses) {
        if (!key.empty() && key.front() == '^')
            continue;
        std::string cate = cls->m_customui.category;
        if (m_cates.find(cate) == m_cates.end())
            m_cates.insert(std::make_pair(cate, std::vector<std::string>()));
        m_cates[cate].push_back(key);
    }
}

void Session::initReflectNodes() {
    auto& registry = zeno::reflect::ReflectionRegistry::get();
    for (zeno::reflect::TypeBase* type : registry->all()) {
        //TODO: 判断type的基类是不是基于INode
        const zeno::reflect::ReflectedTypeInfo& info = type->get_info();
        zeno::reflect::ITypeConstructor* ctor = type->get_constructor_or_null({});
        assert(ctor);

        defNodeReflectClass([=]()->std::shared_ptr<INode> {
            INode* pNewNode = static_cast<INode*>(ctor->new_instance());
            pNewNode->initTypeBase(type);
            std::shared_ptr<INode> spNode(pNewNode);
            return spNode;
        }, type);
    }
}

ZENO_API zeno::NodeCates Session::dumpCoreCates() {
    if (m_cates.empty()) {
        initNodeCates();
    }
    return m_cates;
}

namespace {
std::string dumpDescriptorToJson(const std::string &key, const Descriptor& descriptor) {
    using namespace rapidjson;
    Document doc;
    doc.SetArray();

    // Inputs array
    Value inputs(kArrayType);
    for (const auto& input : descriptor.inputs) {
        Value inputArray(kArrayType);
        inputArray.PushBack(Value().SetString(input.type.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputArray.PushBack(Value().SetString(input.name.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputArray.PushBack(Value().SetString(input.defl.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputArray.PushBack(Value().SetString(input.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());
        inputs.PushBack(inputArray, doc.GetAllocator());
    }

    // Outputs array
    Value outputs(kArrayType);
    for (const auto& output : descriptor.outputs) {
        Value outputArray(kArrayType);
        outputArray.PushBack(Value().SetString(output.type.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputArray.PushBack(Value().SetString(output.name.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputArray.PushBack(Value().SetString(output.defl.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputArray.PushBack(Value().SetString(output.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());
        outputs.PushBack(outputArray, doc.GetAllocator());
    }

    // Params array
    Value params(kArrayType);
    for (const auto& param : descriptor.params) {
        Value paramArray(kArrayType);
        paramArray.PushBack(Value().SetString(param.type.c_str(), doc.GetAllocator()), doc.GetAllocator());
        paramArray.PushBack(Value().SetString(param.name.c_str(), doc.GetAllocator()), doc.GetAllocator());
        paramArray.PushBack(Value().SetString(param.defl.c_str(), doc.GetAllocator()), doc.GetAllocator());
        paramArray.PushBack(Value().SetString(param.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());
        params.PushBack(paramArray, doc.GetAllocator());
    }

    // Categories array
    Value categories(kArrayType);
    for (const auto& category : descriptor.categories) {
        categories.PushBack(Value().SetString(category.c_str(), doc.GetAllocator()), doc.GetAllocator());
    }

    // Push values into the main document
    doc.PushBack(Value().SetString(key.c_str(), doc.GetAllocator()), doc.GetAllocator());
    doc.PushBack(inputs, doc.GetAllocator());
    doc.PushBack(outputs, doc.GetAllocator());
    doc.PushBack(params, doc.GetAllocator());
    doc.PushBack(categories, doc.GetAllocator());
    doc.PushBack(Value().SetString(descriptor.doc.c_str(), doc.GetAllocator()), doc.GetAllocator());

    // Write the JSON string to stdout
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}
}

ZENO_API std::string Session::dumpDescriptorsJSON() const {
    //deprecated.
    return "";
}

ZENO_API UserData &Session::userData() const {
    return *m_userData;
}

ZENO_API Session &getSession() {
#if 0
    static std::unique_ptr<Session> ptr;
    if (!ptr) {
        ptr = std::make_unique<Session>();
    }
#else
    static std::unique_ptr<Session> ptr = std::make_unique<Session>();
#endif
    return *ptr;
}

}
