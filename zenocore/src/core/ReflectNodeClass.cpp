#include <zeno/zeno.h>
#include <zeno/core/INodeClass.h>
#include <regex>
#include <zeno/utils/helper.h>


using namespace zeno::reflect;
using namespace zeno::types;

namespace zeno {

    struct ParamMappingInfo
    {
        std::set<std::string> reg_inputobjs, reg_inputprims, reg_outputobjs, reg_outputprims;
        std::map<std::string, ParamPrimitive> inputPrims;
        std::map<std::string, ParamObject> inputObjs;
        std::map<std::string, ParamPrimitive> outputPrims;
        std::map<std::string, ParamObject> outputObjs;

        //函数返回值专门存放的地方，由于当时没有名字，所以上述其他结构不能保存返回值
        std::vector<std::variant<ParamObject, ParamPrimitive>> retInfo;

        std::set<std::string> anyInputs;
        std::set<std::string> anyOutputs;
    };

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

    static ParamControl parseControlProps(const zeno::reflect::IRawMetadata* metadata, ParamType type, zeno::reflect::Any& controlProps) {
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
    
        return ctrl;
    }

    //整理customui层级
    static void adjustCustomUiStructure(
        std::shared_ptr<INode> spTempNode,
        zeno::reflect::TypeBase* typebase,
        CustomUI& customui,
        ParamMappingInfo& paramsMapping)
    {
        const auto& makeWildCardPrimParam = [](const std::string& name, bool bInput, const std::string& wildCardGroup) {
            ParamPrimitive wildCardPrim;
            wildCardPrim.name = name;
            wildCardPrim.bInput = bInput;
            wildCardPrim.type = Param_Wildcard;
            wildCardPrim.socketType = Socket_WildCard;
            wildCardPrim.wildCardGroup = wildCardGroup;
            return wildCardPrim;
        };

        auto autofillDefaultOutputName = [&]() {
            for (auto arg : paramsMapping.retInfo) {
                std::visit([&](auto& val) {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (std::is_same_v<T, ParamObject>) {
                        std::string dispName = uniqueName("Output Object", paramsMapping.reg_outputobjs);
                        val.name = dispName;
                        paramsMapping.outputObjs.insert({ dispName, val });
                        paramsMapping.reg_outputobjs.insert(dispName);
                    }
                    else if constexpr (std::is_same_v<T, ParamPrimitive>) {
                        std::string dispName = uniqueName("Output Data", paramsMapping.reg_outputprims);
                        val.name = dispName;
                        paramsMapping.outputPrims.insert({ dispName, val });
                        paramsMapping.reg_outputprims.insert(dispName);
                    }
                }, arg);
            }
        };

        bool bUseReflectUI = false;
        bool bReflectCustomUI = false;

        for (IMemberField* field : typebase->get_member_fields()) {
            zeno::reflect::TypeHandle fieldType = field->get_field_type();
            if (fieldType == zeno::reflect::get_type<ReflectCustomUI>()) {
                zeno::reflect::Any defl = field->get_field_value(spTempNode.get());
                if (defl.has_value()) {
                    bUseReflectUI = true;

                    ReflectCustomUI reflectCustomUi = zeno::reflect::any_cast<ReflectCustomUI>(defl);

                    if (!reflectCustomUi.customUI.empty())
                    {
                        //用户在类定义里定义了一个，直接用定义的覆盖原来的
                        customui.inputPrims = reflectCustomUi.customUI;
                        bReflectCustomUI = true;
                    }

                    //customui如果指定了返回值的名称映射，那么大小就要和返回值大小一致，否则会很混乱
                    int nMappingOutput = reflectCustomUi.outputParams.size();
                    assert(nMappingOutput == 0 || nMappingOutput == paramsMapping.retInfo.size());

                    //首先为返回值设置名字（如果有指定名字映射）
                    if (nMappingOutput > 0) {
                        for (int i = 0; i < nMappingOutput; i++) {
                            _CommonParam& outputParam = reflectCustomUi.outputParams[i];
                            std::visit([&](auto&& arg) {
                                using T = std::decay_t<decltype(arg)>;
                                if constexpr (std::is_same_v<T, ParamObject>) {
                                    if (std::holds_alternative<ParamObject>(outputParam.param)) {
                                        auto out_param = std::get<ParamObject>(outputParam.param);
                                        std::string dispName = out_param.name;
                                        out_param.bInput = false;
                                        out_param.socketType = Socket_Output;
                                        out_param.type = arg.type;  //返回值的类型是最正确的
                                        paramsMapping.outputObjs.insert({ dispName, out_param });
                                        paramsMapping.reg_outputobjs.insert(dispName);
                                    }
                                }
                                else if constexpr (std::is_same_v<T, ParamPrimitive>) {
                                    if (std::holds_alternative<ParamPrimitive>(outputParam.param)) {
                                        auto out_param = std::get<ParamPrimitive>(outputParam.param);
                                        out_param.type = arg.type;  //只有类型是在函数解析时确定下来的
                                        out_param.bInput = false;
                                        out_param.socketType = Socket_Primitve;
                                        std::string dispName = out_param.name;
                                        paramsMapping.outputPrims.insert({ dispName, out_param });
                                        paramsMapping.reg_outputprims.insert(dispName);
                                    }
                                }
                            }, paramsMapping.retInfo[i]);
                        }
                    }
                    else {
                        //没有指定名称，为返回值参数给一个默认名称
                        autofillDefaultOutputName();
                    }

                    //处理输入的名称映射
                    if (reflectCustomUi.inputParams.size() > 0)
                    {
                        for (_CommonParam& input_param : reflectCustomUi.inputParams) {
                            auto iterPrim = paramsMapping.inputPrims.find(input_param.mapTo);
                            if (iterPrim != paramsMapping.inputPrims.end()) {
                                auto handler = paramsMapping.inputPrims.extract(input_param.mapTo);
                                if (std::holds_alternative<ParamPrimitive>(input_param.param)) {
                                    auto in_param = std::get<ParamPrimitive>(input_param.param);
                                    handler.key() = in_param.name;
                                    handler.mapped().bInput = true;
                                    handler.mapped().name = in_param.name;
                                    handler.mapped().control = in_param.control;
                                    handler.mapped().ctrlProps = in_param.ctrlProps;
                                    handler.mapped().constrain = in_param.constrain;
                                    //类型和默认值不要在这里给，apply函数解析时已经解析过了，否则会有冲突不好处理
                                    //handler.defl = param.defl;
                                    handler.mapped().wildCardGroup = in_param.wildCardGroup;
                                    paramsMapping.inputPrims.insert(std::move(handler));
                                }
                            }
                            else{
                                auto iterObject = paramsMapping.inputObjs.find(input_param.mapTo);
                                if (iterObject != paramsMapping.inputObjs.end()) {
                                    auto handler = paramsMapping.inputObjs.extract(input_param.mapTo);
                                    if (std::holds_alternative<ParamObject>(input_param.param)) {
                                        auto in_param = std::get<ParamObject>(input_param.param);
                                        handler.key() = in_param.name;
                                        handler.mapped().bInput = true;
                                        handler.mapped().name = in_param.name;
                                        handler.mapped().constrain = in_param.constrain;
                                        handler.mapped().wildCardGroup = in_param.wildCardGroup;
                                        paramsMapping.inputObjs.insert(std::move(handler));
                                    }
                                }
                            }
                        }
                    }
#if 0
                    ParamTab tab;
                    tab.name = reflectCustomUi.inputPrims.name;
                    for (auto& reflectgroup : reflectCustomUi.inputPrims.groups) {
                        ParamGroup group;
                        group.name = reflectgroup.name;
                        for (auto& param : reflectgroup.params) {
                            if (paramsMapping.anyInputs.find(param.mapTo) != paramsMapping.anyInputs.end()) {   //如果是Any视为wildCard
                                group.params.push_back(std::move(makeWildCardPrimParam(param.dispName, true, param.wildCardGroup)));
                                paramsMapping.anyInputs.erase(param.mapTo);
                            }
                            else if (paramsMapping.inputPrims.find(param.mapTo) != paramsMapping.inputPrims.end()) { //按照ReflectCustomUI的信息更新ParamPrimitive并放入对应的group
                                auto& inprim = inputPrims[param.mapTo];
                                inprim.name = param.dispName;
                                inprim.defl = param.defl.type() == zeno::reflect::type_info<const char*>() ? (std::string)zeno::reflect::any_cast<const char*>(param.defl) : param.defl;
                                convertToEditVar(inprim.defl, param.defl.type().hash_code());
                                inprim.bInnerParam = param.bInnerParam;
                                if (param.bInnerParam) {
                                    inprim.control = NullControl;
                                    inprim.sockProp = Socket_Disable;
                                }
                                else if (param.ctrl != NullControl) {
                                    inprim.control = param.ctrl;
                                }
                                inprim.ctrlProps = param.ctrlProps;
                                group.params.push_back(inprim);
                                inputPrims.erase(param.mapTo);
                            }
                        }
                        tab.groups.push_back(std::move(group));
                    }
                    customui.inputPrims.push_back(std::move(tab));

                    for (auto& reflectInputObj : reflectCustomUi.inputObjs.objs) {
                        if (paramsMapping.anyInputs.find(reflectInputObj.mapTo) != anyInputs.end()) {
                            ParamObject wildCardObj;
                            wildCardObj.name = reflectInputObj.dispName;
                            wildCardObj.bInput = true;
                            wildCardObj.type = Obj_Wildcard;
                            wildCardObj.socketType = Socket_WildCard;
                            wildCardObj.wildCardGroup = reflectInputObj.wildCardGroup;
                            customui.inputObjs.push_back(std::move(wildCardObj));
                            anyInputs.erase(reflectInputObj.mapTo);
                        }
                        else if (paramsMapping.inputObjs.find(reflectInputObj.mapTo) != paramsMapping.inputObjs.end()) {
                            auto& ObjSetting = inputObjs[reflectInputObj.mapTo];
                            ObjSetting.name = reflectInputObj.dispName;
                            if (ObjSetting.socketType != Socket_ReadOnly) {
                                //如果出现了只读，说明是函数签名施加的，因以此为主，否则就由info决定
                                ObjSetting.socketType = reflectInputObj.type;
                            }
                            customui.inputObjs.push_back(ObjSetting);
                            inputObjs.erase(reflectInputObj.mapTo);
                        }
                    }

                    //查看返回值的输出信息，如果有映射，就修改返回值的名称信息：
                    if (!reflectCustomUi.retInfo.dispName.empty())
                    {
                        std::visit([&](auto& arg) {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr (std::is_same_v<T, ParamObject>) {
                                arg.name = reflectCustomUi.retInfo.dispName;
                                arg.wildCardGroup = reflectCustomUi.retInfo.wildCardGroup;
                                customui.refltctReturnName = arg.name;
                            }
                            else if constexpr (std::is_same_v<T, ParamPrimitive>) {
                                arg.name = reflectCustomUi.retInfo.dispName;
                                arg.wildCardGroup = reflectCustomUi.retInfo.wildCardGroup;
                                customui.refltctReturnName = arg.name;
                            }
                            }, paramsMapping.retInfo);
                    }

                    for (auto& reflectOutputObj : reflectCustomUi.retParams.objs) {
                        if (paramsMapping.anyOutputs.find(reflectOutputObj.mapTo) != paramsMapping.anyOutputs.end()) {
                            ParamObject wildCardObj;
                            wildCardObj.name = reflectOutputObj.dispName;
                            wildCardObj.bInput = false;
                            wildCardObj.type = Obj_Wildcard;
                            wildCardObj.socketType = Socket_WildCard;
                            wildCardObj.wildCardGroup = reflectOutputObj.wildCardGroup;
                            customui.outputObjs.push_back(std::move(wildCardObj));
                            paramsMapping.anyOutputs.erase(reflectOutputObj.mapTo);
                        }
                        else if (paramsMapping.outputObjs.find(reflectOutputObj.mapTo) != paramsMapping.outputObjs.end()) {
                            paramsMapping.outputObjs[reflectOutputObj.mapTo].name = reflectOutputObj.dispName;
                            paramsMapping.outputObjs[reflectOutputObj.mapTo].socketType = reflectOutputObj.type == Socket_Owning ? Socket_Output : reflectOutputObj.type;
                            customui.outputObjs.push_back(std::move(paramsMapping.outputObjs[reflectOutputObj.mapTo]));
                            paramsMapping.outputObjs.erase(reflectOutputObj.mapTo);
                        }
                        else if (reflectOutputObj.mapTo.empty()) {
                            if (paramsMapping.outputObjs.find("result") != paramsMapping.outputObjs.end()) {    //空串mapping到返回值,返回值名为"result"
                                paramsMapping.outputObjs["result"].name = reflectOutputObj.dispName;
                                paramsMapping.outputObjs["result"].socketType = reflectOutputObj.type;
                                customui.outputObjs.push_back(std::move(paramsMapping.outputObjs["result"]));
                                paramsMapping.outputObjs.erase("result");
                            }
                            else {    //apply返回值为void但ReflectCustomUI定义了空串，加入一个默认IObject输出
                                ParamObject outputObj;
                                outputObj.name = "result";
                                outputObj.bInput = false;
                                outputObj.socketType = Socket_Output;
                                outputObj.type = gParamType_IObject;
                            }
                        }
                    }
                    for (auto& reflectOutputPrim : reflectCustomUi.outputPrims.params) {
                        if (paramsMapping.anyOutputs.find(reflectOutputPrim.mapTo) != paramsMapping.anyOutputs.end()) {
                            customui.outputPrims.push_back(std::move(makeWildCardPrimParam(reflectOutputPrim.dispName, false, reflectOutputPrim.wildCardGroup)));
                            paramsMapping.anyOutputs.erase(reflectOutputPrim.mapTo);
                        }
                        else if (paramsMapping.outputPrims.find(reflectOutputPrim.mapTo) != paramsMapping.outputPrims.end()) {
                            paramsMapping.outputPrims[reflectOutputPrim.mapTo].name = reflectOutputPrim.dispName;
                            paramsMapping.outputPrims[reflectOutputPrim.mapTo].defl = reflectOutputPrim.defl;
                            customui.outputPrims.push_back(std::move(paramsMapping.outputPrims[reflectOutputPrim.mapTo]));
                            paramsMapping.outputPrims.erase(reflectOutputPrim.mapTo);
                        }
                    }

                    //若有剩余(是成员变量或apply中有,但ReflectCustomUI没有的参数)，再将剩余加入
                    if (customui.inputPrims.empty()) {
                        zeno::ParamTab tab;
                        tab.name = "Tab1";
                        zeno::ParamGroup group;
                        group.name = "Group1";
                        tab.groups.emplace_back(group);
                        customui.inputPrims.emplace_back(tab);
                    }
                    else if (customui.inputPrims[0].groups.empty()) {
                        zeno::ParamGroup group;
                        group.name = "Group1";
                        customui.inputPrims[0].groups.emplace_back(group);
                    }
#endif
                }
            }
        }

        if (!bUseReflectUI) {
            autofillDefaultOutputName();
        }

        //如果没有定义ReflectCustomUI类型成员变量，使用默认
        if (customui.inputPrims.empty())
        {
            zeno::ParamTab tab;
            tab.name = "Tab1";
            zeno::ParamGroup group;
            group.name = "Group1";
            tab.groups.emplace_back(group);
            customui.inputPrims.emplace_back(tab);
        }
        else if (customui.inputPrims[0].groups.empty()) {
            zeno::ParamGroup group;
            group.name = "Group1";
            customui.inputPrims[0].groups.emplace_back(group);
        }

        if (!bReflectCustomUI) {
            for (auto& [name, primParam] : paramsMapping.inputPrims) {
                customui.inputPrims[0].groups[0].params.push_back(primParam);
            }
        }

        for (auto& [name, objParam] : paramsMapping.inputObjs)
            customui.inputObjs.push_back(std::move(objParam));
        for (auto& [name, primParam] : paramsMapping.outputPrims)
            customui.outputPrims.push_back(std::move(primParam));
        for (auto& [name, objParam] : paramsMapping.outputObjs)
            customui.outputObjs.push_back(std::move(objParam));
        for (auto& name : paramsMapping.anyInputs)
            customui.inputPrims[0].groups[0].params.push_back(std::move(makeWildCardPrimParam(name, true, "")));
        for (auto& name : paramsMapping.anyOutputs)
            customui.outputPrims.push_back(std::move(makeWildCardPrimParam(name, false, "")));

        //check whether the value from prim has been set by Mapping data
        for (auto& tab : customui.inputPrims) {
            for (auto& group : tab.groups) {
                for (auto& input_param : group.params) {
                    std::string name = input_param.name;
                    auto iterPrim = paramsMapping.inputPrims.find(name);
                    if (iterPrim != paramsMapping.inputPrims.end()) {
                        ParamPrimitive mappingPrim = iterPrim->second;
                        //这个prim一定有类型和名字信息
                        input_param.name = mappingPrim.name;
                        input_param.type = mappingPrim.type;
                        if (input_param.control == NullControl) {
                            //如果customui没有指定控件，看看mapping那里有没有指定
                            input_param.control = mappingPrim.control;
                            //如果都是空，那就给个默认
                            input_param.control = getDefaultControl(input_param.type);
                        }
                    }
                    else {
                        //应该是成员变量定义的参数
                        //assert(false);
                    }
                    if (!input_param.defl.has_value() && input_param.type != Param_Wildcard) {
                        input_param.defl = initAnyDeflValue(input_param.type);
                        convertToEditVar(input_param.defl, input_param.type);
                        assert(input_param.defl.has_value());
                    }
                }
            }
        }
    }

    static void collectParamsFromMember(
        std::shared_ptr<INode> spTempNode,
        zeno::reflect::TypeBase* typebase,
        ParamMappingInfo& paramsMapping
    )
    {
        for (IMemberField* field : typebase->get_member_fields()) {
            std::string field_name(field->get_name().c_str());
            std::string param_name;
            std::string constrain;
            if (const zeno::reflect::IRawMetadata* metadata = field->get_metadata()) {
                //name:
                if (const zeno::reflect::IMetadataValue* value = metadata->get_value("DisplayName")) {
                    param_name = value->as_string();
                }
                else {
                    param_name = field_name;
                }
                //TODO: 名称合法性判断

                //参数约束：
                if (const zeno::reflect::IMetadataValue* value = metadata->get_value("Constrain")) {
                    constrain = value->as_string();
                }

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

                std::regex matchAny(R"(zeno::reflect::Any)");
                if (std::regex_search(typeInfo.name(), matchAny)) {   //判断Any类型，后续处理
                    if (role == Role_InputObject || role == Role_InputPrimitive) {
                        paramsMapping.anyInputs.insert(field_name);
                    }
                    else {
                        paramsMapping.anyOutputs.insert(field_name);
                    }
                    continue;
                }

                if (role == Role_InputObject)
                {
                    if (paramsMapping.reg_inputobjs.find(param_name) != paramsMapping.reg_inputobjs.end()) {
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
                    inputObj.constrain = constrain;
                    paramsMapping.inputObjs.insert({ field_name,inputObj });
                    paramsMapping.reg_inputobjs.insert(param_name);
                }
                else if (role == Role_OutputObject)
                {
                    if (paramsMapping.reg_outputobjs.find(param_name) != paramsMapping.reg_outputobjs.end()) {
                        //因为是定义在PROPERTY上，所以理论上可以重复写
                        throw makeError<UnimplError>("repeated name on input objs");
                    }

                    ParamObject outputObj;
                    outputObj.name = param_name;
                    outputObj.type = type;
                    outputObj.socketType = Socket_Output;

                    paramsMapping.outputObjs.insert({ field_name,outputObj });
                    paramsMapping.reg_outputobjs.insert(param_name);
                }
                else if (role == Role_InputPrimitive)
                {
                    zeno::reflect::Any defl = field->get_field_value(spTempNode.get());
                    zeno::reflect::Any controlProps;
                    ParamPrimitive prim;
                    ParamControl ctrl = parseControlProps(metadata, type, controlProps);

                    //观察是否内部参数：
                    if (const zeno::reflect::IMetadataValue* value = metadata->get_value("InnerSocket")) {
                        prim.sockProp = Socket_Disable;
                        ctrl = NullControl;
                    }

                    prim.name = param_name;
                    prim.type = type;
                    prim.bInput = true;
                    prim.bSocketVisible = false;
                    prim.control = ctrl;
                    prim.ctrlProps = controlProps;
                    prim.defl = defl;
                    convertToEditVar(prim.defl, prim.type);
                    prim.socketType = Socket_Primitve;
                    prim.constrain = constrain;
                    //TODO:
                    prim.tooltip;
                    prim.wildCardGroup;

                    //缓存在inputrims，后面再移动到正确层级
                    paramsMapping.inputPrims.insert({ field_name, prim });
                }
                else if (role == Role_OutputPrimitive)
                {
                    if (paramsMapping.reg_outputprims.find(param_name) != paramsMapping.reg_outputprims.end()) {
                        //因为是定义在PROPERTY上，所以理论上可以重复写
                        throw makeError<UnimplError>("repeated name on output prims");
                    }

                    ParamPrimitive prim;
                    prim.name = param_name;
                    prim.bInput = false;
                    prim.bSocketVisible = false;
                    prim.control = NullControl;
                    prim.socketType = Socket_Primitve;
                    //TODO:
                    prim.tooltip;
                    prim.wildCardGroup;

                    paramsMapping.outputPrims.insert({ field_name, prim });
                    paramsMapping.reg_outputprims.insert(param_name);
                }
            }
        }
    }

    static void collectParamsFromApply(
        std::shared_ptr<INode> spTempNode,
        zeno::reflect::TypeBase* typebase,
        ParamMappingInfo& paramsMapping
    )
    {
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

            if (ret_type.flags() & TF_IsMultiReturn)
            {
                ArrayList<RTTITypeInfo> rets = func->get_multi_return_rtti();
                for (const RTTITypeInfo& ret_type_rtti : rets) {
                    std::string rettype_name(ret_type_rtti.name());
                    ParamType rettype = ret_type_rtti.get_decayed_hash() == 0 ? ret_type_rtti.hash_code() : ret_type_rtti.get_decayed_hash();
                    if (ret_type_rtti.has_flags(TF_IsObject)) {
                        ParamObject outputObj;
                        outputObj.name = "";        //返回值暂时没有名字可以标识
                        outputObj.bInput = false;
                        outputObj.socketType = Socket_Output;
                        outputObj.type = rettype;
                        paramsMapping.retInfo.emplace_back(outputObj);
                    }
                    else {
                        ParamPrimitive outPrim;
                        outPrim.name = "";
                        outPrim.bInput = false;
                        outPrim.socketType = Socket_Primitve;
                        outPrim.type = rettype;
                        outPrim.bSocketVisible = false;
                        paramsMapping.retInfo.emplace_back(outPrim);
                    }
                }
            }
            else if (type != Param_Null)
            {
                //存在返回类型，说明有输出，需要分配一个输出参数
                int idx = 1;
                std::string param_name = "result";

                //TODO: 返回值不该支持Any类型，会使得parse更为复杂
                std::regex matchAny(R"(zeno::reflect::Any)");
                if (std::regex_search(ret_type.name(), matchAny)) {   //判断Any类型，后续处理
                    paramsMapping.anyOutputs.insert(param_name);
                    continue;
                }
                if (isObject) {
                    ParamObject outputObj;
                    outputObj.name = "";
                    outputObj.bInput = false;
                    outputObj.socketType = Socket_Output;
                    outputObj.type = type;
                    paramsMapping.retInfo.emplace_back(outputObj);
                }
                else {
                    ParamPrimitive outPrim;
                    outPrim.name = "";
                    outPrim.bInput = false;
                    outPrim.socketType = Socket_Primitve;
                    outPrim.type = type;
                    outPrim.bSocketVisible = false;
                    outPrim.wildCardGroup;
                    paramsMapping.retInfo.emplace_back(outPrim);
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
                    //忽略引用返回这种情况，输出参数必须要通过返回值才能体现。
                }
                else {
                    //观察是否为shared_ptr<IObject>
                    std::regex matchAny(R"(zeno::reflect::Any)");
                    if (std::regex_search(param_type.name(), matchAny)) {   //判断Any类型，后续处理
                        paramsMapping.anyInputs.insert(param_name);
                        continue;
                    }
                    if (isObject)
                    {
                        if (paramsMapping.reg_inputobjs.find(param_name) != paramsMapping.reg_inputobjs.end()) {
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

                            paramsMapping.inputObjs.insert({ param_name, inObj });
                            paramsMapping.reg_inputobjs.insert(param_name);
                        }
                    }
                    else {
                        if (paramsMapping.reg_inputprims.find(param_name) == paramsMapping.reg_inputprims.end()) {
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
                            convertToEditVar(inPrim.defl, inPrim.type);

                            inPrim.control = getDefaultControl(type);
                            inPrim.bSocketVisible = false;
                            inPrim.wildCardGroup;

                            //缓存在inputrims，后面再移动到正确层级
                            paramsMapping.inputPrims.insert({ param_name, inPrim });
                            paramsMapping.reg_inputprims.insert(param_name);
                        }
                    }
                }
            }
        }
    }

    ReflectNodeClass::ReflectNodeClass(std::function<std::shared_ptr<INode>()> ctor, std::string const& nodecls, zeno::reflect::TypeBase* pTypeBase)
        : INodeClass(CustomUI(), nodecls)
        , ctor(ctor)
        , typebase(pTypeBase)
    {
        initCustomUI();
    }

    void ReflectNodeClass::initCustomUI() {
        //DEBUG:
        if (this->classname == "Duplicate2") {
            int j;
            j = 0;
        }

        if (!m_customui.inputPrims.empty()) {
            m_customui.inputPrims.clear();
        }
        m_customui.inputObjs.clear();
        m_customui.outputPrims.clear();
        m_customui.outputObjs.clear();

        ParamMappingInfo paramsMapping;

        std::shared_ptr<INode> spTempNode = ctor();     //临时节点用于取初始化信息，不参与后续节点过程

        //先遍历所有成员，并收集其中的参数，目前假定所有成员变量都作为节点的参数存在，后续看情况可以指定
        collectParamsFromMember(spTempNode, typebase, paramsMapping);

        collectParamsFromApply(spTempNode, typebase, paramsMapping);

        adjustCustomUiStructure(spTempNode, typebase, m_customui, paramsMapping);

        //TODO: adjust
        m_customui.category = "reflect";
    }

    std::shared_ptr<INode> ReflectNodeClass::new_instance(std::shared_ptr<Graph> pGraph, std::string const& name) {
        std::shared_ptr<INode> spNode = ctor();
        spNode->initUuid(pGraph, classname);
        spNode->set_name(name);
        initCoreParams(spNode, m_customui);
        return spNode;
    }

}