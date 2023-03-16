#include "uihelper.h"
#include <zeno/utils/logger.h>
#include "modelrole.h"
#include "zassert.h"
#include "curvemodel.h"
#include "variantptr.h"
#include "jsonhelper.h"
#include <zenomodel/include/curveutil.h>
#include <zenomodel/include/viewparammodel.h>
#include <zenomodel/include/nodeparammodel.h>
#include <QUuid>
#include "common_def.h"
#include <zeno/funcs/ParseObjectFromUi.h>


using namespace zeno::iotags;
using namespace zeno::iotags::curve;

VarToggleScope::VarToggleScope(bool* pbVar)
    : m_pbVar(pbVar)
{
    Q_ASSERT(m_pbVar);
    *m_pbVar = true;
}

VarToggleScope::~VarToggleScope()
{
    *m_pbVar = false;
}


BlockSignalScope::BlockSignalScope(QObject* pObj)
    : m_pObject(pObj)
{
    if (m_pObject)
        m_pObject->blockSignals(true);
}

BlockSignalScope::~BlockSignalScope()
{
	if (m_pObject)
		m_pObject->blockSignals(false);
}


NODE_DESCS UiHelper::parseDescs(const rapidjson::Value &jsonDescs)
{
    NODE_DESCS _descs;
    for (const auto &node : jsonDescs.GetObject())
    {
        const QString &name = node.name.GetString();
        const auto &objValue = node.value;
        auto inputs = objValue["inputs"].GetArray();
        auto outputs = objValue["outputs"].GetArray();
        auto params = objValue["params"].GetArray();
        auto categories = objValue["categories"].GetArray();

        NODE_DESC desc;

        for (int i = 0; i < inputs.Size(); i++) {
            if (inputs[i].IsArray()) {
                auto input_triple = inputs[i].GetArray();
                const QString &socketType = input_triple[0].GetString();
                const QString &socketName = input_triple[1].GetString();
                const QString &socketDefl = input_triple[2].GetString();

                CONTROL_INFO ctrlInfo = getControlByType(name, PARAM_INPUT, socketName, socketType);
                INPUT_SOCKET inputSocket;
                inputSocket.info = SOCKET_INFO("", socketName);
                inputSocket.info.type = socketType;
                inputSocket.info.control = ctrlInfo.control;
                inputSocket.info.ctrlProps = ctrlInfo.controlProps.toMap();
                inputSocket.info.defaultValue = parseStringByType(socketDefl, socketType);
                desc.inputs.insert(socketName, inputSocket);
            } else {
                Q_ASSERT(inputs[i].IsNull());
            }
        }

        for (int i = 0; i < params.Size(); i++) {
            if (params[i].IsArray()) {
                auto param_triple = params[i].GetArray();
                const QString &socketType = param_triple[0].GetString();
                const QString &socketName = param_triple[1].GetString();
                const QString &socketDefl = param_triple[2].GetString();
                CONTROL_INFO ctrlInfo = getControlByType(name, PARAM_PARAM, socketName, socketType);
                PARAM_INFO paramInfo;
                paramInfo.bEnableConnect = false;
                paramInfo.control = ctrlInfo.control;
                paramInfo.controlProps = ctrlInfo.controlProps.toMap();
                paramInfo.name = socketName;
                paramInfo.typeDesc = socketType;
                paramInfo.defaultValue = parseStringByType(socketDefl, socketType);

                desc.params.insert(socketName, paramInfo);
            } else {
                Q_ASSERT(params[i].IsNull());
            }
        }

        for (int i = 0; i < outputs.Size(); i++) {
            if (outputs[i].IsArray()) {
                auto output_triple = outputs[i].GetArray();
                const QString &socketType = output_triple[0].GetString();
                const QString &socketName = output_triple[1].GetString();
                const QString &socketDefl = output_triple[2].GetString();
                PARAM_CONTROL ctrlType = getControlByType(socketType);
                OUTPUT_SOCKET outputSocket;
                outputSocket.info = SOCKET_INFO("", socketName);
                outputSocket.info.type = socketType;
                outputSocket.info.control = ctrlType;
                outputSocket.info.defaultValue = parseStringByType(socketDefl, socketType);

                desc.outputs.insert(socketName, outputSocket);
            } else {
                Q_ASSERT(outputs[i].IsNull());
            }
        }
        
        for (int i = 0; i < categories.Size(); i++)
        {
            desc.categories.push_back(categories[i].GetString());
        }

        _descs.insert(name, desc);
    }
    return _descs;
}

bool UiHelper::validateVariant(const QVariant& var, const QString& type)
{
    PARAM_CONTROL control = getControlByType(type);
    QVariant::Type varType = var.type();

    switch (control) {
    case CONTROL_INT:   return QVariant::Int == varType;
    case CONTROL_BOOL:  return (QVariant::Bool == varType || QVariant::Int == varType);
    case CONTROL_FLOAT: return (QMetaType::Float == varType || QVariant::Double == varType);
    case CONTROL_STRING:
    case CONTROL_WRITEPATH:
    case CONTROL_READPATH:
    case CONTROL_ENUM:
        return (QVariant::String == varType);
    case CONTROL_MULTILINE_STRING:
        return var.type() == QVariant::String;
    case CONTROL_COLOR:
    {
        if ((var.type() == QVariant::String) ||
            (varType == QVariant::UserType &&
             var.userType() == QMetaTypeId<QLinearGradient>::qt_metatype_id()))
        {
            return true;
        }
    }
    case CONTROL_CURVE:
    {
        return (varType == QMetaType::User);
    }
    case CONTROL_VEC2_FLOAT:
    case CONTROL_VEC2_INT:
    case CONTROL_VEC3_FLOAT:
    case CONTROL_VEC3_INT:
    case CONTROL_VEC4_FLOAT:
    case CONTROL_VEC4_INT:
    {
        if (varType == QVariant::UserType &&
            var.userType() == QMetaTypeId<UI_VECTYPE>::qt_metatype_id())
        {
            return true;
        }
    }
    case CONTROL_NONE:
        return var.isNull();
    case CONTROL_NONVISIBLE:
        return true;
    default:
        break;
    };
    return true;
}

QVariant UiHelper::initDefaultValue(const QString& type)
{
    if (type == "string") {
        return "";
    }
    else if (type == "float")
    {
        return QVariant((float)0.);
    }
    else if (type == "int")
    {
        return QVariant((int)0);
    }
    else if (type == "bool")
    {
        return QVariant(false);
    }
    else if (type.startsWith("vec"))
    {
        int dim = 0;
        bool bFloat = false;
        if (UiHelper::parseVecType(type, dim, bFloat))
        {
            return QVariant::fromValue(UI_VECTYPE(dim, 0));
        }
    }
    return QVariant();
}

QVariant UiHelper::parseStringByType(const QString &defaultValue, const QString &type)
{
    auto control = getControlByType(type);
    switch (control) {
    case CONTROL_INT:
    {
        bool bOk = false;
        int val = defaultValue.toInt(&bOk);
        if (bOk) {
            return val;
        }
        else {
            //type dismatch, try to convert to float.
            //disable it now because the sync problem is complicated and trivival.
            //float fVal = defaultValue.toFloat(&bOk);
            //if (bOk)
            //{
            //    val = fVal;
            //    return val;
            //}
            return defaultValue;
        }
    }
    case CONTROL_BOOL:
    {
        return (bool)defaultValue.toInt();
    }
    case CONTROL_FLOAT:
    {
        return defaultValue.toFloat();
    }
    case CONTROL_STRING:
    case CONTROL_WRITEPATH:
    case CONTROL_READPATH:
    case CONTROL_MULTILINE_STRING:
    case CONTROL_COLOR:
    case CONTROL_ENUM:
        return defaultValue;
    case CONTROL_VEC2_FLOAT:
    case CONTROL_VEC2_INT:
    case CONTROL_VEC3_FLOAT:
    case CONTROL_VEC3_INT:
    case CONTROL_VEC4_FLOAT:
    case CONTROL_VEC4_INT:
    {
        UI_VECTYPE vec;
        if (!defaultValue.isEmpty())
        {
            QStringList L = defaultValue.split(",");
            vec.resize(L.size());
            bool bOK = false;
            for (int i = 0; i < L.size(); i++)
            {
                vec[i] = L[i].toFloat(&bOK);
                Q_ASSERT(bOK);
            }
        }
        else
        {
            vec.resize(3);
        }
        return QVariant::fromValue(vec);
    }
    case CONTROL_CURVE:
    {
        if (defaultValue.isEmpty())
        {
            return QVariant::fromValue(curve_util::deflCurves());
        }
    }
    default:
        return defaultValue;
    };
}

QVariant UiHelper::parseTextValue(PARAM_CONTROL editCtrl, const QString& textValue)
{
	QVariant varValue;
	switch (editCtrl) {
	case CONTROL_INT: varValue = textValue.isEmpty() ? 0 : std::stoi(textValue.toStdString()); break;
	case CONTROL_FLOAT: varValue = textValue.isEmpty() ? 0.0 : std::stod(textValue.toStdString()); break;
	case CONTROL_BOOL: varValue = textValue.isEmpty() ? false : textValue == "true" ? true :
		(textValue == "false" ? false : (bool)std::stoi(textValue.toStdString())); break;
	case CONTROL_READPATH:
	case CONTROL_WRITEPATH:
	case CONTROL_MULTILINE_STRING:
    case CONTROL_COLOR:
    case CONTROL_CURVE:
    case CONTROL_ENUM:
	case CONTROL_STRING: varValue = textValue; break;
	}
    return varValue;
}

QSizeF UiHelper::viewItemTextLayout(QTextLayout& textLayout, int lineWidth, int maxHeight, int* lastVisibleLine)
{
	if (lastVisibleLine)
		*lastVisibleLine = -1;
	qreal height = 0;
	qreal widthUsed = 0;
	textLayout.beginLayout();
	int i = 0;
	while (true) {
		QTextLine line = textLayout.createLine();
		if (!line.isValid())
			break;
		line.setLineWidth(lineWidth);
		line.setPosition(QPointF(0, height));
		height += line.height();
		widthUsed = qMax(widthUsed, line.naturalTextWidth());
		// we assume that the height of the next line is the same as the current one
		if (maxHeight > 0 && lastVisibleLine && height + line.height() > maxHeight) {
			const QTextLine nextLine = textLayout.createLine();
			*lastVisibleLine = nextLine.isValid() ? i : -1;
			break;
		}
		++i;
	}
	textLayout.endLayout();
	return QSizeF(widthUsed, height);
}

QString UiHelper::generateUuid(const QString& name)
{
    QUuid uuid = QUuid::createUuid();
    return QString::number(uuid.data1, 16) + "-" + name;
}

uint UiHelper::generateUuidInt()
{
    QUuid uuid = QUuid::createUuid();
    return uuid.data1;
}

bool UiHelper::parseVecType(const QString& type, int& dim, bool& bFloat)
{
    static QRegExp rx("vec(2|3|4)(i|f)?");
    bool ret = rx.exactMatch(type);
    if (!ret) return false;

    rx.indexIn(type);
    QStringList list = rx.capturedTexts();
    if (list.length() == 3)
    {
        dim = list[1].toInt();
        bFloat = list[2] != 'i';
        return true;
    }
    else
    {
        return false;
    }
}

QString UiHelper::getControlDesc(PARAM_CONTROL ctrl)
{
    switch (ctrl)
    {
    case CONTROL_INT:               return "Integer";
    case CONTROL_FLOAT:             return "Float";
    case CONTROL_STRING:            return "String";
    case CONTROL_BOOL:              return "Boolean";
    case CONTROL_MULTILINE_STRING:  return "Multiline String";
    case CONTROL_READPATH:          return "read path";
    case CONTROL_WRITEPATH:         return "write path";
    case CONTROL_ENUM:              return "Enum";
    case CONTROL_VEC4_FLOAT:        return "Float Vector 4";
    case CONTROL_VEC3_FLOAT:        return "Float Vector 3";
    case CONTROL_VEC2_FLOAT:        return "Float Vector 2";
    case CONTROL_VEC4_INT:          return "Integer Vector 4";
    case CONTROL_VEC3_INT:          return "Integer Vector 3";
    case CONTROL_VEC2_INT:          return "Integer Vector 2";
    case CONTROL_COLOR:             return "Color";
    case CONTROL_CURVE:             return "Curve";
    case CONTROL_HSPINBOX:          return "SpinBox";
    case CONTROL_HDOUBLESPINBOX: return "DoubleSpinBox";
    case CONTROL_HSLIDER:           return "Slider";
    case CONTROL_SPINBOX_SLIDER:    return "SpinBoxSlider";
    case CONTROL_DICTPANEL:         return "Dict Panel";
    case CONTROL_GROUP_LINE:             return "group-line";
    default:
        return "";
    }
}

PARAM_CONTROL UiHelper::getControlByDesc(const QString& descName)
{
    if (descName == "Integer")
    {
        return CONTROL_INT;
    }
    else if (descName == "Float")
    {
        return CONTROL_FLOAT;
    }
    else if (descName == "String")
    {
        return CONTROL_STRING;
    }
    else if (descName == "Boolean")
    {
        return CONTROL_BOOL;
    }
    else if (descName == "Multiline String")
    {
        return CONTROL_MULTILINE_STRING;
    }
    else if (descName == "read path")
    {
        return CONTROL_READPATH;
    }
    else if (descName == "write path")
    {
        return CONTROL_WRITEPATH;
    }
    else if (descName == "Enum")
    {
        return CONTROL_ENUM;
    }
    else if (descName == "Float Vector 4")
    {
        return CONTROL_VEC4_FLOAT;
    }
    else if (descName == "Float Vector 3")
    {
        return CONTROL_VEC3_FLOAT;
    }
    else if (descName == "Float Vector 2")
    {
        return CONTROL_VEC2_FLOAT;
    }
    else if (descName == "Integer Vector 4")
    {
        return CONTROL_VEC4_INT;
    }
    else if (descName == "Integer Vector 3")
    {
        return CONTROL_VEC3_INT;
    }
    else if (descName == "Integer Vector 2")
    {
        return CONTROL_VEC2_INT;
    }
    else if (descName == "Color")
    {
        return CONTROL_COLOR;
    }
    else if (descName == "Curve")
    {
        return CONTROL_CURVE;
    }
    else if (descName == "SpinBox")
    {
        return CONTROL_HSPINBOX;
    } 
    else if (descName == "DoubleSpinBox") 
    {
        return CONTROL_HDOUBLESPINBOX;
    }
    else if (descName == "Slider")
    {
        return CONTROL_HSLIDER;
    }
    else if (descName == "SpinBoxSlider")
    {
        return CONTROL_SPINBOX_SLIDER;
    }
    else if (descName == "Dict Panel")
    {
        return CONTROL_DICTPANEL;
    }
    else if (descName == "group-line")
    {
        return CONTROL_GROUP_LINE;
    }
    else
    {
        return CONTROL_NONE;
    }
}

QStringList UiHelper::getCoreTypeList()
{
    static QStringList types = {
        "",
        "int",
        "bool",
        "float",
        "string",
        "vec2f",
        "vec2i",
        "vec3f",
        "vec3i",
        "vec4f",
        "vec4i",
        //"writepath",
        //"readpath",
        "color",
        "curve",
        "list",
        "dict"
    };
    return types;
}

QStringList UiHelper::getAllControls()
{
    return { "Integer", "Float", "String", "Boolean", "Multiline String", "read path", "write path", "Enum",
        "Float Vector 4", "Float Vector 3", "Float Vector 2","Integer Vector 4", "Integer Vector 3",
        "Integer Vector 2", "Color", "Curve", "SpinBox", "DoubleSpinBox", "Slider", "SpinBoxSlider" };
}

QStringList UiHelper::getControlLists(const QString& type, bool isNodeUI)
{
    QList<PARAM_CONTROL> ctrls;
    if (type == "int") 
    { 
        if (isNodeUI)
            ctrls = {CONTROL_INT, CONTROL_HSPINBOX}; 
        else
            ctrls = {CONTROL_INT, CONTROL_HSPINBOX, CONTROL_HSLIDER, CONTROL_SPINBOX_SLIDER}; 
    }
    else if (type == "bool") { ctrls = { CONTROL_BOOL }; }
    else if (type == "float") { ctrls = { CONTROL_FLOAT, CONTROL_HDOUBLESPINBOX}; }    //todo: slider/spinbox for float.
    else if (type == "string") { ctrls = { CONTROL_STRING, CONTROL_MULTILINE_STRING, CONTROL_ENUM}; }
    else if (type == "vec2f") { ctrls = { CONTROL_VEC2_FLOAT }; }
    else if (type == "vec2i") { ctrls = { CONTROL_VEC2_INT }; }
    else if (type == "vec3f") { ctrls = { CONTROL_VEC3_FLOAT }; }
    else if (type == "vec3i") { ctrls = { CONTROL_VEC3_INT }; }
    else if (type == "vec4f") { ctrls = { CONTROL_VEC4_FLOAT }; }
    else if (type == "vec4i") { ctrls = { CONTROL_VEC4_INT }; }
    else if (type == "writepath") { ctrls = { CONTROL_WRITEPATH }; }
    else if (type == "readpath") { ctrls = { CONTROL_READPATH }; }
    else if (type == "multiline_string") { ctrls = { CONTROL_STRING, CONTROL_MULTILINE_STRING }; }
    else if (type == "color") {   //color is more general than heatmap.
        ctrls = { CONTROL_COLOR };
    }
    else if (type == "curve") { ctrls = { CONTROL_CURVE }; }
    else if (type.startsWith("enum ")) {
        ctrls = { CONTROL_ENUM }; //todo
    }
    else if (type == "NumericObject") { ctrls = { CONTROL_FLOAT }; }
    else { ctrls = { CONTROL_NONE }; }

    QStringList ctrlNames;
    for (PARAM_CONTROL ctrl : ctrls)
    {
        ctrlNames.append(getControlDesc(ctrl));
    }
    return ctrlNames;
}


PARAM_CONTROL UiHelper::getControlByType(const QString &type)
{
    if (type.isEmpty()) {
        return CONTROL_NONE;
    } else if (type == "int") {
        return CONTROL_INT;
    } else if (type == "bool") {
        return CONTROL_BOOL;
    } else if (type == "float") {
        return CONTROL_FLOAT;
    } else if (type == "string") {
        return CONTROL_STRING;
    } else if (type.startsWith("vec")) {
        // support legacy type "vec3"
        int dim = 0;
        bool bFloat = false;
        if (parseVecType(type, dim, bFloat)) {
            switch (dim)
            {
            case 2: return bFloat ? CONTROL_VEC2_FLOAT : CONTROL_VEC2_INT;
            case 3: return bFloat ? CONTROL_VEC3_FLOAT : CONTROL_VEC3_INT;
            case 4: return bFloat ? CONTROL_VEC4_FLOAT : CONTROL_VEC4_INT;
            default:
                return CONTROL_NONE;
            }
        }
        else {
            return CONTROL_NONE;
        }
    } else if (type == "writepath") {
        return CONTROL_WRITEPATH;
    } else if (type == "readpath") {
        return CONTROL_READPATH;
    } else if (type == "multiline_string") {
        return CONTROL_MULTILINE_STRING;
    } else if (type == "color") {   //color is more general than heatmap.
        return CONTROL_COLOR;
    } else if (type == "curve") {
        return CONTROL_CURVE;
    } else if (type.startsWith("enum ")) {
        return CONTROL_ENUM;
    } else if (type == "NumericObject") {
        return CONTROL_FLOAT;
    } else if (type.isEmpty()) {
        return CONTROL_NONE;
    }
    else if (type == "dict")
    {
        //control by multilink socket property. see SOCKET_PROPERTY
        return CONTROL_NONE;
    } else if (type == "group-line") {
        return CONTROL_GROUP_LINE;
    }
    else {
        zeno::log_trace("parse got undefined control type {}", type.toStdString());
        return CONTROL_NONE;
    }
}

CONTROL_INFO UiHelper::getControlByType(const QString &nodeCls, PARAM_CLASS cls, const QString &socketName, const QString &socketType) 
{
    return GlobalControlMgr::instance().controlInfo(nodeCls, cls, socketName, socketType);
}

QString UiHelper::getTypeByControl(PARAM_CONTROL ctrl)
{
    switch (ctrl) {
    case CONTROL_INT: return "int";
    case CONTROL_FLOAT: 
    case CONTROL_HDOUBLESPINBOX:
        return "float";
    case CONTROL_BOOL:  return "bool";
    case CONTROL_MULTILINE_STRING:
    case CONTROL_STRING: return "string";
    case CONTROL_VEC2_FLOAT: return "vec2f";
    case CONTROL_VEC2_INT: return "vec2i";
    case CONTROL_VEC3_FLOAT: return "vec3f";
    case CONTROL_VEC3_INT: return "vec3i";
    case CONTROL_VEC4_FLOAT:    return "vec4f";
    case CONTROL_VEC4_INT: return "vec4i";
    case CONTROL_WRITEPATH: return "string";
    case CONTROL_READPATH: return "string";
    case CONTROL_COLOR: return "color";     //todo: is vec3?
    case CONTROL_CURVE: return "curve";
    case CONTROL_ENUM: return "string";
    case CONTROL_HSLIDER:
    case CONTROL_HSPINBOX:
    case CONTROL_SPINBOX_SLIDER:
         return "int";
    default:
        return "";
    }
}

void UiHelper::getSocketInfo(const QString& objPath,
                             QString& subgName,
                             QString& nodeIdent,
                             QString& paramPath)
{
    //see GraphsModel::indexFromPath
    QStringList lst = objPath.split(cPathSeperator, QtSkipEmptyParts);
    //format like: [subgraph-name]:[node-ident]:[node|panel]/[param-layer-path]/[dict-key]
    //example: main:xxxxx-wrangle:[node]inputs/params/key1
    if (lst.size() >= 3)
    {
        subgName = lst[0];
        nodeIdent = lst[1];
        paramPath = lst[2];
    }
}

QString UiHelper::constructObjPath(const QString& subgraph, const QString& node, const QString& group, const QString& sockName)
{
    QStringList seq = {subgraph, node, group + sockName};
    return seq.join(cPathSeperator);
}

QString UiHelper::constructObjPath(const QString& subgraph, const QString& node, const QString& paramPath)
{
    QStringList seq = {subgraph, node, paramPath};
    return seq.join(cPathSeperator);
}

QString UiHelper::getSockNode(const QString& sockPath)
{
    QStringList lst = sockPath.split(cPathSeperator, QtSkipEmptyParts);
    if (lst.size() > 1)
        return lst[1];
    return "";
}

QString UiHelper::getParamPath(const QString& sockPath)
{
    QStringList lst = sockPath.split(cPathSeperator, QtSkipEmptyParts);
    if (lst.size() > 2)
        return lst[2];
    return "";
}

QString UiHelper::getSockName(const QString& sockPath)
{
    QStringList lst = sockPath.split(cPathSeperator, QtSkipEmptyParts);
    if (lst.size() > 2)
    {
        lst = lst[2].split("/", QtSkipEmptyParts);
        if (!lst.isEmpty())
        {
            //format: main:xxxxx-wrangle:[node]inputs/params/key1
            if (lst.size() == 4)
            {
                return lst[2] + "/" + lst[3];
            }
            else
            {
                return lst.last();
            }
        }
    }
    return "";
}

QString UiHelper::getSockSubgraph(const QString& sockPath)
{
    QStringList lst = sockPath.split(cPathSeperator, QtSkipEmptyParts);
    if (lst.size() > 0)
        return lst[0];
    return "";
}

QString UiHelper::variantToString(const QVariant& var)
{
	QString value;
	if (var.type() == QVariant::String)
	{
		value = var.toString();
	}
	else if (var.type() == QVariant::Double)
	{
		value = QString::number(var.toDouble());
	}
    else if (var.type() == QMetaType::Float)
    {
        value = QString::number(var.toFloat());
    }
	else if (var.type() == QVariant::Int)
	{
		value = QString::number(var.toInt());
	}
	else if (var.type() == QVariant::Bool)
	{
		value = var.toBool() ? "true" : "false";
	}
	else if (var.type() == QVariant::Invalid)
	{
		zeno::log_debug("got null qt variant");
		value = "";
	}
	else if (var.type() == QVariant::Bool)
	{
		value = var.toBool() ? "true" : "false";
	}
	else if (var.type() == QVariant::UserType)
    {
        UI_VECTYPE vec = var.value<UI_VECTYPE>();
        if (vec.isEmpty()) {
            zeno::log_warn("unexpected qt variant {}", var.typeName());
        } else {
            QString res;
            for (int i = 0; i < vec.size(); i++) {
                res.append(QString::number(vec[i]));
                if (i < vec.size() - 1)
                    res.append(",");
            }
            return res;
        }
    }
	else
    {
        zeno::log_warn("bad qt variant {}", var.typeName());
    }

    return value;
}

float UiHelper::parseNumeric(const QVariant& val, bool castStr, bool& bSucceed)
{
    float num = 0;
    QVariant::Type type = val.type();
    if (type == QMetaType::Float || type == QVariant::Double || type == QVariant::Int)
    {
        num = val.toFloat(&bSucceed);
    }
    else if (castStr && type == QVariant::String)
    {
        num = val.toString().toFloat(&bSucceed);
    }
    return num;
}

QVariant UiHelper::initVariantByControl(PARAM_CONTROL ctrl)
{
    switch (ctrl)
    {
        case CONTROL_INT:
        case CONTROL_FLOAT:
        case CONTROL_HSLIDER:
        case CONTROL_HSPINBOX:
        case CONTROL_HDOUBLESPINBOX:
        case CONTROL_SPINBOX_SLIDER:
            return 0;
        case CONTROL_BOOL:
            return false;
        case CONTROL_ENUM:
        case CONTROL_WRITEPATH:
        case CONTROL_READPATH:
        case CONTROL_MULTILINE_STRING:
        case CONTROL_STRING:
            return "";
        case CONTROL_COLOR:
        {
            return QVariant::fromValue(QLinearGradient());
        }
        case CONTROL_CURVE:
        {
            return QVariant::fromValue(curve_util::deflCurves());
        }
        case CONTROL_VEC4_FLOAT:
        case CONTROL_VEC4_INT:
        {
            UI_VECTYPE vec(4);
            return QVariant::fromValue(vec);
        }
        case CONTROL_VEC3_FLOAT:
        case CONTROL_VEC3_INT:
        {
            UI_VECTYPE vec(3);
            return QVariant::fromValue(vec);
        }
        case CONTROL_VEC2_FLOAT:
        case CONTROL_VEC2_INT:
        {
            UI_VECTYPE vec(2);
            return QVariant::fromValue(vec);
        }
        default:
        {
            return QVariant();
        }
    }
}

float UiHelper::parseJsonNumeric(const rapidjson::Value& val, bool castStr, bool& bSucceed)
{
    float num = 0;
    if (val.IsFloat())
    {
        num = val.GetFloat();
        bSucceed = true;
    }
    else if (val.IsDouble())
    {
        num = val.GetDouble();
        bSucceed = true;
    }
    else if (val.IsInt())
    {
        num = val.GetInt();
        bSucceed = true;
    }
    else if (val.IsString() && castStr)
    {
        QString numStr(val.GetString());
        num = numStr.toFloat(&bSucceed);    //may be empty string, no need to assert.
    }
    else
    {
        ZASSERT_EXIT(false, 0.0);
        bSucceed = false;
    }
    return num;
}

QPointF UiHelper::parsePoint(const rapidjson::Value& ptObj, bool& bSucceed)
{
    QPointF pt;

    RAPIDJSON_ASSERT(ptObj.IsArray());
    const auto &arr_ = ptObj.GetArray();
    RAPIDJSON_ASSERT(arr_.Size() == 2);

    const auto &xObj = arr_[0];
    pt.setX(UiHelper::parseJsonNumeric(xObj, false, bSucceed));
    RAPIDJSON_ASSERT(bSucceed);
    if (!bSucceed)
        return pt;

    const auto &yObj = arr_[1];
    pt.setY(UiHelper::parseJsonNumeric(yObj, false, bSucceed));
    RAPIDJSON_ASSERT(bSucceed);
    if (!bSucceed)
        return pt;

    return pt;
}

NODE_TYPE UiHelper::nodeType(const QString& name)
{
    if (name == "Blackboard")
    {
        return BLACKBOARD_NODE;
    }
    else if (name == "Group")
    {
        return GROUP_NODE;
    }
    else if (name == "SubInput")
    {
        return SUBINPUT_NODE;
    }
    else if (name == "SubOutput")
    {
        return SUBOUTPUT_NODE;
    }
    else if (name == "MakeHeatmap")
    {
        return HEATMAP_NODE;
    }
    else
    {
        return NORMAL_NODE;
    }
}

int UiHelper::getMaxObjId(const QList<QString> &lst)
{
    int maxObjId = -1;
    for (QString key : lst)
    {
        QRegExp rx("obj(\\d+)");
        if (rx.indexIn(key) != -1)
        {
            auto caps = rx.capturedTexts();
            if (caps.length() == 2) {
                int id = caps[1].toInt();
                maxObjId = qMax(maxObjId, id);
            }
        }
    }
    return maxObjId;
}

QString UiHelper::getUniqueName(const QList<QString>& existNames, const QString& prefix, bool bWithBrackets)
{
    int n = 0;
    QString name;
    do
    {
        if (bWithBrackets)
            name = prefix + "(" + QString::number(n++) + ")";
        else
            name = prefix + QString::number(n++);
    } while (existNames.contains(name));
    return name;
}

std::pair<qreal, qreal> UiHelper::getRxx2(QRectF r, qreal xRadius, qreal yRadius, bool AbsoluteSize)
{
    if (AbsoluteSize) {
        qreal w = r.width() / 2;
        qreal h = r.height() / 2;

        if (w == 0) {
            xRadius = 0;
        } else {
            xRadius = 100 * qMin(xRadius, w) / w;
        }
        if (h == 0) {
            yRadius = 0;
        } else {
            yRadius = 100 * qMin(yRadius, h) / h;
        }
    } else {
        if (xRadius > 100)// fix ranges
            xRadius = 100;

        if (yRadius > 100)
            yRadius = 100;
    }

    qreal w = r.width();
    qreal h = r.height();
    qreal rxx2 = w * xRadius / 100;
    qreal ryy2 = h * yRadius / 100;
    return std::make_pair(rxx2, ryy2);
}

QPainterPath UiHelper::getRoundPath(QRectF r, int lt_radius, int rt_radius, int lb_radius, int rb_radius, bool bFixRadius) {
    QPainterPath path;
    if (r.isNull())
        return path;

    if (lt_radius <= 0 && rt_radius <= 0 && lb_radius <= 0 && rb_radius <= 0) {
        path.addRect(r);
        return path;
    }

    qreal x = r.x();
    qreal y = r.y();
    qreal w = r.width();
    qreal h = r.height();

    auto pair = getRxx2(r, lt_radius, lt_radius, bFixRadius);
    qreal rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.moveTo(x, y);
    } else {
        path.arcMoveTo(x, y, rxx2, ryy2, 180);
        path.arcTo(x, y, rxx2, ryy2, 180, -90);
    }

    pair = getRxx2(r, rt_radius, rt_radius, bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y);
    } else {
        path.arcTo(x + w - rxx2, y, rxx2, ryy2, 90, -90);
    }

    pair = getRxx2(r, rb_radius, rb_radius, bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y + h);
    } else {
        path.arcTo(x + w - rxx2, y + h - rxx2, rxx2, ryy2, 0, -90);
    }

    pair = getRxx2(r, lb_radius, lb_radius, bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x, y + h);
    } else {
        path.arcTo(x, y + h - rxx2, rxx2, ryy2, 270, -90);
    }

    path.closeSubpath();
    return path;
}

QVector<qreal> UiHelper::getSlideStep(const QString& name, PARAM_CONTROL ctrl)
{
    QVector<qreal> steps;
    if (ctrl == CONTROL_INT)
    {
        steps = { 1, 10, 100 };
    }
    else if (ctrl == CONTROL_FLOAT)
    {
        steps = { .0001, .001, .01, .1, 1, 10, 100 };
    }
    return steps;
}

QString UiHelper::nthSerialNumName(QString name)
{
    QRegExp rx("\\((\\d+)\\)");
    int idx = rx.lastIndexIn(name);
    if (idx == -1) {
        return name + "(1)";
    }
    else {
        name = name.mid(0, idx);
        QStringList lst = rx.capturedTexts();
        ZASSERT_EXIT(lst.size() == 2, "");
        bool bConvert = false;
        int ith = lst[1].toInt(&bConvert);
        ZASSERT_EXIT(bConvert, "");
        return name + "(" + QString::number(ith + 1) + ")";
    }
}

QString UiHelper::correctSubIOName(IGraphsModel* pModel, const QString& subgName, const QString& newName, bool bInput)
{
    ZASSERT_EXIT(pModel, "");

    NODE_DESCS descs = pModel->descriptors();
    if (descs.find(subgName) == descs.end())
        return "";

    const NODE_DESC& desc = descs[subgName];
    QString finalName = newName;
    int i = 1;
    if (bInput)
    {
        while (desc.inputs.find(finalName) != desc.inputs.end())
        {
            finalName = finalName + QString("_%1").arg(i);
            i++;
        }
    }
    else
    {
        while (desc.outputs.find(finalName) != desc.outputs.end())
        {
            finalName = finalName + QString("_%1").arg(i);
            i++;
        }
    }
    return finalName;
}

QVariant UiHelper::parseVarByType(const QString& descType, const QVariant& var, QObject* parentRef)
{
    const QVariant::Type varType = var.type();
    if (descType == "int" ||
        descType == "float" ||
        descType == "NumericObject" ||
        descType == "numeric:float" ||
        descType == "floatslider")
    {
        switch (varType)
        {
            case QVariant::Int:
            case QMetaType::Float:
            case QVariant::Double:
            case QVariant::UInt:
            case QVariant::LongLong:
            case QVariant::ULongLong: return var;
            case QVariant::String:
            {
                //string numeric, try to parse to numeric.
                bool bOk = false;
                float fVal = var.toString().toFloat(&bOk);
                if (bOk)
                    return QVariant(fVal);
            }
            default:
                return QVariant();
        }
    }
    else if ((descType == "string" ||
              descType == "writepath" ||
              descType == "readpath" ||
              descType == "multiline_string" ||
              descType.startsWith("enum ")))
    {
        if (varType == QVariant::String)
            return var;
        else
            return QVariant();
    }
    else if (descType == "bool")
    {
        if (varType == QVariant::Int)
        {
            return var.toInt() != 0;
        }
        else if (varType == QMetaType::Float)
        {
            return var.toFloat() != 0;
        }
        else if (varType == QVariant::Double)
        {
            return var.toDouble() != 0;
        }
        else if (varType == QVariant::Bool)
        {
            return var;
        }
        else if (varType == QVariant::String)
        {
            QString boolStr = var.toString();
            if (boolStr == "true")
                return true;
            if (boolStr == "false")
                return false;
        }
        return QVariant();
    }
    else if (descType.startsWith("vec") && varType == QVariant::UserType &&
             var.userType() == QMetaTypeId<UI_VECTYPE>::qt_metatype_id())
    {
        if (varType == QVariant::UserType && var.userType() == QMetaTypeId<UI_VECTYPE>::qt_metatype_id())
        {
            return var;
        }
        else if (varType == QVariant::String)
        {
            auto lst = var.toString().split(",");
            if (lst.isEmpty())
                return QVariant();
            UI_VECTYPE vec;
            for (int i = 0; i < lst.size(); i++)
            {
                QString str = lst[i];
                bool bOk = false;
                float fVal = str.toFloat(&bOk);
                if (!bOk)
                    return QVariant();
                vec.append(fVal);
            }
            return QVariant::fromValue(vec);
        }
        return QVariant();
    }
    else if (descType == "curve")
    {
        if (varType == QMetaType::User)
        {
            return var;
        }
        //legacy curve is expressed as string, and no desc type associated with it.
        return QVariant();
    }

    //string:
    if (varType == QVariant::String)
    {
        QString str = var.toString();
        if (str.isEmpty()) {
            // the default value of many types, for example primitive, are empty string,
            // skip it and return a invalid variant.
            return QVariant();
        }
        //try to convert to numeric.
        bool bOk = false;
        float fVal = str.toFloat(&bOk);
        if (bOk)
            return fVal;
    }
    //unregister type or unknown data, return itself.
    return var;
}

QVariant UiHelper::parseJsonByType(const QString& descType, const rapidjson::Value& val, QObject* parentRef)
{
    QVariant res;
    auto jsonType = val.GetType();
    if (descType == "int")
    {
        bool bSucc = false;
        int iVal = parseJsonNumeric(val, true, bSucc);
        if (!bSucc)
            return QVariant();  //will not be serialized when return null variant.
        res = iVal;
    }
    else if (descType == "float" ||
             descType == "NumericObject")
    {
        bool bSucc = false;
        float fVal = parseJsonNumeric(val, true, bSucc);
        if (!bSucc)
            return QVariant();
        res = fVal;
    }
    else if (descType == "string" ||
             descType == "writepath"||
             descType == "readpath" ||
             descType == "multiline_string" ||
             descType.startsWith("enum "))
    {
        if (val.IsString())
            res = val.GetString();
        else
            return QVariant();
    }
    else if (descType == "bool")
    {
        if (val.IsBool())
            res = val.GetBool();
        else if (val.IsInt())
            res = val.GetInt() != 0;
        else if (val.IsFloat())
            res = val.GetFloat() != 0;
        else
            return QVariant();
    }
    else if (descType.startsWith("vec"))
    {
        int dim = 0;
        bool bFloat = false;
        if (UiHelper::parseVecType(descType, dim, bFloat))
        {
            res = QVariant::fromValue(UI_VECTYPE(dim, 0));
            UI_VECTYPE vec;
            if (val.IsArray())
            {
                auto values = val.GetArray();
                for (int i = 0; i < values.Size(); i++)
                {
                    vec.append(values[i].GetFloat());
                }
            }
            else if (val.IsString())
            {
                const QString& vecStr = val.GetString();
                QStringList lst = vecStr.split(",");
                for (int i = 0; i < lst.size(); i++)
                {
                    bool bSucc = false;
                    if (lst[i].isEmpty()) {
                        vec.append(0);
                    }
                    else {
                        float fVal = lst[i].toFloat(&bSucc);
                        if (!bSucc)
                            return QVariant();
                        vec.append(fVal);
                    }
                }
            }
            res = QVariant::fromValue(vec);
        }
        else
        {
            return QVariant();
        }
    } else if (descType == "curve") {
        ZASSERT_EXIT(val.HasMember(key_objectType), QVariant());
        QString type = val[key_objectType].GetString();
        if (type != "curve") {
            return QVariant();
        }
        CURVES_DATA curves;
        if (!val.HasMember("x") && !val.HasMember("y") && !val.HasMember("z") && val.HasMember(key_range)) { //compatible old version zsg file
            CURVE_DATA xCurve = JsonHelper::parseCurve("x", val);
            curves.insert("x", xCurve);
        } else {
            bool timeLine = false;
            if (val.HasMember(key_timeline))
            {
                timeLine = val[key_timeline].GetBool();
            }
            for (auto i = val.MemberBegin(); i != val.MemberEnd(); i++) {
                if (i->value.IsObject()) {
                    CURVE_DATA curve = JsonHelper::parseCurve(i->name.GetString(), i->value);
                    curve.timeline = timeLine;
                    curves.insert(i->name.GetString(), curve);
                }
            }
        }
        res = QVariant::fromValue(curves);
    }
    else
    {
        // omitted or legacy type, need to parse by json value.
        if (val.IsString() && val.GetStringLength() == 0)
        {
            // the default value of many types, for example primitive, are empty string,
            // skip it and return a invalid variant.
            return QVariant();
        }
        // unregisted or new type, convert by json value.
        return parseJsonByValue(descType, val, parentRef);
    }
    return res;
}

QVariant UiHelper::parseJsonByValue(const QString& type, const rapidjson::Value& val, QObject* parentRef)
{
    if (val.GetType() == rapidjson::kStringType)
    {
        bool bSucc = false;
        float fVal = parseJsonNumeric(val, true, bSucc);
        if (bSucc)
            return fVal;
        return val.GetString();
    }
    else if (val.GetType() == rapidjson::kNumberType)
    {
        if (val.IsDouble())
            return val.GetDouble();
        else if (val.IsInt())
            return val.GetInt();
        else {
            zeno::log_warn("bad rapidjson number type {}", val.GetType());
            return QVariant();
        }
    }
    else if (val.GetType() == rapidjson::kTrueType)
    {
        return val.GetBool();
    }
    else if (val.GetType() == rapidjson::kFalseType)
    {
        return val.GetBool();
    }
    else if (val.GetType() == rapidjson::kNullType)
    {
        return QVariant();
    }
    else if (val.GetType() == rapidjson::kArrayType)
    {
        UI_VECTYPE vec;
        auto values = val.GetArray();
        for (int i = 0; i < values.Size(); i++)
        {
            vec.append(values[i].GetFloat());
        }
        return QVariant::fromValue(vec);
    }
    else if (val.GetType() == rapidjson::kObjectType)
    {
        if (type == "curve")
        {
            ZASSERT_EXIT(val.HasMember(key_objectType), QVariant());
            QString type = val[key_objectType].GetString();
            if (type != "curve") {
                return QVariant();
            }
            CURVES_DATA curves;
            if (!val.HasMember("x") && !val.HasMember("y") && !val.HasMember("z") && val.HasMember(key_range)) { //compatible old version zsg file
                CURVE_DATA xCurve = JsonHelper::parseCurve("x", val);
                curves.insert("x", xCurve);
            } else {
                for (auto i = val.MemberBegin(); i != val.MemberEnd(); i++) {
                    if (i->value.IsObject()) {
                        CURVE_DATA curve = JsonHelper::parseCurve(i->name.GetString(), i->value);
                        //pModel->setTimeline(val[key_timeline].GetBool()); //todo: timeline
                        curves.insert(i->name.GetString(), curve);
                    }
                }
            }
            return QVariant::fromValue(curves);
        }
    }

    zeno::log_warn("bad rapidjson value type {}", val.GetType());
    return QVariant();
}

QVariant UiHelper::parseJson(const rapidjson::Value& val, QObject* parentRef)
{
    if (val.GetType() == rapidjson::kStringType)
    {
        bool bSucc = false;
        float fVal = parseJsonNumeric(val, true, bSucc);
        if (bSucc)
            return fVal;
        return val.GetString();
    }
    else if (val.GetType() == rapidjson::kNumberType)
    {
        if (val.IsDouble())
            return val.GetDouble();
        else if (val.IsInt())
            return val.GetInt();
        else {
            zeno::log_warn("bad rapidjson number type {}", val.GetType());
            return QVariant();
        }
    }
    else if (val.GetType() == rapidjson::kTrueType)
    {
        return val.GetBool();
    }
    else if (val.GetType() == rapidjson::kFalseType)
    {
        return val.GetBool();
    }
    else if (val.GetType() == rapidjson::kNullType)
    {
        return QVariant();
    }
    else if (val.GetType() == rapidjson::kArrayType)
    {
        //detect whether it is a numeric vector.
        auto values = val.GetArray();
        bool bNumeric = true;
        for (int i = 0; i < values.Size(); i++)
        {
            if (!values[i].IsNumber())
            {
                bNumeric = false;
                break;
            }
        }
        if (bNumeric)
        {
            UI_VECTYPE vec;
            for (int i = 0; i < values.Size(); i++)
            {
                const auto& numObj = values[i];
                if (numObj.IsInt() || numObj.IsInt64() || numObj.IsUint() || numObj.IsUint64())
                    vec.append(values[i].GetInt());
                else if (numObj.IsFloat() || numObj.IsDouble())
                    vec.append(values[i].GetFloat());
            }
            return QVariant::fromValue(vec);
        }
        else
        {
            QStringList lst;
            for (int i = 0; i < values.Size(); i++)
            {
                const auto& obj = values[i];
                if (obj.IsNumber()) {
                    lst.append(QString::number(obj.GetFloat()));
                }
                else if (obj.IsString()) {
                    lst.append(QString::fromUtf8(obj.GetString()));
                }
            }
            return lst;
        }
    }
    else if (val.GetType() == rapidjson::kObjectType)
    {
        //if (type == "curve")
        //{
        //    CurveModel* pModel = JsonHelper::_parseCurveModel(val, parentRef);
        //    return QVariantPtr<CurveModel>::asVariant(pModel);
        //}
    }

    zeno::log_warn("bad rapidjson value type {}", val.GetType());
    return QVariant();
}

QString UiHelper::gradient2colorString(const QLinearGradient& grad)
{
    QString colorStr;
    const QGradientStops &stops = grad.stops();
    colorStr += QString::number(stops.size());
    colorStr += "\n";
    for (QGradientStop grad : stops) {
        colorStr += QString::number(grad.first);
        colorStr += " ";
        colorStr += QString::number(grad.second.redF());
        colorStr += " ";
        colorStr += QString::number(grad.second.greenF());
        colorStr += " ";
        colorStr += QString::number(grad.second.blueF());
        colorStr += "\n";
    }
    return colorStr;
}

QVariant UiHelper::getParamValue(const QModelIndex& idx, const QString& name)
{
    PARAMS_INFO params = idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    if (params.find(name) == params.end())
        return QVariant();
    return params[name].value;
}

int UiHelper::tabIndexOfName(const QTabWidget* pTabWidget, const QString& name)
{
    if (!pTabWidget)
        return -1;
    for (int i = 0; i < pTabWidget->count(); i++)
    {
        if (pTabWidget->tabText(i) == name)
        {
            return i;
        }
    }
    return -1;
}

QModelIndex UiHelper::findSubInOutputIdx(IGraphsModel *pModel, bool bSubInput, const QString &paramName,
                                         const QModelIndex &subgIdx)
{
    QModelIndexList nodes = pModel->searchInSubgraph(bSubInput ? "SubInput" : "SubOutput", subgIdx);
    for (QModelIndex subInOutput : nodes)
    {
        NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(subInOutput.data(ROLE_NODE_PARAMS));
        QModelIndex nameIdx = nodeParams->getParam(PARAM_PARAM, "name");
        if (nameIdx.data(ROLE_PARAM_VALUE) == paramName)
        {
            return subInOutput;
        }
    }
    return QModelIndex();
}

void UiHelper::getAllParamsIndex(
        const QModelIndex& nodeIdx,
        QModelIndexList& inputs,
        QModelIndexList& params,
        QModelIndexList& outputs,
        bool bEnsureSRCDST_lastKey)
{
    NodeParamModel* nodeParams = QVariantPtr<NodeParamModel>::asPtr(nodeIdx.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(nodeParams);

    inputs = nodeParams->getInputIndice();
    params = nodeParams->getInputIndice();
    outputs = nodeParams->getInputIndice();

    if (bEnsureSRCDST_lastKey)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            if (inputs[i].data(ROLE_PARAM_NAME).toString() == "SRC")
            {
                QModelIndex tmp = inputs[i];
                inputs.removeAt(i);
                inputs.append(tmp);
                break;
            }
        }
        for (int i = 0; i < outputs.size(); i++)
        {
            if (outputs[i].data(ROLE_PARAM_NAME).toString() == "DST")
            {
                QModelIndex tmp = outputs[i];
                outputs.removeAt(i);
                outputs.append(tmp);
                break;
            }
        }
    }
}

QVector<qreal> UiHelper::scaleFactors()
{
    static QVector<qreal> lst({0.05, .1, .15, .2, .25, .5, .75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0});
    return lst;
}

QPair<NODES_DATA, LINKS_DATA> UiHelper::dumpNodes(const QModelIndexList &nodeIndice,
                                                  const QModelIndexList &linkIndice)
{
    NODES_DATA nodes;
    QList<EdgeInfo> links;

    QSet<QString> existedNodes;
    for (auto idx : nodeIndice)
    {
        existedNodes.insert(idx.data(ROLE_OBJID).toString());
    }

    for (auto idx : linkIndice)
    {
        QModelIndex outSockIdx = idx.data(ROLE_OUTSOCK_IDX).toModelIndex();
        QModelIndex inSockIdx = idx.data(ROLE_INSOCK_IDX).toModelIndex();
        QString outPath = outSockIdx.data(ROLE_OBJPATH).toString();
        QString inPath = inSockIdx.data(ROLE_OBJPATH).toString();
        QString outId = idx.data(ROLE_OUTNODE).toString();
        QString inId = idx.data(ROLE_INNODE).toString();

        if (existedNodes.find(outId) != existedNodes.end() &&
            existedNodes.find(inId) != existedNodes.end())
        {
            links.append(EdgeInfo(outPath, inPath));
        }
    }

    for (auto idx : nodeIndice)
    {
        NODE_DATA node = idx.data(ROLE_OBJDATA).value<NODE_DATA>();
        INPUT_SOCKETS inputs = node[ROLE_INPUTS].value<INPUT_SOCKETS>();
        for (INPUT_SOCKET& inSocket : inputs)
        {
            for (QList<EdgeInfo>::iterator it = inSocket.info.links.begin(); it != inSocket.info.links.end(); )
            {
                if (links.indexOf(*it) == -1)
                {
                    it = inSocket.info.links.erase(it);
                }
                else
                {
                    it++;
                }
            }

            for (DICTKEY_INFO& keyItem : inSocket.info.dictpanel.keys)
            {
                for (QList<EdgeInfo>::iterator it = keyItem.links.begin(); it != keyItem.links.end(); )
                {
                    if (links.indexOf(*it) == -1)
                    {
                        it = keyItem.links.erase(it);
                    }
                    else
                    {
                        it++;
                    }
                }
            }
        }

        OUTPUT_SOCKETS outputs = node[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        for (OUTPUT_SOCKET& outSocket : outputs)
        {
            for (QList<EdgeInfo>::iterator it = outSocket.info.links.begin();
                 it != outSocket.info.links.end();)
            {
                if (links.indexOf(*it) == -1)
                {
                    it = outSocket.info.links.erase(it);
                }
                else
                {
                    it++;
                }
            }

            for (DICTKEY_INFO& keyItem : outSocket.info.dictpanel.keys)
            {
                for (QList<EdgeInfo>::iterator it = keyItem.links.begin(); it != keyItem.links.end(); )
                {
                    if (links.indexOf(*it) == -1)
                    {
                        it = keyItem.links.erase(it);
                    }
                    else
                    {
                        it++;
                    }
                }
            }
        }

        node[ROLE_INPUTS] = QVariant::fromValue(inputs);
        node[ROLE_OUTPUTS] = QVariant::fromValue(outputs);

        const QString& oldId = node[ROLE_OBJID].toString();
        nodes.insert(oldId, node);
    }

    return QPair<NODES_DATA, LINKS_DATA>(nodes, links);
}

void UiHelper::reAllocIdents(const QString& targetSubgraph,
                              const NODES_DATA& inNodes,
                              const LINKS_DATA& inLinks,
                              NODES_DATA& outNodes,
                              LINKS_DATA& outLinks)
{
    QMap<QString, QString> old2new;
    for (QString key : inNodes.keys())
    {
        const NODE_DATA data = inNodes[key];
        const QString& oldId = data[ROLE_OBJID].toString();
        const QString& name = data[ROLE_OBJNAME].toString();
        const QString& newId = UiHelper::generateUuid(name);
        NODE_DATA newData = data;
        newData[ROLE_OBJID] = newId;
        outNodes.insert(newId, newData);
        old2new.insert(oldId, newId);
    }
    //replace all the old-id in newNodes, and clear cached links.
    for (QString newId : outNodes.keys())
    {
        NODE_DATA& data = outNodes[newId];
        INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
        for (INPUT_SOCKET inputSocket : inputs)
        {
            inputSocket.info.nodeid = newId;
            inputSocket.info.links.clear();
            for (DICTKEY_INFO& key : inputSocket.info.dictpanel.keys)
            {
                key.links.clear();
            }
        }

        OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
        for (OUTPUT_SOCKET outputSocket : outputs)
        {
            outputSocket.info.nodeid = newId;
            outputSocket.info.links.clear();
            for (DICTKEY_INFO& key : outputSocket.info.dictpanel.keys)
            {
                key.links.clear();
            }
        }

        data[ROLE_INPUTS] = QVariant::fromValue(inputs);
        data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
    }

    for (const EdgeInfo& link : inLinks)
    {
        QString outputNode = UiHelper::getSockNode(link.outSockPath);
        QString outParamPath = UiHelper::getParamPath(link.outSockPath);
        QString inputNode = UiHelper::getSockNode(link.inSockPath);
        QString inParamPath = UiHelper::getParamPath(link.inSockPath);

        ZASSERT_EXIT(old2new.find(inputNode) != old2new.end() &&
                     old2new.find(outputNode) != old2new.end());
        QString newInputNode = old2new[inputNode];
        QString newOutputNode = old2new[outputNode];

        const QString& newInSock = UiHelper::constructObjPath(targetSubgraph, newInputNode, inParamPath);
        const QString& newOutSock = UiHelper::constructObjPath(targetSubgraph, newOutputNode, outParamPath);

        outLinks.append(EdgeInfo(newOutSock, newInSock));
    }
}

static std::string getZenoVersion()
{
    const char *date = __DATE__;
    const char *table[] = {
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    };
    int month = std::find(table, table + 12, std::string(date, 3)) - table + 1;
    int day = std::stoi(std::string(date + 4, 2));
    int year = std::stoi(std::string(date + 7, 4));
    return zeno::format("{:04d}.{:02d}.{:02d}", year, month, day);
}

QString UiHelper::nativeWindowTitle(const QString& currentFilePath)
{
    QString ver = QString::fromStdString(getZenoVersion());
    if (currentFilePath.isEmpty())
    {
        return QString("Zeno Editor (%1)").arg(ver);
    }
    else
    {
        QString title = QString::fromUtf8("%1 - Zeno Editor (%2)").arg(currentFilePath).arg(ver);
        return title;
    }
}
