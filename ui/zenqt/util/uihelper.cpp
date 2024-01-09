#include "uihelper.h"
#include <zeno/utils/logger.h>
#include "uicommon.h"
#include <zeno/core/data.h>
#include "zassert.h"
#include "model/curvemodel.h"
#include "variantptr.h"
#include "jsonhelper.h"
#include "model/graphmodel.h"
#include "util/curveutil.h"
#include "model/parammodel.h"
#include <QUuid>
#include <zeno/funcs/ParseObjectFromUi.h>
#include "util/globalcontrolmgr.h"

const char* g_setKey = "setKey";

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

QString UiHelper::createNewNode(QModelIndex subgIdx, const QString& descName, const QPointF& pt)
{
    if (!subgIdx.isValid())
        return "";

    zeno::NodeData node;
    //NODE_DATA node = newNodeData(pModel, descName, pt);
    QAbstractItemModel* graphM = const_cast<QAbstractItemModel*>(subgIdx.model());
    if (GraphModel* pModel = qobject_cast<GraphModel*>(graphM))
    {
        node = pModel->createNode(descName, pt);
    }
    return QString::fromStdString(node.ident);
}

QVariant UiHelper::parseTextValue(const zeno::ParamType& type, const QString& textValue)
{
    //TODO
    return QVariant();
}

QVariant UiHelper::initDefaultValue(const zeno::ParamType& type)
{
    if (type == zeno::Param_String) {
        return "";
    }
    else if (type == zeno::Param_Float)
    {
        return QVariant((float)0.);
    }
    else if (type == zeno::Param_Int)
    {
        return QVariant((int)0);
    }
    else if (type == zeno::Param_Bool)
    {
        return QVariant(false);
    }
    /*
    else if (type.startsWith("vec"))
    {
        int dim = 0;
        bool bFloat = false;
        if (UiHelper::parseVecType(type, dim, bFloat))
        {
            return QVariant::fromValue(UI_VECTYPE(dim, 0));
        }
    }
    */
    return QVariant();
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

#if 0
QString UiHelper::getControlDesc(zeno::ParamControl ctrl)
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
    case CONTROL_PURE_COLOR:        return "Pure Color";
    case CONTROL_COLOR_VEC3F:       return "Color Vec3f";
    case CONTROL_CURVE:             return "Curve";
    case CONTROL_HSPINBOX:          return "SpinBox";
    case CONTROL_HDOUBLESPINBOX: return "DoubleSpinBox";
    case CONTROL_HSLIDER:           return "Slider";
    case CONTROL_SPINBOX_SLIDER:    return "SpinBoxSlider";
    case CONTROL_DICTPANEL:         return "Dict Panel";
    case CONTROL_GROUP_LINE:             return "group-line";
    case CONTROL_PYTHON_EDITOR: return "PythonEditor";
    default:
        return "";
    }
}
#endif

zeno::ParamControl UiHelper::getControlByDesc(const QString& descName)
{
    //compatible with zsg2
    if (descName == "Integer")
    {
        return zeno::Lineedit;
    }
    else if (descName == "Float")
    {
        return zeno::Lineedit;
    }
    else if (descName == "String")
    {
        return zeno::Lineedit;
    }
    else if (descName == "Boolean")
    {
        return zeno::Checkbox;
    }
    else if (descName == "Multiline String")
    {
        return zeno::Multiline;
    }
    else if (descName == "read path")
    {
        return zeno::Pathedit;
    }
    else if (descName == "write path")
    {
        return zeno::Pathedit;
    }
    else if (descName == "Enum")
    {
        return zeno::Combobox;
    }
    else if (descName == "Float Vector 4")
    {
        return zeno::Vec4edit;
    }
    else if (descName == "Float Vector 3")
    {
        return zeno::Vec3edit;
    }
    else if (descName == "Float Vector 2")
    {
        return zeno::Vec2edit;
    }
    else if (descName == "Integer Vector 4")
    {
        return zeno::Vec4edit;
    }
    else if (descName == "Integer Vector 3")
    {
        return zeno::Vec3edit;
    }
    else if (descName == "Integer Vector 2")
    {
        return zeno::Vec2edit;
    }
    else if (descName == "Color")
    {
        return zeno::Heatmap;
    } 
    else if (descName == "Pure Color") 
    {
        return zeno::Color;
    }
    else if (descName == "Color Vec3f")
    {
        return zeno::ColorVec;
    }
    else if (descName == "Curve")
    {
        return zeno::CurveEditor;
    }
    else if (descName == "SpinBox")
    {
        return zeno::SpinBox;
    } 
    else if (descName == "DoubleSpinBox") 
    {
        return zeno::DoubleSpinBox;
    }
    else if (descName == "Slider")
    {
        return zeno::Slider;
    }
    else if (descName == "SpinBoxSlider")
    {
        return zeno::SpinBoxSlider;
    }
    else if (descName == "Dict Panel")
    {
        return zeno::NullControl;
    }
    else if (descName == "group-line")
    {
        return zeno::NullControl;
    }
    else if (descName == "PythonEditor")
    {
        return zeno::PythonEditor;
    }
    else
    {
        return zeno::NullControl;
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

QList<zeno::ParamControl> UiHelper::getControlLists(const zeno::ParamType& type)
{
    QList<zeno::ParamControl> ctrls;
    switch (type)
    {
    case zeno::Param_Int:
        return { zeno::Lineedit, zeno::SpinBox, zeno::SpinBoxSlider, zeno::DoubleSpinBox };
    case zeno::Param_Bool:
        return { zeno::Checkbox };
    case zeno::Param_Float:
        return { zeno::Lineedit, zeno::DoubleSpinBox };
    case zeno::Param_String:
        return { zeno::Lineedit, zeno::Multiline, zeno::Combobox };
    case zeno::Param_Vec2i:
    case zeno::Param_Vec2f:
        return { zeno::Vec2edit };
    case zeno::Param_Vec3i:
    case zeno::Param_Vec3f:
        return { zeno::Vec3edit };
    case zeno::Param_Vec4f:
    case zeno::Param_Vec4i:
        return { zeno::Vec4edit };
    case zeno::Param_Curve:
        return { zeno::CurveEditor };
    default:
        return {};
    }
}


zeno::ParamControl UiHelper::getControlByType(const QString &type)
{
    if (type.isEmpty()) {
        return zeno::NullControl;
    } else if (type == "int") {
        return zeno::Lineedit;
    } else if (type == "bool") {
        return zeno::Checkbox;
    } else if (type == "float") {
        return zeno::Lineedit;
    } else if (type == "string") {
        return zeno::Lineedit;
    } else if (type.startsWith("vec")) {
        // support legacy type "vec3"
        int dim = 0;
        bool bFloat = false;
        if (parseVecType(type, dim, bFloat)) {
            switch (dim)
            {
            case 2: return zeno::Vec2edit;
            case 3: return zeno::Vec3edit;
            case 4: return zeno::Vec4edit;
            default:
                return zeno::NullControl;
            }
        }
        else {
            return zeno::NullControl;
        }
    } else if (type == "writepath") {
        return zeno::Pathedit;
    } else if (type == "readpath") {
        return zeno::Pathedit;
    } else if (type == "multiline_string") {
        return zeno::Multiline;
    } else if (type == "color") {   //color is more general than heatmap.
        return zeno::Heatmap;
    } else if (type == "purecolor") {   
        return zeno::Color;
    } else if (type == "colorvec3f") {   //colorvec3f is for coloreditor, color is heatmap? ^^^^
        return zeno::Color;
    } else if (type == "curve") {
        return zeno::CurveEditor;
    } else if (type.startsWith("enum ")) {
        return zeno::Combobox;
    } else if (type == "NumericObject") {
        return zeno::Lineedit;
    } else if (type.isEmpty()) {
        return zeno::NullControl;
    }
    else if (type == "dict")
    {
        //control by multilink socket property. see SOCKET_PROPERTY
        return zeno::NullControl;
    } else if (type == "group-line") {
        return zeno::NullControl;
    }
    else {
        zeno::log_trace("parse got undefined control type {}", type.toStdString());
        return zeno::NullControl;
    }
}

CONTROL_INFO UiHelper::getControlByType(const QString &nodeCls, bool bInput, const QString &socketName, const QString &socketType)
{
    return GlobalControlMgr::instance().controlInfo(nodeCls, bInput, socketName, socketType);
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

QVector<qreal> UiHelper::getSlideStep(const QString& name, zeno::ParamType type)
{
    QVector<qreal> steps;
    if (type == zeno::Param_Int)
    {
        steps = { 1, 10, 100 };
    }
    else if (type == zeno::Param_Float)
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
        if (!bSucc) {
            if (val.IsString()) {
                return val.GetString();
            } else {
                return QVariant(); //will not be serialized when return null variant.
            }
        }
        res = iVal;
    }
    else if (descType == "float" ||
             descType == "NumericObject")
    {
        bool bSucc = false;
        float fVal = parseJsonNumeric(val, true, bSucc);
        if (!bSucc) {
           if (val.IsString()) {
                return val.GetString();
           } else {
                return QVariant();
           }
        }
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
            UI_VECSTRING strVec;
            if (val.IsArray())
            {
                auto values = val.GetArray();
                for (int i = 0; i < values.Size(); i++)
                {
                    if (values[i].IsFloat())
                    {
                        vec.append(values[i].GetFloat());
                    }
                    else if (values[i].IsDouble())
                    {
                        vec.append(values[i].GetDouble());
                    }
                    else if (values[i].IsInt())
                    {
                        vec.append(values[i].GetInt());
                    }
                    else if (values[i].IsString())
                    {
                        strVec.append(values[i].GetString());
                    }
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
            if (!vec.isEmpty())
                res = QVariant::fromValue(vec);
            else
                res = QVariant::fromValue(strVec);
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
    else if (descType == "color" && val.IsString()) 
    {
        if (QColor(val.GetString()).isValid())
            res = QVariant::fromValue(QColor(val.GetString()));
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

        if (type == "color")
            return QVariant::fromValue(QColor(val.GetString()));

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

QVector<qreal> UiHelper::scaleFactors()
{
    static QVector<qreal> lst({0.01, 0.025, 0.05, .1, .15, .2, .25, .5, .75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0});
    return lst;
}

QPair<zeno::NodesData, zeno::LinksData>
    UiHelper::dumpNodes(const QModelIndexList &nodeIndice, const QModelIndexList &linkIndice)
{
    zeno::NodesData nodes;
    zeno::LinksData links;

    QSet<QString> existedNodes;
    for (auto idx : nodeIndice)
    {
        existedNodes.insert(idx.data(ROLE_OBJID).toString());
    }

    for (auto idx : linkIndice)
    {
        QVariantList outInfo = idx.data(ROLE_LINK_FROMPARAM_INFO).toList();
        ZASSERT_EXIT(outInfo.size() == 3, {});

        QVariantList inInfo = idx.data(ROLE_LINK_TOPARAM_INFO).toList();
        ZASSERT_EXIT(inInfo.size() == 3, {});

        QString outId = outInfo[0].toString();
        QString inId = inInfo[0].toString();

        if (existedNodes.find(outId) != existedNodes.end() &&
            existedNodes.find(inId) != existedNodes.end())
        {
            const QString& outParam = outInfo[1].toString();
            const QString& inParam = inInfo[1].toString();
            zeno::EdgeInfo edge = {
                outId.toStdString(),
                outParam.toStdString(),
                "",
                inId.toStdString(),
                inParam.toStdString(),
                ""
            };
            links.push_back(edge);
        }
    }

    for (auto idx : nodeIndice)
    {
        zeno::NodeData node = idx.data(ROLE_NODEDATA).value<zeno::NodeData>();
        for (zeno::ParamInfo& param : node.inputs)
        {
            for (auto it = param.links.begin(); it != param.links.end(); )
            {
                if (std::find(links.begin(), links.end(), *it) == links.end())
                {
                    it = param.links.erase(it);
                }
                else
                {
                    it++;
                }
            }
        }

        for (zeno::ParamInfo& param : node.outputs)
        {
            for (auto it = param.links.begin(); it != param.links.end(); )
            {
                if (std::find(links.begin(), links.end(), *it) == links.end())
                {
                    it = param.links.erase(it);
                }
                else
                {
                    it++;
                }
            }
        }

        const std::string& oldId = node.ident;
        nodes.insert(std::make_pair(oldId, node));
    }

    return { nodes, links };
}

void UiHelper::reAllocIdents(const QString& targetSubgraph,
                              const zeno::NodesData& inNodes,
                              const zeno::LinksData& inLinks,
                              zeno::NodesData& outNodes,
                              zeno::LinksData& outLinks)
{
    QMap<QString, QString> old2new;
    for (const auto& [key, data] : inNodes)
    {
        const auto& oldId = data.ident;
        const auto& name = data.cls;
        const QString& newId = UiHelper::generateUuid(QString::fromStdString(name));
        zeno::NodeData newData = data;
        newData.ident = newId.toStdString();
        outNodes.insert(std::make_pair(newData.ident, newData));
        old2new.insert(QString::fromStdString(oldId), newId);
    }
    //replace all the old-id in newNodes, and clear cached links.
    for (auto& [key, data] : outNodes)
    {
        for (auto& param : data.inputs)
        {
            param.links.clear();
        }
        for (auto& param : data.outputs)
        {
            param.links.clear();
        }
    }

    for (const zeno::EdgeInfo& link : inLinks)
    {
        QString outputNode = QString::fromStdString(link.outNode);
        QString outParam = QString::fromStdString(link.outParam);
        QString outKey = QString::fromStdString(link.outKey);

        QString inputNode = QString::fromStdString(link.inNode);
        QString inParam = QString::fromStdString(link.inParam);
        QString inKey = QString::fromStdString(link.inKey);

        ZASSERT_EXIT(old2new.find(inputNode) != old2new.end() &&
                     old2new.find(outputNode) != old2new.end());
        QString newInputNode = old2new[inputNode];
        QString newOutputNode = old2new[outputNode];

        zeno::EdgeInfo newLink = link;
        newLink.outNode = newOutputNode.toStdString();
        newLink.inNode = newInputNode.toStdString();

        outLinks.push_back(newLink);
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
