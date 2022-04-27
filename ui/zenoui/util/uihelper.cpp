#include "uihelper.h"
#include <zeno/utils/logger.h>
#include <zenoui/model/modelrole.h>
#include <QUuid>


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
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                INPUT_SOCKET inputSocket;
                inputSocket.info = SOCKET_INFO("", socketName);
                inputSocket.info.type = socketType;
                inputSocket.info.control = _getControlType(socketType);
                inputSocket.info.defaultValue = _parseDefaultValue(socketDefl, socketType);
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
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                PARAM_INFO paramInfo;
                paramInfo.bEnableConnect = false;
                paramInfo.control = ctrlType;
                paramInfo.name = socketName;
                paramInfo.typeDesc = socketType;
                paramInfo.defaultValue = _parseDefaultValue(socketDefl, socketType);

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
                PARAM_CONTROL ctrlType = _getControlType(socketType);
                OUTPUT_SOCKET outputSocket;
                outputSocket.info = SOCKET_INFO("", socketName);
                outputSocket.info.type = socketType;
                outputSocket.info.control = _getControlType(socketType);
                outputSocket.info.defaultValue = _parseDefaultValue(socketDefl, socketType);

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

QVariant UiHelper::_parseDefaultValue(const QString &defaultValue, const QString &type)
{
    auto control = _getControlType(type);
    switch (control) {
    case CONTROL_INT:
        return defaultValue.toInt();
    case CONTROL_BOOL:
        return (bool)defaultValue.toInt();
    case CONTROL_FLOAT:
        return defaultValue.toDouble();
    case CONTROL_STRING:
    case CONTROL_WRITEPATH:
    case CONTROL_READPATH:
    case CONTROL_MULTILINE_STRING:
    case CONTROL_HEATMAP:
    case CONTROL_ENUM:
        return defaultValue;
    case CONTROL_VEC3F:
    {
        QVector<qreal> vec;
        if (!defaultValue.isEmpty())
        {
            QStringList L = defaultValue.split(",");
            vec.resize(qMax(L.size(), 3));
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
    default:
        return QVariant();
    };
}

QVariant UiHelper::parseVariantValue(const rapidjson::Value& val)
{
    if (val.GetType() == rapidjson::kStringType) {
        return val.GetString();
    }
    else if (val.GetType() == rapidjson::kNumberType) {
        if (val.IsDouble())
            return val.GetDouble();
        else if (val.IsInt())
            return val.GetInt();
        else {
            zeno::log_warn("bad rapidjson number type {}", val.GetType());
            return QVariant();
        }
    }
    else if (val.GetType() == rapidjson::kTrueType) {
        return val.GetBool();
    }
    else if (val.GetType() == rapidjson::kFalseType) {
        return val.GetBool();
    }
    else {
        return QVariant();
    }
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

PARAM_CONTROL UiHelper::_getControlType(const QString &type)
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
    } else if (type == "vec3f") {
        return CONTROL_VEC3F;
    } else if (type == "writepath") {
        return CONTROL_WRITEPATH;
    } else if (type == "readpath") {
        return CONTROL_READPATH;
    } else if (type == "multiline_string") {
        return CONTROL_MULTILINE_STRING;
    } else if (type == "_RAMPS") {
        return CONTROL_HEATMAP;
    } else if (type.startsWith("enum ")) {
        return CONTROL_ENUM;
    } else if (type.isEmpty()) {
        return CONTROL_NONE;
    } else {
        zeno::log_trace("parse got undefined control type {}", type.toStdString());
        return CONTROL_NONE;
    }
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
    // TODO: LUZH how to detect QVector<qreal> for vec3f?
	else zeno::log_warn("bad qt variant {}", var.typeName());

    return value;
}

QMap<QString, NODE_DATA> UiHelper::dumpItems(IGraphsModel* pGraphsModel, const QPersistentModelIndex& subgIdx, 
    const QModelIndexList& nodesIndice, const QModelIndexList& linkIndice)
{
	QMap<QString, QString> oldToNew;
	QMap<QString, NODE_DATA> newNodes;
	QList<NODE_DATA> vecNodes;
	for (const QModelIndex& index : nodesIndice)
	{
        const QString& currNode = index.data(ROLE_OBJID).toString();
		NODE_DATA data = pGraphsModel->itemData(index, subgIdx);
		QString oldId = data[ROLE_OBJID].toString();
		const QString& newId = UiHelper::generateUuid(data[ROLE_OBJNAME].toString());
		data[ROLE_OBJID] = newId;
		oldToNew[oldId] = newId;

		//clear any connections.
		OUTPUT_SOCKETS outputs = data[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
		INPUT_SOCKETS inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
		for (auto it = outputs.begin(); it != outputs.end(); it++)
		{
			it->second.linkIndice.clear();
			it->second.inNodes.clear();
		}
		for (auto it = inputs.begin(); it != inputs.end(); it++)
		{
			it->second.linkIndice.clear();
			it->second.outNodes.clear();
		}

		data[ROLE_INPUTS] = QVariant::fromValue(inputs);
		data[ROLE_OUTPUTS] = QVariant::fromValue(outputs);
		newNodes[newId] = data;
	}

	for (const QPersistentModelIndex& linkIdx : linkIndice)
	{
		const QString& outOldId = linkIdx.data(ROLE_OUTNODE).toString();
		const QString& inOldId = linkIdx.data(ROLE_INNODE).toString();
		const QString& outSock = linkIdx.data(ROLE_OUTSOCK).toString();
		const QString& inSock = linkIdx.data(ROLE_INSOCK).toString();

		const QString& outId = oldToNew[outOldId];
		const QString& inId = oldToNew[inOldId];

		//out link
		NODE_DATA& outData = newNodes[outId];
		OUTPUT_SOCKETS outputs = outData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
		SOCKET_INFO& newOutSocket = outputs[outSock].inNodes[inId][inSock];

		QModelIndex tempIdx = pGraphsModel->index(outOldId, subgIdx);

		NODE_DATA outOldData = pGraphsModel->itemData(tempIdx, subgIdx);
		OUTPUT_SOCKETS oldOutputs = outOldData[ROLE_OUTPUTS].value<OUTPUT_SOCKETS>();
		const SOCKET_INFO& oldOutSocket = oldOutputs[outSock].inNodes[inOldId][inSock];
		newOutSocket = oldOutSocket;
		newOutSocket.nodeid = inId;
		outData[ROLE_OUTPUTS] = QVariant::fromValue(outputs);

		//in link
		NODE_DATA& inData = newNodes[inId];
		INPUT_SOCKETS inputs = inData[ROLE_INPUTS].value<INPUT_SOCKETS>();
		SOCKET_INFO& newInSocket = inputs[inSock].outNodes[outId][outSock];

		tempIdx = pGraphsModel->index(inOldId, subgIdx);
		NODE_DATA inOldData = pGraphsModel->itemData(tempIdx, subgIdx);
		INPUT_SOCKETS oldInputs = inOldData[ROLE_INPUTS].value<INPUT_SOCKETS>();
		const SOCKET_INFO& oldInSocket = oldInputs[inSock].outNodes[outOldId][outSock];
		newInSocket = oldInSocket;
		newInSocket.nodeid = outId;
		inData[ROLE_INPUTS] = QVariant::fromValue(inputs);
	}
    return newNodes;
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

