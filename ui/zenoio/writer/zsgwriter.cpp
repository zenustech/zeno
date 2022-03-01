#include "zsgwriter.h"
#include <zenoui/model/modelrole.h>
#include <zeno/utils/logger.h>


ZsgWriter::ZsgWriter()
{
}

ZsgWriter& ZsgWriter::getInstance()
{
	static ZsgWriter writer;
	return writer;
}

QString ZsgWriter::dumpProgramStr(IGraphsModel* pModel)
{
	QJsonObject obj = dumpGraphs(pModel);
	QJsonDocument doc(obj);
	QString strJson(doc.toJson(QJsonDocument::Indented));
	return strJson;
}

QJsonObject ZsgWriter::dumpGraphs(IGraphsModel* pModel)
{
	QJsonObject obj;
	if (!pModel)
		return obj;

	QJsonObject graphObj;
	for (int i = 0; i < pModel->rowCount(); i++)
	{
		const QModelIndex& subgIdx = pModel->index(i, 0);
		const QString& subgName = pModel->name(subgIdx);
		graphObj.insert(subgName, _dumpSubGraph(pModel, subgIdx));
	}
	obj.insert("graph", graphObj);
	obj.insert("views", QJsonObject());     //todo

	NODE_DESCS descs = pModel->descriptors();
	obj.insert("descs", _dumpDescriptors(descs));

	return obj;
}

QJsonObject ZsgWriter::_dumpSubGraph(IGraphsModel* pModel, const QModelIndex& subgIdx)
{
	QJsonObject obj;
	if (!pModel)
		return obj;

	QJsonObject nodesObj;

	const NODES_DATA& nodes = pModel->nodes(subgIdx);
	for (const NODE_DATA& node : nodes)
	{
		const QString& id = node[ROLE_OBJID].toString();
		nodesObj.insert(id, dumpNode(node));
	}
	obj.insert("nodes", nodesObj);

	QRectF viewRc = pModel->viewRect(subgIdx);
	if (!viewRc.isNull())
	{
		QJsonObject rcObj;
		rcObj.insert("x", viewRc.x());
		rcObj.insert("y", viewRc.y());
		rcObj.insert("width", viewRc.width());
		rcObj.insert("height", viewRc.height());
		obj.insert("view_rect", rcObj);
	}
	return obj;
}

QJsonObject ZsgWriter::dumpNode(const NODE_DATA& data)
{
	QJsonObject obj;
	obj.insert("name", data[ROLE_OBJNAME].toString());

	const INPUT_SOCKETS& inputs = data[ROLE_INPUTS].value<INPUT_SOCKETS>();
	QJsonObject inputsArr;
	for (const INPUT_SOCKET& inSock : inputs)
	{
		QJsonValue deflVal = QJsonValue::fromVariant(inSock.info.defaultValue);
		if (!inSock.linkIndice.isEmpty())
		{
			for (QPersistentModelIndex linkIdx : inSock.linkIndice)
			{
				QString outNode = linkIdx.data(ROLE_OUTNODE).toString();
				QString outSock = linkIdx.data(ROLE_OUTSOCK).toString();

				QJsonArray arr;
				arr.push_back(outNode);
				arr.push_back(outSock);
				arr.push_back(deflVal);
				inputsArr.insert(inSock.info.name, arr);
			}
		}
		else
		{
			QJsonArray arr = { QJsonValue::Null, QJsonValue::Null, deflVal };
			inputsArr.insert(inSock.info.name, arr);
		}
	}
	obj.insert("inputs", inputsArr);

	const PARAMS_INFO& params = data[ROLE_PARAMETERS].value<PARAMS_INFO>();
	QJsonObject paramsObj;
	for (const PARAM_INFO& info : params)
	{
		QVariant val = info.value;
		if (val.type() == QVariant::String)
			paramsObj.insert(info.name, val.toString());
		else if (val.type() == QVariant::Double)
			paramsObj.insert(info.name, val.toDouble());
		else if (val.type() == QVariant::Int)
			paramsObj.insert(info.name, val.toInt());
		else if (val.type() == QVariant::Bool)
			paramsObj.insert(info.name, val.toBool());
        else if (val.type() != QVariant::Invalid)
                zeno::log_warn("bad param info qvariant type {}", val.typeName() ? val.typeName() : "(null)");
	}
	obj.insert("params", paramsObj);

	QPointF pos = data[ROLE_OBJPOS].toPointF();
	obj.insert("uipos", QJsonArray({ pos.x(), pos.y() }));

	QJsonArray options;
	int opts = data[ROLE_OPTIONS].toInt();
	if (opts & OPT_ONCE) {
		options.push_back("ONCE");
	}
	if (opts & OPT_MUTE) {
		options.push_back("MUTE");
	}
	if (opts & OPT_PREP) {
		options.push_back("PREP");
	}
	if (opts & OPT_VIEW) {
		options.push_back("VIEW");
	}
	if (data[ROLE_COLLASPED].toBool())
	{
		options.push_back("collapsed");
	}
	obj.insert("options", options);

	QJsonArray socketKeys = data[ROLE_SOCKET_KEYS].value<QJsonArray>();
	if (!socketKeys.isEmpty())
	{
		obj.insert("socket_keys", socketKeys);
	}

	return obj;
}

QJsonObject ZsgWriter::_dumpDescriptors(const NODE_DESCS& descs)
{
	QJsonObject descsObj;
	for (auto name : descs.keys())
	{
		QJsonObject descObj;
		const NODE_DESC& desc = descs[name];

		if (name == "SDFToPoly")
		{
			int j;
			j = 0;
		}

		QJsonArray inputs;
		for (INPUT_SOCKET inSock : desc.inputs)
		{
			QJsonArray socketInfo;
			socketInfo.push_back(inSock.info.type);
			socketInfo.push_back(inSock.info.name);

			const QVariant& defl = inSock.info.defaultValue;
			if (defl.type() == QVariant::String)
				socketInfo.push_back(defl.toString());
			else if (defl.type() == QVariant::Double)
				socketInfo.push_back(QString::fromStdString(std::to_string(defl.toDouble())));
			else if (defl.type() == QVariant::Int)
				socketInfo.push_back(QString::fromStdString(std::to_string(defl.toInt())));
			else if (defl.type() == QVariant::Bool)
				socketInfo.push_back(QString::fromStdString(std::to_string((int)defl.toBool())));
			else {
                if (defl.type() != QVariant::Invalid)
                    zeno::log_warn("bad input qvariant type {}", defl.typeName() ? defl.typeName() : "(null)");
				socketInfo.push_back("");
            }

			inputs.push_back(socketInfo);
		}

		QJsonArray params;
		for (PARAM_INFO param : desc.params)
		{
			QJsonArray paramInfo;
			paramInfo.push_back(param.typeDesc);
			paramInfo.push_back(param.name);

			const QVariant& defl = param.defaultValue;
			if (defl.type() == QVariant::String)
				paramInfo.push_back(defl.toString());
			else if (defl.type() == QVariant::Double)
				paramInfo.push_back(QString::fromStdString(std::to_string(defl.toDouble())));
			else if (defl.type() == QVariant::Int)
				paramInfo.push_back(QString::fromStdString(std::to_string(defl.toInt())));
			else if (defl.type() == QVariant::Bool)
				paramInfo.push_back(QString::fromStdString(std::to_string((int)defl.toBool())));
			else {
                if (defl.type() != QVariant::Invalid)
                    zeno::log_warn("bad param qvariant type {}", defl.typeName() ? defl.typeName() : "(null)");
				paramInfo.push_back("");
            }

			params.push_back(paramInfo);
		}

		QJsonArray outputs;
		for (OUTPUT_SOCKET outSock : desc.outputs)
		{
			QJsonArray socketInfo;
			socketInfo.push_back(outSock.info.type);
			socketInfo.push_back(outSock.info.name);

			const QVariant& defl = outSock.info.defaultValue;
			if (defl.type() == QVariant::String)
				socketInfo.push_back(defl.toString());
			else if (defl.type() == QVariant::Double)
				socketInfo.push_back(QString::fromStdString(std::to_string(defl.toDouble())));
			else if (defl.type() == QVariant::Int)
				socketInfo.push_back(QString::fromStdString(std::to_string(defl.toInt())));
			else if (defl.type() == QVariant::Bool)
				socketInfo.push_back(QString::fromStdString(std::to_string((int)defl.toBool())));
			else {
                if (defl.type() != QVariant::Invalid)
                    zeno::log_warn("bad output qvariant type {}", defl.typeName() ? defl.typeName() : "(null)");
				socketInfo.push_back("");
            }

			outputs.push_back(socketInfo);
		}

		QJsonArray categories;
		for (QString cate : desc.categories)
		{
			categories.push_back(cate);
		}

		descObj.insert("inputs", inputs);
		descObj.insert("outputs", outputs);
		descObj.insert("params", params);
		descObj.insert("categories", categories);
		if (desc.is_subgraph)
			descObj.insert("is_subgraph", desc.is_subgraph);

		descsObj.insert(name, descObj);
	}
	return descsObj;
}
