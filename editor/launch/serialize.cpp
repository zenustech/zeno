#include "serialize.h"
#include <model/graphsmodel.h>
#include <model/modeldata.h>
#include <model/modelrole.h>


void serializeScene(GraphsModel* pModel, QJsonArray& ret)
{
	QJsonArray item = { "clearAllState" };
    ret.push_back(item);

	QStringList graphs;
	for (int i = 0; i < pModel->rowCount(); i++)
		graphs.push_back(pModel->subGraph(i)->name());

    for (int i = 0; i < pModel->rowCount(); i++)
    {
        SubGraphModel* pSubModel = pModel->subGraph(i);
		serializeGraph(pSubModel, graphs, ret);
    }
}

QJsonArray serializeGraphs(GraphsModel* pModel)
{
	QJsonArray ret;
    return ret;
}

void serializeGraph(SubGraphModel* pModel, const QStringList& graphNames, QJsonArray& ret)
{
	const QString& name = pModel->name();
	ret.push_back(QJsonArray({ "switchGraph", name }));
	const NODES_DATA& nodes = pModel->nodes();

	for (const NODE_DATA& node : nodes)
	{
		QString ident = node[ROLE_OBJID].toString();
		QString name = node[ROLE_OBJNAME].toString();
		const INPUT_SOCKETS& inputs = node[ROLE_INPUTS].value<INPUT_SOCKETS>();
		PARAMS_INFO params = node[ROLE_PARAMETERS].value<PARAMS_INFO>();
		int opts = node[ROLE_OPTIONS].toInt();
		QStringList options;
		if (opts & OPT_ONCE)
		{
			options.push_back("ONCE");
		}
		if (opts & OPT_MUTE)
		{
			options.push_back("MUTE");
		}
		if (opts & OPT_PREP)
		{
			options.push_back("PREP");
		}
		if (opts & OPT_VIEW)
		{
			options.push_back("VIEW");
		}

		if (graphNames.indexOf(name) != -1)
		{
			params["name"].name = "name";
			params["name"].value = name;
			name = "Subgraph";
		}
		else if (name == "ExecutionOutput")
		{
			name = "Route";
		}

		//temp code for debug
		if (name == "MakeDict")
		{
			int j;
			j = 0;
		}
		if (ident == "730549a0-Make2DGridPrimitive")
		{
			int j;
			j = 0;
		}

		ret.push_back(QJsonArray({ "addNode", name, ident }));

		for (INPUT_SOCKET input : inputs)
		{
			if (input.outNodes.isEmpty())
			{
				const QVariant& defl = input.info.defaultValue;
				if (!defl.isNull())
				{
					QVariant::Type varType = defl.type();
					if (varType == QMetaType::Float)
					{
						ret.push_back(QJsonArray({ "setNodeInput", ident, input.info.name, defl.toFloat() }));
					}
					else if (varType == QVariant::String)
					{
						ret.push_back(QJsonArray({ "setNodeInput", ident, input.info.name, defl.toString() }));
					}
					else if (varType == QVariant::Bool)
					{
						ret.push_back(QJsonArray({ "setNodeInput", ident, input.info.name, defl.toBool() }));
					}
					else
					{
						Q_ASSERT(false);
					}
				}
			}
			else
			{
				for (QString outId : input.outNodes.keys())
				{
					Q_ASSERT(!input.outNodes[outId].isEmpty());
					for (SOCKET_INFO outSock : input.outNodes[outId])
					{
						ret.push_back(QJsonArray({"bindNodeInput", ident, input.info.name, outId, outSock.name}));
					}
				}
			}
		}

		for (PARAM_INFO param_info : params)
		{
			QVariant value = param_info.value;
			QVariant::Type varType = value.type();

			if (param_info.name == "_KEYS")
			{
				int j;
				j = 0;
			}

			if (varType == QVariant::Double || varType == QMetaType::Float)
			{
				ret.push_back(QJsonArray({"setNodeParam", ident, param_info.name, value.toFloat()}));
			}
			else if (varType == QVariant::String)
			{
				ret.push_back(QJsonArray({"setNodeParam", ident, param_info.name, value.toString()}));
			}
			else if (varType == QVariant::Bool)
			{
				ret.push_back(QJsonArray({ "setNodeParam", ident, param_info.name, value.toBool() }));
			}
			else
			{
				Q_ASSERT(false);
			}
		}

		for (QString optionName : options)
		{
			ret.push_back(QJsonArray({"setNodeOption", ident, optionName}));
		}

		ret.push_back(QJsonArray({"completeNode", ident}));
	}
}

QJsonArray serializeScene(const QJsonObject& graphs)
{
    QJsonArray item = { "clearAllState" };
	QJsonArray arr;
    arr.push_back(item);

	const QStringList& graphsNames = graphs.keys();
	for (const QString& name : graphsNames)
	{
		const QJsonObject& graphObj = graphs[name].toObject();
		serializeGraph(graphObj["nodes"].toObject(), graphsNames, arr);
	}
	return arr;
}

QJsonArray serializeGraphs(const QJsonObject& graphs, bool has_subgraphs)
{
	return QJsonArray();
}

void serializeGraph(const QJsonObject& nodes, const QStringList& subgkeys, QJsonArray& ret)
{
	for (const QString& nodeId : nodes.keys())
	{
		const QJsonObject& nodeObj = nodes[nodeId].toObject();
		if (nodeObj.find("special") != nodeObj.end())
			continue;
		QString name = nodeObj["name"].toString();
		const QJsonObject& inputs = nodeObj["inputs"].toObject();
		QJsonObject params = nodeObj["params"].toObject();
		const QJsonArray& options = nodeObj["options"].toArray();
		const QJsonArray& uipos = nodeObj["uipos"].toArray();

		if (subgkeys.contains(name))
		{
			params["name"] = name;
			name = "Subgraph";
		}
		else if (name == "ExecutionOutput")
		{
			name = "Route";
		}
		ret.push_back(QJsonArray({ "addNode", name, nodeId }));

		for (const QString& inputSock : inputs.keys())
		{
			QJsonObject inputObj = inputs[inputSock].toObject();
            if (inputObj.isEmpty()) {
				continue;
			}
			else {
				QJsonArray inputArr = inputs[inputSock].toArray();
				QJsonValue srcIdent = inputArr[0];
				QJsonValue srcSockName = inputArr[1];
				QJsonValue sockDeflVal;
				if (inputArr.size() > 2)
					sockDeflVal = inputArr[2];
				if (srcIdent.isNull())
				{
					if (!sockDeflVal.isNull())
					{
						ret.push_back(QJsonArray({ "setNodeInput", srcIdent, srcSockName, sockDeflVal }));
					}
				}
				else
				{
					ret.push_back(QJsonArray({ "bindNodeInput", srcIdent, inputSock, sockDeflVal, srcSockName }));
				}
			}
		}

		for (const QString& paramName : params.keys())
		{
			ret.push_back(QJsonArray({"setNodeParam", nodeId, paramName, params[paramName] }));
		}
		for (auto option : options)
		{
			ret.push_back(QJsonArray({"setNodeOption", nodeId, option.toString()}));
		}
		ret.push_back(QJsonArray({ "completeNode", nodeId }));
	}
}