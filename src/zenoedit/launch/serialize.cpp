#include "serialize.h"
#include <model/graphsmodel.h>
#include <model/modeldata.h>
#include <model/modelrole.h>


static void serializeGraph(SubGraphModel* pModel, GraphsModel* pGraphsModel, QStringList const &graphNames, QJsonArray& ret)
{
	const QString& name = pModel->name();
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
            SubGraphModel* pSubModel = pGraphsModel->subGraph(name);
            ret.push_back(QJsonArray({ "pushSubgraph", ident, name }));
            serializeGraph(pSubModel, pGraphsModel, graphNames, ret);
            ret.push_back(QJsonArray({ "popSubgraph", ident, name }));

		} else {
            ret.push_back(QJsonArray({ "addNode", name, ident }));
        }

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

        // TODO(bate): luzh please mock options at editor side!
		/*for (QString optionName : options)
		{
			ret.push_back(QJsonArray({"setNodeOption", ident, optionName}));
		}*/

		ret.push_back(QJsonArray({"completeNode", ident}));
	}
}

void serializeScene(GraphsModel* pModel, QJsonArray& ret)
{
	//QJsonArray item = { "clearAllState" };
    //ret.push_back(item);

	QStringList graphs;
	for (int i = 0; i < pModel->rowCount(); i++)
		graphs.push_back(pModel->subGraph(i)->name());

    SubGraphModel* pSubModel = pModel->subGraph("main");
    serializeGraph(pSubModel, pModel, graphs, ret);
}
