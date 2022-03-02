#include "modelacceptor.h"
#include "model/graphsmodel.h"
#include <zenoui/model/modelrole.h>
#include "../nodesys/nodesmgr.h"
#include <zeno/utils/logger.h>
#include "nodesys/zenosubgraphscene.h"
#include "magic_enum.hpp"


ModelAcceptor::ModelAcceptor(GraphsModel* pModel, bool bImport)
	: m_pModel(pModel)
	, m_currentGraph(nullptr)
	, m_bImport(bImport)
{
}


void ModelAcceptor::setDescriptors(const NODE_DESCS& nodesParams)
{
	m_pModel->setDescriptors(nodesParams);
}

void ModelAcceptor::BeginSubgraph(const QString& name)
{
	if (m_bImport && name == "main")
	{
		m_currentGraph = nullptr;
		return;
	}

	if (m_bImport)
		zeno::log_info("Importing subgraph {}", name.toStdString());

	Q_ASSERT(m_pModel && !m_currentGraph);
	SubGraphModel* pSubModel = new SubGraphModel(m_pModel);
	pSubModel->setName(name);
	m_pModel->appendSubGraph(pSubModel);
	m_currentGraph = pSubModel;
}

void ModelAcceptor::EndSubgraph()
{
	if (!m_currentGraph)
		return;

	//init output ports for each node.
	int n = m_currentGraph->rowCount();
	for (int r = 0; r < n; r++)
	{
		const QModelIndex& idx = m_currentGraph->index(r, 0);
		const QString& inNode = idx.data(ROLE_OBJID).toString();
		INPUT_SOCKETS inputs = idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
		foreach(const QString & inSockName, inputs.keys())
		{
			const INPUT_SOCKET& inSocket = inputs[inSockName];

			//init connection
			for (const QString& outNode : inSocket.outNodes.keys())
			{
				for (const QString& outSock : inSocket.outNodes[outNode].keys())
				{
					const QModelIndex& outIdx = m_currentGraph->index(outNode);
					if (outIdx.isValid())
					{
						GraphsModel* pGraphsModel = m_currentGraph->getGraphsModel();
						const QModelIndex& subgIdx = pGraphsModel->indexBySubModel(m_currentGraph);
						pGraphsModel->addLink(EdgeInfo(outNode, inNode, outSock, inSockName), subgIdx);
					}
				}
			}
		}
	}

	m_currentGraph->onModelInited();
	m_currentGraph = nullptr;
}

void ModelAcceptor::setFilePath(const QString& fileName)
{
	if (!m_bImport)
		m_pModel->setFilePath(fileName);
}

void ModelAcceptor::switchSubGraph(const QString& graphName)
{
	m_pModel->switchSubGraph(graphName);
}

void ModelAcceptor::addNode(const QString& nodeid, const QString& name, const NODE_DESCS& descriptors)
{// called on load-zsg!!
	if (!m_currentGraph)
		return;

	NODE_DATA data;
	data[ROLE_OBJID] = nodeid;
	data[ROLE_OBJNAME] = name;
	data[ROLE_COLLASPED] = false;
	data[ROLE_NODETYPE] = NodesMgr::nodeType(name);

    //Q_ASSERT(data.find(ROLE_INPUTS) == data.end());

    //zeno::log_warn("zsg has Inputs {}", data.find(ROLE_PARAMETERS) != data.end());
	m_currentGraph->appendItem(data, false);
}

void ModelAcceptor::setViewRect(const QRectF& rc)
{
	if (!m_currentGraph)
		return;
	m_currentGraph->setViewRect(rc);
}

void ModelAcceptor::_initSockets(const QString& id, const QString& name, INPUT_SOCKETS& inputs, PARAMS_INFO& params, OUTPUT_SOCKETS& outputs)
{
	if (!m_currentGraph)
		return;
	//TODO
	if (name == "MakeDict")
	{
		const QStringList& socketKeys = m_currentGraph->data(m_currentGraph->index(id), ROLE_SOCKET_KEYS).toStringList();
		for (const QString& key : socketKeys)
		{
			INPUT_SOCKET inputSock;
			inputSock.info.name = key;
			inputSock.info.nodeid = id;
			inputs[key] = inputSock;
		}
		PARAM_INFO info;
		info.name = "_KEYS";
		info.value = socketKeys.join("\n");
		params.insert(info.name, info);
	}
	else if (name == "MakeList")
	{

	}
	else if (name == "ExtractList")
	{
		//todo: dynamic_sockets_is_output
	}
	else if (name == "MakeHeatmap")
	{

	}
	else if (name == "MakeCurvemap")
	{

	}
	else
	{

	}
}

void ModelAcceptor::initSockets(const QString& id, const QString& name, const NODE_DESCS& descs)
{
	if (!m_currentGraph)
		return;

	//params
	INPUT_SOCKETS inputs;
	PARAMS_INFO params;
	OUTPUT_SOCKETS outputs;

	for (PARAM_INFO descParam : descs[name].params)
	{
		PARAM_INFO param;
		param.name = descParam.name;
		param.control = descParam.control;
		param.typeDesc = descParam.typeDesc;
		param.defaultValue = descParam.defaultValue;
		params.insert(param.name, param);
	}
	
	for (INPUT_SOCKET descInput : descs[name].inputs)
	{
		INPUT_SOCKET input;
		input.info.nodeid = id;
		input.info.control = descInput.info.control;
		input.info.type = descInput.info.type;
		input.info.name = descInput.info.name;

		QVariant deflVal = descInput.info.defaultValue;
		QString type = descInput.info.type;
		if (type == "NumericObject")
		{
			type = "float";
		}
		if (!type.startsWith("enum "))
		{
			static QStringList acceptTypes = { "int", "bool", "float", "string", "writepath", "readpath" };
			if (type.isEmpty() || acceptTypes.indexOf(type) == -1)
			{
				deflVal = QVariant();
			}
		}
		input.info.defaultValue = deflVal;
		inputs.insert(input.info.name, input);
	}
	
	for (OUTPUT_SOCKET descOutput : descs[name].outputs)
	{
		OUTPUT_SOCKET output;
		output.info.nodeid = id;
		output.info.control = descOutput.info.control;
		output.info.type = descOutput.info.type;
		output.info.name = descOutput.info.name;
		outputs[output.info.name] = output;
	}

	_initSockets(id, name, inputs, params, outputs);

	m_currentGraph->setData(m_currentGraph->index(id), QVariant::fromValue(inputs), ROLE_INPUTS);
	m_currentGraph->setData(m_currentGraph->index(id), QVariant::fromValue(params), ROLE_PARAMETERS);
	m_currentGraph->setData(m_currentGraph->index(id), QVariant::fromValue(outputs), ROLE_OUTPUTS);
}

void ModelAcceptor::setSocketKeys(const QString& id, const QStringList& keys)
{
	if (!m_currentGraph)
		return;
	m_currentGraph->setData(m_currentGraph->index(id), keys, ROLE_SOCKET_KEYS);
}

void ModelAcceptor::setInputSocket(const QString& id, const QString& inSock, const QString& outId, const QString& outSock, const QVariant& defaultValue)
{
	if (!m_currentGraph)
		return;
	QModelIndex idx = m_currentGraph->index(id);
	Q_ASSERT(idx.isValid());
	INPUT_SOCKETS inputs = m_currentGraph->data(idx, ROLE_INPUTS).value<INPUT_SOCKETS>();
	if (inputs.find(inSock) != inputs.end())
	{
		if (!defaultValue.isNull())
			inputs[inSock].info.defaultValue = defaultValue;	//default value?
		if (!outId.isEmpty() && !outSock.isEmpty())
		{
			inputs[inSock].outNodes[outId][outSock] = SOCKET_INFO(outId, outSock);
			m_currentGraph->setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
		}
	}
}

void ModelAcceptor::setParamValue(const QString& id, const QString& name, const QVariant& var)
{
	if (!m_currentGraph)
		return;
	QModelIndex idx = m_currentGraph->index(id);
	Q_ASSERT(idx.isValid());
	PARAMS_INFO params = m_currentGraph->data(idx, ROLE_PARAMETERS).value<PARAMS_INFO>();

	if (id == "3e5f8949-MakeMultilineString" && name == "value")
	{
		int j;
		j = 0;
	}

	if (params.find(name) != params.end())
	{
        zeno::log_trace("found param name {}", name.toStdString());
		params[name].value = var;
		m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
	} else {
        zeno::log_warn("not found param name {}", name.toStdString());
    }
}

void ModelAcceptor::setPos(const QString& id, const QPointF& pos)
{
	if (!m_currentGraph)
		return;

	QModelIndex idx = m_currentGraph->index(id);
	Q_ASSERT(idx.isValid());
	m_currentGraph->setData(idx, pos, ROLE_OBJPOS);
}

void ModelAcceptor::setOptions(const QString& id, const QStringList& options)
{
	if (!m_currentGraph)
		return;

	QModelIndex idx = m_currentGraph->index(id);
	Q_ASSERT(idx.isValid());
	int opts = 0;
	for (int i = 0; i < options.size(); i++)
	{
		const QString& optName = options[i];
		if (optName == "ONCE")
		{
			opts |= OPT_ONCE;
		}
		else if (optName == "PREP")
		{
			opts |= OPT_PREP;
		}
		else if (optName == "VIEW")
		{
			opts |= OPT_VIEW;
		}
		else if (optName == "MUTE")
		{
			opts |= OPT_MUTE;
		}
		else if (optName == "collapsed")
		{
			m_currentGraph->setData(idx, true, ROLE_COLLASPED);
		}
		else
		{
			Q_ASSERT(false);
		}
	}
	m_currentGraph->setData(idx, opts, ROLE_OPTIONS);
}

void ModelAcceptor::setColorRamps(const QString& id, const COLOR_RAMPS& colorRamps)
{
	if (!m_currentGraph)
		return;

	QModelIndex idx = m_currentGraph->index(id);
	Q_ASSERT(idx.isValid());
	m_currentGraph->setData(idx, QVariant::fromValue(colorRamps), ROLE_COLORRAMPS);
}

void ModelAcceptor::setBlackboard(const QString& id, const BLACKBOARD_INFO& blackboard)
{
	if (!m_currentGraph)
		return;

	QModelIndex idx = m_currentGraph->index(id);
	Q_ASSERT(idx.isValid());
	m_currentGraph->setData(idx, QVariant::fromValue(blackboard), ROLE_BLACKBOARD);
}
