#include "modelacceptor.h"
#include "model/graphsmodel.h"
#include <zenoui/model/modelrole.h>
#include "../nodesys/nodesmgr.h"
#include <zeno/utils/logger.h>
#include "nodesys/zenosubgraphscene.h"
#include "magic_enum.hpp"
#include "util/log.h"
#include <zenoio/reader/zsgreader.h>


ModelAcceptor::ModelAcceptor(GraphsModel* pModel, bool bImport)
	: m_pModel(pModel)
	, m_currentGraph(nullptr)
	, m_bImport(bImport)
{
}

void ModelAcceptor::setLegacyDescs(const rapidjson::Value& graphObj, const NODE_DESCS& legacyDescs)
{
	//discard legacy desc except subnet desc.
	QStringList subgraphs;
	for (const auto& subgraph : graphObj.GetObject())
	{
		if (subgraph.name != "main") {
            subgraphs.append(QString::fromUtf8(subgraph.name.GetString()));
		}
	}
	QList<NODE_DESC> subnetDescs;
	for (QString name : subgraphs)
	{
		ZASSERT_EXIT(legacyDescs.find(name) != legacyDescs.end());
        subnetDescs.append(legacyDescs[name]);
	}
	m_pModel->appendDescriptors(subnetDescs);
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

	ZASSERT_EXIT(m_pModel && !m_currentGraph);
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
				const QModelIndex& outIdx = m_currentGraph->index(outNode);
				if (outIdx.isValid())
				{
					//the items in outputs are descripted by core descriptors.
					OUTPUT_SOCKETS outputs = outIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
					for (const QString &outSock : inSocket.outNodes[outNode].keys())
					{
						//checkout whether outSock is existed.
						if (outputs.find(outSock) == outputs.end())
						{
							const QString& nodeName = outIdx.data(ROLE_OBJNAME).toString();
							zeno::log_warn("no such output socket {} in {}", outSock.toStdString(), nodeName.toStdString());
							continue;
						}
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

bool ModelAcceptor::addNode(const QString& nodeid, const QString& name, const NODE_DESCS& legacyDescs)
{
	if (!m_currentGraph)
		return false;

	if (!m_pModel->hasDescriptor(name)) {
		zeno::log_warn("no node class named [{}]", name.toStdString());
		return false;
	}

	NODE_DATA data;
	data[ROLE_OBJID] = nodeid;
	data[ROLE_OBJNAME] = name;
	data[ROLE_COLLASPED] = false;
	data[ROLE_NODETYPE] = NodesMgr::nodeType(name);

    //zeno::log_warn("zsg has Inputs {}", data.find(ROLE_PARAMETERS) != data.end());
	m_currentGraph->appendItem(data, false);
	return true;
}

void ModelAcceptor::setViewRect(const QRectF& rc)
{
	if (!m_currentGraph)
		return;
	m_currentGraph->setViewRect(rc);
}

void ModelAcceptor::initSockets(const QString& id, const QString& name, const NODE_DESCS& legacyDescs)
{
	if (!m_currentGraph)
		return;

	NODE_DESC desc;
	bool ret = m_pModel->getDescriptor(name, desc);
	ZASSERT_EXIT(ret);

	//params
	INPUT_SOCKETS inputs;
	PARAMS_INFO params;
	OUTPUT_SOCKETS outputs;

	for (PARAM_INFO descParam : desc.params)
	{
		PARAM_INFO param;
		param.name = descParam.name;
		param.control = descParam.control;
		param.typeDesc = descParam.typeDesc;
		param.defaultValue = descParam.defaultValue;
		params.insert(param.name, param);
	}
	
	for (INPUT_SOCKET descInput : desc.inputs)
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
	
	for (OUTPUT_SOCKET descOutput : desc.outputs)
	{
		OUTPUT_SOCKET output;
		output.info.nodeid = id;
		output.info.control = descOutput.info.control;
		output.info.type = descOutput.info.type;
		output.info.name = descOutput.info.name;
		outputs[output.info.name] = output;
	}

	QModelIndex idx = m_currentGraph->index(id);

	m_currentGraph->setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
	m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
	m_currentGraph->setData(idx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
}

void ModelAcceptor::setSocketKeys(const QString& id, const QStringList& keys)
{
	if (!m_currentGraph)
		return;

	//legacy io formats.

	//there is no info about whether the key belongs to input or output.
	//have to classify by nodecls.
	QModelIndex idx = m_currentGraph->index(id);
    const QString& nodeName = idx.data(ROLE_OBJNAME).toString();
	if (nodeName == "MakeDict")
	{
        INPUT_SOCKETS inputs = m_currentGraph->data(idx, ROLE_INPUTS).value<INPUT_SOCKETS>();
		for (auto keyName : keys)
		{
            addDictKey(id, keyName, true);
		}
	}
	else if (nodeName == "ExtractDict")
	{
        OUTPUT_SOCKETS outputs = m_currentGraph->data(idx, ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
		for (auto keyName : keys)
		{
            addDictKey(id, keyName, false);
		}
	}
}

void ModelAcceptor::addDictKey(const QString& id, const QString& keyName, bool bInput)
{
    if (!m_currentGraph)
        return;

	QModelIndex idx = m_currentGraph->index(id);
	if (bInput)
	{
        INPUT_SOCKETS inputs = m_currentGraph->data(idx, ROLE_INPUTS).value<INPUT_SOCKETS>();
		if (inputs.find(keyName) == inputs.end())
		{
            INPUT_SOCKET inputSocket;
			inputSocket.info.name = keyName;
            inputSocket.info.nodeid = id;
			inputSocket.info.control = CONTROL_DICTKEY;
            inputSocket.info.type = "";
			inputs[keyName] = inputSocket;
            m_currentGraph->setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
		}
	}
	else
	{
        OUTPUT_SOCKETS outputs = m_currentGraph->data(idx, ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
		if (outputs.find(keyName) == outputs.end())
		{
            OUTPUT_SOCKET outputSocket;
			outputSocket.info.name = keyName;
            outputSocket.info.nodeid = id;
            outputSocket.info.control = CONTROL_DICTKEY;
            outputSocket.info.type = "";
            outputs[keyName] = outputSocket;
            m_currentGraph->setData(idx, QVariant::fromValue(outputs), ROLE_OUTPUTS);
		}
	}
}

void ModelAcceptor::setInputSocket(
				const QString& nodeCls,
				const QString& id,
				const QString& inSock,
                const QString& outId,
				const QString& outSock,
				const rapidjson::Value& defaultVal,
				const NODE_DESCS& legacyDescs)
{
	if (!m_currentGraph)
		return;

	NODE_DESC desc;
	bool ret = m_pModel->getDescriptor(nodeCls, desc);
	ZASSERT_EXIT(ret);

	//parse default value.
	QVariant defaultValue;
	if (!defaultVal.IsNull())
	{
		SOCKET_INFO descInfo;
		if (desc.inputs.find(inSock) != desc.inputs.end()) {
			descInfo = desc.inputs[inSock].info;
		}
		defaultValue = ZsgReader::getInstance()._parseToVariant(descInfo.type, defaultVal, m_currentGraph);
	}

	QModelIndex idx = m_currentGraph->index(id);
	ZASSERT_EXIT(idx.isValid());

	//standard inputs desc by latest descriptors. 
	INPUT_SOCKETS inputs = m_currentGraph->data(idx, ROLE_INPUTS).value<INPUT_SOCKETS>();

	if (inputs.find(inSock) != inputs.end())
	{
		if (!defaultValue.isNull())
			inputs[inSock].info.defaultValue = defaultValue;
        //if (defaultValue.type() == QVariant::Int)
            //zeno::log_critical("rehappy {}", defaultValue.toInt());
		if (!outId.isEmpty() && !outSock.isEmpty())
		{
			inputs[inSock].outNodes[outId][outSock] = SOCKET_INFO(outId, outSock);
		}
		m_currentGraph->setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
	}
	else
	{
		//TODO: optimize the code.
		if (nodeCls == "MakeList" || nodeCls == "MakeDict")
		{
			INPUT_SOCKET inSocket;
			inSocket.info.name = inSock;
			if (nodeCls == "MakeDict")
			{
				inSocket.info.control = CONTROL_DICTKEY;
			}
			inputs[inSock] = inSocket;

			if (!outId.isEmpty() && !outSock.isEmpty())
			{
				inputs[inSock].outNodes[outId][outSock] = SOCKET_INFO(outId, outSock);
			}
			m_currentGraph->setData(idx, QVariant::fromValue(inputs), ROLE_INPUTS);
		}
		else
		{
			zeno::log_warn("no such input socket {}", inSock.toStdString());
		}
    }
}

void ModelAcceptor::setParamValue(const QString& id, const QString& name, const QVariant& var)
{
	if (!m_currentGraph)
		return;
	QModelIndex idx = m_currentGraph->index(id);
	ZASSERT_EXIT(idx.isValid());
	PARAMS_INFO params = m_currentGraph->data(idx, ROLE_PARAMETERS).value<PARAMS_INFO>();

	if (params.find(name) != params.end())
	{
        zeno::log_trace("found param name {}", name.toStdString());
		params[name].value = var;
		m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
	}
	else
	{
		PARAMS_INFO _params = m_currentGraph->data(idx, ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
		_params[name].value = var;
		m_currentGraph->setData(idx, QVariant::fromValue(_params), ROLE_PARAMS_NO_DESC);
        zeno::log_warn("not found param name {}", name.toStdString());
    }
}

void ModelAcceptor::setPos(const QString& id, const QPointF& pos)
{
	if (!m_currentGraph)
		return;

	QModelIndex idx = m_currentGraph->index(id);
	ZASSERT_EXIT(idx.isValid());
	m_currentGraph->setData(idx, pos, ROLE_OBJPOS);
}

void ModelAcceptor::setOptions(const QString& id, const QStringList& options)
{
	if (!m_currentGraph)
		return;

	QModelIndex idx = m_currentGraph->index(id);
	ZASSERT_EXIT(idx.isValid());
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
	}
	m_currentGraph->setData(idx, opts, ROLE_OPTIONS);
}

void ModelAcceptor::setColorRamps(const QString& id, const COLOR_RAMPS& colorRamps)
{
	if (!m_currentGraph)
		return;

	QLinearGradient linearGrad;
	for (COLOR_RAMP ramp : colorRamps)
	{
		linearGrad.setColorAt(ramp.pos, QColor::fromRgbF(ramp.r, ramp.g, ramp.b));
	}

	QModelIndex idx = m_currentGraph->index(id);
	ZASSERT_EXIT(idx.isValid());

	PARAMS_INFO params = m_currentGraph->data(idx, ROLE_PARAMETERS).value<PARAMS_INFO>();

	PARAM_INFO param;
	param.name = "color";
	param.control = CONTROL_HEATMAP;
	param.value = QVariant::fromValue(linearGrad);
	params.insert(param.name, param);

	m_currentGraph->setData(idx, QVariant::fromValue(params), ROLE_PARAMETERS);
}

void ModelAcceptor::setBlackboard(const QString& id, const BLACKBOARD_INFO& blackboard)
{
	if (!m_currentGraph)
		return;

	QModelIndex idx = m_currentGraph->index(id);
	ZASSERT_EXIT(idx.isValid());
	m_pModel->updateBlackboard(id, blackboard, m_pModel->indexBySubModel(m_currentGraph), false);
}

QObject* ModelAcceptor::currGraphObj()
{
    return m_currentGraph;
}
