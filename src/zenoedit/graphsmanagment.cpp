#include "graphsmanagment.h"
#include <zenoui/model/graphsmodel.h>
#include <zenoui/model/modelrole.h>
#include "model/graphstreemodel.h"
#include <zenoio/reader/zsgreader.h>
#include <zenoio/acceptor/modelacceptor.h>
#include <zenoui/util/uihelper.h>
#include "nodesys/zenosubgraphscene.h"
#include <zeno/utils/log.h>


GraphsManagment::GraphsManagment(QObject* parent)
    : QObject(parent)
    , m_model(nullptr)
{

}

IGraphsModel* GraphsManagment::currentModel()
{
    return m_model;
}

IGraphsModel* GraphsManagment::openZsgFile(const QString& fn)
{
    GraphsModel* pModel = new GraphsModel(this);
    ModelAcceptor acceptor(pModel);
    m_model = pModel;

    ZsgReader::getInstance().loadZsgFile(fn, &acceptor);
    pModel->clearDirty();
    for (int row = 0; row < pModel->rowCount(); row++)
    {
        SubGraphModel* pSubGraphModel = pModel->subGraph(row);
        QModelIndex subgIdx = pModel->index(row, 0);
        ZenoSubGraphScene* pScene = new ZenoSubGraphScene(this);
        pScene->initModel(subgIdx);
        m_scenes[pSubGraphModel->name()] = pScene;
    }
    m_model = pModel;
    return m_model;
}

IGraphsModel* GraphsManagment::importGraph(const QString& fn)
{
    IGraphsModel *pModel = openZsgFile(fn);
    Q_ASSERT(pModel);
    m_model->setDescriptors(pModel->descriptors());
    for (int i = 0; i < pModel->rowCount(); i++)
    {
        QModelIndex subgIdx = pModel->index(i, 0, QModelIndex());
        QString name = pModel->name(subgIdx);
        //if (SubGraphModel *pExist = m_model->subGraph(name)) {
        //    //todo: reload
        //} else {
        //    
        //    //m_model->appendSubGraph(pSubGraphModel->clone(m_model));
        //}
    }
    return m_model;
}

void GraphsManagment::reloadGraph(const QString& graphName)
{
    if (m_model)
        m_model->reloadSubGraph(graphName);
}

bool GraphsManagment::saveCurrent()
{
    if (!m_model || !m_model->isDirty())
        return false;

    int flag = QMessageBox::question(nullptr, "Save", "Save changes?", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (flag & QMessageBox::Yes)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void GraphsManagment::clear()
{
    if (m_model) {
        //m_model->clear();
        delete m_model;
        m_model = nullptr;
    }
}

void GraphsManagment::removeCurrent()
{
    if (m_model) {
        
    }
}

QList<QAction*> GraphsManagment::getCategoryActions(QModelIndex subgIdx, QPointF scenePos)
{
    NODE_CATES cates = m_model->getCates();
    QList<QAction*> acts;
    if (cates.isEmpty())
    {
        QAction* pAction = new QAction("ERROR: no descriptors loaded!");
        pAction->setEnabled(false);
        acts.push_back(pAction);
        return acts;
    }
    for (const NODE_CATE& cate : cates)
    {
        QAction* pAction = new QAction(cate.name);
        QMenu* pChildMenu = new QMenu;
        pChildMenu->setToolTipsVisible(true);
        for (const QString& name : cate.nodes)
        {
            QAction* pChildAction = pChildMenu->addAction(name);
            //todo: tooltip
            connect(pChildAction, &QAction::triggered, this, [=]() {
                onNewNodeCreated(subgIdx, name, scenePos);
                });
        }
        pAction->setMenu(pChildMenu);
        acts.push_back(pAction);
    }
    return acts;
}

void GraphsManagment::onNewNodeCreated(QModelIndex subgIdx, const QString& descName, const QPointF& pt)
{//called on right-click!!
    zeno::log_info("onNewNodeCreated");
    NODE_DESCS descs = m_model->descriptors();
    const NODE_DESC& desc = descs[descName];

    const QString& nodeid = UiHelper::generateUuid(descName);
    NODE_DATA node;
    node[ROLE_OBJID] = nodeid;
    node[ROLE_OBJNAME] = descName;
    node[ROLE_NODETYPE] = NORMAL_NODE;
    node[ROLE_INPUTS] = QVariant::fromValue(desc.inputs);
    node[ROLE_OUTPUTS] = QVariant::fromValue(desc.outputs);
    node[ROLE_PARAMETERS] = QVariant::fromValue(desc.params);
    node[ROLE_OBJPOS] = pt;

    /*
	NODE_DATA data;
	data[ROLE_OBJID] = nodeid;
	data[ROLE_OBJNAME] = name;
	data[ROLE_COLLASPED] = false;
	if (name == "Blackboard")
	{
		data[ROLE_NODETYPE] = BLACKBOARD_NODE;
	}
	else if (name == "SubInput")
	{
		data[ROLE_NODETYPE] = SUBINPUT_NODE;
	}
	else if (name == "SubOutput")
	{
		data[ROLE_NODETYPE] = SUBOUTPUT_NODE;
	}
	else
	{
		data[ROLE_NODETYPE] = NORMAL_NODE;
	}*/

    //zeno::log_warn("rclk has Inputs {}", node.find(ROLE_PARAMETERS) != node.end());
    m_model->appendItem(node, subgIdx);
}

ZenoSubGraphScene* GraphsManagment::scene(const QString& subGraphName)
{
    return m_scenes[subGraphName];
}
