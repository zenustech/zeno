#include "graphsmanagment.h"
#include <io/zsgreader.h>
#include <model/graphsmodel.h>
#include <zenoio/reader/zsgreader.h>
#include <zenoio/acceptor/modelacceptor.h>
#include <zenoui/util/uihelper.h>


GraphsManagment::GraphsManagment(QObject* parent)
    : QObject(parent)
    , m_model(nullptr)
{

}

GraphsModel* GraphsManagment::currentModel()
{
    return m_model;
}

GraphsModel* GraphsManagment::openZsgFile(const QString& fn)
{
    m_model = new GraphsModel(this);
    ModelAcceptor acceptor(m_model);
    ZsgReader::getInstance().loadZsgFile(fn, &acceptor);
    m_model->clearDirty();
    return m_model;
}

GraphsModel* GraphsManagment::importGraph(const QString& fn)
{
    GraphsModel *pModel = openZsgFile(fn);
    Q_ASSERT(pModel);
    m_model->setDescriptors(pModel->descriptors());
    for (int i = 0; i < pModel->rowCount(); i++)
    {
        SubGraphModel *pSubGraphModel = pModel->subGraph(i);
        QString name = pSubGraphModel->name();
        if (SubGraphModel *pExist = m_model->subGraph(name)) {
            //todo: reload
        } else {
            m_model->appendSubGraph(pSubGraphModel->clone(m_model));
        }
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
        m_model->clear();
        delete m_model;
        m_model = nullptr;
    }
}

void GraphsManagment::removeCurrent()
{
    if (m_model) {
        m_model->onRemoveCurrentItem();
    }
}

QList<QAction*> GraphsManagment::getCategoryActions(QPointF scenePos)
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
                onNewNodeCreated(name, scenePos);
                });
        }
        pAction->setMenu(pChildMenu);
        acts.push_back(pAction);
    }
    return acts;
}

void GraphsManagment::onNewNodeCreated(const QString& descName, const QPointF& pt)
{
    NODE_DESCS descs = m_model->descriptors();
    const NODE_DESC& desc = descs[descName];

    const QString& nodeid = UiHelper::generateUuid(descName);
    NODE_DATA node;
    node[ROLE_OBJID] = nodeid;
    node[ROLE_OBJNAME] = descName;
    node[ROLE_OBJTYPE] = NORMAL_NODE;
    node[ROLE_INPUTS] = QVariant::fromValue(desc.inputs);
    node[ROLE_OUTPUTS] = QVariant::fromValue(desc.outputs);
    node[ROLE_PARAMETERS] = QVariant::fromValue(desc.params);
    node[ROLE_OBJPOS] = pt;

    SubGraphModel* pModel = m_model->currentGraph();
    int row = pModel->rowCount();
    pModel->appendItem(node, true);
}