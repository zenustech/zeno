#include "graphsmanagment.h"
#include <zenoui/model/graphsmodel.h>
#include <zenoui/model/modelrole.h>
#include "model/graphstreemodel.h"
#include <zenoio/reader/zsgreader.h>
#include "acceptor/modelacceptor.h"
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

    bool ret = ZsgReader::getInstance().loadZsgFile(fn, &acceptor);
    if (!ret) return nullptr;

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
    if (!pModel) {
        zeno::log_error("failed to open zsg file: {}", fn.toStdString());
        return nullptr;
    }
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

ZenoSubGraphScene* GraphsManagment::scene(const QString& subGraphName)
{
    return m_scenes[subGraphName];
}
