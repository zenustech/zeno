#include "graphsmanagment.h"
#include <io/zsgreader.h>
#include <model/graphsmodel.h>
#include <zenoio/reader/zsgreader.h>
#include <zenoio/acceptor/modelacceptor.h>


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