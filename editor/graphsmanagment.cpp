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
    if (m_model != nullptr)
    {
        //save first
        m_model = nullptr;
    }
    else
    {
        m_model = new GraphsModel;
        ModelAcceptor acceptor(m_model);
        ZsgReader::getInstance().loadZsgFile(fn, &acceptor);
    }
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