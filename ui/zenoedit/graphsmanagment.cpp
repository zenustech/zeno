#include "graphsmanagment.h"
#include "model/graphsmodel.h"
#include <zenoui/model/modelrole.h>
#include "model/graphstreemodel.h"
#include "model/graphsplainmodel.h"
#include <zenoio/reader/zsgreader.h>
#include "acceptor/modelacceptor.h"
#include <zenoui/util/uihelper.h>
#include "nodesys/zenosubgraphscene.h"
#include <zeno/utils/log.h>
#include "zenoapplication.h"


class IOBreakingBatch
{
public:
    IOBreakingBatch() {
        zenoApp->setIOProcessing(true);
    }

    ~IOBreakingBatch() {
        zenoApp->setIOProcessing(false);
    }
};


GraphsManagment::GraphsManagment(QObject* parent)
    : QObject(parent)
    , m_model(nullptr)
    , m_pTreeModel(nullptr)
{

}

IGraphsModel* GraphsManagment::currentModel()
{
    return m_model;
}

void GraphsManagment::setCurrentModel(IGraphsModel* model)
{
    m_model = model;
    m_pTreeModel = new GraphsTreeModel(this);
    m_pTreeModel->init(model);
    emit modelInited(m_model);
}

GraphsTreeModel* GraphsManagment::treeModel()
{
    return m_pTreeModel;
}

IGraphsModel* GraphsManagment::openZsgFile(const QString& fn)
{
    GraphsModel* pModel = new GraphsModel(this);

    {
        IOBreakingBatch batch;
		ModelAcceptor acceptor(pModel, false);
		if (!ZsgReader::getInstance().openFile(fn, &acceptor))
			return nullptr;
    }

    pModel->clearDirty();
    pModel->initDescriptors();
    setCurrentModel(pModel);
    return pModel;
}

IGraphsModel* GraphsManagment::newFile()
{
    GraphsModel* pModel = new GraphsModel(this);
    SubGraphModel* pSubModel = new SubGraphModel(pModel);
    pSubModel->setName("main");
    pModel->appendSubGraph(pSubModel);
    setCurrentModel(pModel);
    return pModel;
}

void GraphsManagment::importGraph(const QString& fn)
{
    if (!m_model)
        return;

	ModelAcceptor acceptor(qobject_cast<GraphsModel*>(m_model), true);
	bool ret = ZsgReader::getInstance().openFile(fn, &acceptor);
	if (!ret)
	{
		zeno::log_error("failed to open zsg file: {}", fn.toStdString());
		return;
	}
    m_model->initDescriptors();
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
    if (m_model)
    {
        m_model->clear();

        delete m_model;
        m_model = nullptr;

        delete m_pTreeModel;
        m_pTreeModel = nullptr;
    }
}

void GraphsManagment::removeCurrent()
{
    if (m_model) {
        
    }
}
