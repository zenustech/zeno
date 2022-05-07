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
    , m_logModel(nullptr)
{
     m_logModel = new QStandardItemModel(this);
}

IGraphsModel* GraphsManagment::currentModel()
{
    return m_model;
}

QStandardItemModel* GraphsManagment::logModel() const
{
    return m_logModel;
}

void GraphsManagment::setCurrentModel(IGraphsModel* model)
{
    m_model = model;
    m_pTreeModel = new GraphsTreeModel(this);
    m_pTreeModel->init(model);
    emit modelInited(m_model);
    connect(m_model, &IGraphsModel::_dataChanged, this, &GraphsManagment::onModelDataChanged);
    connect(m_model, &IGraphsModel::_rowsInserted, this, [=]() {
        emit modelDataChanged();
    });
    connect(m_model, &IGraphsModel::_rowsRemoved, this, [=]() {
        emit modelDataChanged();
    });
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

void GraphsManagment::onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role)
{
    switch (role)
    {
    case ROLE_OBJPOS:
    case ROLE_COLLASPED:
        break;
    default:
        emit modelDataChanged();
        break;
    }
}

void GraphsManagment::removeCurrent()
{
    if (m_model) {
        
    }
}

void GraphsManagment::appendErr(const QString& nodeName, const QString& msg)
{
    QMutexLocker lock(&m_mutex);

    QStandardItem* item = new QStandardItem(msg);
    item->setData(QtFatalMsg, ROLE_LOGTYPE);
    item->setData(nodeName, ROLE_NODENAME);
    item->setEditable(false);
    item->setData(QBrush(QColor(200, 84, 79)), Qt::ForegroundRole);
    m_logModel->appendRow(item);
}

void GraphsManagment::appendLog(QtMsgType type, QString fileName, int ln, const QString &msg)
{
    QMutexLocker lock(&m_mutex);

    QStandardItem *item = new QStandardItem(msg);
    item->setData(type, ROLE_LOGTYPE);
    item->setData(fileName, ROLE_FILENAME);
    item->setData(ln, ROLE_LINENO);
    item->setEditable(false);
    switch (type)
    {
        //todo: time
        case QtDebugMsg:
        {
            item->setData(QBrush(QColor(200, 200, 200, 0.7 * 255)), Qt::ForegroundRole);
            m_logModel->appendRow(item);
            break;
        }
        case QtCriticalMsg:
        {
            item->setData(QBrush(QColor(80, 154, 200)), Qt::ForegroundRole);
            m_logModel->appendRow(item);
            break;
        }
        case QtInfoMsg:
        {
            item->setData(QBrush(QColor(51, 148, 85)), Qt::ForegroundRole);
            m_logModel->appendRow(item);
            break;
        }
        case QtWarningMsg:
        {
            item->setData(QBrush(QColor(200, 154, 80)), Qt::ForegroundRole);
            m_logModel->appendRow(item);
            break;
        }
        case QtFatalMsg:
        {
            item->setData(QBrush(QColor(200, 84, 79)), Qt::ForegroundRole);
            m_logModel->appendRow(item);
            break;
        }
    default:
        delete item;
        break;
    }
}
