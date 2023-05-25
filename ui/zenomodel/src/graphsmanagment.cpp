#include "graphsmanagment.h"
#include <zenomodel/include/zenomodel.h>
#include <zenomodel/include/modelrole.h>
#include <zenoio/reader/zsgreader.h>
#include <zenomodel/include/uihelper.h>
#include <zeno/utils/log.h>
#include <zeno/utils/scope_exit.h>
#include "common_def.h"
#include <zenoio/writer/zsgwriter.h>
#include "graphstreemodel.h"


class IOBreakingScope
{
public:
    IOBreakingScope(IGraphsModel* model) : m_model(model) {
        if (m_model)
            m_model->setIOProcessing(true);
    }

    ~IOBreakingScope() {
        if (m_model)
            m_model->setIOProcessing(false);
    }

private:
    IGraphsModel* m_model;
};


GraphsManagment::GraphsManagment(QObject* parent)
    : QObject(parent)
    , m_pNodeModel(nullptr)
    , m_logModel(nullptr)
{
     m_logModel = new QStandardItemModel(this);
}

GraphsManagment::~GraphsManagment()
{
}

GraphsManagment& GraphsManagment::instance() {
    static GraphsManagment inst;
    return inst;
}

IGraphsModel* GraphsManagment::currentModel()
{
    return m_pNodeModel;
}

IGraphsModel* GraphsManagment::sharedSubgraphs()
{
    return m_pSharedGraphs;
}

QStandardItemModel* GraphsManagment::logModel() const
{
    return m_logModel;
}

void GraphsManagment::setGraphsModel(IGraphsModel* model, IGraphsModel* pSubgraphsModel)
{
    clear();
    m_pNodeModel = model;
    m_pSharedGraphs = pSubgraphsModel;

    emit modelInited(m_pNodeModel, m_pSharedGraphs);
    connect(m_pNodeModel, SIGNAL(apiBatchFinished()), this, SIGNAL(modelDataChanged()));
    connect(m_pNodeModel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex&, int, int)),
        this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, int, int)));
    connect(m_pNodeModel, &IGraphsModel::dirtyChanged, this, [=]() {
        emit dirtyChanged(m_pNodeModel->isDirty());
    });
}

IGraphsModel* GraphsManagment::openZsgFile(const QString& fn)
{
    IGraphsModel* pNodeModel = zeno_model::createModel(false, this);
    //todo: when the io version is 2.5, should reuse the io code to init subgraph model.
    IGraphsModel *pSubgraphsModel = zeno_model::createModel(true, this);
    {
        IOBreakingScope batch(pNodeModel);
        std::shared_ptr<IAcceptor> acceptor(zeno_model::createIOAcceptor(pNodeModel, pSubgraphsModel, false));
        ZASSERT_EXIT(acceptor, nullptr);
        bool ret = ZsgReader::getInstance().openFile(fn, acceptor.get());
        m_timerInfo = acceptor->timeInfo();
        if (!ret)
            return nullptr;
    }

    pNodeModel->clearDirty();
    setGraphsModel(pNodeModel, pSubgraphsModel);
    emit fileOpened(fn);
    return pNodeModel;
}

bool GraphsManagment::saveFile(const QString& filePath, APP_SETTINGS settings)
{
    if (m_pNodeModel == nullptr) {
        zeno::log_error("The current model is empty.");
        return false;
    }

    QString strContent = ZsgWriter::getInstance().dumpProgramStr(m_pNodeModel, settings);
    QFile f(filePath);
    zeno::log_debug("saving {} chars to file [{}]", strContent.size(), filePath.toStdString());
    if (!f.open(QIODevice::WriteOnly)) {
        qWarning() << Q_FUNC_INFO << "Failed to open" << filePath << f.errorString();
        zeno::log_error("Failed to open file for write: {} ({})", filePath.toStdString(),
                        f.errorString().toStdString());
        return false;
    }

    f.write(strContent.toUtf8());
    f.close();
    zeno::log_debug("saved successfully");

    m_pNodeModel->setFilePath(filePath);
    m_pNodeModel->clearDirty();

    QFileInfo info(filePath);
    emit fileSaved(filePath);
    return true;
}

IGraphsModel* GraphsManagment::newFile()
{
    IGraphsModel *pSubgrahsModel = zeno_model::createModel(true, this);
    IGraphsModel* pModel = zeno_model::createModel(false, this);
    pModel->initMainGraph();

    GraphsTreeModel *pNodeModel = qobject_cast<GraphsTreeModel *>(pModel);
    ZASSERT_EXIT(pNodeModel, nullptr);
    pNodeModel->initSubgraphs(pSubgrahsModel);

    setGraphsModel(pModel, pSubgrahsModel);
    return pModel;
}

void GraphsManagment::importGraph(const QString& fn)
{
    if (!m_pSharedGraphs)
        return;

    IOBreakingScope batch(m_pSharedGraphs);
    std::shared_ptr<IAcceptor> acceptor(zeno_model::createIOAcceptor(m_pSharedGraphs, nullptr, true));
    ZASSERT_EXIT(acceptor);
    if (!ZsgReader::getInstance().openFile(fn, acceptor.get()))
    {
        zeno::log_error("failed to open zsg file: {}", fn.toStdString());
        return;
    }
}

void GraphsManagment::clear()
{
    if (m_pNodeModel)
    {
        m_pNodeModel->clear();

        delete m_pNodeModel;
        m_pNodeModel = nullptr;

        for (auto scene : m_scenes)
        {
            delete scene;
        }
        m_scenes.clear();
    }
    if (m_pSharedGraphs)
    {
        m_pSharedGraphs->clear();
        delete m_pSharedGraphs;
        m_pSharedGraphs = nullptr;
    }
    emit fileClosed();
}

void GraphsManagment::onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    const QModelIndex& idx = m_pNodeModel->index(first, 0);
    if (idx.isValid())
    {
        const QString& subgName = idx.data(ROLE_OBJNAME).toString();
        if (m_scenes.find(subgName) != m_scenes.end())
        {
            delete m_scenes[subgName];
            m_scenes.remove(subgName);
        }
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
    if (m_pNodeModel) {
        
    }
}

QGraphicsScene* GraphsManagment::gvScene(const QModelIndex& subgIdx) const
{
    if (!subgIdx.isValid())
        return nullptr;

    const QString& subgName = subgIdx.data(ROLE_OBJPATH).toString();
    if (m_scenes.find(subgName) == m_scenes.end())
        return nullptr;
    return m_scenes[subgName];
}

void GraphsManagment::addScene(const QModelIndex& subgIdx, QGraphicsScene* scene)
{
    const QString& subgName = subgIdx.data(ROLE_OBJPATH).toString();
    if (m_scenes.find(subgName) != m_scenes.end() || !scene)
        return;
    m_scenes.insert(subgName, scene);
}

TIMELINE_INFO GraphsManagment::timeInfo() const
{
    return m_timerInfo;
}

void GraphsManagment::appendErr(const QString& nodeName, const QString& msg)
{
    if (msg.trimmed().isEmpty())
        return;

    QStandardItem* item = new QStandardItem(msg);
    item->setData(QtFatalMsg, ROLE_LOGTYPE);
    item->setData(nodeName, ROLE_NODE_IDENT);
    item->setEditable(false);
    item->setData(QBrush(QColor(200, 84, 79)), Qt::ForegroundRole);
    m_logModel->appendRow(item);
}

void GraphsManagment::appendLog(QtMsgType type, QString fileName, int ln, const QString &msg)
{
    if (msg.trimmed().isEmpty())
        return;

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
