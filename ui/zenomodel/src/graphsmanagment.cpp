#include "graphsmanagment.h"
#include <zenomodel/include/zenomodel.h>
#include <zenomodel/include/modelrole.h>
#include <zenoio/reader/zsgreader.h>
#include <zenomodel/include/uihelper.h>
#include <zeno/utils/log.h>
#include <zeno/utils/scope_exit.h>
#include "common_def.h"
#include <zenoio/writer/zsgwriter.h>


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
    , m_model(nullptr)
    , m_pTreeModel(nullptr)
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
    return m_model;
}

QStandardItemModel* GraphsManagment::logModel() const
{
    return m_logModel;
}

void GraphsManagment::setCurrentModel(IGraphsModel* model)
{
    clear();
    m_model = model;
    m_pTreeModel = zeno_model::treeModel(m_model, this);

    emit modelInited(m_model);
    connect(m_model, SIGNAL(apiBatchFinished()), this, SIGNAL(modelDataChanged()));
    connect(m_model, SIGNAL(rowsAboutToBeRemoved(const QModelIndex&, int, int)),
        this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, int, int)));
    connect(m_model, &IGraphsModel::dirtyChanged, this, [=]() {
        emit dirtyChanged(m_model->isDirty());
    });
}

QAbstractItemModel* GraphsManagment::treeModel()
{
    return m_pTreeModel;
}

IGraphsModel* GraphsManagment::openZsgFile(const QString& fn)
{
    IGraphsModel* pModel = zeno_model::createModel(this);

    {
        IOBreakingScope batch(pModel);
        std::shared_ptr<IAcceptor> acceptor(zeno_model::createIOAcceptor(pModel, false));
        bool ret = ZsgReader::getInstance().openFile(fn, acceptor.get());
        m_timerInfo = acceptor->timeInfo();
        if (!ret)
            return nullptr;
    }

    pModel->clearDirty();
    setCurrentModel(pModel);
    emit fileOpened(fn);
    return pModel;
}

bool GraphsManagment::saveFile(const QString& filePath, APP_SETTINGS settings)
{
    if (m_model == nullptr) {
        zeno::log_error("The current model is empty.");
        return false;
    }

    QString strContent = ZsgWriter::getInstance().dumpProgramStr(m_model, settings);
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

    m_model->setFilePath(filePath);
    m_model->clearDirty();

    QFileInfo info(filePath);
    emit fileSaved(info.fileName());
    return true;
}

IGraphsModel* GraphsManagment::newFile()
{
    IGraphsModel* pModel = zeno_model::createModel(this);
    pModel->initMainGraph();
    setCurrentModel(pModel);
    return pModel;
}

void GraphsManagment::importGraph(const QString& fn)
{
    if (!m_model)
        return;

    IOBreakingScope batch(m_model);
    std::shared_ptr<IAcceptor> acceptor(zeno_model::createIOAcceptor(m_model, true));
	if (!ZsgReader::getInstance().openFile(fn, acceptor.get()))
	{
		zeno::log_error("failed to open zsg file: {}", fn.toStdString());
		return;
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

        for (auto scene : m_scenes)
        {
            delete scene;
        }
        m_scenes.clear();
    }
    emit fileClosed();
}

void GraphsManagment::onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    const QModelIndex& idx = m_model->index(first, 0);
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
    if (m_model) {
        
    }
}

QGraphicsScene* GraphsManagment::gvScene(const QModelIndex& subgIdx) const
{
    if (!subgIdx.isValid())
        return nullptr;

    const QString& subgName = subgIdx.data(ROLE_OBJNAME).toString();
    if (m_scenes.find(subgName) == m_scenes.end())
        return nullptr;

    return m_scenes[subgName];
}

void GraphsManagment::addScene(const QModelIndex& subgIdx, QGraphicsScene* scene)
{
    const QString& subgName = subgIdx.data(ROLE_OBJNAME).toString();
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

    bool ret = m_mutex.tryLock(0);
    if (!ret)
        return;

    zeno::scope_exit sp([=]() { m_mutex.unlock(); });

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

    bool ret = m_mutex.tryLock(0);
    if (!ret)
        return;

    zeno::scope_exit sp([=]() { m_mutex.unlock(); });

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
