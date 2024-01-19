#include "graphsmanager.h"
#include "model/graphstreemodel.h"
#include "model/assetsmodel.h"
#include "uicommon.h"
#include <zenoio/reader/zsg2reader.h>
#include <zeno/utils/log.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/core/Graph.h>
#include "util/uihelper.h"
#include <zenoio/writer/zsgwriter.h>
#include <zeno/core/Session.h>
#include <zeno/types/UserData.h>
#include "nodeeditor/gv/zenosubgraphscene.h"
#include "zassert.h"


GraphsManager::GraphsManager(QObject* parent)
    : QObject(parent)
    , m_model(nullptr)
    , m_logModel(nullptr)
    , m_assets(nullptr)
{
    m_logModel = new QStandardItemModel(this);
    m_model = new GraphsTreeModel(this);
    GraphModel* main = new GraphModel(zeno::getSession().mainGraph, m_model, m_model);
    m_model->init(main);
    m_assets = new AssetsModel(this);
}

GraphsManager::~GraphsManager()
{
}

GraphsManager& GraphsManager::instance() {
    static GraphsManager inst;
    return inst;
}

void GraphsManager::registerCoreNotify() {

}

GraphsTreeModel* GraphsManager::currentModel() const
{
    return m_model;
}

AssetsModel* GraphsManager::assetsModel() const
{
    return m_assets;
}

QStandardItemModel* GraphsManager::logModel() const
{
    return m_logModel;
}

GraphModel* GraphsManager::getGraph(const QString& objPath) const
{
    if (objPath.startsWith("/")) {
        return m_model ? m_model->getGraphByPath(objPath) : nullptr;
    }
    else {
        return m_assets ? m_assets->getAsset(objPath) : nullptr;
    }
}

GraphsTreeModel* GraphsManager::openZsgFile(const QString& fn)
{
    zeno::ZSG_PARSE_RESULT result;
    bool ret = zenoio::Zsg2Reader::getInstance().openFile(fn.toStdString(), result);
    m_timerInfo = result.timeline;
    if (!ret)
        return nullptr;

    createGraphs(result);
    emit fileOpened(fn);
    return m_model;
}

void GraphsManager::createGraphs(const zeno::ZSG_PARSE_RESULT ioresult)
{
    //todo
    /*
    if (m_assets)
        m_assets->clear();

    delete m_assets;
    m_assets = new AssetsModel(this);
    */
    ZASSERT_EXIT(m_assets);
    m_assets->init(ioresult.assetGraphs);
}

bool GraphsManager::saveFile(const QString& filePath, APP_SETTINGS settings)
{
    if (m_model == nullptr) {
        zeno::log_error("The current model is empty.");
        return false;
    }

    //todo: writer.
    QString strContent;// = ZsgWriter::getInstance().dumpProgramStr(m_model, settings);
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
    zeno::log_info("saved '{}' successfully", filePath.toStdString());

    m_filePath = filePath;

    m_model->clearDirty();

    QFileInfo info(filePath);
    emit fileSaved(filePath);
    return true;
}

GraphsTreeModel* GraphsManager::newFile()
{
    clear();

    if (!m_model) {
        m_model = new GraphsTreeModel(this);
        auto main = new GraphModel(zeno::getSession().mainGraph, m_model, m_model);
        m_model->init(main);
    }

    //TODO: assets may be kept.

    emit modelInited();

    connect(m_model, SIGNAL(apiBatchFinished()), this, SIGNAL(modelDataChanged()));
    connect(m_model, SIGNAL(rowsAboutToBeRemoved(const QModelIndex&, int, int)),
        this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, int, int)));
    connect(m_model, &GraphsTreeModel::dirtyChanged, this, [=]() {
        emit dirtyChanged(m_model->isDirty());
    });

    return m_model;
}

void GraphsManager::importGraph(const QString& fn)
{
    //todo: the function needs to be refactor.
}

void GraphsManager::importSubGraphs(const QString& fn, const QMap<QString, QString>& map)
{
    //todo: the function needs to be refactor.
}

void GraphsManager::clear()
{
    if (m_model)
    {
        m_model->clear();

        //no need this delete.
        //delete m_model;
        //m_model = nullptr;

        for (auto scene : m_scenes)
        {
            delete scene;
        }
        m_scenes.clear();
    }
    emit fileClosed();
}

void GraphsManager::onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    const QModelIndex& idx = m_model->index(first, 0);
    if (idx.isValid())
    {
        const QString& subgName = idx.data(ROLE_CLASS_NAME).toString();
        if (m_scenes.find(subgName) != m_scenes.end())
        {
            delete m_scenes[subgName];
            m_scenes.remove(subgName);
        }
    }
}

void GraphsManager::onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role)
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

void GraphsManager::removeCurrent()
{
    if (m_model) {
        
    }
}

QGraphicsScene* GraphsManager::gvScene(const QString& tabName) const
{
    if (m_scenes.find(tabName) != m_scenes.end())
        return nullptr;
    return m_scenes[tabName];
}

QGraphicsScene* GraphsManager::gvScene(const QModelIndex& subgIdx) const
{
    if (!subgIdx.isValid())
        return nullptr;

    const QString& subgName = subgIdx.data(ROLE_CLASS_NAME).toString();
    if (m_scenes.find(subgName) == m_scenes.end())
        return nullptr;

    return m_scenes[subgName];
}

void GraphsManager::addScene(const QModelIndex& subgIdx, ZenoSubGraphScene* scene)
{
    const QString& subgName = subgIdx.data(ROLE_CLASS_NAME).toString();
    if (m_scenes.find(subgName) != m_scenes.end() || !scene)
        return;
    m_scenes.insert(subgName, scene);
}

void GraphsManager::addScene(const QString& tabName, ZenoSubGraphScene* scene)
{
    if (m_scenes.find(tabName) != m_scenes.end())
        return;
    m_scenes.insert(tabName, scene);
}

zeno::TimelineInfo GraphsManager::timeInfo() const
{
    return m_timerInfo;
}

QString GraphsManager::zsgPath() const
{
    return m_filePath;
}

QString GraphsManager::zsgDir() const
{
    const QString& zsgpath = zsgPath();
    QFileInfo fp(zsgpath);
    return fp.absolutePath();
}

USERDATA_SETTING GraphsManager::userdataInfo() const
{
    //TODO
    return USERDATA_SETTING();
}

RECORD_SETTING GraphsManager::recordSettings() const
{
    return RECORD_SETTING();
}

zeno::NodeCates GraphsManager::getCates() const
{
    zeno::NodeCates cates = zeno::getSession().dumpCoreCates();
    //TODO: collect assets.
    return cates;
}

zeno::ZSG_VERSION GraphsManager::ioVersion() const
{
    return zeno::VER_3;
}

void GraphsManager::setIOVersion(zeno::ZSG_VERSION ver)
{
    //??
}

void GraphsManager::clearMarkOnGv() {
    for (QString name : m_scenes.keys()) {
        if (name.startsWith("/")) {
            m_scenes[name]->clearMark();
        }
    }
}

void GraphsManager::appendErr(const QString& nodeName, const QString& msg)
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

void GraphsManager::appendLog(QtMsgType type, QString fileName, int ln, const QString &msg)
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
