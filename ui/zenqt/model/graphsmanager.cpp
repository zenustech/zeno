#include "graphsmanager.h"
#include "GraphsTreeModel.h"
#include "assetsmodel.h"
#include "uicommon.h"
#include "dialog/zeditparamlayoutdlg.h"
#include "dialog/znewassetdlg.h"
#include "customuimodel.h"
#include <zeno/io/zsg2reader.h>
#include <zeno/utils/log.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/io/iohelper.h>
#include "util/uihelper.h"
#include <zeno/io/zenwriter.h>
#include <zeno/io/zenreader.h>
#include <zeno/core/Session.h>
#include "nodeeditor/gv/zenosubgraphscene.h"
#include "zassert.h"
#include "variantptr.h"
#include "parammodel.h"
#include "pluginsmodel.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "settings/zsettings.h"
#include <QQmlContext>


GraphsTotalView::GraphsTotalView()
    : _graphMgr(zenoApp->graphsManager())
{
    connect(_graphMgr, &GraphsManager::modelInited, this, &GraphsTotalView::modelInited);
    connect(_graphMgr, &GraphsManager::modelDataChanged, this, &GraphsTotalView::modelDataChanged);
    connect(_graphMgr, &GraphsManager::fileOpened, this, &GraphsTotalView::fileOpened);
    connect(_graphMgr, &GraphsManager::fileClosed, this, &GraphsTotalView::fileClosed);
    connect(_graphMgr, &GraphsManager::fileSaved, this, &GraphsTotalView::fileSaved);
    connect(_graphMgr, &GraphsManager::dirtyChanged, this, &GraphsTotalView::dirtyChanged);
}

GraphsTotalView::GraphsTotalView(const GraphsTotalView& graphsview) {
    _graphMgr = graphsview._graphMgr;
}

void GraphsTotalView::setGraphsMgr(GraphsManager* graph) {
    _graphMgr = zenoApp->graphsManager();
}

void GraphsTotalView::newFile() {
    zenoApp->getMainWindow()->onNewFile();
}

void GraphsTotalView::openFile() {
    zenoApp->getMainWindow()->openFileDialog();
}


GraphsManager::GraphsManager(QObject* parent)
    : QObject(parent)
    , m_treemodel(nullptr)
    , m_logModel(nullptr)
    , m_assets(nullptr)
    , m_main(nullptr)
    , m_version(zeno::VER_3)
    , m_bIniting(false)
    , m_bImporting(false)
    , m_nodeDelegate(nullptr)
    , m_edgeDelegate(nullptr)
{
    m_logModel = new QStandardItemModel(this);
    m_treemodel = new GraphsTreeModel(this);
    //m_main = new GraphModel("/main", false, m_model, this);
    //m_model->init(m_main);
    m_assets = new AssetsModel(this);
    m_plugins = new PluginsModel(this);
}

GraphsManager::~GraphsManager()
{
}

void GraphsManager::initRootObjects() {
    QQmlApplicationEngine* engine = zenoApp->getQmlEngine();
    engine->rootContext()->setContextProperty("graphsmanager", this);
    if (m_main)
        engine->rootContext()->setContextProperty("nodesModel", m_main);
    if (m_treemodel)
        engine->rootContext()->setContextProperty("treeModel", m_treemodel);
    if (m_assets)
        engine->rootContext()->setContextProperty("assetsModel", m_assets);
    if (m_plugins)
        engine->rootContext()->setContextProperty("pluginsModel", m_plugins);
}

void GraphsManager::registerCoreNotify() {

}

QQmlComponent* GraphsManager::nodeDelegate() {
    if (!m_nodeDelegate) {
        m_nodeDelegate = new QQmlComponent(zenoApp->getQmlEngine(), "qrc:/NormalNode.qml",
            QQmlComponent::PreferSynchronous, this);
    }
    return m_nodeDelegate;
}

QQmlComponent* GraphsManager::edgeDelegate() {
    if (!m_edgeDelegate) {
        m_edgeDelegate = new QQmlComponent(zenoApp->getQmlEngine(), "qrc:/QuickQanava/Edge.qml",
            QQmlComponent::PreferSynchronous, this);
    }
    return m_edgeDelegate;
}

AssetsModel* GraphsManager::assetsModel() const
{
    return m_assets;
}

PluginsModel* GraphsManager::pluginModel() const
{
    return m_plugins;
}

QStandardItemModel* GraphsManager::logModel() const
{
    return m_logModel;
}

GraphModel* GraphsManager::getGraph(const QStringList& objPath) const
{
    if (objPath.empty())
        return nullptr;

    if (objPath[0] == "main") {
        QStringList _path = objPath;
        _path.removeFirst();
        if (!m_main) return nullptr;
        return m_main->getGraphByPath(_path);
    }
    else {
        QStringList assetGraphPath = objPath;
        GraphModel* pModel = m_assets->getAssetGraph(assetGraphPath[0]);
        if (!pModel) return nullptr;

        assetGraphPath.removeAt(0);
        if (assetGraphPath.isEmpty())
            return pModel;

        return pModel->getGraphByPath(assetGraphPath);
    }
}

QString GraphsManager::currentGraphPath() const
{
    return m_graphPath;
}

void GraphsManager::setCurrentGraphPath(const QString& path)
{
    m_graphPath = path;
    emit currentPathChanged(m_graphPath);
}

GraphsTreeModel* GraphsManager::openZsgFile(const QString& fn, zenoio::ERR_CODE& code)
{
    zenoio::ZSG_PARSE_RESULT result;

    result.path = fn.toStdWString();
    zeno::getSession().globalState->clearState();
    zeno::getSession().init_project_path(result.path);
    zeno::ZSG_VERSION ver = zenoio::getVersion(fn.toStdWString());

    m_bIniting = true;
    zeno::scope_exit sp([=] { m_bIniting = false; });
    /*if (ver == zeno::VER_2_5) {
        zenoio::Zsg2Reader reader;
        result = reader.openFile(result.path);
    }
    else */if (ver == zeno::VER_3) {
        zenoio::ZenReader reader;
        result = reader.openFile(result.path);
        result.num_of_nodes = reader.numOfNodes();
    }
    else {
        m_version = zeno::UNKNOWN_VER;
        result.code = zenoio::PARSE_VERSION_UNKNOWN;
    }

    if (result.code != zenoio::PARSE_NOERROR)
    {
        code = result.code;
        return nullptr;
    }

    m_version = ver;

    zenoApp->getMainWindow()->updateStatusTip(false, "parse finish, now importing");

    if (false) {
        zeno::getSession().registerIOCallback([&](const std::string& info, int inc) {
        auto& gState = zeno::getSession().globalState;
        gState->inc_io_processed(inc);
        int io_processed = gState->get_io_processed();
        float value = 0;
        bool bShowProgress = false;
        QString text;
        if (io_processed >= result.num_of_nodes) {
            value = 1.0;
            text = "Open Project Finish";
        }
        else {
            value = (float)io_processed / result.num_of_nodes;
            text = QString::fromStdString(info);
            bShowProgress = true;
        }
        zenoApp->getMainWindow()->updateStatusTip(bShowProgress, text, value);
    });
    }

    m_timerInfo = result.timeline;
    createGraphs(result);
    //reset model.
    newFile();
    m_filePath = fn;

    emit fileOpened(fn);
    m_treemodel->markDirty(false);
    return m_treemodel;
}

bool GraphsManager::isInitializing() const
{
    return m_bIniting;
}

bool GraphsManager::isImporting() const
{
    return m_bImporting;
}

void GraphsManager::createGraphs(const zenoio::ZSG_PARSE_RESULT ioresult)
{
    ZASSERT_EXIT(m_assets);
    zeno::getSession().initEnv(ioresult);
}

bool GraphsManager::saveFile(const QString& filePath, APP_SETTINGS)
{
    if (m_treemodel == nullptr) {
        zeno::log_error("The current model is empty.");
        return false;
    }

    zenoio::AppSettings settings;       //TODO:
    settings.timeline = zenoApp->getMainWindow()->timelineInfo();

    zeno::GraphData graph = zeno::getSession().mainGraph->exportGraph();

    zenoio::ZenWriter writer;
    writer.set_proj_path(filePath.toStdString());
    std::string strContent = writer.dumpProgramStr(graph, settings);
    QFile f(filePath);
    zeno::log_debug("saving {} chars to file [{}]", strContent.size(), filePath.toStdString());
    if (!f.open(QIODevice::WriteOnly)) {
        qWarning() << Q_FUNC_INFO << "Failed to open" << filePath << f.errorString();
        zeno::log_error("Failed to open file for write: {} ({})", filePath.toStdString(),
                        f.errorString().toStdString());
        return false;
    }

    f.write(strContent.c_str());
    f.close();
    zeno::log_info("saved '{}' successfully", filePath.toStdString());

    m_filePath = filePath;

    m_treemodel->clearDirty();

    QFileInfo info(filePath);
    emit fileSaved(filePath);
    return true;
}

GraphsTreeModel* GraphsManager::newFile()
{
    auto& sess = zeno::getSession();
    clear();
    if (!sess.mainGraph) {
        sess.resetMainGraph();
    }
    m_main = new GraphModel("/main", false, m_treemodel, nullptr, this);
    m_treemodel->init(m_main);

    m_graphPath = "/main";

    //TODO: assets may be kept.
    initRootObjects();

    emit modelInited();

    connect(m_treemodel, SIGNAL(rowsAboutToBeRemoved(const QModelIndex&, int, int)),
        this, SLOT(onRowsAboutToBeRemoved(const QModelIndex&, int, int)));
    connect(m_treemodel, &GraphsTreeModel::dirtyChanged, this, [=]() {
        emit dirtyChanged(m_treemodel->isDirty());
    });

    return m_treemodel;
}

void GraphsManager::importGraph(const QString& fn)
{
    m_bImporting = true;
    zeno::scope_exit sp([=] { m_bImporting = false; });
    //todo: the function needs to be refactor.
}

void GraphsManager::importSubGraphs(const QString& fn, const QMap<QString, QString>& map)
{
    m_bImporting = true;
    zeno::scope_exit sp([=] { m_bImporting = false; });
    //todo: the function needs to be refactor.
}

void GraphsManager::clear()
{
    if (m_treemodel)
    {
        m_treemodel->clear();
        for (auto scene : m_scenes)
        {
            delete scene;
        }
        m_scenes.clear();
    }

    //前面delete main的时候会removeNode，导致treemodel又脏了})
    zeno::scope_exit sp([&] {    m_treemodel->markDirty(false); });

    //先清理UIModel，再清理内核模型
    if (m_main) {
        m_main->clear();
        delete m_main;
        m_main = nullptr;

        zeno::getSession().clearMainGraph();
        zeno::getSession().clearAssets();
    }
    m_filePath = "";
    emit fileClosed();
}

void GraphsManager::onRowsAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    //TODO: deprecated.
    /*
    const QModelIndex& idx = m_model->index(first, 0);
    if (idx.isValid())
    {
        const QString& subgName = idx.data(QtRole::ROLE_CLASS_NAME).toString();
        if (m_scenes.find(subgName) != m_scenes.end())
        {
            delete m_scenes[subgName];
            m_scenes.remove(subgName);
        }
    }
    */
}

void GraphsManager::onModelDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role)
{
    switch (role)
    {
    case QtRole::ROLE_OBJPOS:
    case QtRole::ROLE_COLLASPED:
        break;
    default:
        emit modelDataChanged();
        break;
    }
}

void GraphsManager::removeCurrent()
{
    if (m_treemodel) {
        
    }
}

QGraphicsScene* GraphsManager::gvScene(const QStringList& graphPath) const
{
    for (auto scene : m_scenes) {
        auto pModel = scene->getGraphModel();
        auto path = pModel->currentPath();
        if (path == graphPath)
            return scene;
    }
    return nullptr;
}

QGraphicsScene* GraphsManager::gvScene(const QModelIndex& subgIdx) const
{
    return nullptr;
    /*
    if (!subgIdx.isValid())
        return nullptr;

    const QString& subgName = subgIdx.data(QtRole::ROLE_CLASS_NAME).toString();
    if (m_scenes.find(subgName) == m_scenes.end())
        return nullptr;

    return m_scenes[subgName];
    */
}

QMap<QStringList, QGraphicsScene*> GraphsManager::gvSubScenes(const QStringList& graphName) const
{
    QMap<QStringList, QGraphicsScene*> pathSceneMap;
    QString graphPath = graphName.join("");
    for (auto scene : m_scenes) {
        auto pModel = scene->getGraphModel();
        auto path = pModel->currentPath();
        if (path.join("").contains(graphPath)) {
            pathSceneMap.insert(path, scene);
        }
    }
    return pathSceneMap;
}

void GraphsManager::addScene(const QModelIndex& subgIdx, ZenoSubGraphScene* scene)
{
    //TODO: deprecated
    /*
    const QString& subgName = subgIdx.data(QtRole::ROLE_CLASS_NAME).toString();
    if (m_scenes.find(subgName) != m_scenes.end() || !scene)
        return;
    m_scenes.insert(subgName, scene);
    */
}

void GraphsManager::addScene(const QStringList& graphPath, ZenoSubGraphScene* scene)
{
    for (auto scene : m_scenes) {
        auto path = scene->getGraphModel()->currentPath();
        if (path == graphPath)
            return;
    }
    m_scenes.push_back(scene);
}

bool GraphsManager::removeScene(const QStringList& graphPath)
{
    for (int i = 0; i < m_scenes.size(); i++) {
        auto path = m_scenes[i]->getGraphModel()->currentPath();
        if (path == graphPath)
        {
            delete m_scenes[i];
            m_scenes.removeAt(i);
            return true;
        }
    }
    return false;
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

void GraphsManager::saveProject(const QString& name)
{
    if (name == "main") {
        zenoApp->getMainWindow()->save();
    }
    else {
        assetsModel()->saveAsset(name);
        if (m_main)
        {
            m_main->syncToAssetsInstance(name);
        }
    }
}

void GraphsManager::undo(const QStringList& path)
{
    if (path.isEmpty()) return;

    QString graph(path[0]);
    if (graph == "main") {
        m_main->undo();
    }
    else {
        GraphModel* pModel = m_assets->getAssetGraph(graph);
        ZASSERT_EXIT(pModel);
        pModel->undo();
    }
}

void GraphsManager::redo(const QStringList& path)
{
    if (path.isEmpty()) return;

    QString graph(path[0]);
    if (graph == "main") {
        m_main->redo();
    }
    else {
        GraphModel* pModel = m_assets->getAssetGraph(graph);
        ZASSERT_EXIT(pModel);
        pModel->redo();
    }
}

void GraphsManager::openProject(const QString& zsgpath) {
    zenoApp->getMainWindow()->openFile(zsgpath);
}

void GraphsManager::openCustomUIDialog(CustomUIModel* customUIM) {
    if (!customUIM)
        return;
    auto mainWin = zenoApp->getMainWindow();
    ZEditParamLayoutDlg dlg(customUIM, mainWin);
    if (QDialog::Accepted == dlg.exec())
    {
        zeno::ParamsUpdateInfo info = dlg.getEdittedUpdateInfo();
        ParamsModel* paramsM = customUIM->coreModel();
        const zeno::CustomUI& ui = dlg.getCustomUiInfo();
        paramsM->resetCustomUi(ui);
        paramsM->batchModifyParams(info);
    }
}

void GraphsManager::onAssetsCustomUIDialog(const QString& assetsName) {
    auto& assetsMgr = zeno::getSession().assets;
    const auto& name = assetsName.toStdString();
    const zeno::Asset& asset = assetsMgr->getAsset(name);
    auto mainWin = zenoApp->getMainWindow();

    //ensure the graph be loaded.
    assetsMgr->getAssetGraph(name, true);

    ParamsModel paramsM(asset.m_customui);
    CustomUIModel* pCustomM = paramsM.customUIModel();
    ZEditParamLayoutDlg dlg(pCustomM, mainWin);
    if (QDialog::Accepted == dlg.exec())
    {
        auto graphsMgr = zenoApp->graphsManager();
        zeno::ParamsUpdateInfo info = dlg.getEdittedUpdateInfo();
        zeno::CustomUI customui = dlg.getCustomUiInfo();
        updateAssets(assetsName, info, customui);
    }
}

void GraphsManager::onNodeSelected(const QStringList& graphs_path, const QModelIndex& idx) {
    if (graphs_path.empty())
        return;
    auto mainWin = zenoApp->getMainWindow();
    auto graphM = getGraph(graphs_path);
    mainWin->onNodesSelected(graphM, { idx }, true);
}

void GraphsManager::createAssetDialog()
{
    ZNewAssetDlg dlg(zenoApp->getMainWindow());
    if (QDialog::Accepted == dlg.exec())
    {
        zeno::AssetInfo asset = dlg.getAsset();
        AssetsModel* pModel = zenoApp->graphsManager()->assetsModel();
        pModel->newAsset(asset);
    }
}

void GraphsManager::loadAssetDialog()
{
    //TODO
}

void GraphsManager::addPlugin()
{
    ZenoMainWindow* mainWin = zenoApp->getMainWindow();
    QString filePath = QFileDialog::getOpenFileName(mainWin, "File to Open", "", "Zeno Module (*.dll)");
    if (!filePath.isEmpty()) {
        m_plugins->addPlugin(filePath);
    }
}

void GraphsManager::copy(const QModelIndexList& selNodes)
{
    if (selNodes.empty())
        return;
    zeno::NodesData datas = UiHelper::dumpNodes(selNodes);
    zenoio::ZenWriter writer;
    QString strJson = QString::fromStdString(writer.dumpToClipboard(datas));
    QMimeData* pMimeData = new QMimeData;
    pMimeData->setText(strJson);
    QApplication::clipboard()->setMimeData(pMimeData);
}

QStringList GraphsManager::paste(const QPointF& pos, const QStringList& path_of_graphM)
{
    const QMimeData* pMimeData = QApplication::clipboard()->mimeData();

    GraphModel* pTargetModel = getGraph(path_of_graphM);
    QStringList newnodes_name;

    if (pMimeData->hasText() && pTargetModel)
    {
        const QString& strJson = pMimeData->text();
        rapidjson::Document doc;
        doc.Parse(strJson.toStdString().c_str());
        if (doc.HasMember("version"))
        {
            std::string ver = doc["version"].GetString();
            std::pair<zeno::NodesData, zeno::LinksData> datas;
            zeno::ReferencesData refs;
            if (ver == "v2") {
                zenoio::Zsg2Reader reader;
                reader.importNodes(strJson.toStdString(), datas.first, datas.second, refs);
            }
            else {
                zenoio::ZenReader reader;
                reader.importNodes(strJson.toStdString(), datas.first, datas.second, refs);
            }
            newnodes_name = pTargetModel->pasteNodes(datas.first, datas.second, pos);
        }
    }
    return newnodes_name;
}

QModelIndex GraphsManager::getNodeIndexByUuidPath(const QString& objPath)
{
    if (!m_main)
        return QModelIndex();

    return m_main->indexFromUuidPath(objPath.toStdString());
}

static QModelIndex recursive_search(GraphModel* pCurrentGraph, QStringList pathList) {
    /*pathList是一个相对于当前pCurrentGraph的路径，不是完整的路径*/
    if (pathList.empty())
        return QModelIndex();

    if (pathList.size() == 1) {
        //最后一项必须是节点的名字(name)
        QString name = pathList[0];
        return pCurrentGraph->indexFromName(name);
    }
    else {
        QString subnetnode = pathList.front();
        pathList.pop_front();
        //观察subnetnode是不是一个子图节点
        QModelIndex subnetnode_idx = pCurrentGraph->indexFromName(subnetnode);
        if (!subnetnode_idx.isValid())
            return QModelIndex();

        GraphModel* pSubgraph = pCurrentGraph->subgraph(subnetnode_idx);
        if (!pSubgraph)
            return QModelIndex();

        return recursive_search(pSubgraph, pathList);
    }
}

QModelIndex GraphsManager::getNodeIndexByPath(const QString& path) {
    if (!m_main)
        return QModelIndex();

    QStringList fullPathList = path.split('/', Qt::SkipEmptyParts);
    if (fullPathList.first() == "main") {
        fullPathList.pop_front();
    }
    else {
        //TODO: assets相关
    }
    return recursive_search(m_main, fullPathList);
}

QStringList GraphsManager::recentFiles() const
{
    QSettings settings(QSettings::UserScope, zsCompanyName, zsEditor);
    settings.beginGroup("Recent File List");
    QStringList lst = settings.childKeys();
    zenoApp->getMainWindow()->sortRecentFile(lst);

    QStringList paths;
    for (int i = 0; i < lst.size(); i++)
    {
        const QString& key = lst[i];
        const QString& path = settings.value(key).toString();
        if (path.endsWith(".zen"))
            paths.append(path);
    }
    return paths;
}

NodeCates GraphsManager::getCates() const
{
    zeno::NodeRegistry nodeRegs = zeno::getSession().dumpCoreCates();
    QVector<zeno::NodeInfo> assetsNames;
    NodeCates cates;

    for (int r = 0; r < m_assets->rowCount(); r++)
    {
        QModelIndex idx = m_assets->index(r);
        const QString& asset = idx.data(QtRole::ROLE_CLASS_NAME).toString();
        zeno::NodeInfo info;
        info.name = asset.toStdString();
        info.status = zeno::ZModule_Loaded;
        info.module_path = "";  //先默认算到主模块
        assetsNames.push_back(std::move(info));
    }

    cates["assets"] = assetsNames;

    for (const zeno::NodeInfo& nodereg : nodeRegs) {
        QString category = QString::fromStdString(nodereg.cate);
        if (!category.isEmpty()) {
            cates[category].append(nodereg);
        }
    }

    //1. foreach-count
    QVector<zeno::NodeInfo> control_cases = {
        {"Foreach-Count", "", "control", zeno::ZModule_Loaded},
        {"Foreach-Geometry-attr", "", "control", zeno::ZModule_Loaded},
        {"Foreach-StopCond", "", "control", zeno::ZModule_Loaded}
    };
    cates["control"] = control_cases;
    return cates;
}

void GraphsManager::updateAssets(const QString& assetsName, zeno::ParamsUpdateInfo info, const zeno::CustomUI& customui)
{
    const std::string& sasset = assetsName.toStdString();
    zeno::getSession().assets->updateAssets(sasset, info, customui);
    //update to each assets node on the tree
    ZASSERT_EXIT(m_main);
    m_main->syncToAssetsInstance_customui(assetsName, info, customui);

    //also need to sync all other assets.
    for (int i = 0; i < m_assets->rowCount(); i++)
    {
        GraphModel* pAssetM = m_assets->getAssetGraph(i);
        if (pAssetM && pAssetM->name() != assetsName)
        {
            pAssetM->syncToAssetsInstance_customui(assetsName, info, customui);
        }
    }
}

zeno::ZSG_VERSION GraphsManager::ioVersion() const
{
    return m_version;
}

void GraphsManager::setIOVersion(zeno::ZSG_VERSION ver)
{
    m_version = ver;
}

void GraphsManager::clearMarkOnGv() {
    for (auto scene : m_scenes) {
        if (scene->getGraphModel()->currentPath().startsWith("main")) {
            scene->clearMark();
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
