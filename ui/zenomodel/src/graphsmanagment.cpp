#include "graphsmanagment.h"
#include <zenomodel/include/zenomodel.h>
#include <zenomodel/include/modelrole.h>
#include <zenoio/reader/zsgreader.h>
#include <zenomodel/include/uihelper.h>
#include <zeno/zeno.h>
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
    initCoreDescriptors();
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
    connect(m_pNodeModel, &IGraphsModel::dirtyChanged, this, [=]() {
        emit dirtyChanged(m_pNodeModel->isDirty());
    });
}

NODE_DESCS GraphsManagment::getCoreDescs()
{
    NODE_DESCS descs;
    QString strDescs = QString::fromStdString(zeno::getSession().dumpDescriptors());
    //zeno::log_critical("EEEE {}", strDescs.toStdString());
    //ZENO_P(strDescs.toStdString());
    QStringList L = strDescs.split("\n");
    for (int i = 0; i < L.size(); i++)
    {
        QString line = L[i];
        if (line.startsWith("DESC@"))
        {
            line = line.trimmed();
            int idx1 = line.indexOf("@");
            int idx2 = line.indexOf("@", idx1 + 1);
            ZASSERT_EXIT(idx1 != -1 && idx2 != -1, descs);
            QString wtf = line.mid(0, idx1);
            QString z_name = line.mid(idx1 + 1, idx2 - idx1 - 1);
            QString rest = line.mid(idx2 + 1);
            ZASSERT_EXIT(rest.startsWith("{") && rest.endsWith("}"), descs);
            auto _L = rest.mid(1, rest.length() - 2).split("}{");
            QString inputs = _L[0], outputs = _L[1], params = _L[2], categories = _L[3];
            QStringList z_categories = categories.split('%', QtSkipEmptyParts);

            NODE_DESC desc;
            for (QString input : inputs.split("%", QtSkipEmptyParts)) {
                QString type, name;
                QVariant defl;

                parseDescStr(input, name, type, defl);

                INPUT_SOCKET socket;
                socket.info.type = type;
                socket.info.name = name;
                CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_INPUT, name, type);
                socket.info.control = ctrlInfo.control;
                socket.info.ctrlProps = ctrlInfo.controlProps.toMap();
                socket.info.defaultValue = defl;
                desc.inputs[name] = socket;
            }
            for (QString output : outputs.split("%", QtSkipEmptyParts)) {
                QString type, name;
                QVariant defl;

                parseDescStr(output, name, type, defl);

                OUTPUT_SOCKET socket;
                socket.info.type = type;
                socket.info.name = name;
                CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_OUTPUT, name, type);
                socket.info.control = ctrlInfo.control;
                socket.info.ctrlProps = ctrlInfo.controlProps.toMap();
                socket.info.defaultValue = defl;
                desc.outputs[name] = socket;
            }
            for (QString param : params.split("%", QtSkipEmptyParts)) {
                QString type, name;
                QVariant defl;

                parseDescStr(param, name, type, defl);

                PARAM_INFO paramInfo;
                paramInfo.bEnableConnect = false;
                paramInfo.name = name;
                paramInfo.typeDesc = type;
                CONTROL_INFO ctrlInfo = UiHelper::getControlByType(z_name, PARAM_PARAM, name, type);
                paramInfo.control = ctrlInfo.control;
                paramInfo.controlProps = ctrlInfo.controlProps;
                paramInfo.defaultValue = defl;
                //thers is no "value" in descriptor, but it's convient to initialize param value.
                paramInfo.value = paramInfo.defaultValue;
                desc.params[name] = paramInfo;
            }
            desc.categories = z_categories;
            desc.name = z_name;

            descs.insert(z_name, desc);
        }
    }
    return descs;
}

void GraphsManagment::parseDescStr(const QString& descStr, QString& name, QString& type, QVariant& defl)
{
    auto _arr = descStr.split('@', QtSkipEmptyParts);
    ZASSERT_EXIT(!_arr.isEmpty());

    if (_arr.size() == 1)
    {
        name = _arr[0];
    }
    else if (_arr.size() == 2)
    {
        type = _arr[0];
        name = _arr[1];
        if (type == "string")
            defl = UiHelper::parseStringByType("", type);
    }
    else if (_arr.size() == 3)
    {
        type = _arr[0];
        name = _arr[1];
        QString strDefl = _arr[2];
        defl = UiHelper::parseStringByType(strDefl, type);
    }
}

void GraphsManagment::registerCate(const NODE_DESC& desc)
{
    for (auto cate : desc.categories)
    {
        m_nodesCate[cate].name = cate;
        m_nodesCate[cate].nodes.push_back(desc.name);
    }
}

void GraphsManagment::initCoreDescriptors()
{
    m_nodesDesc = getCoreDescs();
    m_nodesCate.clear();
    for (auto it = m_nodesDesc.constBegin(); it != m_nodesDesc.constEnd(); it++) {
        const QString &name = it.key();
        const NODE_DESC &desc = it.value();
        registerCate(desc);
    }

    //add Blackboard
    NODE_DESC desc;
    desc.name = "Blackboard";
    desc.categories.push_back("layout");
    m_nodesDesc.insert(desc.name, desc);
    registerCate(desc);

    //add Group
    NODE_DESC groupDesc;
    groupDesc.name = "Group";
    groupDesc.categories.push_back("layout");
    m_nodesDesc.insert(groupDesc.name, groupDesc);
    registerCate(groupDesc);
}

void GraphsManagment::initSubnetDescriptors(const QList<QString>& subgraphs, const zenoio::ZSG_PARSE_RESULT& res)
{
    for (QString name : subgraphs)
    {
        if (res.descs.find(name) == res.descs.end())
        {
            zeno::log_warn("subgraph {} isn't described by the file descs.", name.toStdString());
            continue;
        }
        NODE_DESC desc = res.descs[name];
        if (m_subgsDesc.find(desc.name) == m_subgsDesc.end())
        {
            desc.is_subgraph = true;
            m_subgsDesc.insert(desc.name, desc);
            registerCate(desc);
        }
        else
        {
            zeno::log_error("The graph \"{}\" exists!", desc.name.toStdString());
        }
    }
}

bool GraphsManagment::getSubgDesc(const QString& subgName, NODE_DESC& desc)
{
    if (m_subgsDesc.find(subgName) != m_subgsDesc.end()) {
        desc = m_subgsDesc[subgName];
        return true;
    }
    return false;
}

bool GraphsManagment::getDescriptor(const QString& descName, NODE_DESC& desc)
{
    //internal node or subgraph node? if same name.
    if (m_subgsDesc.find(descName) != m_subgsDesc.end()) {
        desc = m_subgsDesc[descName];
        return true;
    }
    if (m_nodesDesc.find(descName) != m_nodesDesc.end()) {
        desc = m_nodesDesc[descName];
        return true;
    }
    return false;
}

bool GraphsManagment::updateSubgDesc(const QString& descName, const NODE_DESC& desc)
{
    if (m_subgsDesc.find(descName) != m_subgsDesc.end()) {
        m_subgsDesc[descName] = desc;
        return true;
    }
    return false;
}

void GraphsManagment::renameSubGraph(const QString& oldName, const QString& newName)
{
    ZASSERT_EXIT(m_subgsDesc.find(oldName) != m_subgsDesc.end() &&
                 m_subgsDesc.find(newName) == m_subgsDesc.end());

    NODE_DESC desc = m_subgsDesc[oldName];
    desc.name = newName;
    m_subgsDesc[newName] = desc;
    m_subgsDesc.remove(oldName);

    for (QString cate : desc.categories) {
        m_nodesCate[cate].nodes.removeAll(oldName);
        m_nodesCate[cate].nodes.append(newName);
    }
}

void GraphsManagment::removeGraph(const QString& subgName)
{
    ZASSERT_EXIT(m_subgsDesc.find(subgName) != m_subgsDesc.end());
    NODE_DESC desc = m_subgsDesc[subgName];
    m_subgsDesc.remove(subgName);
    for (QString cate : desc.categories) {
        m_nodesCate[cate].nodes.removeAll(subgName);
    }
}

void GraphsManagment::clearSubgDesc()
{
    for (auto desc : m_subgsDesc)
    {
        for (QString cate : desc.categories) {
            m_nodesCate[cate].nodes.removeAll(desc.name);
            if (m_nodesCate[cate].nodes.isEmpty())
                m_nodesCate.remove(cate);
        }
    }
    m_subgsDesc.clear();
}

void GraphsManagment::appendSubGraph(const NODE_DESC& desc)
{
    if (!desc.name.isEmpty() && m_subgsDesc.find(desc.name) == m_subgsDesc.end())
    {
        m_subgsDesc.insert(desc.name, desc);
        registerCate(desc);
    }
}

NODE_CATES GraphsManagment::getCates()
{
    return m_nodesCate;
}

NODE_TYPE GraphsManagment::nodeType(const QString& name)
{
    if (m_subgsDesc.find(name) != m_subgsDesc.end()) {
        return SUBGRAPH_NODE;
    } else if (name == "Blackboard") {
        return BLACKBOARD_NODE;
    } else if (name == "Group") {
        return GROUP_NODE;
    } else if (name == "SubInput") {
        return SUBINPUT_NODE;
    } else if (name == "SubOutput") {
        return SUBOUTPUT_NODE;
    } else if (name == "MakeHeatmap") {
        return HEATMAP_NODE;
    } else {
        return NORMAL_NODE;
    }
}

QString GraphsManagment::filePath() const 
{
    return m_filePath;
}

NODE_DESCS GraphsManagment::descriptors()
{
    NODE_DESCS descs;
    for (QString descName : m_subgsDesc.keys()) {
        descs.insert(descName, m_subgsDesc[descName]);
    }
    for (QString nodeName : m_nodesDesc.keys()) {
        //subgraph node has high priority than core node.
        if (descs.find(nodeName) == descs.end()) {
            descs.insert(nodeName, m_nodesDesc[nodeName]);
        }
    }
    return descs;
}

void GraphsManagment::onParseResult(
        const zenoio::ZSG_PARSE_RESULT &res,
        IGraphsModel *pNodeModel,
        IGraphsModel *pSubgraphs)
{
    //init descriptor first.
    auto subgraphs = res.subgraphs.keys();
    initSubnetDescriptors(subgraphs, res);

    pNodeModel->setIOVersion(res.ver);
    pSubgraphs->setIOVersion(res.ver);

    //only based on tree layout.
    ZASSERT_EXIT(pNodeModel && pSubgraphs);
    for (QString subgName : subgraphs)
    {
        const SUBGRAPH_DATA& subg = res.subgraphs[subgName];
        pSubgraphs->newSubgraph(subgName);
        const QModelIndex &subgIdx = pSubgraphs->index(subgName);
        pSubgraphs->importNodes(subg.nodes, subg.links, QPointF(0, 0), subgIdx, false);
    }
    const QModelIndex& mainIdx = pNodeModel->index("main");
    pNodeModel->importNodes(res.mainGraph.nodes, res.mainGraph.links, QPointF(0, 0), mainIdx, false);
}

IGraphsModel* GraphsManagment::openZsgFile(const QString& fn)
{
    IGraphsModel* pNodeModel = zeno_model::createModel(false, this);
    //todo: when the io version is 2.5, should reuse the io code to init subgraph model.
    IGraphsModel *pSubgraphsModel = zeno_model::createModel(true, this);
    {
        IOBreakingScope batch(pNodeModel);

        zenoio::ZSG_PARSE_RESULT result;
        bool ret = zenoio::ZsgReader::getInstance().openFile(fn, result);
        m_timerInfo = result.timeline;
        if (!ret)
            return nullptr;

        onParseResult(result, pNodeModel, pSubgraphsModel);
    }

    m_filePath = fn;
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

    QString strContent = ZsgWriter::getInstance().dumpProgramStr(m_pNodeModel, m_pSharedGraphs, settings);
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

    m_filePath = filePath;
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
    m_filePath.clear();
    return pModel;
}

void GraphsManagment::importGraph(const QString& fn)
{
    if (!m_pSharedGraphs)
        return;

    IOBreakingScope batch(m_pSharedGraphs);
    zenoio::ZSG_PARSE_RESULT res;
    if (!zenoio::ZsgReader::getInstance().openFile(fn, res))
    {
        zeno::log_error("failed to open zsg file: {}", fn.toStdString());
        return;
    }

    for (QString subgName : res.subgraphs.keys())
    {
        const SUBGRAPH_DATA& subg = res.subgraphs[subgName];
        m_pSharedGraphs->newSubgraph(subgName);
        const QModelIndex &subgIdx = m_pSharedGraphs->index(subgName);
        m_pSharedGraphs->importNodes(subg.nodes, subg.links, QPointF(0, 0), subgIdx, false);
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
        delete m_pSharedGraphs;
        m_pSharedGraphs = nullptr;
        clearSubgDesc();
    }
    emit fileClosed();
}

void GraphsManagment::removeScene(const QString& subgName)
{
    if (m_scenes.find(subgName) != m_scenes.end())
    {
        delete m_scenes[subgName];
        m_scenes.remove(subgName);
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
