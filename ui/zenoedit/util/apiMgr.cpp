#include "apiMgr.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "viewport/displaywidget.h"
#include "settings/zsettings.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>
#include <QSettings>
#include <QTextStream>
#include <QTimer>
#include <QCoreApplication>
#include "launch/corelaunch.h"
#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
#include "launch/ztcpserver.h"
#endif
#include <zeno/extra/GlobalStatus.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <zeno/utils/log.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>
#include <zenovis/Camera.h>
#include <zenovis/Session.h>
#include <zenovis/DrawOptions.h>
#include "viewport/optixviewport.h"
#include "dock/ztabdockwidget.h"
#include <QBuffer>
#include <QUuid>
#include <glm/gtc/quaternion.hpp>

#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/nodesmgr.h>
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/uihelper.h>
#include <zenomodel/include/enum.h>
#include <zenomodel/include/api.h>
#include "../nodesys/zenosubgraphscene.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/select.h>
#include <unistd.h>
#endif

namespace
{
constexpr ZENO_HANDLE ZENO_INVALID_HANDLE = static_cast<ZENO_HANDLE>(-1);

#ifndef ZENO_OPTIX_PROC
// Returns the standalone optix viewport instance used for capturing rendered
// frames. Under ZENO_OPTIX_PROC the optix viewport runs out-of-process and
// has no local screenshot path, so the calls report an error instead.
ZOptixViewport* optixViewport()
{
    auto* main = zenoApp->getMainWindow();
    if (!main)
        return nullptr;

    ZOptixViewport* fallback = nullptr;
    for (DisplayWidget* view : main->viewports()) {
        if (!view || view->isGLViewport())
            continue;
        ZOptixViewport* optix = view->optixViewport();
        // Prefer a live (render thread running) ray-tracing window, so the
        // "does one exist?" semantics hold: as long as any single running optix
        // window is present, it counts as existing. Fall back to the first optix
        // viewport only when none is currently running.
        if (optix && optix->isOptixViewRunning())
            return optix;
        if (!fallback)
            fallback = optix;
    }
    return fallback;
}

// Returns a ray-tracing viewport whose render thread is alive, creating the
// whole chain on demand when none is available:
//     main window -> DockContent_View -> DisplayWidget -> ZOptixViewport
// Zeno only ever needs a single ray-tracing window, so the first existing
// live viewport is reused; a new one is created only when none is running.
ZOptixViewport* ensureOptixViewport()
{
    if (ZOptixViewport* vp = optixViewport()) {
        if (vp->isOptixViewRunning())
            return vp;
    }

    ZenoMainWindow* main = zenoApp->getMainWindow();
    if (!main) {
        main = new ZenoMainWindow;
        main->resize(1600, 900);
        main->show();
    }
    if (!main)
        return nullptr;

    // Create a single optix view tab and keep the new dock floating so the
    // main managed dock layout stays untouched; viewports() still finds a
    // floating ZTabDockWidget among the top-level windows.
    ZTabDockWidget* dock = new ZTabDockWidget(main);
    dock->setObjectName("apiOptixDock");
    dock->onAddTab(PANEL_OPTIX_VIEW);
    dock->setFloating(true);
    dock->resize(1280, 720);
    dock->show();

    for (DisplayWidget* view : dock->viewports()) {
        if (!view->isGLViewport())
            return view->optixViewport();
    }
    return nullptr;
}
#endif

// Reads whatever bytes are immediately available from stdin without blocking.
// Returns >0 bytes read, 0 if nothing is available yet, -1 on closed/error.
// Polling instead of a blocking read lets stop() return promptly instead of
// hanging forever on a stdin pipe that the parent process keeps open.
int readAvailable(char* buf, size_t cap)
{
#ifdef _WIN32
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    if (!hStdin || hStdin == INVALID_HANDLE_VALUE)
        return -1;
    DWORD avail = 0;
    if (!PeekNamedPipe(hStdin, nullptr, 0, nullptr, &avail, nullptr))
        return -1;  // pipe closed by the parent process or broken.
    if (avail == 0)
        return 0;
    DWORD rd = 0;
    if (!ReadFile(hStdin, buf, (DWORD)std::min(cap, (size_t)avail), &rd, nullptr) || rd == 0)
        return -1;
    return (int)rd;
#else
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    timeval tv{0, 0};
    int ret = select(STDIN_FILENO + 1, &rfds, nullptr, nullptr, &tv);
    if (ret <= 0)
        return ret < 0 ? -1 : 0;
    ssize_t n = read(STDIN_FILENO, buf, cap);
    return n > 0 ? (int)n : -1;
#endif
}

// Accumulates raw stdin bytes and splits them into newline-delimited frames.
class LineBuffer
{
public:
    std::vector<std::string> feed(const char* data, size_t n)
    {
        m_buf.append(data, n);
        std::vector<std::string> lines;
        size_t pos = 0;
        while ((pos = m_buf.find('\n')) != std::string::npos)
        {
            std::string line = m_buf.substr(0, pos);
            m_buf.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (!line.empty())
                lines.push_back(std::move(line));
        }
        return lines;
    }

private:
    std::string m_buf;
};

QString compactJson(const QJsonValue& value)
{
    QJsonDocument doc;
    if (value.isObject())
        doc.setObject(value.toObject());
    else if (value.isArray())
        doc.setArray(value.toArray());
    else
        doc = QJsonDocument(QJsonObject{{"result", value}});
    return QString::fromUtf8(doc.toJson(QJsonDocument::Compact));
}

QString argAt(const QJsonArray& args, int idx)
{
    return args.at(idx).toString();
}

// Parses "true/false/1/0/on/off" (case-insensitive) for the node-flag toggle
// commands (setView/setOnce). Returns false when the text is none of them.
bool parseToggleText(const QString& text, bool& out)
{
    const QString t = text.trimmed().toLower();
    if (t == "true" || t == "1" || t == "on") { out = true; return true; }
    if (t == "false" || t == "0" || t == "off") { out = false; return true; }
    return false;
}

// Converts a ZVARIANT to a plain-text representation used in responses.
QString variantToString(const ZVARIANT& var)
{
    switch (var.index())
    {
    case 0:
        return QString::fromStdString(std::get<std::string>(var));
    case 1:
        return QString::number(std::get<int>(var));
    case 2:
        return QString::number((double)std::get<float>(var), 'g', 9);
    case 3:
        return QString::number(std::get<double>(var), 'g', 15);
    case 4:
        return std::get<bool>(var) ? "true" : "false";
    case 5:
    {
        const auto& v = std::get<zeno::vec2i>(var);
        return QString("%1,%2").arg(v[0]).arg(v[1]);
    }
    case 6:
    {
        const auto& v = std::get<zeno::vec2f>(var);
        return QString("%1,%2").arg((double)v[0]).arg((double)v[1]);
    }
    case 7:
    {
        const auto& v = std::get<zeno::vec3i>(var);
        return QString("%1,%2,%3").arg(v[0]).arg(v[1]).arg(v[2]);
    }
    case 8:
    {
        const auto& v = std::get<zeno::vec3f>(var);
        return QString("%1,%2,%3").arg((double)v[0]).arg((double)v[1]).arg((double)v[2]);
    }
    case 9:
    {
        const auto& v = std::get<zeno::vec4i>(var);
        return QString("%1,%2,%3,%4").arg(v[0]).arg(v[1]).arg(v[2]).arg(v[3]);
    }
    case 10:
    {
        const auto& v = std::get<zeno::vec4f>(var);
        return QString("%1,%2,%3,%4").arg((double)v[0]).arg((double)v[1]).arg((double)v[2]).arg((double)v[3]);
    }
    }
    return "";
}

bool splitVecList(const QString& text, QList<double>& out)
{
    QString t = text.trimmed();
    if (t.startsWith('[') && t.endsWith(']'))
        t = t.mid(1, t.size() - 2);
    const QStringList parts = t.split(',');
    for (const QString& p : parts)
    {
        bool ok = false;
        double v = p.trimmed().toDouble(&ok);
        if (!ok)
            return false;
        out.append(v);
    }
    return !out.isEmpty();
}

// Parses a string value into a ZVARIANT according to the declared param type.
bool variantFromString(const QString& text, const QString& typeDesc, ZVARIANT& var)
{
    const QString raw = typeDesc.trimmed();
    // Enum types look like "enum png jpg exr" — the value is stored as the
    // selected option string (see UiHelper::CONTROL_ENUM). Accept any listed
    // option case-insensitively and store its canonical spelling.
    if (raw.startsWith("enum ", Qt::CaseInsensitive)) {
        const QStringList options = raw.mid(5).split(' ', Qt::SkipEmptyParts);
        const QString wanted = text.trimmed();
        for (const QString& opt : options) {
            if (opt.compare(wanted, Qt::CaseInsensitive) == 0) {
                var = opt.toStdString();
                return true;
            }
        }
        return false;  // not a listed option; caller reports typeDesc (the option list)
    }

    const QString t = raw.toLower();
    if (t == "int")
    {
        bool ok = false;
        int v = text.trimmed().toInt(&ok);
        if (ok)
        {
            var = v;
            return true;
        }
    }
    else if (t == "float")
    {
        bool ok = false;
        float v = text.trimmed().toFloat(&ok);
        if (ok)
        {
            var = v;
            return true;
        }
    }
    else if (t == "double")
    {
        bool ok = false;
        double v = text.trimmed().toDouble(&ok);
        if (ok)
        {
            var = v;
            return true;
        }
    }
    else if (t == "bool")
    {
        const QString b = text.trimmed().toLower();
        if (b == "true" || b == "1")
        {
            var = true;
            return true;
        }
        if (b == "false" || b == "0")
        {
            var = false;
            return true;
        }
    }
    else if (t == "string" || t == "readpath")
    {
        // readpath (asset/file path inputs like HDRSky.path, ReadAlembic.path)
        // is stored as a plain string, same as in the zsg file.
        var = text.toStdString();
        return true;
    }
    else if (t == "vec2i" || t == "vec2f" || t == "vec3i" || t == "vec3f" || t == "vec4i" || t == "vec4f")
    {
        QList<double> vals;
        if (splitVecList(text, vals))
        {
            const bool isInt = t.contains('i');
            const int need = t.mid(3, 1).toInt();
            if (vals.size() == need)
            {
                if (isInt)
                {
                    if (need == 2)
                        var = zeno::vec2i((int)vals[0], (int)vals[1]);
                    else if (need == 3)
                        var = zeno::vec3i((int)vals[0], (int)vals[1], (int)vals[2]);
                    else
                        var = zeno::vec4i((int)vals[0], (int)vals[1], (int)vals[2], (int)vals[3]);
                }
                else
                {
                    if (need == 2)
                        var = zeno::vec2f((float)vals[0], (float)vals[1]);
                    else if (need == 3)
                        var = zeno::vec3f((float)vals[0], (float)vals[1], (float)vals[2]);
                    else
                        var = zeno::vec4f((float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
                }
                return true;
            }
        }
    }
    return false;
}

// Serializes a node/socket default value (QVariant) into a JSON value.
QJsonValue qVariantToJson(const QVariant& var)
{
    switch (var.type())
    {
    case QVariant::Int:
        return QJsonValue(var.toInt());
    case QVariant::Bool:
        return QJsonValue(var.toBool());
    case QMetaType::Float:
        return QJsonValue((double)var.toFloat());
    case QVariant::Double:
        return QJsonValue(var.toDouble());
    case QVariant::String:
        return QJsonValue(var.toString());
    case QVariant::UserType:
        if (var.userType() == QMetaTypeId<UI_VECTYPE>::qt_metatype_id())
        {
            const UI_VECTYPE v = var.value<UI_VECTYPE>();
            QJsonArray arr;
            for (qreal f : v)
                arr.append(f);
            return arr;
        }
        return QJsonValue(var.toString());
    default:
        return QJsonValue(var.toString());
    }
}

// Resolves a subgraph by name into a valid handle.
bool resolveGraph(IGraphsModel* pModel, const QString& graphName, ZENO_HANDLE& hGraph)
{
    const QModelIndex subgIdx = pModel->index(graphName);
    if (!subgIdx.isValid())
        return false;
    hGraph = (ZENO_HANDLE)subgIdx.internalId();
    return true;
}

// Resolves a node by uuid inside a subgraph into a valid handle.
bool resolveNode(IGraphsModel* pModel, ZENO_HANDLE hGraph, const QString& uuid, ZENO_HANDLE& hNode)
{
    auto nodeHandleOpt = Zeno_GetNode(hGraph, uuid.toStdString());
    if (!nodeHandleOpt.has_value())
        return false;
    hNode = nodeHandleOpt.value();
    return true;
}

// Reads a node's on-canvas geometry (bounding-box width/height). The size is a
// view-layer property computed from the node item in the subgraph scene
// (ZenoSubGraphScene::getNode + QGraphicsItem::boundingRect), NOT stored in the
// model. It is therefore an OPTIONAL field: returns false (and plays no part
// in the reply) when no scene exists for the graph or the node is absent from
// it (e.g. the graph has never been opened in the editor).
bool nodeSize(IGraphsModel* pModel, ZENO_HANDLE hGraph, ZENO_HANDLE hNode, float& w, float& h)
{
    const QModelIndex subgIdx = pModel->subgIndex(hGraph);
    const QModelIndex nodeIdx = pModel->nodeIndex(hGraph, hNode);
    if (!subgIdx.isValid() || !nodeIdx.isValid())
        return false;

    ZenoSubGraphScene* scene = qobject_cast<ZenoSubGraphScene*>(
        GraphsManagment::instance().gvScene(subgIdx));
    if (!scene)
        return false;

    QGraphicsItem* item = scene->getNode(nodeIdx.data(ROLE_OBJID).toString());
    if (!item || item->boundingRect().isEmpty())
        return false;

    const QSizeF size = item->boundingRect().size();
    w = (float)size.width();
    h = (float)size.height();
    return true;
}

// Creates a node and, once its on-canvas item is ready, fills 'obj' with
// {class, graph, pos, width, height} plus an optional "name". Returns false and
// sets 'err' on any failure, so the single (createNode) and batch
// (createNodeList) paths can share exactly the same semantics.
bool createNodeAndGeometry(IGraphsModel* pModel, ZENO_HANDLE hGraph, const QString& graphName,
                           const QString& nodeCls, const QString& customName,
                           QJsonObject& obj, QString& err)
{
    ZENO_HANDLE hNode = Zeno_AddNode(hGraph, nodeCls.toStdString());
    if (hNode == ZENO_INVALID_HANDLE) {
        err = "failed to create node: " + nodeCls;
        return false;
    }

    std::string uuid;
    if (Zeno_GetNodeUuid(hGraph, hNode, uuid) != Err_NoError || uuid.empty()) {
        err = "failed to create node: " + nodeCls;
        return false;
    }

    const QModelIndex subgIdx = pModel->subgIndex(hGraph);
    const QModelIndex nodeIdx = pModel->nodeIndex(hGraph, hNode);
    if (!subgIdx.isValid() || !nodeIdx.isValid()) {
        err = "node created but its model index is invalid: " + nodeCls;
        return false;
    }

    // Sizes only exist once the item is in the subgraph scene; see nodeSize().
    if (!qobject_cast<ZenoSubGraphScene*>(GraphsManagment::instance().gvScene(subgIdx))) {
        err = "graph scene not available for '" + graphName + "'; open the graph in the editor before creating nodes";
        return false;
    }

    QCoreApplication::processEvents();

    float w = 0.f, h = 0.f;
    if (!nodeSize(pModel, hGraph, hNode, w, h)) {
        err = "failed to retrieve on-canvas size for node " + QString::fromStdString(uuid);
        return false;
    }

    obj["node_uuid"] = QString::fromStdString(uuid);
    obj["class"] = nodeCls;
    obj["graph"] = graphName;
    const QPointF pos = nodeIdx.data(ROLE_OBJPOS).toPointF();
    obj["pos"] = QJsonArray{pos.x(), pos.y()};
    obj["width"] = (double)w;
    obj["height"] = (double)h;

    if (!customName.isEmpty() && pModel->setCustomName(subgIdx, nodeIdx, customName))
        obj["name"] = customName;
    return true;
}

// Builds the full JSON info of a node: class/uuid/pos/options, inputs (with
// upstream link when connected), params (with current value), outputs.
QJsonObject buildNodeInfo(IGraphsModel* pModel, ZENO_HANDLE hGraph, ZENO_HANDLE hNode)
{
    const QModelIndex subgIdx = pModel->subgIndex(hGraph);
    const QModelIndex nodeIdx = pModel->nodeIndex(hGraph, hNode);

    QJsonObject info;
    info["class"] = nodeIdx.data(ROLE_OBJNAME).toString();
    info["uuid"] = nodeIdx.data(ROLE_OBJID).toString();

    // Optional custom display name: only present when a non-empty custom name
    // has been set (see setNodeName / createNode's optional name argument).
    const QString customName = nodeIdx.data(ROLE_CUSTOM_OBJNAME).toString();
    if (!customName.isEmpty())
        info["name"] = customName;

    const QPointF pos = nodeIdx.data(ROLE_OBJPOS).toPointF();
    info["pos"] = QJsonArray{pos.x(), pos.y()};
    info["options"] = nodeIdx.data(ROLE_OPTIONS).toInt();

    // Optional on-canvas size: present only when the graph's scene is available
    // (see nodeSize), otherwise omitted so clients can treat it as unavailable.
    float w = 0.f, h = 0.f;
    if (nodeSize(pModel, hGraph, hNode, w, h)) {
        info["width"] = (double)w;
        info["height"] = (double)h;
    }

    QJsonArray inputsArr;
    const INPUT_SOCKETS inputs = nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    for (auto it = inputs.constBegin(); it != inputs.constEnd(); ++it) {
        const SOCKET_INFO& sock = it->value().info;
        QJsonObject o;
        o["name"] = sock.name;
        o["type"] = sock.type;
        o["default"] = qVariantToJson(sock.defaultValue);

        std::pair<ZENO_HANDLE, std::string> ret;
        if (Zeno_GetInput(hGraph, hNode, sock.name.toStdString(), ret) == Err_NoError && ret.first != 0) {
            std::string outUuid;
            if (Zeno_GetNodeUuid(hGraph, ret.first, outUuid) == Err_NoError && !outUuid.empty()) {
                QJsonObject link;
                link["node"] = QString::fromStdString(outUuid);
                link["socket"] = QString::fromStdString(ret.second);
                o["link"] = link;
            }
        }
        inputsArr.append(o);
    }
    info["inputs"] = inputsArr;

    QJsonArray paramsArr;
    const PARAMS_INFO params = nodeIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    for (auto it = params.constBegin(); it != params.constEnd(); ++it) {
        const PARAM_INFO& param = it.value();
        QJsonObject o;
        o["name"] = param.name;
        o["type"] = param.typeDesc;
        o["value"] = qVariantToJson(param.value);
        o["default"] = qVariantToJson(param.defaultValue);
        paramsArr.append(o);
    }
    info["params"] = paramsArr;

    QJsonArray outputsArr;
    const OUTPUT_SOCKETS outputs = nodeIdx.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
    for (auto it = outputs.constBegin(); it != outputs.constEnd(); ++it) {
        const SOCKET_INFO& sock = it->value().info;
        QJsonObject o;
        o["name"] = sock.name;
        o["type"] = sock.type;
        outputsArr.append(o);
    }
    info["outputs"] = outputsArr;

    return info;
}

// Builds the JSON representation of a node class descriptor (structure only).
QJsonObject buildDescJson(const NODE_DESC& desc)
{
    QJsonObject o;
    o["class"] = desc.name;
    QJsonArray cats;
    for (const QString& c : desc.categories)
        cats.append(c);
    o["categories"] = cats;
    o["is_subgraph"] = desc.is_subgraph;

    QJsonArray inputs;
    for (auto it = desc.inputs.constBegin(); it != desc.inputs.constEnd(); ++it) {
        const SOCKET_INFO& sock = it->value().info;
        QJsonObject io;
        io["name"] = sock.name;
        io["type"] = sock.type;
        io["default"] = qVariantToJson(sock.defaultValue);
        inputs.append(io);
    }
    o["inputs"] = inputs;

    QJsonArray params;
    for (auto it = desc.params.constBegin(); it != desc.params.constEnd(); ++it) {
        const PARAM_INFO& param = it.value();
        QJsonObject po;
        po["name"] = param.name;
        po["type"] = param.typeDesc;
        po["default"] = qVariantToJson(param.defaultValue);
        params.append(po);
    }
    o["params"] = params;

    QJsonArray outputs;
    for (auto it = desc.outputs.constBegin(); it != desc.outputs.constEnd(); ++it) {
        const SOCKET_INFO& sock = it->value().info;
        QJsonObject oo;
        oo["name"] = sock.name;
        oo["type"] = sock.type;
        outputs.append(oo);
    }
    o["outputs"] = outputs;

    return o;
}
} // namespace

ApiMgr::ApiMgr(ZenoApplication* app, QObject* parent)
    : QObject(parent)
    , m_app(app)
    , m_bRunning(false)
    , m_pThread(nullptr)
{
    connect(this, &ApiMgr::commandReceived, this, &ApiMgr::handleCommand, Qt::BlockingQueuedConnection);
}

ApiMgr::~ApiMgr()
{
    stop();
}

void ApiMgr::start()
{
    if (m_bRunning) {
        return;
    }

    m_running = true;
    m_bRunning = true;
    m_pThread = new std::thread([this]() {
        inputListenerThread();
    });
}

void ApiMgr::stop()
{
    if (!m_bRunning) {
        return;
    }

    // The listener thread polls stdin and checks m_running every few
    // milliseconds, so it exits promptly instead of blocking forever on a
    // pipe that may never deliver EOF.
    m_running = false;

    if (m_pThread && m_pThread->joinable()) {
        m_pThread->join();
        delete m_pThread;
        m_pThread = nullptr;
    }

    m_bRunning = false;
}

void ApiMgr::sendResponse(const QString& requestId, bool ok, const QString& result)
{
    QJsonObject response;
    response.insert("type", "response");
    response.insert("id", requestId);
    response.insert("ok", ok);
    response.insert("result", result);

    QTextStream out(stdout);
    out.setCodec("UTF-8");
    out << QString::fromUtf8(QJsonDocument(response).toJson(QJsonDocument::Compact)) << endl;
    out.flush();
}

void ApiMgr::inputListenerThread()
{
    zeno::log_critical("API listener thread started");

    LineBuffer lineBuffer;
    char buf[4096];
    while (m_running) {
        const int n = readAvailable(buf, sizeof(buf));
        if (n > 0) {
            for (std::string& line : lineBuffer.feed(buf, (size_t)n)) {
                handleFrame(QString::fromStdString(line));
            }
        } else if (n < 0) {
            zeno::log_critical("Input stream closed, thread exiting");
            break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    zeno::log_critical("API listener thread stopped");
}

void ApiMgr::handleFrame(const QString& line)
{
    QJsonParseError parseError;
    QJsonDocument requestDocument = QJsonDocument::fromJson(line.toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !requestDocument.isObject()) {
        sendResponse("", false, "invalid request frame");
        return;
    }

    QJsonObject request = requestDocument.object();
    const QString requestId = request.value("id").toString();
    const QString name = request.value("name").toString();
    const QJsonArray args = request.value("args").toArray();
    if (request.value("type").toString() != "request" || requestId.isEmpty() || name.isEmpty()) {
        sendResponse(requestId, false, "request requires a non-empty id and name");
        return;
    }

    zeno::log_critical("API thread received request " + requestId.toStdString() + ": " + name.toStdString());
    emit commandReceived(requestId, name, args);
}

void ApiMgr::handleCommand(const QString& requestId, const QString& name, const QJsonArray& args)
{
    if (name == "ping") {
        sendResponse(requestId, true, "pong");
        return;
    }

    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel) {
        std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
        ZASSERT_EXIT(mgr);
        mgr->setNewCacheDir(true);

        zenoApp->graphsManagment()->newFile();
        pModel = GraphsManagment::instance().currentModel();

        if (!pModel) {
            sendResponse(requestId, false, "failed to create the default graph");
            return;
        }
    }

    if (name == "createGraph") { handleCreateGraph(requestId, args, pModel); return; }
    if (name == "deleteGraph") { handleDeleteGraph(requestId, args, pModel); return; }
    if (name == "createNode") { handleCreateNode(requestId, args, pModel); return; }
    if (name == "deleteNode") { handleDeleteNode(requestId, args, pModel); return; }
    if (name == "setNodeName") { handleSetNodeName(requestId, args, pModel); return; }
    if (name == "getNodeSize") { handleGetNodeSize(requestId, args, pModel); return; }
    if (name == "graphGeometry") { handleGraphGeometry(requestId, args, pModel); return; }
    if (name == "addLink") { handleAddLink(requestId, args, pModel); return; }
    if (name == "removeLink") { handleRemoveLink(requestId, args, pModel); return; }
    if (name == "setParam") { handleSetParam(requestId, args, pModel); return; }
    if (name == "getParam") { handleGetParam(requestId, args, pModel); return; }
    if (name == "listGraphs") { handleListGraphs(requestId, args, pModel); return; }
    if (name == "listNodes") { handleListNodes(requestId, args, pModel); return; }
    if (name == "nodeInfo") { handleNodeInfo(requestId, args, pModel); return; }
    if (name == "graphInfo") { handleGraphInfo(requestId, args, pModel); return; }
    if (name == "listNodeClasses") { handleListNodeClasses(requestId, args, pModel); return; }
    if (name == "listNodeCategories") { handleListNodeCategories(requestId, args, pModel); return; }
    if (name == "nodeClassesInfo") { handleNodeClassesInfo(requestId, args, pModel); return; }
    if (name == "runGraph") { handleRunGraph(requestId, args, pModel); return; }
    if (name == "killGraph") { handleKillGraph(requestId, args, pModel); return; }
    if (name == "setTimelineRange") { handleSetTimelineRange(requestId, args, pModel); return; }
    if (name == "setView") { handleSetView(requestId, args, pModel); return; }
    if (name == "setOnce") { handleSetOnce(requestId, args, pModel); return; }
    if (name == "setNodePos") { handleSetNodePos(requestId, args, pModel); return; }
    if (name == "setNodePosList") { handleSetNodePosList(requestId, args, pModel); return; }
    if (name == "setViewList") { handleSetViewList(requestId, args, pModel); return; }
    if (name == "addLinkList") { handleAddLinkList(requestId, args, pModel); return; }
    if (name == "deleteNodeList") { handleDeleteNodeList(requestId, args, pModel); return; }
    if (name == "createNodeList") { handleCreateNodeList(requestId, args, pModel); return; }
    if (name == "setNodeNameList") { handleSetNodeNameList(requestId, args, pModel); return; }
    if (name == "removeLinkList") { handleRemoveLinkList(requestId, args, pModel); return; }
    if (name == "setOnceList") { handleSetOnceList(requestId, args, pModel); return; }
    if (name == "setParamList") { handleSetParamList(requestId, args, pModel); return; }
    if (name == "openFile") { handleOpenFile(requestId, args, pModel); return; }
    if (name == "saveAs") { handleSaveAs(requestId, args, pModel); return; }
    if (name == "getCameraOptix") { handleGetCameraOptix(requestId, args, pModel); return; }
    if (name == "setCameraOptix") { handleSetCameraOptix(requestId, args, pModel); return; }
    if (name == "screenShotOptix") { handleScreenShotOptix(requestId, args, pModel); return; }
    if (name == "getOptixBackground") { handleGetOptixBackground(requestId, args, pModel); return; }
    if (name == "setOptixBackground") { handleSetOptixBackground(requestId, args, pModel); return; }
    if (name == "getOptixStatus") { handleGetOptixStatus(requestId, args, pModel); return; }
    if (name == "openOptixViewport") { handleOpenOptixViewport(requestId, args, pModel); return; }

    sendResponse(requestId, false, "unknown command: " + name);
}

void ApiMgr::handleCreateGraph(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    if (graphName.isEmpty()) {
        sendResponse(requestId, false, "createGraph requires a graph name");
        return;
    }

    // The default "main" graph always exists; treat it as already present.
    if (pModel->index(graphName).isValid()) {
        sendResponse(requestId, true, compactJson(QJsonObject{
            {"graph", graphName}, {"created", false}, {"already_exists", true}}));
        return;
    }

    ZENO_HANDLE hGraph = Zeno_CreateGraph(graphName.toStdString());
    if (hGraph == ZENO_INVALID_HANDLE || !pModel->index(graphName).isValid()) {
        sendResponse(requestId, false, "failed to create graph: " + graphName);
        return;
    }
    sendResponse(requestId, true, compactJson(QJsonObject{
        {"graph", graphName}, {"created", true}}));
}

void ApiMgr::handleDeleteGraph(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    if (graphName.isEmpty()) {
        sendResponse(requestId, false, "deleteGraph requires a graph name");
        return;
    }

    // GraphsModel::removeSubGraph silently refuses to delete "main"; report it
    // as an error up front instead of pretending the call succeeded.
    if (graphName.compare("main", Qt::CaseInsensitive) == 0) {
        sendResponse(requestId, false, "the main graph cannot be deleted");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }

    ZENO_ERROR err = Zeno_DeleteGraph(hGraph);
    if (err == Err_NoError && !pModel->index(graphName).isValid()) {
        sendResponse(requestId, true, "deleted graph " + graphName);
    } else {
        sendResponse(requestId, false, "failed to delete graph, error code " + QString::number((uint)err));
    }
}

void ApiMgr::handleCreateNode(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeCls = argAt(args, 1);
    if (graphName.isEmpty() || nodeCls.isEmpty()) {
        sendResponse(requestId, false, "createNode requires graph name and node class");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }

    QJsonObject obj;
    QString err;
    if (!createNodeAndGeometry(pModel, hGraph, graphName, nodeCls, argAt(args, 2).trimmed(), obj, err)) {
        sendResponse(requestId, false, err);
        return;
    }
    sendResponse(requestId, true, compactJson(obj));
}

void ApiMgr::handleDeleteNode(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeUuid = argAt(args, 1);
    if (graphName.isEmpty() || nodeUuid.isEmpty()) {
        sendResponse(requestId, false, "deleteNode requires graph name and node uuid");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hNode = 0;
    if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    ZENO_ERROR err = Zeno_DeleteNode(hGraph, hNode);
    if (err == Err_NoError) {
        sendResponse(requestId, true, "deleted node " + nodeUuid);
    } else {
        sendResponse(requestId, false, "failed to delete node, error code " + QString::number((uint)err));
    }
}

void ApiMgr::handleSetNodeName(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeUuid = argAt(args, 1);
    const QString customName = argAt(args, 2).trimmed();
    if (graphName.isEmpty() || nodeUuid.isEmpty() || customName.isEmpty()) {
        sendResponse(requestId, false, "setNodeName requires graph name, node uuid and a non-empty name");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hNode = 0;
    if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    const QModelIndex subgIdx = pModel->subgIndex(hGraph);
    const QModelIndex nodeIdx = pModel->nodeIndex(hGraph, hNode);
    if (!subgIdx.isValid() || !nodeIdx.isValid() || !pModel->setCustomName(subgIdx, nodeIdx, customName)) {
        sendResponse(requestId, false, "failed to set custom name for node " + nodeUuid);
        return;
    }

    // Path to clear the custom name (revert to the class name) when name is
    // empty is not offered here: an empty name is rejected above so the meaning
    // of this command stays unambiguous.
    sendResponse(requestId, true, compactJson(QJsonObject{
        {"node_uuid", nodeUuid},
        {"name", customName}}));
}

void ApiMgr::handleGetNodeSize(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeUuid = argAt(args, 1);
    if (graphName.isEmpty() || nodeUuid.isEmpty()) {
        sendResponse(requestId, false, "getNodeSize requires graph name and node uuid");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hNode = 0;
    if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    const QModelIndex subgIdx = pModel->subgIndex(hGraph);
    const QModelIndex nodeIdx = pModel->nodeIndex(hGraph, hNode);
    if (!subgIdx.isValid() || !nodeIdx.isValid()) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    // Position always comes from the model (same data as setNodePos / Zeno_GetPos).
    const QPointF pos = nodeIdx.data(ROLE_OBJPOS).toPointF();

    // Size is optional (view-layer): only present when the graph's scene exists.
    QJsonObject obj;
    obj["node_uuid"] = nodeUuid;
    obj["class"] = nodeIdx.data(ROLE_OBJNAME).toString();
    obj["pos"] = QJsonArray{pos.x(), pos.y()};

    float w = 0.f, h = 0.f;
    if (nodeSize(pModel, hGraph, hNode, w, h)) {
        obj["width"] = (double)w;
        obj["height"] = (double)h;
    }
    sendResponse(requestId, true, compactJson(obj));
}

void ApiMgr::handleGraphGeometry(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    if (graphName.isEmpty()) {
        sendResponse(requestId, false, "graphGeometry requires a graph name");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }

    const QModelIndex subgIdx = pModel->subgIndex(hGraph);
    if (!subgIdx.isValid()) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }

    // Sizes need the on-canvas items, so the graph must have been opened.
    ZenoSubGraphScene* scene = qobject_cast<ZenoSubGraphScene*>(
        GraphsManagment::instance().gvScene(subgIdx));
    if (!scene) {
        sendResponse(requestId, false, "graph scene not available for '" + graphName
                     + "'; open the graph in the editor first");
        return;
    }

    QJsonArray nodesArr;
    const int count = pModel->itemCount(subgIdx);
    for (int r = 0; r < count; ++r) {
        const QModelIndex nodeIdx = pModel->index(r, subgIdx);
        const QString uuid = nodeIdx.data(ROLE_OBJID).toString();
        if (uuid.isEmpty())
            continue;
        const QPointF pos = nodeIdx.data(ROLE_OBJPOS).toPointF();

        QJsonObject o;
        o["node_uuid"] = uuid;
        o["class"] = nodeIdx.data(ROLE_OBJNAME).toString();
        o["pos"] = QJsonArray{pos.x(), pos.y()};

        float w = 0.f, h = 0.f;
        if (nodeSize(pModel, hGraph, (ZENO_HANDLE)nodeIdx.internalId(), w, h)) {
            o["width"] = (double)w;
            o["height"] = (double)h;
        }
        nodesArr.append(o);
    }

    // Whole-graph bounding box (scene-local makes it independent of any zoom).
    const QRectF bbox = scene->nodesBoundingRect();

    QJsonObject obj;
    obj["graph"] = graphName;
    obj["nodes"] = nodesArr;
    obj["bounding_box"] = QJsonObject{
        {"x", bbox.x()},
        {"y", bbox.y()},
        {"width", bbox.width()},
        {"height", bbox.height()},
    };
    sendResponse(requestId, true, compactJson(obj));
}

void ApiMgr::handleAddLink(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString outNodeUuid = argAt(args, 1);
    const QString outSock = argAt(args, 2);
    const QString inNodeUuid = argAt(args, 3);
    const QString inSock = argAt(args, 4);
    if (graphName.isEmpty() || outNodeUuid.isEmpty() || outSock.isEmpty() || inNodeUuid.isEmpty() || inSock.isEmpty()) {
        sendResponse(requestId, false, "addLink requires graph, out_node_uuid, out_socket, in_node_uuid, in_socket");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hOutNode = 0;
    if (!resolveNode(pModel, hGraph, outNodeUuid, hOutNode)) {
        sendResponse(requestId, false, "output node not found: " + outNodeUuid);
        return;
    }
    ZENO_HANDLE hInNode = 0;
    if (!resolveNode(pModel, hGraph, inNodeUuid, hInNode)) {
        sendResponse(requestId, false, "input node not found: " + inNodeUuid);
        return;
    }

    ZENO_ERROR err = Zeno_AddLink(hGraph, hOutNode, outSock.toStdString(), hInNode, inSock.toStdString());
    if (err == Err_NoError) {
        sendResponse(requestId, true, "added link from " + outNodeUuid + ":" + outSock + " to " + inNodeUuid + ":" + inSock);
    } else {
        sendResponse(requestId, false, "failed to add link, error code " + QString::number((uint)err));
    }
}

void ApiMgr::handleRemoveLink(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString outNodeUuid = argAt(args, 1);
    const QString outSock = argAt(args, 2);
    const QString inNodeUuid = argAt(args, 3);
    const QString inSock = argAt(args, 4);
    if (graphName.isEmpty() || outNodeUuid.isEmpty() || outSock.isEmpty() || inNodeUuid.isEmpty() || inSock.isEmpty()) {
        sendResponse(requestId, false, "removeLink requires graph, out_node_uuid, out_socket, in_node_uuid, in_socket");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hOutNode = 0;
    if (!resolveNode(pModel, hGraph, outNodeUuid, hOutNode)) {
        sendResponse(requestId, false, "output node not found: " + outNodeUuid);
        return;
    }
    ZENO_HANDLE hInNode = 0;
    if (!resolveNode(pModel, hGraph, inNodeUuid, hInNode)) {
        sendResponse(requestId, false, "input node not found: " + inNodeUuid);
        return;
    }

    ZENO_ERROR err = Zeno_RemoveLink(hGraph, hOutNode, outSock.toStdString(), hInNode, inSock.toStdString());
    if (err == Err_NoError) {
        sendResponse(requestId, true, "removed link from " + outNodeUuid + ":" + outSock + " to " + inNodeUuid + ":" + inSock);
    } else {
        sendResponse(requestId, false, "failed to remove link, error code " + QString::number((uint)err));
    }
}

// Collects the settable names of a node: params first, then input sockets.
// Used to produce actionable error messages for setParam/getParam.
// (INPUT_SOCKETS is a FuckQMap whose iterators point at FuckQPair, hence
// the it->key() syntax instead of QMap's it.key().)
QStringList settableNames(const QModelIndex& nodeIdx)
{
    QStringList names;
    const PARAMS_INFO params = nodeIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    for (auto it = params.constBegin(); it != params.constEnd(); ++it)
        names << it.key();
    const INPUT_SOCKETS inputs = nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    for (auto it = inputs.constBegin(); it != inputs.constEnd(); ++it)
        names << it->key();
    return names;
}

void ApiMgr::handleSetParam(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeUuid = argAt(args, 1);
    const QString paramName = argAt(args, 2);
    const QString valueStr = argAt(args, 3);
    if (graphName.isEmpty() || nodeUuid.isEmpty() || paramName.isEmpty()) {
        sendResponse(requestId, false, "setParam requires graph, node uuid, param name and value");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hNode = 0;
    if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    const QModelIndex nodeIdx = pModel->nodeIndex(hGraph, hNode);
    if (!nodeIdx.isValid()) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    // 1) Regular params panel entries.
    const PARAMS_INFO params = nodeIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    if (params.contains(paramName)) {
        const QString typeDesc = params.value(paramName).typeDesc;
        ZVARIANT var;
        if (!variantFromString(valueStr, typeDesc, var)) {
            sendResponse(requestId, false, "cannot parse value '" + valueStr + "' for param '" + paramName + "' of type " + typeDesc);
            return;
        }
        ZENO_ERROR err = Zeno_SetParam(hGraph, hNode, paramName.toStdString(), var);
        if (err == Err_NoError) {
            sendResponse(requestId, true, "set param " + paramName + " = " + valueStr);
        } else {
            sendResponse(requestId, false, "failed to set param, error code " + QString::number((uint)err));
        }
        return;
    }

    // 2) Input sockets: setting the default value is how an unconnected input
    //    receives data, and nodeClassesInfo reports most knobs as inputs.
    const INPUT_SOCKETS inputs = nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    if (inputs.contains(paramName)) {
        // FuckQMap::value(key, def) is declared but not implemented; use the
        // const operator[] instead (key existence is guaranteed by contains).
        const QString typeDesc = inputs[paramName].info.type;
        ZVARIANT var;
        if (!variantFromString(valueStr, typeDesc, var)) {
            sendResponse(requestId, false, "cannot parse value '" + valueStr + "' for input '" + paramName + "' of type " + typeDesc);
            return;
        }
        ZENO_ERROR err = Zeno_SetInputDefl(hGraph, hNode, paramName.toStdString(), var);
        if (err == Err_NoError) {
            sendResponse(requestId, true, "set input " + paramName + " = " + valueStr);
        } else {
            sendResponse(requestId, false, "failed to set input, error code " + QString::number((uint)err));
        }
        return;
    }

    // 3) Nothing matched: tell the client what is actually settable.
    sendResponse(requestId, false, "param not found: " + paramName
                 + "; settable params/inputs on this node: " + settableNames(nodeIdx).join(", "));
}

void ApiMgr::handleGetParam(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeUuid = argAt(args, 1);
    const QString paramName = argAt(args, 2);
    if (graphName.isEmpty() || nodeUuid.isEmpty() || paramName.isEmpty()) {
        sendResponse(requestId, false, "getParam requires graph, node uuid and param name");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hNode = 0;
    if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    ZVARIANT var;
    std::string type;
    ZENO_ERROR err = Zeno_GetParam(hGraph, hNode, paramName.toStdString(), var, type);
    if (err == Err_NoError) {
        QJsonObject obj;
        obj["name"] = paramName;
        obj["kind"] = "param";
        obj["type"] = QString::fromStdString(type);
        obj["value"] = variantToString(var);
        sendResponse(requestId, true, compactJson(obj));
        return;
    }

    // Fall back to input sockets (unconnected inputs keep a default value).
    err = Zeno_GetInputDefl(hGraph, hNode, paramName.toStdString(), var, type);
    if (err == Err_NoError) {
        QJsonObject obj;
        obj["name"] = paramName;
        obj["kind"] = "input";
        obj["type"] = QString::fromStdString(type);
        obj["value"] = variantToString(var);
        sendResponse(requestId, true, compactJson(obj));
        return;
    }

    const QModelIndex nodeIdx = pModel->nodeIndex(hGraph, hNode);
    sendResponse(requestId, false, "param not found: " + paramName
                 + "; settable params/inputs on this node: "
                 + (nodeIdx.isValid() ? settableNames(nodeIdx).join(", ") : QString()));
}

void ApiMgr::handleListGraphs(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QStringList names = pModel->subgraphsName();
    QJsonArray arr;
    for (const QString& name : names)
        arr.append(name);
    sendResponse(requestId, true, compactJson(QJsonObject{{"graphs", arr}}));
}

void ApiMgr::handleListNodes(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    if (graphName.isEmpty()) {
        sendResponse(requestId, false, "listNodes requires a graph name");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }

    const QModelIndex subgIdx = pModel->subgIndex(hGraph);
    const int count = pModel->itemCount(subgIdx);
    QJsonArray arr;
    for (int r = 0; r < count; ++r) {
        const QModelIndex nodeIdx = pModel->index(r, subgIdx);
        QJsonObject o;
        o["uuid"] = nodeIdx.data(ROLE_OBJID).toString();
        o["class"] = nodeIdx.data(ROLE_OBJNAME).toString();
        arr.append(o);
    }
    sendResponse(requestId, true, compactJson(QJsonObject{{"nodes", arr}}));
}

void ApiMgr::handleNodeInfo(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeUuid = argAt(args, 1);
    if (graphName.isEmpty() || nodeUuid.isEmpty()) {
        sendResponse(requestId, false, "nodeInfo requires graph name and node uuid");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hNode = 0;
    if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    sendResponse(requestId, true, compactJson(buildNodeInfo(pModel, hGraph, hNode)));
}

void ApiMgr::handleListNodeClasses(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const NODE_DESCS descs = pModel->descriptors();
    QJsonArray arr;
    for (auto it = descs.constBegin(); it != descs.constEnd(); ++it)
        arr.append(it.key());
    sendResponse(requestId, true, compactJson(QJsonObject{{"classes", arr}}));
}

void ApiMgr::handleListNodeCategories(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const NODE_CATES cates = pModel->getCates();
    QJsonArray arr;
    for (auto it = cates.constBegin(); it != cates.constEnd(); ++it)
        arr.append(it.key());
    sendResponse(requestId, true, compactJson(QJsonObject{{"categories", arr}}));
}

void ApiMgr::handleNodeClassesInfo(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString category = argAt(args, 0).trimmed();
    const QString keyword = argAt(args, 1).trimmed();
    int page = argAt(args, 2).toInt();
    if (page <= 0)
        page = 1;
    int size = argAt(args, 3).toInt();
    if (size <= 0)
        size = 200;

    const NODE_DESCS descs = pModel->descriptors();
    const int startIdx = (page - 1) * size;
    const int endIdx = startIdx + size;

    QJsonArray arr;
    int idx = 0;
    for (auto it = descs.constBegin(); it != descs.constEnd(); ++it) {
        const NODE_DESC& desc = it.value();
        if (desc.name.isEmpty())
            continue;
        if (!category.isEmpty() && !desc.categories.contains(category))
            continue;
        if (!keyword.isEmpty() && !desc.name.contains(keyword, Qt::CaseInsensitive))
            continue;
        if (idx >= startIdx && idx < endIdx)
            arr.append(buildDescJson(desc));
        ++idx;
    }

    QJsonObject resultObj;
    resultObj["total"] = idx;
    resultObj["page"] = page;
    resultObj["size"] = size;
    resultObj["classes"] = arr;
    sendResponse(requestId, true, compactJson(resultObj));
}

void ApiMgr::handleGraphInfo(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    if (graphName.isEmpty()) {
        sendResponse(requestId, false, "graphInfo requires a graph name");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }

    const QModelIndex subgIdx = pModel->subgIndex(hGraph);
    const int count = pModel->itemCount(subgIdx);

    QJsonObject graphObj;
    graphObj["name"] = graphName;
    QJsonArray nodesArr;
    QJsonArray linksArr;
    for (int r = 0; r < count; ++r) {
        const QModelIndex nodeIdx = pModel->index(r, subgIdx);
        const ZENO_HANDLE hNode = (ZENO_HANDLE)nodeIdx.internalId();
        const QJsonObject info = buildNodeInfo(pModel, hGraph, hNode);

        // Aggregate the full topology from each node's input links.
        const QJsonArray inputs = info["inputs"].toArray();
        for (const QJsonValue& v : inputs) {
            const QJsonObject io = v.toObject();
            if (io.contains("link")) {
                const QJsonObject link = io["link"].toObject();
                QJsonObject e;
                e["from"] = link["node"];
                e["from_socket"] = link["socket"];
                e["to"] = info["uuid"];
                e["to_socket"] = io["name"];
                linksArr.append(e);
            }
        }
        nodesArr.append(info);
    }

    graphObj["nodes"] = nodesArr;
    graphObj["links"] = linksArr;
    sendResponse(requestId, true, compactJson(graphObj));
}

void ApiMgr::handleRunGraph(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    ZenoMainWindow* main = zenoApp->getMainWindow();
    if (!main) {
        sendResponse(requestId, false, "no main window available to run the graph");
        return;
    }

    // Optional run type as args[0]; defaults to runAll. The mapping mirrors the
    // GUI run button in docktabcontent.cpp.
    const QString typeStr = argAt(args, 0).trimmed();
    runType runtype = RunALL;
    if (!typeStr.isEmpty()) {
        const QString t = typeStr.toLower();
        if (t == "loadasset") runtype = LoadAsset;
        else if (t == "runall") runtype = RunALL;
        else if (t == "runmatrix") runtype = RunMatrix;
        else if (t == "runlightcamera") runtype = RunLightCamera;
        else if (t == "runmaterial") runtype = RunMaterial;
        else {
            sendResponse(requestId, false, "unknown run type: " + typeStr
                         + " (expected loadAsset, runAll, runMatrix, runLightCamera or runMaterial)");
            return;
        }
    }

    // RunMatrix/RunLightCamera/RunMaterial only work in cache mode, same guard
    // as the GUI run button.
    if (runtype == RunMatrix || runtype == RunLightCamera || runtype == RunMaterial) {
        QSettings settings(zsCompanyName, zsEditor);
        if (!settings.value(zsCacheEnable).toBool()) {
            sendResponse(requestId, false, "run type " + typeStr + " can only be used in cache mode");
            return;
        }
    }

    // Optional timeout in seconds as args[1] is intentionally NOT supported:
    // the call waits for the run to finish (or be killed via killGraph), and
    // terminating a run is the caller's decision.

    // Optional clearCache flag as args[1]: whether to wipe history caches before
    // launching. Absent/false keeps them. This only takes effect because we mark
    // the launch as fromCmd, which suppresses the modal "clear cache folder?"
    // dialog in ZTcpServer::startProc and decides silently from this flag.
    bool clearCache = false;
    const QString cacheStr = argAt(args, 1).trimmed();
    if (!cacheStr.isEmpty()) {
        if (!parseToggleText(cacheStr, clearCache)) {
            sendResponse(requestId, false, "runGraph clearCache must be a boolean (true/false/1/0/on/off), got: " + cacheStr);
            return;
        }
    }

    // Same launch preparation as the GUI run button: tell the non-GL viewports
    // to render separately and select the matching cache strategy.
    std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
    ZASSERT_EXIT(mgr);
    for (DisplayWidget* view : main->viewports()) {
        if (!view->isGLViewport())
            view->setRenderSeparately(runtype);
    }
    switch (runtype) {
    case LoadAsset:
        mgr->setCacheOpt(ZCacheMgr::Opt_LoadAsset);
        break;
    case RunALL:
        mgr->setCacheOpt(ZCacheMgr::Opt_RunAll);
        break;
    case RunMatrix:
        mgr->setCacheOpt(ZCacheMgr::Opt_RunMatrix);
        break;
    case RunLightCamera:
    case RunMaterial:
        mgr->setCacheOpt(ZCacheMgr::Opt_RunLightCameraMaterial);
        break;
    }

#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
    // Waiting for completion is only possible when the runner is a separate
    // process that reports its exit via ZenoMainWindow::runFinished
    // (ZTcpServer::onProcFinished). handleCommand runs in the main thread, so
    // replying has to be deferred via signals instead of blocking here.
    ZTcpServer* server = zenoApp->getServer();
    if (!server) {
        sendResponse(requestId, false, "no runner server available to run the graph");
        return;
    }
    // startProc silently refuses to launch while a runner is still alive, which
    // would leave this request without a response; refuse up front instead.
    if (!m_pendingRunRequestId.isEmpty() || server->isProcRunning()) {
        sendResponse(requestId, false, "a run is already in progress; call killGraph to stop it first");
        return;
    }

    connectRunSignalsOnce(main);

    // The reply is deferred until the runner process exits. Graph failures
    // (bad links, parameter/type errors, ...) are delivered by the runner as
    // "reportStatus" packets (runnermain.cpp) and decoded into
    // zeno::getSession().globalStatus (viewdecode.cpp); onRunGraphFinished()
    // turns them into the error response.
    m_runnerErrored = false;
    m_killRequested = false;
    m_pendingRunRequestId = requestId;
    beginRunLogCapture();

    main->onRunTriggered(runtype, /*fromCmd=*/true, /*cmdRmHistoryCacheBeforeRun=*/clearCache);
#else
    main->onRunTriggered(runtype, /*fromCmd=*/true, /*cmdRmHistoryCacheBeforeRun=*/clearCache);
    // Pipe / single-process builds have no runFinished signal to wait for.
    sendResponse(requestId, true, "run started (completion tracking requires the TCP multiprocess build)");
#endif
}

void ApiMgr::connectRunSignalsOnce(ZenoMainWindow* main)
{
    if (m_runSignalsConnected)
        return;
    connect(main, &ZenoMainWindow::runFinished, this, &ApiMgr::onRunGraphFinished);
#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
    if (ZTcpServer* server = zenoApp->getServer()) {
        connect(server, &ZTcpServer::runnerError, this, &ApiMgr::onRunnerError);
        connect(server, &ZTcpServer::startProcFailed, this, &ApiMgr::onStartProcFailed);
    }
#endif
    m_runSignalsConnected = true;
}

void ApiMgr::finishPendingRun(bool ok, const QString& result)
{
    const QString requestId = m_pendingRunRequestId;
    m_pendingRunRequestId.clear();
    if (!requestId.isEmpty())
        sendResponse(requestId, ok, result);
}

void ApiMgr::onRunGraphFinished()
{
    if (m_pendingRunRequestId.isEmpty())
        return;   // the run was started from the GUI, not via runGraph

    if (m_killRequested) {
        finishPendingRun(false, "run killed");
        return;
    }

    // Graph failures arrive as "reportStatus" packets from the runner and are
    // decoded into globalStatus; see viewdecode.cpp / runnermain.cpp.
    const auto& stat = zeno::getSession().globalStatus;
    if (stat && stat->failed()) {
        const QString node = QString::fromStdString(stat->nodeName);
        const QString message = stat->error ? QString::fromStdString(stat->error->message)
                                            : QStringLiteral("unknown error");
        finishPendingRun(false, runFailureMessage(
            QString("run failed in node '%1': %2").arg(node, message)));
        return;
    }

    if (m_runnerErrored) {
        finishPendingRun(false, runFailureMessage("runner process exited abnormally"));
        return;
    }

    finishPendingRun(true, "run finished");
}

void ApiMgr::beginRunLogCapture()
{
    QStandardItemModel* logModel = zenoApp->logModel();
    m_runLogStartRow = logModel ? logModel->rowCount() : 0;
}

QString ApiMgr::collectRunErrorLogs() const
{
    QStandardItemModel* logModel = zenoApp->logModel();
    if (!logModel)
        return QString();

    // Level mapping happens in ZWidgetErrStream: the runner line's leading
    // tag selects the QMessageLogger call ("[E"->warning, "[C"->critical,
    // "[W"->warning, ...), and customMsgHandler then upgrades any "[E "-prefixed
    // message to QtFatalMsg. So the error/warn/critical rows below cover the
    // runner's E/C/W lines (and unrecognized lines, which default to critical).
    QStringList lines;
    const int rowCount = logModel->rowCount();
    for (int r = m_runLogStartRow; r < rowCount; r++) {
        QStandardItem* item = logModel->item(r);
        if (!item)
            continue;
        const int type = item->data(ROLE_LOGTYPE).toInt();
        if (type == QtFatalMsg || type == QtCriticalMsg || type == QtWarningMsg)
            lines.append(item->text());
    }
    return lines.join('\n');
}

QString ApiMgr::runFailureMessage(const QString& summary) const
{
    const QString logs = collectRunErrorLogs();
    return logs.isEmpty() ? summary : summary + '\n' + logs;
}

void ApiMgr::onRunnerError()
{
    // ZTcpServer::onProcFinished emits runnerError (crash / non-zero exit)
    // before runFinished; remember it for the deferred response.
    m_runnerErrored = true;
}

void ApiMgr::onStartProcFailed(const QString& reason)
{
    // ZTcpServer::startProc abandoned the launch (invalid cache dir / process
    // failed to start). No process exists, so runFinished will never fire;
    // settle the deferred runGraph response right away. For GUI-launched runs
    // there is no pending request, keep the legacy dialog-free silent return.
    if (m_pendingRunRequestId.isEmpty())
        return;
    finishPendingRun(false, "run failed to start: " + reason);
}

void ApiMgr::handleKillGraph(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    ZenoMainWindow* main = zenoApp->getMainWindow();
    if (!main) {
        sendResponse(requestId, false, "no main window available to kill the run");
        return;
    }

#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
    ZTcpServer* server = zenoApp->getServer();
    if (!server) {
        sendResponse(requestId, false, "no runner server available to kill");
        return;
    }
    const bool hasPending = !m_pendingRunRequestId.isEmpty();
    if (!hasPending && !server->isProcRunning()) {
        sendResponse(requestId, false, "no run in progress to kill");
        return;
    }
    // Same as the GUI kill button (docktabcontent.cpp m_btnKill). If a runGraph
    // call is still waiting, mark it so its deferred response reports the kill.
    m_killRequested = hasPending;
    killProgram();
    // killProc() destroys the runner QProcess, so onProcFinished (and with it
    // runFinished) may never fire for a killed run; settle the pending reply
    // here instead. If it did fire synchronously inside killProgram(), the
    // request was already settled and this is a no-op.
    if (hasPending && !m_pendingRunRequestId.isEmpty())
        finishPendingRun(false, "run killed");
    sendResponse(requestId, true, hasPending
        ? "kill requested; the pending runGraph call will report the termination"
        : "kill requested");
#else
    killProgram();
    sendResponse(requestId, true, "kill requested");
#endif
}

void ApiMgr::handleSetTimelineRange(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    ZenoMainWindow* main = zenoApp->getMainWindow();
    if (!main || !main->timeline()) {
        sendResponse(requestId, false, "no main window timeline available");
        return;
    }

    bool okStart = false, okEnd = false;
    const int startFrame = argAt(args, 0).trimmed().toInt(&okStart);
    const int endFrame = argAt(args, 1).trimmed().toInt(&okEnd);
    if (!okStart || !okEnd) {
        sendResponse(requestId, false, "setTimelineRange requires integer start and end frames");
        return;
    }
    if (startFrame < 0 || endFrame < startFrame) {
        sendResponse(requestId, false, QString("invalid range [%1, %2]: need 0 <= start <= end")
                     .arg(startFrame).arg(endFrame));
        return;
    }

    // Reuse the current timeline state (fps etc.) and only move the frame
    // range; this is the same reset path the GUI takes when a zsg file
    // restores its time info.
    TIMELINE_INFO info = main->timelineInfo();
    info.beginFrame = startFrame;
    info.endFrame = endFrame;
    main->resetTimeline(info);

    const TIMELINE_INFO updated = main->timelineInfo();
    sendResponse(requestId, true, compactJson(QJsonObject{
        {"start_frame", updated.beginFrame},
        {"end_frame", updated.endFrame},
        {"fps", updated.timelinefps}}));
}

void ApiMgr::handleSetView(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeUuid = argAt(args, 1);
    bool bOn = false;
    if (graphName.isEmpty() || nodeUuid.isEmpty() || !parseToggleText(argAt(args, 2), bOn)) {
        sendResponse(requestId, false, "setView requires graph name, node uuid and on/off (true/false)");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hNode = 0;
    if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    ZENO_ERROR err = Zeno_SetView(hGraph, hNode, bOn);
    if (err == Err_NoError) {
        sendResponse(requestId, true, QString("set view ") + (bOn ? "on" : "off") + " for node " + nodeUuid);
    } else {
        sendResponse(requestId, false, "failed to set view, error code " + QString::number((uint)err));
    }
}

void ApiMgr::handleSetOnce(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeUuid = argAt(args, 1);
    bool bOn = false;
    if (graphName.isEmpty() || nodeUuid.isEmpty() || !parseToggleText(argAt(args, 2), bOn)) {
        sendResponse(requestId, false, "setOnce requires graph name, node uuid and on/off (true/false)");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hNode = 0;
    if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    ZENO_ERROR err = Zeno_SetOnce(hGraph, hNode, bOn);
    if (err == Err_NoError) {
        sendResponse(requestId, true, QString("set once ") + (bOn ? "on" : "off") + " for node " + nodeUuid);
    } else {
        sendResponse(requestId, false, "failed to set once, error code " + QString::number((uint)err));
    }
}

void ApiMgr::handleSetNodePos(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString nodeUuid = argAt(args, 1);
    if (graphName.isEmpty() || nodeUuid.isEmpty()) {
        sendResponse(requestId, false, "setNodePos requires graph name, node uuid and 'x,y' position");
        return;
    }

    // Accept "x,y" (splitVecList also tolerates "[x,y]").
    QList<double> pos;
    if (!splitVecList(argAt(args, 2), pos) || pos.size() != 2) {
        sendResponse(requestId, false, "setNodePos requires the position as 'x,y', got '" + argAt(args, 2) + "'");
        return;
    }

    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    ZENO_HANDLE hNode = 0;
    if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
        sendResponse(requestId, false, "node not found: " + nodeUuid);
        return;
    }

    ZENO_ERROR err = Zeno_SetPos(hGraph, hNode, {(float)pos[0], (float)pos[1]});
    if (err == Err_NoError) {
        sendResponse(requestId, true, compactJson(QJsonObject{
            {"node_uuid", nodeUuid},
            {"pos", QJsonArray{pos[0], pos[1]}}}));
    } else {
        sendResponse(requestId, false, "failed to set node pos, error code " + QString::number((uint)err));
    }
}

void ApiMgr::handleSetNodePosList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString items = argAt(args, 1);
    if (graphName.isEmpty() || items.trimmed().isEmpty()) {
        sendResponse(requestId, false, "setNodePosList requires a graph name and items as 'uuid:x,y;uuid:x,y;...'");
        return;
    }
    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    QJsonArray results;
    bool allOk = true;
    const QStringList entries = items.split(';', Qt::SkipEmptyParts);
    for (const QString& entry : entries) {
        const QString item = entry.trimmed();
        const int colon = item.indexOf(':');
        const QString nodeUuid = colon >= 0 ? item.left(colon).trimmed() : "";
        const QString posText = colon >= 0 ? item.mid(colon + 1) : item;
        QJsonObject res;
        res["node_uuid"] = nodeUuid;
        QList<double> pos;
        if (nodeUuid.isEmpty() || !splitVecList(posText, pos) || pos.size() != 2) {
            res["ok"] = false;
            res["error"] = "malformed item '" + item + "', expected uuid:x,y";
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hNode = 0;
        if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
            res["ok"] = false;
            res["error"] = "node not found: " + nodeUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_ERROR err = Zeno_SetPos(hGraph, hNode, {(float)pos[0], (float)pos[1]});
        if (err == Err_NoError) {
            res["ok"] = true;
            res["pos"] = QJsonArray{pos[0], pos[1]};
        } else {
            res["ok"] = false;
            res["error"] = "failed to set pos, error code " + QString::number((uint)err);
            allOk = false;
        }
        results.append(res);
    }
    sendResponse(requestId, allOk, compactJson(QJsonObject{
        {"graph", graphName},
        {"count", results.size()},
        {"results", results}}));
}

void ApiMgr::handleSetViewList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString items = argAt(args, 1);
    if (graphName.isEmpty() || items.trimmed().isEmpty()) {
        sendResponse(requestId, false, "setViewList requires a graph name and items as 'uuid:on/off;uuid:on/off;...'");
        return;
    }
    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    QJsonArray results;
    bool allOk = true;
    const QStringList entries = items.split(';', Qt::SkipEmptyParts);
    for (const QString& entry : entries) {
        const QString item = entry.trimmed();
        const int colon = item.indexOf(':');
        const QString nodeUuid = colon >= 0 ? item.left(colon).trimmed() : "";
        const QString flagText = colon >= 0 ? item.mid(colon + 1) : item;
        bool bOn = false;
        QJsonObject res;
        res["node_uuid"] = nodeUuid;
        if (nodeUuid.isEmpty() || !parseToggleText(flagText, bOn)) {
            res["ok"] = false;
            res["error"] = "malformed item '" + item + "', expected uuid:on/off";
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hNode = 0;
        if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
            res["ok"] = false;
            res["error"] = "node not found: " + nodeUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_ERROR err = Zeno_SetView(hGraph, hNode, bOn);
        if (err == Err_NoError) {
            res["ok"] = true;
            res["view"] = bOn;
        } else {
            res["ok"] = false;
            res["error"] = "failed to set view, error code " + QString::number((uint)err);
            allOk = false;
        }
        results.append(res);
    }
    sendResponse(requestId, allOk, compactJson(QJsonObject{
        {"graph", graphName},
        {"count", results.size()},
        {"results", results}}));
}

void ApiMgr::handleAddLinkList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString items = argAt(args, 1);
    if (graphName.isEmpty() || items.trimmed().isEmpty()) {
        sendResponse(requestId, false, "addLinkList requires a graph name and items as 'outNode:outSock->inNode:inSock;...'");
        return;
    }
    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    QJsonArray results;
    bool allOk = true;
    const QStringList entries = items.split(';', Qt::SkipEmptyParts);
    for (const QString& entry : entries) {
        const QString item = entry.trimmed();
        const int arrow = item.indexOf("->");
        const QString outPart = arrow >= 0 ? item.left(arrow).trimmed() : "";
        const QString inPart = arrow >= 0 ? item.mid(arrow + 2).trimmed() : "";
        const int outp = outPart.indexOf(':');
        const int inp = inPart.indexOf(':');
        const QString outUuid = outp >= 0 ? outPart.left(outp).trimmed() : "";
        const QString outSock = outp >= 0 ? outPart.mid(outp + 1).trimmed() : "";
        const QString inUuid = inp >= 0 ? inPart.left(inp).trimmed() : "";
        const QString inSock = inp >= 0 ? inPart.mid(inp + 1).trimmed() : "";
        QJsonObject res;
        if (outUuid.isEmpty() || outSock.isEmpty() || inUuid.isEmpty() || inSock.isEmpty()) {
            res["ok"] = false;
            res["error"] = "malformed item '" + item + "', expected outNode:outSock->inNode:inSock";
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hOutNode = 0;
        if (!resolveNode(pModel, hGraph, outUuid, hOutNode)) {
            res["ok"] = false;
            res["error"] = "output node not found: " + outUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hInNode = 0;
        if (!resolveNode(pModel, hGraph, inUuid, hInNode)) {
            res["ok"] = false;
            res["error"] = "input node not found: " + inUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        const QString link = outUuid + ":" + outSock + "->" + inUuid + ":" + inSock;
        ZENO_ERROR err = Zeno_AddLink(hGraph, hOutNode, outSock.toStdString(), hInNode, inSock.toStdString());
        if (err == Err_NoError) {
            res["ok"] = true;
            res["link"] = link;
        } else {
            res["ok"] = false;
            res["error"] = "failed to add link, error code " + QString::number((uint)err);
            allOk = false;
        }
        results.append(res);
    }
    sendResponse(requestId, allOk, compactJson(QJsonObject{
        {"graph", graphName},
        {"count", results.size()},
        {"results", results}}));
}

void ApiMgr::handleDeleteNodeList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString items = argAt(args, 1);
    if (graphName.isEmpty() || items.trimmed().isEmpty()) {
        sendResponse(requestId, false, "deleteNodeList requires a graph name and a 'uuid;uuid;...' list");
        return;
    }
    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    QJsonArray results;
    bool allOk = true;
    const QStringList entries = items.split(';', Qt::SkipEmptyParts);
    for (const QString& entry : entries) {
        const QString nodeUuid = entry.trimmed();
        QJsonObject res;
        res["node_uuid"] = nodeUuid;
        if (nodeUuid.isEmpty()) {
            res["ok"] = false;
            res["error"] = "empty node uuid in item '" + entry + "'";
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hNode = 0;
        if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
            res["ok"] = false;
            res["error"] = "node not found: " + nodeUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_ERROR err = Zeno_DeleteNode(hGraph, hNode);
        if (err == Err_NoError) {
            res["ok"] = true;
        } else {
            res["ok"] = false;
            res["error"] = "failed to delete node, error code " + QString::number((uint)err);
            allOk = false;
        }
        results.append(res);
    }
    sendResponse(requestId, allOk, compactJson(QJsonObject{
        {"graph", graphName},
        {"count", results.size()},
        {"results", results}}));
}

void ApiMgr::handleCreateNodeList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString items = argAt(args, 1);
    if (graphName.isEmpty() || items.trimmed().isEmpty()) {
        sendResponse(requestId, false, "createNodeList requires a graph name and a 'class;class:name;...' list");
        return;
    }
    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    QJsonArray results;
    bool allOk = true;
    const QStringList entries = items.split(';', Qt::SkipEmptyParts);
    for (const QString& entry : entries) {
        const QString item = entry.trimmed();
        const int colon = item.indexOf(':');
        const QString nodeCls = (colon >= 0 ? item.left(colon) : item).trimmed();
        const QString customName = colon >= 0 ? item.mid(colon + 1).trimmed() : "";
        QJsonObject res;
        if (nodeCls.isEmpty()) {
            res["ok"] = false;
            res["error"] = "malformed item '" + item + "', expected 'class' or 'class:name'";
            allOk = false;
            results.append(res);
            continue;
        }
        QJsonObject created;
        QString err;
        if (createNodeAndGeometry(pModel, hGraph, graphName, nodeCls, customName, created, err)) {
            res["ok"] = true;
            // Flatten the shared created payload into the per-item result.
            const QStringList keys = created.keys();
            for (const QString& k : keys)
                res[k] = created.value(k);
        } else {
            res["ok"] = false;
            res["error"] = err;
            res["class"] = nodeCls;
            allOk = false;
        }
        results.append(res);
    }
    sendResponse(requestId, allOk, compactJson(QJsonObject{
        {"graph", graphName},
        {"count", results.size()},
        {"results", results}}));
}

void ApiMgr::handleSetNodeNameList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString items = argAt(args, 1);
    if (graphName.isEmpty() || items.trimmed().isEmpty()) {
        sendResponse(requestId, false, "setNodeNameList requires a graph name and a 'uuid:name;uuid:name;...' list");
        return;
    }
    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    QJsonArray results;
    bool allOk = true;
    const QStringList entries = items.split(';', Qt::SkipEmptyParts);
    for (const QString& entry : entries) {
        const QString item = entry.trimmed();
        const int colon = item.indexOf(':');
        const QString nodeUuid = colon >= 0 ? item.left(colon).trimmed() : "";
        const QString customName = colon >= 0 ? item.mid(colon + 1).trimmed() : "";
        QJsonObject res;
        res["node_uuid"] = nodeUuid;
        if (nodeUuid.isEmpty() || customName.isEmpty()) {
            res["ok"] = false;
            res["error"] = "malformed item '" + item + "', expected uuid:name";
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hNode = 0;
        if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
            res["ok"] = false;
            res["error"] = "node not found: " + nodeUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        const QModelIndex subgIdx = pModel->subgIndex(hGraph);
        const QModelIndex nodeIdx = pModel->nodeIndex(hGraph, hNode);
        if (!subgIdx.isValid() || !nodeIdx.isValid() || !pModel->setCustomName(subgIdx, nodeIdx, customName)) {
            res["ok"] = false;
            res["error"] = "failed to set custom name for node " + nodeUuid;
            allOk = false;
        } else {
            res["ok"] = true;
            res["name"] = customName;
        }
        results.append(res);
    }
    sendResponse(requestId, allOk, compactJson(QJsonObject{
        {"graph", graphName},
        {"count", results.size()},
        {"results", results}}));
}

void ApiMgr::handleRemoveLinkList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString items = argAt(args, 1);
    if (graphName.isEmpty() || items.trimmed().isEmpty()) {
        sendResponse(requestId, false, "removeLinkList requires a graph name and a 'outNode:outSock->inNode:inSock;...' list");
        return;
    }
    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    QJsonArray results;
    bool allOk = true;
    const QStringList entries = items.split(';', Qt::SkipEmptyParts);
    for (const QString& entry : entries) {
        const QString item = entry.trimmed();
        const int arrow = item.indexOf("->");
        const QString outPart = arrow >= 0 ? item.left(arrow).trimmed() : "";
        const QString inPart = arrow >= 0 ? item.mid(arrow + 2).trimmed() : "";
        const int outp = outPart.indexOf(':');
        const int inp = inPart.indexOf(':');
        const QString outUuid = outp >= 0 ? outPart.left(outp).trimmed() : "";
        const QString outSock = outp >= 0 ? outPart.mid(outp + 1).trimmed() : "";
        const QString inUuid = inp >= 0 ? inPart.left(inp).trimmed() : "";
        const QString inSock = inp >= 0 ? inPart.mid(inp + 1).trimmed() : "";
        QJsonObject res;
        if (outUuid.isEmpty() || outSock.isEmpty() || inUuid.isEmpty() || inSock.isEmpty()) {
            res["ok"] = false;
            res["error"] = "malformed item '" + item + "', expected outNode:outSock->inNode:inSock";
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hOutNode = 0;
        if (!resolveNode(pModel, hGraph, outUuid, hOutNode)) {
            res["ok"] = false;
            res["error"] = "output node not found: " + outUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hInNode = 0;
        if (!resolveNode(pModel, hGraph, inUuid, hInNode)) {
            res["ok"] = false;
            res["error"] = "input node not found: " + inUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        const QString link = outUuid + ":" + outSock + "->" + inUuid + ":" + inSock;
        ZENO_ERROR err = Zeno_RemoveLink(hGraph, hOutNode, outSock.toStdString(), hInNode, inSock.toStdString());
        if (err == Err_NoError) {
            res["ok"] = true;
            res["link"] = link;
        } else {
            res["ok"] = false;
            res["error"] = "failed to remove link, error code " + QString::number((uint)err);
            allOk = false;
        }
        results.append(res);
    }
    sendResponse(requestId, allOk, compactJson(QJsonObject{
        {"graph", graphName},
        {"count", results.size()},
        {"results", results}}));
}

void ApiMgr::handleSetOnceList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString items = argAt(args, 1);
    if (graphName.isEmpty() || items.trimmed().isEmpty()) {
        sendResponse(requestId, false, "setOnceList requires a graph name and a 'uuid:on/off;...' list");
        return;
    }
    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    QJsonArray results;
    bool allOk = true;
    const QStringList entries = items.split(';', Qt::SkipEmptyParts);
    for (const QString& entry : entries) {
        const QString item = entry.trimmed();
        const int colon = item.indexOf(':');
        const QString nodeUuid = colon >= 0 ? item.left(colon).trimmed() : "";
        const QString flagText = colon >= 0 ? item.mid(colon + 1) : item;
        bool bOn = false;
        QJsonObject res;
        res["node_uuid"] = nodeUuid;
        if (nodeUuid.isEmpty() || !parseToggleText(flagText, bOn)) {
            res["ok"] = false;
            res["error"] = "malformed item '" + item + "', expected uuid:on/off";
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hNode = 0;
        if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
            res["ok"] = false;
            res["error"] = "node not found: " + nodeUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_ERROR err = Zeno_SetOnce(hGraph, hNode, bOn);
        if (err == Err_NoError) {
            res["ok"] = true;
            res["once"] = bOn;
        } else {
            res["ok"] = false;
            res["error"] = "failed to set once, error code " + QString::number((uint)err);
            allOk = false;
        }
        results.append(res);
    }
    sendResponse(requestId, allOk, compactJson(QJsonObject{
        {"graph", graphName},
        {"count", results.size()},
        {"results", results}}));
}

void ApiMgr::handleSetParamList(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString graphName = argAt(args, 0);
    const QString items = argAt(args, 1);
    if (graphName.isEmpty() || items.trimmed().isEmpty()) {
        sendResponse(requestId, false, "setParamList requires a graph name and a 'uuid:param:value;...' list");
        return;
    }
    ZENO_HANDLE hGraph = 0;
    if (!resolveGraph(pModel, graphName, hGraph)) {
        sendResponse(requestId, false, "graph not found: " + graphName);
        return;
    }
    QJsonArray results;
    bool allOk = true;
    const QStringList entries = items.split(';', Qt::SkipEmptyParts);
    for (const QString& entry : entries) {
        const QString item = entry.trimmed();
        const int c1 = item.indexOf(':');
        const int c2 = c1 >= 0 ? item.indexOf(':', c1 + 1) : -1;
        const QString nodeUuid = c1 >= 0 ? item.left(c1).trimmed() : "";
        // Value keeps everything after the second ':' verbatim (may contain ':').
        const QString paramName = c2 >= 0 ? item.mid(c1 + 1, c2 - c1 - 1).trimmed() : "";
        const QString valueStr = c2 >= 0 ? item.mid(c2 + 1) : "";
        QJsonObject res;
        res["node_uuid"] = nodeUuid;
        // An empty value is allowed (e.g. clearing a string field); only the
        // node uuid and param name are required to form a valid item.
        if (nodeUuid.isEmpty() || paramName.isEmpty()) {
            res["ok"] = false;
            res["error"] = "malformed item '" + item + "', expected uuid:param:value";
            allOk = false;
            results.append(res);
            continue;
        }
        ZENO_HANDLE hNode = 0;
        if (!resolveNode(pModel, hGraph, nodeUuid, hNode)) {
            res["ok"] = false;
            res["error"] = "node not found: " + nodeUuid;
            allOk = false;
            results.append(res);
            continue;
        }
        const QModelIndex nodeIdx = pModel->nodeIndex(hGraph, hNode);
        if (!nodeIdx.isValid()) {
            res["ok"] = false;
            res["error"] = "node not found: " + nodeUuid;
            allOk = false;
            results.append(res);
            continue;
        }

        QString err;
        bool ok = false;
        // 1) Regular params panel entries.
        const PARAMS_INFO params = nodeIdx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        if (params.contains(paramName)) {
            ZVARIANT var;
            if (variantFromString(valueStr, params.value(paramName).typeDesc, var)) {
                ok = Zeno_SetParam(hGraph, hNode, paramName.toStdString(), var) == Err_NoError;
                if (!ok) err = "failed to set param";
            } else {
                err = "cannot parse value '" + valueStr + "' for param '" + paramName + "'";
            }
        } else if (nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>().contains(paramName)) {
            const INPUT_SOCKETS inputs = nodeIdx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
            ZVARIANT var;
            if (variantFromString(valueStr, inputs[paramName].info.type, var)) {
                ok = Zeno_SetInputDefl(hGraph, hNode, paramName.toStdString(), var) == Err_NoError;
                if (!ok) err = "failed to set input";
            } else {
                err = "cannot parse value '" + valueStr + "' for input '" + paramName + "'";
            }
        } else {
            err = "param not found: " + paramName + "; settable params/inputs on this node: "
                + settableNames(nodeIdx).join(", ");
        }

        if (ok) {
            res["ok"] = true;
            res["name"] = paramName;
            res["value"] = valueStr;
        } else {
            res["ok"] = false;
            res["error"] = err;
            allOk = false;
        }
        results.append(res);
    }
    sendResponse(requestId, allOk, compactJson(QJsonObject{
        {"graph", graphName},
        {"count", results.size()},
        {"results", results}}));
}

void ApiMgr::handleOpenFile(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString path = argAt(args, 0);
    if (path.isEmpty()) {
        sendResponse(requestId, false, "openFile requires a file path");
        return;
    }

    ZENO_ERROR err = Zeno_OpenFile(path.toStdString());
    if (err == Err_NoError) {
        sendResponse(requestId, true, "opened file: " + path);
    } else {
        sendResponse(requestId, false, "failed to open file, error code " + QString::number((uint)err));
    }
}

void ApiMgr::handleSaveAs(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const QString path = argAt(args, 0);
    if (path.isEmpty()) {
        sendResponse(requestId, false, "saveAs requires a file path");
        return;
    }

    ZENO_ERROR err = Zeno_SaveAs(path.toStdString());
    if (err == Err_NoError) {
        sendResponse(requestId, true, "saved file: " + path);
    } else {
        sendResponse(requestId, false, "failed to save file, error code " + QString::number((uint)err));
    }
}

void ApiMgr::handleGetCameraOptix(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
#ifndef ZENO_OPTIX_PROC
    // The camera is owned by the render loop and only accessed safely on the
    // optix worker thread, so route the query through the viewport (which
    // marshals it onto that thread) rather than touching scene->camera here.
    ZOptixViewport* view = optixViewport();
    if (!view || !view->isOptixViewRunning()) {
        sendResponse(requestId, false, "no running optix viewport available");
        return;
    }
    const QJsonObject obj = view->getCameraJson();
    if (obj.isEmpty()) {
        sendResponse(requestId, false, "failed to read the optix camera");
        return;
    }
    sendResponse(requestId, true, compactJson(obj));
#else
    sendResponse(requestId, false, "getCameraOptix is not supported in the proc optix build");
#endif
}

void ApiMgr::handleSetCameraOptix(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
#ifndef ZENO_OPTIX_PROC
    ZOptixViewport* view = optixViewport();
    if (!view || !view->isOptixViewRunning()) {
        sendResponse(requestId, false, "no running optix viewport available");
        return;
    }

    // args[0] is a JSON object string holding the subset of fields to update.
    const QJsonDocument doc = QJsonDocument::fromJson(argAt(args, 0).toUtf8());
    if (!doc.isObject()) {
        sendResponse(requestId, false, "setCameraOptix requires a JSON object argument (e.g. {\"pos\":[...],\"fov\":45})");
        return;
    }

    // Apply on the optix worker thread; the echo lets the client verify the
    // camera state that was actually applied.
    const QJsonObject obj = view->setCameraJson(doc.object());
    if (obj.isEmpty()) {
        sendResponse(requestId, false, "failed to apply the optix camera");
        return;
    }
    sendResponse(requestId, true, compactJson(obj));
#else
    sendResponse(requestId, false, "setCameraOptix is not supported in the proc optix build");
#endif
}

void ApiMgr::handleScreenShotOptix(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
#ifdef ZENO_OPTIX_PROC
    sendResponse(requestId, false, "screenshot is not supported in the proc optix build");
    return;
#else
    // Make sure a live ray-tracing viewport exists before asking it to render.
    auto* view = ensureOptixViewport();
    if (!view || !view->isOptixViewRunning()) {
        sendResponse(requestId, false, "no running optix viewport available for screenshot");
        return;
    }

    int numSamples = 0;  // 0 = keep the viewport's own sample count
    const QString samplesStr = argAt(args, 0).trimmed();
    if (!samplesStr.isEmpty()) {
        bool ok = false;
        numSamples = samplesStr.toInt(&ok);
        if (!ok || numSamples < 1) {
            sendResponse(requestId, false, "screenShotOptix sample rate must be a positive integer, got: " + samplesStr);
            return;
        }
    }

    QImage img = view->captureScreen(numSamples);
    if (img.isNull()) {
        sendResponse(requestId, false, "screenshot failed: rendering did not produce an image");
        return;
    }

    static const qint64 kMaxBytes = 5 * 1024 * 1024;  // 5 MiB transfer budget

    // Encode to PNG first; it is lossless and preferred when small enough.
    QByteArray png;
    {
        QBuffer buf(&png);
        buf.open(QIODevice::WriteOnly);
        img.save(&buf, "PNG");
        buf.close();
    }

    QByteArray bytes;
    QString fmt;
    if (png.size() <= kMaxBytes) {
        bytes = png;
        fmt = "png";
    } else {
        // Too large: re-encode as JPEG, dropping quality until it fits. JPEG
        // discards alpha, which is acceptable for a screenshot transfer.
        QByteArray fallback;
        for (int q = 95; q >= 1; q -= 5) {
            QByteArray jpg;
            QBuffer buf(&jpg);
            buf.open(QIODevice::WriteOnly);
            img.save(&buf, "JPEG", q);
            buf.close();
            fallback = jpg;
            if (jpg.size() <= kMaxBytes) {
                bytes = jpg;
                fmt = "jpeg";
                break;
            }
        }
        if (bytes.isEmpty()) {  // still over budget at quality 1: send the smallest we have
            bytes = fallback;
            fmt = "jpeg";
        }
    }

    QJsonObject obj;
    obj["format"] = fmt;
    obj["width"] = img.width();
    obj["height"] = img.height();
    obj["sample_rate"] = numSamples ? numSamples : QJsonValue();
    obj["image_base64"] = QString::fromLatin1(bytes.toBase64());
    sendResponse(requestId, true, compactJson(obj));
#endif
}

// The optix background on/off toggle backs the GUI's m_background checkbox
// (DockContent_View). State is stored in session userData as
// "optix_show_background" and consumed by the optix worker on each render.
void ApiMgr::handleGetOptixBackground(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
    const std::string key = "optix_show_background";
    auto &ud = zeno::getSession().userData();
    bool bShow = ud.has(key) ? ud.get2<bool>(key, false) : false;
    sendResponse(requestId, true, compactJson(QJsonObject{{"background", bShow}}));
}

void ApiMgr::handleSetOptixBackground(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
#ifndef ZENO_OPTIX_PROC
    const QString valStr = argAt(args, 0).trimmed();
    if (valStr.isEmpty()) {
        sendResponse(requestId, false, "setOptixBackground requires a boolean argument (true/false/1/0/on/off)");
        return;
    }
    bool bShow = false;
    if (!parseToggleText(valStr, bShow)) {
        sendResponse(requestId, false, "setOptixBackground must be a boolean (true/false/1/0/on/off), got: " + valStr);
        return;
    }

    // Route through the live optix view so the worker both records the flag in
    // userData and tells the render engine to show/hide the background, keeping
    // this consistent with the GUI's m_background toggle.
    ZOptixViewport* view = ensureOptixViewport();
    if (!view || !view->isOptixViewRunning()) {
        sendResponse(requestId, false, "no running optix viewport to apply the background");
        return;
    }
    view->showBackground(bShow);
    sendResponse(requestId, true, compactJson(QJsonObject{{"background", bShow}}));
#else
    sendResponse(requestId, false, "setOptixBackground is not supported in the proc optix build");
#endif
}

// Reports whether the ray-tracing (optix) render window is currently live. This
// is the API-level probe for "screenshot came back black": if running is false,
// screenShotOptix would likely produce an empty/black image because no render
// thread exists to draw the scene, so the caller should start one first.
void ApiMgr::handleGetOptixStatus(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
#ifndef ZENO_OPTIX_PROC
    ZOptixViewport* view = optixViewport();
    const bool bRunning = view && view->isOptixViewRunning();
#else
    // Under ZENO_OPTIX_PROC the ray-tracing window runs out-of-process, so the
    // in-process viewport is never present; report it as not running.
    const bool bRunning = false;
#endif
    sendResponse(requestId, true, compactJson(QJsonObject{{"running", bRunning}}));
}

// Actively opens a ray-tracing (optix) render window. Only one such window is
// ever created: if a live optix viewport already exists it is reused, otherwise
// the whole chain is created on demand. Returns the resulting running state.
void ApiMgr::handleOpenOptixViewport(const QString& requestId, const QJsonArray& args, IGraphsModel* pModel)
{
#ifndef ZENO_OPTIX_PROC
    ZOptixViewport* view = ensureOptixViewport();
    if (!view) {
        sendResponse(requestId, false, "failed to open an optix viewport");
        return;
    }
    const bool bRunning = view->isOptixViewRunning();
    sendResponse(requestId, true, compactJson(QJsonObject{{"running", bRunning}}));
#else
    sendResponse(requestId, false, "openOptixViewport is not supported in the proc optix build");
#endif
}
