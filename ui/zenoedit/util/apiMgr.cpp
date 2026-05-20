#include "apiMgr.h"
#include "zenoapplication.h"
#include <QTextStream>
#include <iostream>
#include <QRegularExpression>
#include <zeno/utils/log.h>
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>

#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/modelrole.h>
#include <zenomodel/include/nodesmgr.h>
#include <zenomodel/include/nodeparammodel.h>
#include <zenomodel/include/uihelper.h>
#include <zenomodel/include/enum.h>
#include <zenomodel/include/api.h>

#ifdef _WIN32
#include <io.h>
#define STDIN_FILENO _fileno(stdin)
#else
#include <unistd.h>
#endif

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

    m_running = false;

    if (m_pThread && m_pThread->joinable()) {
        m_pThread->join();
        delete m_pThread;
        m_pThread = nullptr;
    }

    m_bRunning = false;
}

void ApiMgr::sendResponse(const QString& response)
{
    QTextStream out(stdout);
    out << response << endl;
    out.flush();
}

void ApiMgr::inputListenerThread()
{
    zeno::log_critical("API listener thread started");

    while (m_running) {
        std::string line;
        if (std::getline(std::cin, line)) {
            if (!line.empty()) {
                QString command = QString::fromStdString(line);
                zeno::log_critical("API thread received: " + line);

                emit commandReceived(command);
            }
        } else {
            if (std::cin.eof()) {
                zeno::log_critical("Input stream ended, thread exiting");
                break;
            } else if (std::cin.fail()) {
                zeno::log_critical("Input stream error, thread exiting");
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    zeno::log_critical("API listener thread stopped");
}

void ApiMgr::handleCommand(const QString& command)
{
    IGraphsModel* pModel = GraphsManagment::instance().currentModel();
    if (!pModel) {
        std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
        ZASSERT_EXIT(mgr);
        mgr->setNewCacheDir(true);

        zenoApp->graphsManagment()->newFile();
        pModel = GraphsManagment::instance().currentModel();

        QModelIndex subgIdx = pModel->index(0, 0);
        const QString& subgName = subgIdx.data(ROLE_OBJNAME).toString();
        sendResponse("success: created graph " + subgName);
    }

    QString cmd = command.trimmed();

    QRegularExpression reCreateNode("^createNode\\(\"([^\"]+)\",\\s*\"([^\"]+)\"\\)$");
    QRegularExpressionMatch matchCreateNode = reCreateNode.match(cmd);
    if (matchCreateNode.hasMatch()) {
        //ZENO_HANDLE hGraph = Zeno_GetItem(0);
        QString graphName = matchCreateNode.captured(1);
        QString nodeCls = matchCreateNode.captured(2);
        ZENO_HANDLE hGraph = Zeno_GetGraph(graphName.toStdString());
        if (hGraph < 0) {
            sendResponse("error: graph not found: " + graphName);
            return;
        }
        ZENO_HANDLE hNode = Zeno_AddNode(hGraph, nodeCls.toStdString());
        if (hNode >= 0) {
            std::string uuid;
            Zeno_GetNodeUuid(hGraph, hNode, uuid);
            sendResponse("success: created node " + nodeCls + " in graph " + graphName + " with node_uuid " + QString::fromStdString(uuid));
        } else {
            sendResponse("error: failed to create node " + nodeCls);
        }
        return;
    }

    QRegularExpression reDeleteNode("^deleteNode\\(\"([^\"]+)\",\\s*\"([^\"]+)\"\\)$");
    QRegularExpressionMatch matchDeleteNode = reDeleteNode.match(cmd);
    if (matchDeleteNode.hasMatch()) {
        QString graphName = matchDeleteNode.captured(1);
        QString nodeUuid = matchDeleteNode.captured(2);
        ZENO_HANDLE hSubg = Zeno_GetGraph(graphName.toStdString());
        if (hSubg < 0) {
            sendResponse("error: graph not found: " + graphName);
            return;
        }
        auto nodeHandleOpt = Zeno_GetNode(hSubg, nodeUuid.toStdString());
        if (!nodeHandleOpt.has_value()) {
            sendResponse("error: node not found: " + nodeUuid);
            return;
        }
        ZENO_HANDLE hNode = nodeHandleOpt.value();
        ZENO_ERROR err = Zeno_DeleteNode(hSubg, hNode);
        if (err == Err_NoError) {
            sendResponse("success: deleted node " + nodeUuid);
        } else {
            sendResponse("error: failed to delete node, error code " + QString::number(err));
        }
        return;
    }

    QRegularExpression reAddLink("^addLink\\(\"([^\"]+)\",\\s*\"([^\"]+)\",\\s*\"([^\"]+)\",\\s*\"([^\"]+)\",\\s*\"([^\"]+)\"\\)$");
    QRegularExpressionMatch matchAddLink = reAddLink.match(cmd);
    if (matchAddLink.hasMatch()) {
        QString graphName = matchAddLink.captured(1);
        QString outNodeUuid = matchAddLink.captured(2);
        QString outSock = matchAddLink.captured(3);
        QString inNodeUuid = matchAddLink.captured(4);
        QString inSock = matchAddLink.captured(5);

        ZENO_HANDLE hSubg = Zeno_GetGraph(graphName.toStdString());
        if (hSubg < 0) {
            sendResponse("error: graph not found: " + graphName);
            return;
        }

        auto outNodeHandleOpt = Zeno_GetNode(hSubg, outNodeUuid.toStdString());
        if (!outNodeHandleOpt.has_value()) {
            sendResponse("error: output node not found: " + outNodeUuid);
            return;
        }
        ZENO_HANDLE hOutNode = outNodeHandleOpt.value();

        auto inNodeHandleOpt = Zeno_GetNode(hSubg, inNodeUuid.toStdString());
        if (!inNodeHandleOpt.has_value()) {
            sendResponse("error: input node not found: " + inNodeUuid);
            return;
        }
        ZENO_HANDLE hInNode = inNodeHandleOpt.value();

        ZENO_ERROR err = Zeno_AddLink(hSubg, hOutNode, outSock.toStdString(), hInNode, inSock.toStdString());
        if (err == Err_NoError) {
            sendResponse("success: added link from " + outNodeUuid + ":" + outSock + " to " + inNodeUuid + ":" + inSock);
        } else {
            sendResponse("error: failed to add link, error code " + QString::number(err));
        }
        return;
    }

    QRegularExpression reRemoveLink("^removeLink\\(\"([^\"]+)\",\\s*\"([^\"]+)\",\\s*\"([^\"]+)\",\\s*\"([^\"]+)\",\\s*\"([^\"]+)\"\\)$");
    QRegularExpressionMatch matchRemoveLink = reRemoveLink.match(cmd);
    if (matchRemoveLink.hasMatch()) {
        QString graphName = matchRemoveLink.captured(1);
        QString outNodeUuid = matchRemoveLink.captured(2);
        QString outSock = matchRemoveLink.captured(3);
        QString inNodeUuid = matchRemoveLink.captured(4);
        QString inSock = matchRemoveLink.captured(5);

        ZENO_HANDLE hSubg = Zeno_GetGraph(graphName.toStdString());
        if (hSubg < 0) {
            sendResponse("error: graph not found: " + graphName);
            return;
        }

        auto outNodeHandleOpt = Zeno_GetNode(hSubg, outNodeUuid.toStdString());
        if (!outNodeHandleOpt.has_value()) {
            sendResponse("error: output node not found: " + outNodeUuid);
            return;
        }
        ZENO_HANDLE hOutNode = outNodeHandleOpt.value();

        auto inNodeHandleOpt = Zeno_GetNode(hSubg, inNodeUuid.toStdString());
        if (!inNodeHandleOpt.has_value()) {
            sendResponse("error: input node not found: " + inNodeUuid);
            return;
        }
        ZENO_HANDLE hInNode = inNodeHandleOpt.value();

        ZENO_ERROR err = Zeno_RemoveLink(hSubg, hOutNode, outSock.toStdString(), hInNode, inSock.toStdString());
        if (err == Err_NoError) {
            sendResponse("success: removed link from " + outNodeUuid + ":" + outSock + " to " + inNodeUuid + ":" + inSock);
        } else {
            sendResponse("error: failed to remove link, error code " + QString::number(err));
        }
        return;
    }

    sendResponse("error: unknown command");
}