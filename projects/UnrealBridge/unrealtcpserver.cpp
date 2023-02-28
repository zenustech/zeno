
#include "unrealclient.h"
#include "unrealtcpserver.h"
#include <zeno/utils/log.h>
#include <QThread>
#include <QTcpServer>

UnrealTcpServer::UnrealTcpServer(QObject* parent)
    : QObject(parent) {
}

UnrealTcpServer::~UnrealTcpServer() {
    shutdown();
}

bool UnrealTcpServer::isRunning() {
    return nullptr != m_server && m_server->isListening();
}

UnrealTcpServer& UnrealTcpServer::getStaticClass() {
    static UnrealTcpServer sUnrealTcpServer;

    return sUnrealTcpServer;
}

void UnrealTcpServer::start(QThread* qThread, const QHostAddress& inAddress, int32_t inPort) {
    if (isRunning()) return;

    auto startServer = [inAddress, inPort, this] {
        m_server = new QTcpServer(this);
        if (!m_server->listen(inAddress, inPort)) {
            zeno::log_error("failed to bind unreal bridge server at '{}:{}'", inAddress.toString().toStdString(), inPort);
        }

        connect(m_server, SIGNAL(newConnection()), this, SLOT(onNewConnection()));
    };

    if (nullptr != qThread) {
        moveToThread(qThread);
        m_currentThread = qThread;
        connect(m_currentThread, &QThread::started, this, startServer, Qt::QueuedConnection);
    } else {
        startServer();
    }

}

void UnrealTcpServer::shutdown() {
    if (nullptr != m_currentThread) {
        disconnect(m_currentThread, SIGNAL(started(QPrivateSignal)));
        m_currentSocket = nullptr;
    }
    if (nullptr != m_server) {
        disconnect(m_server, SIGNAL(newConnection()), this, SLOT(onNewConnection()));
        m_server->deleteLater();
        m_server = nullptr;
    }
}

#pragma region signal_handler
void UnrealTcpServer::onNewConnection() {
    if (!isRunning()) {
        shutdown();
        return;
    }

    // keep socket of the connection
    m_currentSocket = m_server->nextPendingConnection();

    // create a client instance for socket
    IUnrealLiveLinkClient * newClient = new UnrealLiveLinkTcpClient(this, m_currentSocket);
    m_clients.push_back(newClient);
    connect(newClient, SIGNAL(invalid(IUnrealLiveLinkClient*)), this, SLOT(onClientInvalided(IUnrealLiveLinkClient*)));
    newClient->init();

    m_currentSocket = nullptr;
}

void UnrealTcpServer::onClientInvalided(IUnrealLiveLinkClient * who) {
    if (nullptr != who) who->cleanupSocket();
    m_clients.erase(std::remove_if(m_clients.begin(), m_clients.end(), [who](const auto v){
        return v == who;
    }), m_clients.end());
    who->deleteLater();
}

#pragma endregion signal_handler

void zeno::startUnrealTcpServer(const QHostAddress &inAddress, int32_t inPort) {
    static auto* pThread = new QThread;
    UnrealTcpServer::getStaticClass().start(pThread, inAddress, inPort);
    pThread->start(QThread::HighPriority);
}
