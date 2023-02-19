
#include "unrealtcpserver.h"
#include "launch/unreal/model/transform.h"
#include <zeno/utils/log.h>
#include "msgpack.h"

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

void UnrealTcpServer::cleanUpSocket() {
    if (nullptr == m_currentSocket) return;
    if (m_currentSocket->isOpen()) m_currentSocket->close();

    m_currentSocket = nullptr;
    // TODO: darc clear buffers
}

void UnrealTcpServer::start(const QHostAddress& inAddress, int32_t inPort) {
    m_server = new QTcpServer(this);
    if (!m_server->listen(inAddress, inPort)) {
        zeno::log_error("failed to bind unreal bridge server at '{}:{}'", inAddress.toString().toStdString(), inPort);
    }

    connect(m_server, SIGNAL(newConnection()), this, SLOT(onNewConnection()));
}

void UnrealTcpServer::shutdown() {
    if (nullptr != m_server) {
        disconnect(m_server, SIGNAL(newConnection()), this, SLOT(onNewConnection()));
    }

    delete m_server;
    delete m_currentSocket;
}

#pragma region signal_handler
void UnrealTcpServer::onNewConnection() {
    if (!isRunning()) {
        shutdown();
        return;
    }

    cleanUpSocket();

    // keep socket of the connection
    m_currentSocket = m_server->nextPendingConnection();

    // create a client instance for socket
    UnrealLiveLinkClient* newClient = new UnrealLiveLinkTcpClient(this, m_currentSocket);
    m_clients.push_back(newClient);
    connect(newClient, SIGNAL(invalid(UnrealLiveLinkClient*)), this, SLOT(onClientInvalided(UnrealLiveLinkClient*)));
    newClient->init();

    m_currentSocket = nullptr;
}

void UnrealTcpServer::onClientInvalided(UnrealLiveLinkClient* who) {
    if (nullptr != who) who->cleanupSocket();
    m_clients.erase(std::remove_if(m_clients.begin(), m_clients.end(), [who](const auto v){
        return v == who;
    }), m_clients.end());
    who->deleteLater();
}

#pragma endregion signal_handler
