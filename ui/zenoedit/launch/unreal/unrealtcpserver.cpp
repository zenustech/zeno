
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

    if (nullptr != m_currentSocket) {
        connect(m_currentSocket, SIGNAL(disconnected()), this, SLOT(onCurrentConnectionClosed()));
        connect(m_currentSocket, SIGNAL(readyRead()), this, SLOT(onCurrentConnectionReceiveData()));
    }
}

void UnrealTcpServer::onCurrentConnectionClosed() {
    const QTcpSocket* tmp = m_currentSocket;
    cleanUpSocket();
    disconnect(tmp, SIGNAL(disconnected()), this, SLOT(onCurrentConnectionClosed()));
    disconnect(tmp, SIGNAL(readyRead()), this, SLOT(onCurrentConnectionReceiveData()));
}

void UnrealTcpServer::onCurrentConnectionReceiveData() {
    if (nullptr == m_currentSocket || !m_currentSocket->isReadable()) {
        return;
    }

    QByteArray byteArray = m_currentSocket->readAll();
    qint64 size = byteArray.size();

    TestModel model { 1 };
    auto data = msgpack::pack(model);
    m_currentSocket->write(reinterpret_cast<const char *>(data.data()), data.size());
}
#pragma endregion signal_handler
