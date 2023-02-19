#include "unrealclient.h"

UnrealLiveLinkTcpClient::UnrealLiveLinkTcpClient(QObject* parent, QTcpSocket* inTcpSocket)
    : UnrealLiveLinkClient(parent),
      m_socket(inTcpSocket)
{
}

UnrealLiveLinkTcpClient::~UnrealLiveLinkTcpClient() = default;

void UnrealLiveLinkTcpClient::init() {
    connect(m_socket, SIGNAL(disconnected()), this, SLOT(onSocketClosed()));
    connect(m_socket, SIGNAL(readyRead()), this, SLOT(onSocketReceiveData()));
}

void UnrealLiveLinkTcpClient::cleanupSocket() {
    disconnect(m_socket, SIGNAL(disconnected()), this, SLOT(onSocketClosed()));
    disconnect(m_socket, SIGNAL(readyRead()), this, SLOT(onSocketReceiveData()));
    m_socket->deleteLater();
}

#pragma region tcp_socket_events
void UnrealLiveLinkTcpClient::onSocketClosed() {
    emit invalid(this);
}

void UnrealLiveLinkTcpClient::onSocketReceiveData() {
    if (nullptr == m_socket || !m_socket->isReadable()) {
        emit invalid(this);
        return;
    }

    QByteArray byteArray = m_socket->readAll();
    qint64 size = byteArray.size();
    m_socket->write(byteArray);
}
#pragma endregion tcp_socket_events
