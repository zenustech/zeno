#include "unrealclient.h"
#include "launch/unreal/model/transform.h"
#include "msgpack.h"

UnrealLiveLinkTcpClient::UnrealLiveLinkTcpClient(QObject* parent, QTcpSocket* inTcpSocket)
    : IUnrealLiveLinkClient(parent),
      m_socket(inTcpSocket)
{
}

UnrealLiveLinkTcpClient::~UnrealLiveLinkTcpClient() = default;

void UnrealLiveLinkTcpClient::init() {
    connect(m_socket, SIGNAL(disconnected()), this, SLOT(onSocketClosed()));
    connect(m_socket, SIGNAL(readyRead()), this, SLOT(onSocketReceiveData()));
    connect(m_socket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(onError(QAbstractSocket::SocketError)));
}

void UnrealLiveLinkTcpClient::cleanupSocket() {
    disconnect(m_socket, SIGNAL(disconnected()), this, SLOT(onSocketClosed()));
    disconnect(m_socket, SIGNAL(readyRead()), this, SLOT(onSocketReceiveData()));
    disconnect(m_socket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(onError(QAbstractSocket::SocketError)));
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

    // TODO: darc read data to buffer and then handle it in new thread
    QByteArray byteArray = m_socket->readAll();
    qint64 size = byteArray.size();
    Translation translation { 1.4131, M_PI, 1.f };
    auto data = msgpack::pack(translation);
    m_socket->write(reinterpret_cast<const char *>(data.data()), data.size());
}

void UnrealLiveLinkTcpClient::onError(QAbstractSocket::SocketError error) {
    using E = QAbstractSocket::SocketError;

    Q_UNUSED(error);
}
#pragma endregion tcp_socket_events
