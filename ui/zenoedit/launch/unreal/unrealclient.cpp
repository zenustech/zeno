#include "unrealclient.h"
#include <atomic>

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

bool UnrealLiveLinkTcpClient::sendPacket(const ZBTControlPacketType packetType, uint8_t* data, const uint16_t size) {
    static std::atomic<uint16_t> currentPacketIndex = 0;

    ZBTPacketHeader header {
        currentPacketIndex.fetch_add(1),
        size,
        packetType,
    };

    return QtSocketHelper::writeData(m_socket, header, data);
}

void UnrealLiveLinkTcpClient::timerEvent(QTimerEvent *event) {
    const int16_t packetSize = m_buffer.getNextPacketSize();
    if (packetSize != -1) {
        auto* tmp = static_cast<uint8_t *>(malloc(packetSize));
        bool bIsSuccess = m_buffer.readSinglePacket(tmp);
        auto header = reinterpret_cast<const ZBTPacketHeader*>(tmp);

        bool bHasRespond = false;
        ZBTControlPacketType respondPacketType = ZBTControlPacketType::Start;
        OutPacketBufferType respondData;
        uint16_t respondSize = 0;

        PacketHandlerMap::get().tryCall(header->type, tmp, bHasRespond, respondPacketType, respondData, respondSize);

        if (bHasRespond && respondData.has_value()) {
            sendPacket(respondPacketType, respondData->data(), respondSize);
        }

    }

    m_buffer.cleanUp();
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

    QtSocketHelper::readToByteBuffer(m_socket, m_buffer);
}

void UnrealLiveLinkTcpClient::onError(QAbstractSocket::SocketError error) {
    using E = QAbstractSocket::SocketError;

    Q_UNUSED(error);
    emit invalid(this);
}
#pragma endregion tcp_socket_events

/**
 * Copy handlers to here and only here.
 * We need to run static initializer to map handler at runtime.
 */
#include "handler/unrealhandler.h"
