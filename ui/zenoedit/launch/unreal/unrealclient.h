#ifndef ZENO_UNREALCLIENT_H
#define ZENO_UNREALCLIENT_H

#include "model/bytebuffer.h"
#include "model/packethandler.h"
#include "model/networktypes.h"
#include <QObject>
#include <QAbstractSocket>
#include <QTcpSocket>
#include <QTimer>

class IUnrealLiveLinkClient : public QObject {
    Q_OBJECT

public:
    using ByteBuffer = ByteBuffer<2048>;

    explicit IUnrealLiveLinkClient(QObject* parent) : QObject(parent) {
        startTimer(1);
    }
    ~IUnrealLiveLinkClient() override = default;

public:
    /** binding socket events */
    virtual void init() = 0;
    /** close and destroy socket */
    virtual void cleanupSocket() = 0;

    /** check client authority */
    virtual bool isAuthority() { return true; }

    /**
     * Send packet to client by socket
     * @param packetType packet type
     * @param data data to size
     * @param size data size in bytes
     * @return true if success
     */
    virtual bool sendPacket(ZBTControlPacketType packetType, uint8_t* data, uint16_t size) = 0;

    void timerEvent(QTimerEvent *event) override = 0;

signals:
    void invalid(IUnrealLiveLinkClient * who);
};

class UnrealLiveLinkTcpClient : public IUnrealLiveLinkClient {
    Q_OBJECT

public:
    void init() override;
    void cleanupSocket() override;
    bool sendPacket(ZBTControlPacketType packetType, uint8_t *data, uint16_t size) override;

    explicit UnrealLiveLinkTcpClient(QObject* parent, QTcpSocket* inTcpSocket);
    ~UnrealLiveLinkTcpClient() override;

    void timerEvent(QTimerEvent *event) override;

private:

private slots:
    void onSocketClosed();
    void onSocketReceiveData();
    void onError(QAbstractSocket::SocketError error);

private:
    QTcpSocket* m_socket;
    ByteBuffer m_buffer;

};

// TODO: darc support LocalSocket for unix-like OS

struct QtSocketHelper {
    /**
     * Write wrapped data to target socket
     * @param target input socket
     * @param header packet header
     * @param data data to send
     */
    static bool writeData(QAbstractSocket* target, const ZBTPacketHeader& header, const uint8_t* const data) {
        if (nullptr == target || (nullptr == data && header.length != 0) || !target->isOpen() || !target->isWritable()) {
            return false;
        }

        target->write(reinterpret_cast<const char *>(&header), sizeof(ZBTPacketHeader));
        if (header.length != 0) {
            target->write(reinterpret_cast<const char *>(data), header.length);
        }
        target->write(reinterpret_cast<const char *>(g_packetSplit.data()), g_packetSplit.size());

        return true;
    }

    template <size_t Size>
    static bool readToByteBuffer(QAbstractSocket* socket, ByteBuffer<Size>& buffer) {
        if (nullptr != socket) {
            const int64_t socketBufSize = socket->bytesAvailable();
            if (buffer.moveCursor(socketBufSize)) {
                const int64_t bytesRead = socket->read(reinterpret_cast<char*>(*buffer), socketBufSize);
                assert(bytesRead == socketBufSize);
            }
        }

        return false;
    }
};

#endif //ZENO_UNREALCLIENT_H
