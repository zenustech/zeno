#ifndef ZENO_UNREALUDPSERVER_H
#define ZENO_UNREALUDPSERVER_H

#include <QHostAddress>
#include <QNetworkDatagram>
#include <QObject>
#include <QtEndian>
#include <atomic>
#include "include/msgpack.h"
#include "model/networktypes.h"

class QUdpSocket;

class UnrealUdpServer : public QObject {
    Q_OBJECT

public:
    /**
     * Get static instance of UnrealUdpServer.
     * Thread safe.
     * @return instance
     */
    static UnrealUdpServer& getStaticClass();

    explicit UnrealUdpServer(QObject* parent = nullptr);
    ~UnrealUdpServer() override;

    /**
     * Start the server
     * @param qThread thread that this server running at. Pass nullptr to run on caller event loop.
     * @param inAddress clients from which internet interface are allowed
     * @param inPort binding port, default 23343
     */
    void start(QThread* qThread = nullptr, const QHostAddress& inAddress = QHostAddress::LocalHost, int32_t inPort = 23343);

    /**
     * shutdown and release server instance
     */
    void shutdown();

    /**
     * Check does server valid
     * @return status
     */
    bool isRunning();

    void timerEvent(QTimerEvent *event) override;

    void sendDatagram(const QNetworkDatagram& datagram);

    void tryMakeupFile(uint32_t fileId);

signals:
    void newFile(ZBFileType, std::vector<uint8_t>);

private slots:
    void onNewMessage();
    void onNewFile(ZBFileType fileType, std::vector<uint8_t> data);

private:
    QUdpSocket* m_socket;

    std::vector<ZBUFileMessage> m_msg_buffer;
};

namespace zeno {
    void startUnrealUdpServer(const QHostAddress& inAddress = QHostAddress::LocalHost, int32_t inPort = 23343);

    // actual packet size = g_UdpPacketCutSize + sizeof(ZBUMessageHeader)
    constexpr size_t g_UdpPacketCutSize = 256; // 264

    template <typename T>
    std::vector<QNetworkDatagram> makeSendFileDatagrams(T& dataToSend, const ZBFileType fileType ) {
        const static uint8_t timestamp = time(nullptr);
        static std::atomic<uint32_t> currentFileId {0};

        std::vector<QNetworkDatagram> datagrams;

        auto data = msgpack::pack(dataToSend);
        uint16_t dataPartSize = std::ceil(data.size() / static_cast<double>(g_UdpPacketCutSize));
        uint32_t fileId = currentFileId++;
        fileId <<= 8;
        fileId += timestamp;
        for (uint16_t i = 0; i < dataPartSize; i++) {
            uint32_t size = g_UdpPacketCutSize;
            if ((i+1) * g_UdpPacketCutSize > data.size()) {
                size = data.size() - (g_UdpPacketCutSize * i);
            }

            ZBUFileMessageHeader header {
                qToLittleEndian(fileType),
                qToLittleEndian(size),
                qToLittleEndian(fileId),
                qToLittleEndian(dataPartSize),
                qToLittleEndian(i),
            };

            QByteArray tmpArr;
            tmpArr.resize(static_cast<int32_t>(sizeof(ZBUFileMessageHeader) + size));
            std::memmove(tmpArr.data(), &header, sizeof(ZBUFileMessageHeader));
            std::memmove(tmpArr.data() + sizeof(ZBUFileMessageHeader), data.data() + (i * g_UdpPacketCutSize), size);
            QNetworkDatagram datagram;
            datagram.setData(tmpArr);
            datagrams.push_back(std::move(datagram));
        }

        return datagrams;
    }
}

#endif //ZENO_UNREALUDPSERVER_H
