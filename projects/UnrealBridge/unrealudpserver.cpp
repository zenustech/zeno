#include "unrealudpserver.h"
#include "model/networktypes.h"
#include "unrealregistry.h"
#include <QThread>
#include <QUdpSocket>
#include <QtEndian>
#include <zeno/utils/log.h>

UnrealUdpServer &UnrealUdpServer::getStaticClass() {
    static UnrealUdpServer sUnrealUdpServer;

    return sUnrealUdpServer;
}

UnrealUdpServer::UnrealUdpServer(QObject* parent) : QObject(parent), m_socket(nullptr) {
    startTimer(5);
}

UnrealUdpServer::~UnrealUdpServer() = default;

void UnrealUdpServer::start(QThread *qThread, const QHostAddress& inAddress, int32_t inPort) {
    // do nothing if server already started
    if (isRunning()) return;

    auto startServer = [this, inAddress, inPort] {
        m_socket = new QUdpSocket(this);
        if (!m_socket->bind(inAddress, inPort)) {
            zeno::log_error("failed to bind unreal udp server at '{}:{}'", inAddress.toString().toStdString(), inPort);
        }
        m_socket->setSocketOption(QAbstractSocket::ReceiveBufferSizeSocketOption, 409600);
        connect(m_socket, SIGNAL(readyRead()), this, SLOT(onNewMessage()));
        connect(m_socket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(onError(QAbstractSocket::SocketError)));
        connect(this, SIGNAL(newFile(ZBFileType,std::vector<uint8_t>)), this, SLOT(onNewFile(ZBFileType,std::vector<uint8_t>)));
    };

    if (nullptr != qThread) {
        moveToThread(qThread);
        connect(qThread, &QThread::started, this, startServer, Qt::QueuedConnection);
        connect(this, SIGNAL(newFile(ZBFileType,std::vector<uint8_t>)), this, SLOT(onNewFile(ZBFileType,std::vector<uint8_t>)));
    } else {
        startServer();
    }
}

void UnrealUdpServer::shutdown() {
    if (nullptr != m_socket) {
        disconnect(m_socket, SIGNAL(readyRead()), this, SLOT(onNewMessage()));
        m_socket->deleteLater();
        m_socket = nullptr;
    }
}

bool UnrealUdpServer::isRunning() {
    return m_socket != nullptr && m_socket->isOpen();
}

void UnrealUdpServer::timerEvent(QTimerEvent* event) {
    // TODO: darc move send height field to unreal to a better place
    if (UnrealSubjectRegistry::getStatic().isDirty()) {
        for (auto& session : UnrealSessionRegistry::getStatic().all()) {
            for (auto& item : UnrealSubjectRegistry::getStatic().height_fields()) {
                std::vector<QNetworkDatagram> datagrams = zeno::makeSendFileDatagrams(const_cast<UnrealHeightFieldSubject&>(item.second), ZBFileType::HeightField);
                for (auto& datagram : datagrams) {
                    datagram.setDestination(QHostAddress{ QString::fromStdString(session.udp_address.value()) }, session.udp_port.value());
                    m_socket->writeDatagram(datagram);
                }
            }
        }
        UnrealSubjectRegistry::getStatic().markDirty(false);
    }
}

void UnrealUdpServer::onNewMessage() {
    // TODO: darc drop unauthorized message
    QNetworkDatagram datagram = m_socket->receiveDatagram();

    QByteArray byteArray = datagram.data();
    unsigned char* rawData = reinterpret_cast<unsigned char*>(byteArray.data());
    const uint16_t dataSize = datagram.data().size();

    if (dataSize < sizeof(ZBUFileMessageHeader)) {
        return;
    }

    ZBFileType fileType = qFromLittleEndian(*reinterpret_cast<ZBFileType*>(rawData));
    if (fileType > ZBFileType::End) {
        return;
    }

    auto* messageHeader = reinterpret_cast<ZBUFileMessageHeader*>(rawData);
    messageHeader->type = qFromLittleEndian(messageHeader->type);
    messageHeader->total_part = qFromLittleEndian(messageHeader->total_part);
    messageHeader->size = qFromLittleEndian(messageHeader->size);
    messageHeader->part_id = qFromLittleEndian(messageHeader->part_id);
    messageHeader->file_id = qFromLittleEndian(messageHeader->file_id);

    const uint32_t fileId = messageHeader->file_id;
    std::vector<uint8_t> messageData;
    messageData.resize(dataSize - sizeof(ZBUFileMessageHeader));
    std::memmove(messageData.data(), rawData + sizeof(ZBUFileMessageHeader), dataSize - sizeof(ZBUFileMessageHeader));
    ZBUFileMessage message {
        *messageHeader,
        std::move(messageData),
    };

    m_lock.lock();
    m_msg_buffer.push_back(std::move(message));
    m_lock.unlock();
    tryMakeupFile(fileId);
}

void UnrealUdpServer::sendDatagram(const QNetworkDatagram &datagram) {
    if (m_socket) {
        m_socket->writeDatagram(datagram);
    }
}

void UnrealUdpServer::tryMakeupFile(const uint32_t fileId) {
    std::set<uint16_t> partSet;
    int32_t fileParts = -1;
    ZBFileType fileType = ZBFileType::End;
    std::unordered_map<uint16_t , size_t> partIdx;
    uint64_t totalSize = 0;
    for (size_t i = 0; i < m_msg_buffer.size(); ++i) {
        const ZBUFileMessage& item1 = m_msg_buffer[i];
        if (item1.header.file_id == fileId) {
            fileParts = std::max(static_cast<int32_t>(item1.header.total_part), fileParts);
            partSet.insert(item1.header.part_id);
            fileType = item1.header.type;
            partIdx.insert_or_assign(item1.header.part_id, i);
            totalSize += item1.data.size();
        }
    }

    if (-1 != fileParts && partSet.size() == fileParts) {
        std::vector<uint8_t> data;
        data.resize(totalSize);
        uint64_t offset = 0;

        for (int32_t i = 0; i < fileParts; ++i) {
            std::vector<uint8_t>& rawData = m_msg_buffer[partIdx[i]].data;
            std::memmove(data.data() + offset, rawData.data(), rawData.size());
            offset += rawData.size();
        }

        for (auto [key, value] : partIdx) {
            if (value < m_msg_buffer.size()) {
                m_msg_buffer.erase(m_msg_buffer.begin() + value);
            }
        }

        emit newFile(fileType, data);
    }
}

void UnrealUdpServer::onNewFile(ZBFileType fileType, const std::vector<uint8_t>& data) {
    if (fileType == ZBFileType::HeightField) {
        try {
            const auto subject = msgpack::unpack<UnrealHeightFieldSubject>(data);

            std::shared_ptr<zeno::UnrealZenoHeightFieldSubject> heightFieldSubject = std::make_shared<zeno::UnrealZenoHeightFieldSubject>();
            heightFieldSubject->heights.resize(data.size());
            std::memmove(heightFieldSubject->heights.data(), data.data(), data.size() * sizeof(float));

            ZenoSubjectRegistry::getStatic().put(subject.m_name, heightFieldSubject);
        } catch (msgpack::UnpackerError) {
        }
    }
}
void UnrealUdpServer::onError(QAbstractSocket::SocketError error) {
    zeno::log_error("Error %s", qt_getEnumName(error));
}

void zeno::startUnrealUdpServer(const QHostAddress &inAddress, int32_t inPort) {
    static auto* pThread = new QThread;
    UnrealUdpServer::getStaticClass().start(pThread, inAddress, inPort);
    pThread->start();
}
