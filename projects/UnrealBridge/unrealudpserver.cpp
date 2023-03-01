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
        connect(m_socket, SIGNAL(readyRead()), this, SLOT(onNewMessage()));
    };

    if (nullptr != qThread) {
        moveToThread(qThread);
        connect(qThread, &QThread::started, this, startServer, Qt::QueuedConnection);
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
    // TODO: darc clean up message buffer to free memory
    std::vector<decltype(m_msg_buffer.begin())> elements_to_remove;

    for (auto i = m_msg_buffer.begin(); i < m_msg_buffer.end(); i++) {
        const QNetworkDatagram& data = *i;
        const QByteArray& body = data.data();
        if (body.size() < sizeof(ZBUFileMessageHeader)) {
            // skip if isn't struct we wanted
            continue;
        }
        auto type =  static_cast<ZBFileType>(qFromLittleEndian(*(uint32_t*)body.constData()));
        if (type > ZBFileType::End) {
            // bad type
            continue;
        }
        uint32_t bodySize = qFromLittleEndian(*(uint32_t*)(body.constData() + sizeof(ZBUFileMessageHeader::type)));
        uint32_t fileId = qFromLittleEndian(*(uint32_t*)(body.constData() + sizeof(ZBUFileMessageHeader::type) + sizeof(ZBUFileMessageHeader::size)));
        uint16_t totalPart = qFromLittleEndian(*(uint32_t*)(body.constData() + sizeof(ZBUFileMessageHeader::type) + sizeof(ZBUFileMessageHeader::size) + sizeof(ZBUFileMessageHeader::file_id)));
        uint16_t partId = qFromLittleEndian(*(uint32_t*)(body.constData() + sizeof(ZBUFileMessageHeader::type) + sizeof(ZBUFileMessageHeader::size) + sizeof(ZBUFileMessageHeader::file_id) + sizeof(ZBUFileMessageHeader::total_part)));

        ZBUFileMessageHeader header {
            type, bodySize, fileId, totalPart, partId,
        };
        std::vector<uint8_t> messageData;
        messageData.resize(body.size() - sizeof(ZBUFileMessageHeader));
        std::memmove(messageData.data(), body.constData() + sizeof(ZBUFileMessageHeader), body.size() - sizeof(ZBUFileMessageHeader));
        ZBUFileMessage message {
            header,
            std::move(messageData),
        };

        elements_to_remove.push_back(i);
    }

    for (const auto& i : elements_to_remove) {
        m_msg_buffer.erase(i);
    }

    if (m_msg_buffer.size() > 1024) {
        m_msg_buffer.erase(m_msg_buffer.begin(), m_msg_buffer.begin() + 512);
    }

    // TODO: darc move send height field to unreal to a better place
    if (UnrealSubjectRegistry::getStatic().isDirty()) {
        for (auto& session : UnrealSessionRegistry::getStatic().all()) {
            for (auto& item : UnrealSubjectRegistry::getStatic().height_fields()) {
                std::vector<QNetworkDatagram> datagrams = zeno::makeSendFileDatagrams(const_cast<UnrealHeightFieldSubject&>(item), ZBFileType::HeightField);
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

    m_msg_buffer.push_back(std::move(datagram));
}

void UnrealUdpServer::sendDatagram(const QNetworkDatagram &datagram) {
    if (m_socket) {
        m_socket->writeDatagram(datagram);
    }
}

void zeno::startUnrealUdpServer(const QHostAddress &inAddress, int32_t inPort) {
    static auto* pThread = new QThread;
    UnrealUdpServer::getStaticClass().start(pThread, inAddress, inPort);
    pThread->start();
}
