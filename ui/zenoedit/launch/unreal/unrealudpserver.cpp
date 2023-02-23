#include "unrealudpserver.h"
#include <zeno/utils/log.h>
#include <QUdpSocket>
#include <QThread>
#include <QNetworkDatagram>

UnrealUdpServer &UnrealUdpServer::getStaticClass() {
    static UnrealUdpServer sUnrealUdpServer;

    return sUnrealUdpServer;
}

UnrealUdpServer::UnrealUdpServer(QObject* parent) : QObject(parent), m_socket(nullptr) {
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

void UnrealUdpServer::onNewMessage() {
    QNetworkDatagram datagram = m_socket->receiveDatagram();
    QByteArray array { };
    array.push_back('1');
    m_socket->writeDatagram(datagram.makeReply(array));
}

void zeno::startUnrealUdpServer(const QHostAddress &inAddress, int32_t inPort) {
    static auto* pThread = new QThread;
    UnrealUdpServer::getStaticClass().start(pThread, inAddress, inPort);
    pThread->start();
}
