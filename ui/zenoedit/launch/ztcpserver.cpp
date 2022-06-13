#ifdef ZENO_MULTIPROCESS
#include "ztcpserver.h"
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/log.h>
#include <QMessageBox>
#include <zeno/zeno.h>
#include "launch/viewdecode.h"
#include "util/log.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"


ZTcpServer::ZTcpServer(QObject *parent)
    : QObject(parent)
    , m_tcpServer(nullptr)
    , m_tcpSocket(nullptr)
{
}

void ZTcpServer::init(const QHostAddress& address, quint16 port)
{
    m_tcpServer = new QTcpServer(this);
    if (!m_tcpServer->listen(QHostAddress::LocalHost, port))
    {
        zeno::log_error("tcp socket listen failure");
        return;
    }
    else
    {
        zeno::log_info("tcp server listend");
    }
    connect(m_tcpServer, SIGNAL(newConnection()), this, SLOT(onNewConnection()));
}

void ZTcpServer::startProc(const std::string& progJson)
{
    ZASSERT_EXIT(m_tcpServer);
    if (m_proc && m_proc->isOpen())
    {
        zeno::log_warn("A program is already running! Please kill first");
        return;
    }

    zeno::log_info("launching program...");
    zeno::log_debug("program JSON: {}", progJson);

    m_proc = std::make_unique<QProcess>();
    m_proc->setInputChannelMode(QProcess::InputChannelMode::ManagedInputChannel);
    m_proc->setReadChannel(QProcess::ProcessChannel::StandardOutput);
    m_proc->setProcessChannelMode(QProcess::ProcessChannelMode::ForwardedErrorChannel);
    int sessionid = zeno::getSession().globalState->sessionid;
    m_proc->start(QCoreApplication::applicationFilePath(), QStringList({"-runner", QString::number(sessionid)}));
    if (!m_proc->waitForStarted(-1)) {
        zeno::log_warn("process failed to get started, giving up");
        return;
    }

    m_proc->write(progJson.data(), progJson.size());
    m_proc->closeWriteChannel();

    connect(m_proc.get(), SIGNAL(readyRead()), this, SLOT(onProcPipeReady()));
}

void ZTcpServer::killProc()
{
    if (m_proc) {
        m_proc->terminate();
        m_proc = nullptr;
    }
}

void ZTcpServer::onNewConnection()
{
    ZASSERT_EXIT(m_tcpServer);
    m_tcpSocket = m_tcpServer->nextPendingConnection();
    if (!m_tcpSocket)
    {
        zeno::log_error("tcp connection recv failed");
    }
    else
    {
        zeno::log_info("tcp connection succeed");

    }
    connect(m_tcpSocket, SIGNAL(readyRead()), this, SLOT(onReadyRead()));
    connect(m_tcpSocket, SIGNAL(disconnected()), this, SLOT(onDisconnect()));
    viewDecodeClear();
}

void ZTcpServer::onReadyRead()
{
    QByteArray arr = m_tcpSocket->readAll();
    qint64 redSize = arr.size();
    zeno::log_debug("qtcpsocket got {} bytes (ping test has 19)", redSize);
    if (redSize > 0) {
        viewDecodeAppend(arr.data(), redSize);
    }
}

void ZTcpServer::onProcPipeReady()
{
    if (!m_proc) {
        return;
    }
    QByteArray arr = m_proc->readAll();
    zenoApp->graphsManagment()->appendMsgStream(arr);
}

void ZTcpServer::onDisconnect()
{
    if (m_proc)
    {
        viewDecodeFinish();
        m_proc->terminate();
        int code = m_proc->exitCode();
        m_proc = nullptr;
        zeno::log_info("runner process exited with {}", code);
    }
}
#endif