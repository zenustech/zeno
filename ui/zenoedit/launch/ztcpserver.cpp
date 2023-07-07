#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
#include "ztcpserver.h"
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/utils/log.h>
#include <QMessageBox>
#include <zeno/zeno.h>
#include "launch/viewdecode.h"
#include "util/log.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include "settings/zsettings.h"
#include "zenoapplication.h"
#include "cache/zcachemgr.h"
#include "zenomainwindow.h"
#include "viewport/displaywidget.h"


struct _Header { // sync with viewdecode.cpp
    size_t total_size;
    size_t info_size;
    size_t magicnum;
    size_t checksum;

    void makeValid() {
        magicnum = 314159265;
        checksum = total_size ^ info_size ^ magicnum;
    }
};



ZTcpServer::ZTcpServer(QObject *parent)
    : QObject(parent)
    , m_tcpServer(nullptr)
    , m_port(0)
{
}

void ZTcpServer::init(const QHostAddress& address)
{
    m_tcpServer = new QTcpServer(this);
    bool bSucceed = false;
    int maxTry = 10, i = 0;

    while (i < maxTry)
    {
        int minPort = 49152;
        int maxPort = 65535;
        m_port = rand() % (maxPort - minPort + 1) + minPort;
        if (m_tcpServer->listen(QHostAddress::LocalHost, m_port))
        {
            zeno::log_info("tcp server listend, port is {}", m_port);
            bSucceed = true;
            break;
        }
        i++;
    }
    if (!bSucceed)
    {
        zeno::log_info("tcp server listend");
    }
    connect(m_tcpServer, SIGNAL(newConnection()), this, SLOT(onNewConnection()));
}

void ZTcpServer::startProc(const std::string& progJson, bool applyLightAndCameraOnly, bool applyMaterialOnly)
{
    ZASSERT_EXIT(m_tcpServer);
    if (m_proc && m_proc->isOpen())
    {
        zeno::log_info("background process already running");
        return;
    }

    zeno::log_info("launching program...");
    zeno::log_debug("program JSON: {}", progJson);

    m_proc = std::make_unique<QProcess>();
    m_proc->setInputChannelMode(QProcess::InputChannelMode::ManagedInputChannel);
    m_proc->setReadChannel(QProcess::ProcessChannel::StandardOutput);
    m_proc->setProcessChannelMode(QProcess::ProcessChannelMode::ForwardedErrorChannel);
    int sessionid = zeno::getSession().globalState->sessionid;
    QSettings settings(zsCompanyName, zsEditor);
    bool bEnableCache = settings.value("zencache-enable").toBool();
    bool bAutoRemove = settings.value("zencache-autoremove", true).toBool();
    QString finalPath;
    if (bEnableCache)
    {
        const QString& cacheRootdir = settings.value("zencachedir").toString();
        QDir dirCacheRoot(cacheRootdir);
        if (!QFileInfo(cacheRootdir).isDir() && !bAutoRemove)
        {
            QMessageBox::warning(nullptr, tr("ZenCache"), tr("The path of cache is invalid, please choose another path."));
            return;
        }

        std::shared_ptr<ZCacheMgr> mgr = zenoApp->getMainWindow()->cacheMgr();
        ZASSERT_EXIT(mgr);
        bool ret = mgr->initCacheDir(bAutoRemove, cacheRootdir);
        ZASSERT_EXIT(ret);
        finalPath = mgr->cachePath();
        int cnum = settings.value("zencachenum").toInt();
        viewDecodeSetFrameCache(finalPath.toStdString().c_str(), cnum);
    }
    else
    {
        viewDecodeSetFrameCache("", 0);
    }

    QStringList args = {
        "-runner", QString::number(sessionid),
        "-port", QString::number(m_port),
        "-cachedir", finalPath,
        "-cacheLightCameraOnly", QString::number(applyLightAndCameraOnly),
        "-cacheMaterialOnly", QString::number(applyMaterialOnly),
    };

    m_proc->start(QCoreApplication::applicationFilePath(), args);

    if (!m_proc->waitForStarted(-1)) {
        zeno::log_warn("process failed to get started, giving up");
        return;
    }

    m_proc->write(progJson.data(), progJson.size());
    m_proc->closeWriteChannel();

    connect(m_proc.get(), SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(onProcFinished(int, QProcess::ExitStatus)));
    connect(m_proc.get(), SIGNAL(readyRead()), this, SLOT(onProcPipeReady()));
}

void ZTcpServer::startOptixCmd(const ZENO_RECORD_RUN_INITPARAM& param)
{
    zeno::log_info("launching optix program...");

    auto optixProc = std::make_unique<QProcess>();
    optixProc->start(QCoreApplication::applicationFilePath(), args);

    if (!optixProc->waitForStarted(-1)) {
        zeno::log_warn("optix process failed to get started, giving up");
        return;
    }

    connect(optixProc.get(), SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(onProcFinished(int, QProcess::ExitStatus)));
    connect(optixProc.get(), SIGNAL(readyRead()), this, SLOT(onProcPipeReady()));
}

void ZTcpServer::startOptixProc()
{
    zeno::log_info("launching optix program...");

    auto optixProc = std::make_unique<QProcess>();
    //optixProc->setInputChannelMode(QProcess::InputChannelMode::ManagedInputChannel);
    //optixProc->setReadChannel(QProcess::ProcessChannel::StandardOutput);
    //optixProc->setProcessChannelMode(QProcess::ProcessChannelMode::ForwardedErrorChannel);

    //check whether there is cached result.
    int cachenum = 0, sFrame = 0, eFrame = 0;
    auto& globalComm = zeno::getSession().globalComm;
    int nRunFrames = globalComm->numOfFinishedFrame();
    auto pair = globalComm->frameRange();

    QStringList args = {
        "-optix", QString::number(0),
        "-port", QString::number(m_port),
        "-cachedir", QString::fromStdString(globalComm->cacheFramePath),
        "-cachenum", QString::number(globalComm->maxCachedFrames),
        "-beginFrame", QString::number(pair.first),
        "-endFrame", QString::number(pair.second)
    };

    //open a new console to show log from optix.
    /*
    optixProc->setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* args) {
        args->flags |= CREATE_NEW_CONSOLE;
        args->startupInfo->dwFlags &= ~STARTF_USESTDHANDLES;
    });
    */

    optixProc->start(QCoreApplication::applicationFilePath(), args);

    if (!optixProc->waitForStarted(-1)) {
        zeno::log_warn("optix process failed to get started, giving up");
        return;
    }

    connect(optixProc.get(), SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(onProcFinished(int, QProcess::ExitStatus)));
    connect(optixProc.get(), SIGNAL(readyRead()), this, SLOT(onProcPipeReady()));

    m_optixProcs.push_back(std::move(optixProc));
}

void ZTcpServer::send_packet(QTcpSocket* socket, std::string_view info, const char* buf, size_t len)
{
    _Header header;
    header.total_size = info.size() + len;
    header.info_size = info.size();
    header.makeValid();

    std::vector<char> headbuffer(4 + sizeof(_Header) + info.size());
    headbuffer[0] = '\a';
    headbuffer[1] = '\b';
    headbuffer[2] = '\r';
    headbuffer[3] = '\t';
    std::memcpy(headbuffer.data() + 4, &header, sizeof(_Header));
    std::memcpy(headbuffer.data() + 4 + sizeof(_Header), info.data(), info.size());

    for (char c : headbuffer) {
        socket->write(&c, 1);
    }
    socket->write(buf, len);
    while (socket->bytesToWrite() > 0) {
        socket->waitForBytesWritten();
    }
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
    QTcpSocket* tcpSocket = m_tcpServer->nextPendingConnection();
    if (!tcpSocket)
    {
        zeno::log_error("tcp connection recv failed");
    }
    else
    {
        zeno::log_debug("tcp connection succeed");
    }

    connect(tcpSocket, SIGNAL(readyRead()), this, SLOT(onReadyRead()));
    connect(tcpSocket, SIGNAL(disconnected()), this, SLOT(onDisconnect()));

    viewDecodeClear();
}

void ZTcpServer::onReadyRead()
{
    if (QTcpSocket* socket = qobject_cast<QTcpSocket*>(sender()))
    {
        QByteArray arr = socket->readAll();
        if (arr == "optixProcStart")
        {
            m_optixSocks.append(socket);
            return;
        }
        qint64 redSize = arr.size();
        for (auto socket : m_optixSocks)
        {
            QString retData = QString::fromUtf8(arr.data(), redSize);
            socket->write(arr.data(), redSize);
            while (socket->bytesToWrite() > 0) {
                socket->waitForBytesWritten();
            }
        }

        zeno::log_debug("qtcpsocket got {} bytes (ping test has 19)", redSize);
        if (redSize > 0) {
            viewDecodeAppend(arr.data(), redSize);
        }
    }
}

void ZTcpServer::onProcPipeReady()
{
    QProcess* proc = qobject_cast<QProcess*>(sender());
    if (!proc) {
        return;
    }
    QByteArray arr = proc->readAll();
    QList<QByteArray> lst = arr.split('\n');
    for (QByteArray line : lst)
    {
        if (!line.isEmpty())
        {
            std::cout << line.data() << std::endl;
            ZWidgetErrStream::appendFormatMsg(line.toStdString());
        }
    }
}

void ZTcpServer::onDisconnect()
{
    QVector<DisplayWidget*> views = zenoApp->getMainWindow()->viewports();
    for (auto pDisplay : views)
    {
        Zenovis* pZenovis = pDisplay->getZenoVis();
        ZASSERT_EXIT(pZenovis);
        auto session = pZenovis->getSession();
        ZASSERT_EXIT(session);
        session->set_curr_frameid(0);
    }

    viewDecodeFinish();

    if (QTcpSocket* socket = qobject_cast<QTcpSocket*>(sender()))
    {
        if (m_optixSocks.indexOf(socket) != -1)
        {
            QString fin = "calcuProcFin";
            socket->write(fin.toStdString().data(), fin.size());
            while (socket->bytesToWrite() > 0) {
                socket->waitForBytesWritten();
            }
        }
    }
}

void ZTcpServer::onProcFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::NormalExit)
    {
        if (m_proc)
            m_proc->terminate();
        m_proc = nullptr;
        zeno::log_info("runner process normally exited with {}", exitCode);
    }
    else if (exitStatus == QProcess::CrashExit)
    {
        if (m_proc)
            m_proc->terminate();
        m_proc= nullptr;
        zeno::log_error("runner process crashed with code {}", exitCode);
    }
    viewDecodeFinish();

    auto mainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(mainWin);
    emit mainWin->runFinished();
}

#endif
