#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)
#include <cstdio>
#include <cstring>
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
#include <zeno/zeno.h>
#include <zeno/extra/GlobalComm.h>
#include "common.h"
#include <zenomodel/include/uihelper.h>
#include "util/apphelper.h"

ZTcpServer::ZTcpServer(QObject *parent)
    : QObject(parent)
    , m_tcpServer(nullptr)
    , m_optixServer(nullptr)
    , m_port(0)
    , m_tcpSocket(nullptr)
{
}

ZTcpServer::~ZTcpServer()
{
    if (m_proc)
    {
        disconnect(m_proc.get(), SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(onProcFinished(int, QProcess::ExitStatus)));
        disconnect(m_proc.get(), SIGNAL(readyRead()), this, SLOT(onProcPipeReady()));
    }
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

void ZTcpServer::startProc(const std::string& progJson, LAUNCH_PARAM param)
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

    QString cachedir;
    if (param.enableCache)
    {
        const QString& cacheRootdir = param.cacheDir;
        QDir dirCacheRoot(cacheRootdir);
        if (!QFileInfo(cacheRootdir).isDir() && !param.tempDir)
        {
            QMessageBox::warning(nullptr, tr("ZenCache"), tr("The path of cache is invalid, please choose another path."));
            return;
        }
        std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
        ZASSERT_EXIT(mgr);
        bool ret = mgr->initCacheDir(param.tempDir, cacheRootdir, param.autoCleanCacheInCacheRoot);
        ZASSERT_EXIT(ret);
        cachedir = mgr->cachePath();
        int cnum = param.cacheNum;
        viewDecodeSetFrameCache(cachedir.toStdString().c_str(), cnum);
    }
    else
    {
        viewDecodeSetFrameCache("", 0);
    }

    //clear last running state
    zeno::getSession().globalComm->clearState();

    if (param.zsgPath.isEmpty())
    {
        auto pGraphsMgr = zenoApp->graphsManagment();
        ZASSERT_EXIT(pGraphsMgr);
        param.zsgPath = pGraphsMgr->zsgDir();
    }

    QStringList args = {
        "--runner", "1",
        "--sessionid", QString::number(sessionid),
        "--port", QString::number(m_port),
        "--enablecache", QString::number(param.enableCache && QFileInfo(cachedir).isDir() && param.cacheNum),
        "--cachenum", QString::number(param.cacheNum),
        "--cachedir", cachedir,
        "--cacheLightCameraOnly", QString::number(param.applyLightAndCameraOnly),
        "--cacheMaterialOnly", QString::number(param.applyMaterialOnly),
        "--cacheautorm", QString::number(param.autoRmCurcache),
        "--zsg", param.zsgPath,
        "--projectFps", QString::number(param.projectFps),
        "--objcachedir", zenoApp->cacheMgr()->objCachePath(),
        "--generator", param.generator
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
    if (ZenoMainWindow* mainwin = zenoApp->getMainWindow())
        emit zenoApp->getMainWindow()->runStarted();
#ifdef ZENO_OPTIX_PROC
    //finally we need to send the cache path to the seperate optix process.
    sendCacheRenderInfoToOptix(cachedir, param.cacheNum, param.applyLightAndCameraOnly, param.applyMaterialOnly);
#endif
}

void ZTcpServer::startOptixCmd(const ZENO_RECORD_RUN_INITPARAM& param)
{
    zeno::log_info("launching optix program...");

    auto optixProc = std::make_unique<QProcess>();
    //optixProc->start(QCoreApplication::applicationFilePath(), args);

    if (!optixProc->waitForStarted(-1)) {
        zeno::log_warn("optix process failed to get started, giving up");
        return;
    }

    connect(optixProc.get(), SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(onProcFinished(int, QProcess::ExitStatus)));
    connect(optixProc.get(), SIGNAL(readyRead()), this, SLOT(onProcPipeReady()));
}

void ZTcpServer::onOptixNewConn()
{
    ZASSERT_EXIT(m_optixServer);
    QLocalSocket* socket = m_optixServer->nextPendingConnection();
    connect(socket, &QLocalSocket::readyRead, this, [=]() {
        while (socket->canReadLine()) {
            QByteArray content = socket->readLine();
            rapidjson::Document doc;
            doc.Parse(content);
            if (doc.IsObject()) {
                ZASSERT_EXIT(doc.HasMember("action"));
                const QString& action(doc["action"].GetString());
                if (action == "runBeforRecord") {
                    ZASSERT_EXIT(doc["launchparam"].IsObject());
                    const auto& param = doc["launchparam"];
                    ZASSERT_EXIT(param.HasMember("beginFrame") && param.HasMember("endFrame"));
                    auto pGraphsMgr = zenoApp->graphsManagment();
                    ZASSERT_EXIT(pGraphsMgr);
                    IGraphsModel* pModel = pGraphsMgr->currentModel();
                    ZASSERT_EXIT(pModel);
                    LAUNCH_PARAM lparam;
                    lparam.beginFrame = param["beginFrame"].GetInt();
                    lparam.endFrame = param["endFrame"].GetInt();
                    auto main = zenoApp->getMainWindow();
                    ZASSERT_EXIT(main);
                    lparam.projectFps = main->timelineInfo().timelinefps;
                    AppHelper::initLaunchCacheParam(lparam);
                    launchProgram(pModel, lparam);
                }else if (action == "removeCache")
                {
                    const RECORD_SETTING& recordSetting = zenoApp->graphsManagment()->recordSettings();
                    ZASSERT_EXIT(doc.HasMember("frame"));
                    int frame = doc["frame"].GetInt();
                    if (recordSetting.bAutoRemoveCache)
                        zeno::getSession().globalComm->removeCache(frame);
                }
                else if (action == "clrearFrameState")
                {
                    QString cachepath = QString::fromStdString(zeno::getSession().globalComm->cachePath());
                    QDir dir(cachepath);
                    if (dir.exists() && dir.isEmpty()) {
                        zeno::log_info("remove dir: {}", cachepath.toStdString());
                        dir.rmdir(cachepath);
                    }
                    zeno::getSession().globalComm->clearFrameState();
                    QMessageBox msgBox(QMessageBox::Information, "", tr("Cache information was deleted during recording."));
                    msgBox.exec();
                }
            }
        }
        });
    m_optixSockets.append(socket);
    initializeNewOptixProc();
    connect(socket, &QLocalSocket::disconnected, this, [=]() {
        m_optixSockets.removeOne(socket);
    });
}

void ZTcpServer::sendCacheRenderInfoToOptix(const QString& finalCachePath, int cacheNum, bool applyLightAndCameraOnly, bool applyMaterialOnly)
{
    QString renderKey = QString("{\"applyLightAndCameraOnly\":%1, \"applyMaterialOnly\":%2}").arg(applyLightAndCameraOnly).arg(applyMaterialOnly);
    QString objKey = QString("{\"cachedir\":\"%1\", \"cachenum\":%2}").arg(finalCachePath).arg(cacheNum);
    QString info = QString("{\"action\":\"initCache\", \"key\":%2, \"render\":%3}\n").arg(objKey).arg(renderKey);
    dispatchPacketToOptix(info);
}

void ZTcpServer::onInitFrameRange(const QString& action, int frameStart, int frameEnd)
{
    if (!m_optixServer || m_optixSockets.isEmpty())
        return;

    QString info = QString("{\"action\":\"%1\", \"beginFrame\":%2, \"endFrame\":%3}\n").arg(action).arg(frameStart).arg(frameEnd);
    dispatchPacketToOptix(info);
}

void ZTcpServer::onClearFrameState()
{
    QString info = QString("{\"action\":\"clearFrameState\"}\n");
    dispatchPacketToOptix(info);
}

void ZTcpServer::onFrameStarted(const QString& action, const QString& keyObj)
{
    bool bOK = false;
    int frame = keyObj.toInt(&bOK);
    ZASSERT_EXIT(bOK);
    QString info = QString("{\"action\":\"%1\", \"key\":%2}\n").arg(action).arg(frame);
    dispatchPacketToOptix(info);
}

void ZTcpServer::onFrameFinished(const QString& action, const QString& keyObj)
{
    bool bOK = false;
    int frame = keyObj.toInt(&bOK);
    ZASSERT_EXIT(bOK);
    QString info = QString("{\"action\":\"%1\", \"key\":%2}\n").arg(action).arg(frame);
    dispatchPacketToOptix(info);
}

void ZTcpServer::dispatchPacketToOptix(const QString& info)
{
    if (m_optixServer) {
        for (QLocalSocket* pSocket : m_optixSockets) {
            pSocket->write(info.toUtf8());
        }
    }
}

void ZTcpServer::initializeNewOptixProc()
{
    std::shared_ptr<ZCacheMgr> mgr = zenoApp->cacheMgr();
    ZASSERT_EXIT(mgr);
    auto& globalComm = zeno::getSession().globalComm;
    sendCacheRenderInfoToOptix(mgr->cachePath(), globalComm->maxCachedFramesNum(), false, false);
    onInitFrameRange(QString::fromStdString("frameRange"), globalComm->frameRange().first, globalComm->frameRange().second);
    QString frameRunningState = QString("{\"action\":\"frameRunningState\", \"initializedFrames\":%1, \"finishedFrame\":%2}\n").arg(globalComm->numOfInitializedFrame()).arg(globalComm->numOfFinishedFrame());
    dispatchPacketToOptix(frameRunningState);
}

void ZTcpServer::startOptixProc()
{
    zeno::log_info("launching optix program...");

    static const QString sessionID = UiHelper::generateUuid("zenooptix");
    if (!m_optixServer) {
        m_optixServer = new QLocalServer(this);
        m_optixServer->listen(sessionID);
        connect(m_optixServer, &QLocalServer::newConnection, this, &ZTcpServer::onOptixNewConn);
    }

    auto optixProc = std::make_unique<QProcess>();
    //optixProc->setInputChannelMode(QProcess::InputChannelMode::ManagedInputChannel);
    //optixProc->setReadChannel(QProcess::ProcessChannel::StandardOutput);
    //optixProc->setProcessChannelMode(QProcess::ProcessChannelMode::ForwardedErrorChannel);

    //check whether there is cached result.
    auto& globalComm = zeno::getSession().globalComm;
    int nRunFrames = globalComm->numOfFinishedFrame();
    auto mainWin = zenoApp->getMainWindow();
    auto pair = globalComm->frameRange();

    QStringList args = {
        "-optix", QString::number(0),
        "-port", QString::number(m_port),
        "-cachedir", QString::fromStdString(globalComm->cacheFramePath),
        "-cachenum", QString::number(globalComm->maxCachedFramesNum()),
        "-beginFrame", QString::number(pair.first),
        "-endFrame", QString::number(pair.second),
        "-finishedFrames", QString::number(nRunFrames),
        "-sessionId", sessionID,
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

void ZTcpServer::killProc()
{
    if (m_proc) {
        m_proc->kill();
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
        zeno::log_debug("tcp connection succeed");

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
            if (zenoApp->isUIApplication())
                ZWidgetErrStream::appendFormatMsg(line.toStdString());
        }
    }
}

void ZTcpServer::onDisconnect()
{
    /*
    auto mainWin = zenoApp->getMainWindow();
    if (mainWin)
    {
        QVector<DisplayWidget*> views = mainWin->viewports();
        for (auto pDisplay : views)
        {
            Zenovis* pZenovis = pDisplay->getZenoVis();
            ZASSERT_EXIT(pZenovis);
            auto session = pZenovis->getSession();
            ZASSERT_EXIT(session);
            session->set_curr_frameid(0);
        }
    }*/

    viewDecodeFinish();
}

void ZTcpServer::onProcFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::NormalExit)
    {
        if (m_proc)
            m_proc->kill();
        m_proc = nullptr;
        zeno::log_info("runner process normally exited with {}", exitCode);
        if (exitCode != 0)
            emit runnerError();
    }
    else if (exitStatus == QProcess::CrashExit)
    {
        if (m_proc)
            m_proc->kill();
        m_proc= nullptr;
        zeno::log_error("runner process crashed with code {}", exitCode);
        emit runnerError();
    }
    viewDecodeFinish();

    auto mainWin = zenoApp->getMainWindow();
    if (mainWin)
        emit mainWin->runFinished();
    else
        emit runFinished();
}

#endif
