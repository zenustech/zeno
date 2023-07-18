#ifndef __ZCORE_TCPSERVER_H__
#define __ZCORE_TCPSERVER_H__

#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)

#include <QObject>
#include <QtNetwork>
#include "launch/corelaunch.h"
#include "common.h"

class ZTcpServer : public QObject
{
    Q_OBJECT
public:
    ZTcpServer(QObject* parent = nullptr);
    void init(const QHostAddress &address);
    void startProc(const std::string &progJson, LAUNCH_PARAM param);
    void startOptixProc();
    void startOptixCmd(const ZENO_RECORD_RUN_INITPARAM& param);
    void killProc();

    void onFrameStarted(const QString& action, const QString& keyObj);
    void onFrameFinished(const QString& action, const QString& keyObj);
    void onInitFrameRange(const QString& action, int frameStart, int frameEnd);

private slots:
    void onNewConnection();
    void onOptixNewConn();
    void onReadyRead();
    void onProcPipeReady();
    void onDisconnect();
    void onProcFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    void send_packet(QTcpSocket* socket, std::string_view info, const char* buf, size_t len);
    void sendCacheRenderInfoToOptix(const QString& finalCachePath, int cacheNum, bool applyLightAndCameraOnly, bool applyMaterialOnly);
    void dispatchPacketToOptix(const QString& info);
    void sendInitInfoToOptixProc();

    QTcpServer* m_tcpServer;
    QTcpSocket* m_tcpSocket;
    QLocalServer* m_optixServer;
    QVector<QLocalSocket*> m_optixSockets;
    std::unique_ptr<QProcess> m_proc;

    std::vector<std::unique_ptr<QProcess>> m_optixProcs;
    int m_port;
};

#endif

#endif
