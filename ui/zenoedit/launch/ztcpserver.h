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

private slots:
    void onNewConnection();
    void onReadyRead();
    void onProcPipeReady();
    void onDisconnect();
    void onProcFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    void send_packet(QTcpSocket* socket, std::string_view info, const char* buf, size_t len);

    QTcpServer* m_tcpServer;
    QVector<QTcpSocket*> m_optixSocks;
    std::unique_ptr<QProcess> m_proc;

    std::vector<std::unique_ptr<QProcess>> m_optixProcs;
    int m_port;
};

#endif

#endif
