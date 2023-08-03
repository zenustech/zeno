#ifndef __ZCORE_TCPSERVER_H__
#define __ZCORE_TCPSERVER_H__

#if defined(ZENO_MULTIPROCESS) && defined(ZENO_IPC_USE_TCP)

#include <QObject>
#include <QtNetwork>
#include "launch/corelaunch.h"

class ZTcpServer : public QObject
{
    Q_OBJECT
public:
    ZTcpServer(QObject* parent = nullptr);
    void init(const QHostAddress &address);
    void startProc(const std::string &progJson, LAUNCH_PARAM param);
    void killProc();

private slots:
    void onNewConnection();
    void onReadyRead();
    void onProcPipeReady();
    void onDisconnect();
    void onProcFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    QTcpServer* m_tcpServer;
    QTcpSocket* m_tcpSocket;
    std::unique_ptr<QProcess> m_proc;
    int m_port;
};

#endif

#endif
