#ifndef ZENO_LIVETCPSERVER_H
#define ZENO_LIVETCPSERVER_H

#include <QObject>
#include <QTcpSocket>
#include <QTcpServer>
#include <QDebug>

#include "include/common.h"

class LiveTcpServer : public QObject
{
    Q_OBJECT
public:
    explicit LiveTcpServer(QObject *parent = 0);

signals:
    void sendVertDone();
    void sendCamDone();
public slots:
    void newConnection();
    void onReadyRead();

private:
    QTcpServer *server;
    QTcpSocket* tcpSocket;
    int receiveType = 0;
    std::string vertTmp;
    std::string cameTmp;
public:
    LiveObjectData liveData;
};

#endif //ZENO_LIVETCPSERVER_H
