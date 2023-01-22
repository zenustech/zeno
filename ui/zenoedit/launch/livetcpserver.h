#ifndef ZENO_LIVETCPSERVER_H
#define ZENO_LIVETCPSERVER_H

#include <QObject>
#include <QTcpSocket>
#include <QTcpServer>
#include <QDebug>

#include "include/common.h"

struct TcpSend{
    std::string host{};
    int port{};
    std::string data{};
};

struct TcpReceive{
    bool success{};
    std::string data{};
};

class LiveTcpServer : public QObject
{
    Q_OBJECT
public:
    explicit LiveTcpServer(QObject *parent = 0);
    ~LiveTcpServer();

signals:
    void sendVertDone();
    void sendCamDone();
public slots:
    void newConnection();
    void onReadyRead();
public:
    TcpReceive sendData(TcpSend s);

private:
    QTcpServer *server;
    QTcpSocket* tcpSocket;
    int receiveType = 0;
    std::string vertTmp;
    std::string cameTmp;
public:
    LiveObjectData liveData;
    QTcpSocket* clientSocket;
};

class LiveSignalsBridge : public QObject{
    Q_OBJECT
  public:
    explicit LiveSignalsBridge(QObject *parent = 0);

  signals:
    void frameMeshSendDone();
};

#endif //ZENO_LIVETCPSERVER_H
