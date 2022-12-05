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

public slots:
    void newConnection();
    void onReadyRead();

private:
    QTcpServer *server;
    QTcpSocket* tcpSocket;
public:
    LiveObjectData liveData;
};

#endif //ZENO_LIVETCPSERVER_H
