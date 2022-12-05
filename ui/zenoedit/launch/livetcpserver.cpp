#include "livetcpserver.h"
#include <iostream>

LiveTcpServer::LiveTcpServer(QObject *parent) :
      QObject(parent)
{
    liveData.verSrc = "";
    liveData.camSrc = "";
    liveData.verLoadCount = 0;
    liveData.camLoadCount = 0;

    server = new QTcpServer(this);

    // whenever a user connects, it will emit signal
    connect(server, SIGNAL(newConnection()),
            this, SLOT(newConnection()));

    if(!server->listen(QHostAddress::Any, 5236))
    {
        std::cout << "Server could not start\n";
    }
    else
    {
        std::cout << "Server started!\n";
    }
}

void LiveTcpServer::newConnection()
{
    // need to grab the socket
    tcpSocket = server->nextPendingConnection();
    std::cout << "newConnection\n";

    connect(tcpSocket, SIGNAL(readyRead()), this, SLOT(onReadyRead()));

    tcpSocket->write("Received\r\n");
    tcpSocket->flush();
    tcpSocket->waitForBytesWritten(3000);
    //tcpSocket->close();
}
void LiveTcpServer::onReadyRead() {
    QByteArray arr = tcpSocket->readAll();
    qint64 redSize = arr.size();
    std::cout << "read size " << redSize << "\n";
    if (redSize > 0) {
        liveData.verSrc = arr.toStdString();
    }
}
