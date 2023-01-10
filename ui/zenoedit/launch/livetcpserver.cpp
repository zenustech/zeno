#include "livetcpserver.h"
#include <iostream>

LiveTcpServer::LiveTcpServer(QObject *parent) :
      QObject(parent)
{
    liveData.verSrc = "";
    liveData.camSrc = "";
    liveData.verLoadCount = 0;
    liveData.camLoadCount = 0;
    vertTmp = "";
    cameTmp = "";

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
        //std::cout << "Server started!\n";
    }
}

void LiveTcpServer::newConnection()
{
    // need to grab the socket
    tcpSocket = server->nextPendingConnection();
    //std::cout << "newConnection\n";

    connect(tcpSocket, SIGNAL(readyRead()), this, SLOT(onReadyRead()));

    tcpSocket->write("Received\r\n");
    tcpSocket->flush();
    tcpSocket->waitForBytesWritten(30000);
    //tcpSocket->close();
}
void LiveTcpServer::onReadyRead() {
    QByteArray arr = tcpSocket->readAll();
    qint64 redSize = arr.size();
    auto src = arr.toStdString();
    if (redSize > 0) {
        //std::cout << "read size " << redSize << "\n";
        if(src == "TYPE VERT"){
            receiveType = 1;
        }else if(src == "TYPE CAME"){
            receiveType = 2;
        }else if(src == "SEND DONE"){
            std::cout << "size " << liveData.verSrc.size() << " " << liveData.camSrc.size() << "\n";
            if(receiveType == 1) {
                liveData.verSrc = vertTmp;
                vertTmp = "";
                emit sendVertDone();
            }
            if(receiveType == 2){
                liveData.camSrc = cameTmp;
                cameTmp = "";
                emit sendCamDone();
            }
        }else{
            if(receiveType == 1) {
                vertTmp += src;
            }
            if(receiveType == 2){
                cameTmp += src;
            }
        }
    }
}
