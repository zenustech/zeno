#include "livetcpserver.h"
#include <iostream>

LiveTcpServer::LiveTcpServer(QObject *parent) :
      QObject(parent)
{
#ifdef ZENO_LIVESYNC
    liveData.verSrc = "";
    liveData.camSrc = "";
    liveData.verLoadCount = 0;
    liveData.camLoadCount = 0;
    vertTmp = "";
    cameTmp = "";
    clientSocket = new QTcpSocket(this);
    server = new QTcpServer(this);

    // whenever a user connects, it will emit signal
    connect(server, SIGNAL(newConnection()), this, SLOT(newConnection()));
    if(!server->listen(QHostAddress::Any, 5236))
        std::cout << "LiveTcp Server could not start\n";
    else
        std::cout << "LiveTcp Server Running On 5236.\n";
#endif
}

LiveTcpServer::~LiveTcpServer(){
#ifdef ZENO_LIVESYNC
    delete clientSocket;
    delete server;
#endif
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

TcpReceive LiveTcpServer::sendData(TcpSend s){
    clientSocket->connectToHost(QString::fromStdString(s.host), s.port);
    TcpReceive r;
    if(clientSocket->waitForConnected(2000))
    {
        // send
        clientSocket->write(QByteArray::fromStdString(s.data));
        clientSocket->waitForBytesWritten(1000);
        clientSocket->waitForReadyRead(2000);
        std::cout << "\tConnected! " << s.host << ":" << s.port << " Reading: " << clientSocket->bytesAvailable() << "\n";

        // receive
        r.data = clientSocket->readAll().toStdString();
        r.success = true;
        clientSocket->close();
    }
    else
    {
        r.success = false;
        std::cout << "Not connected! " << s.host << ":" << s.port << "\n";
    }
    return r;
}

LiveSignalsBridge::LiveSignalsBridge(QObject *parent) : QObject(parent) {

}
