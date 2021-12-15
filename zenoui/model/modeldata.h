#ifndef __MODEL_DATA_H__
#define __MODEL_DATA_H__

#include <QtWidgets>

struct SOCKET_INFO {
    QString nodeid;
    QString name;
    //only used to paint link at gv:
    QPointF pos;
    bool binsock;
};
typedef QMap<QString, SOCKET_INFO> SOCKETS_INFO;


struct INPUT_SOCKET
{
    QVariant defaultValue;
    SOCKET_INFO info;
    QMap<QString, SOCKETS_INFO> outNodes;      //describe is connected which node and socket.
};
typedef QMap<QString, INPUT_SOCKET> INPUT_SOCKETS;
Q_DECLARE_METATYPE(INPUT_SOCKETS)

struct OUTPUT_SOCKET
{
    SOCKET_INFO info;
    QMap<QString, SOCKETS_INFO> inNodes; //describe is connected which node and socket.
};
typedef QMap<QString, OUTPUT_SOCKET> OUTPUT_SOCKETS;
Q_DECLARE_METATYPE(OUTPUT_SOCKETS)

#endif