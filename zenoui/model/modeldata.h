#ifndef __MODEL_DATA_H__
#define __MODEL_DATA_H__

#include <QtWidgets>

struct PARAM_INFO {
    QString name;
    QVariant defaultValue;
    QVariant value;
    bool bEnableConnect;     //enable connection with other out socket.
};
typedef QMap<QString, PARAM_INFO> PARAMS_INFO;
Q_DECLARE_METATYPE(PARAMS_INFO)

struct SOCKET_INFO {
    QString nodeid;
    QString name;
    //only used to paint link at gv:
    QPointF pos;
    bool binsock;

    SOCKET_INFO() : binsock(true) {}
    SOCKET_INFO(const QString& id, const QString& name) : nodeid(id), name(name) {}
    SOCKET_INFO(const QString &id, const QString &name, const QPointF &p, bool bIn)
        : nodeid(id), name(name), pos(p), binsock(bIn) {}
};
typedef QMap<QString, SOCKET_INFO> SOCKETS_INFO;

struct INPUT_SOCKET
{
    QVariant defaultValue;
    SOCKET_INFO info;
    QMap<QString, SOCKETS_INFO> outNodes;      //describe the connection with this socket.
};
typedef QMap<QString, INPUT_SOCKET> INPUT_SOCKETS;
Q_DECLARE_METATYPE(INPUT_SOCKETS)

struct OUTPUT_SOCKET
{
    SOCKET_INFO info;
    QMap<QString, SOCKETS_INFO> inNodes;
};
typedef QMap<QString, OUTPUT_SOCKET> OUTPUT_SOCKETS;
Q_DECLARE_METATYPE(OUTPUT_SOCKETS)

#endif