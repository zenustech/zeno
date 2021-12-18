#ifndef __MODEL_DATA_H__
#define __MODEL_DATA_H__

#include <QtWidgets>

enum PARAM_CONTROL {
    CONTROL_NONE,
    CONTROL_INT,
    CONTROL_BOOL,
    CONTROL_FLOAT,
    CONTROL_STRING,
    CONTROL_ENUM,
    CONTROL_WRITEPATH,
    CONTROL_READPATH,
    CONTROL_MULTILINE_STRING,
};

struct PARAM_INFO {
    QString name;
    QVariant defaultValue;
    QVariant value;
    PARAM_CONTROL control;
    bool bEnableConnect;     //enable connection with other out socket.

    PARAM_INFO() : control(CONTROL_NONE), bEnableConnect(false) {}
};
typedef QMap<QString, PARAM_INFO> PARAMS_INFO;
Q_DECLARE_METATYPE(PARAMS_INFO)


struct SOCKET_INFO {
    QString nodeid;
    QString name;
    PARAM_CONTROL control;
    QString type;
    QVariant defaultValue;

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


struct NODE_PARAMS_PACK {
    INPUT_SOCKETS inputs;
    PARAMS_INFO params;
    OUTPUT_SOCKETS outputs;
};
typedef QMap<QString, NODE_PARAMS_PACK> NODES_PARAMS;


#endif