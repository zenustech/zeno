#ifndef __MODEL_DATA_H__
#define __MODEL_DATA_H__

#include <QtWidgets>
#include "zenoui/util/fuckqmap.h"

enum PARAM_CONTROL {
    CONTROL_NONE,
    CONTROL_INT,
    CONTROL_BOOL,
    CONTROL_FLOAT,
    CONTROL_STRING,
    CONTROL_VEC3F,
    CONTROL_ENUM,
    CONTROL_WRITEPATH,
    CONTROL_READPATH,
    CONTROL_MULTILINE_STRING,
    CONTROL_HEATMAP,
    CONTROL_CURVE,
};

enum NODE_TYPE {
    NORMAL_NODE,
    HEATMAP_NODE,
    BLACKBOARD_NODE,
    SUBINPUT_NODE,
    SUBOUTPUT_NODE,
};

enum NODE_OPTION {
    OPT_ONCE = 1,
    OPT_MUTE = 1 << 1,
    OPT_VIEW = 1 << 2,
    OPT_PREP = 1 << 3
};

struct PARAM_INFO {
    QString name;
    QVariant defaultValue;
    QVariant value;
    PARAM_CONTROL control;
    QString typeDesc;
    bool bEnableConnect;     //enable connection with other out socket.

    PARAM_INFO() : control(CONTROL_NONE), bEnableConnect(false) {}
};
Q_DECLARE_METATYPE(PARAM_INFO)

typedef QMap<QString, PARAM_INFO> PARAMS_INFO;
Q_DECLARE_METATYPE(PARAMS_INFO)

struct EdgeInfo
{
    QString outputNode;
    QString inputNode;
    QString outputSock;
    QString inputSock;
    EdgeInfo() = default;
    EdgeInfo(const QString &outNode, const QString &inNode, const QString &outSock, const QString &inSock)
        : outputNode(outNode), inputNode(inNode), outputSock(outSock), inputSock(inSock) {}
    bool operator==(const EdgeInfo &rhs) const {
        return outputNode == rhs.outputNode && inputNode == rhs.inputNode &&
               outputSock == rhs.outputSock && inputSock == rhs.inputSock;
    }
    bool operator<(const EdgeInfo &rhs) const {
        if (outputNode != rhs.outputNode) {
            return outputNode < rhs.outputNode;
        } else if (inputNode != rhs.inputNode) {
            return inputNode < rhs.inputNode;
        } else if (outputSock != rhs.outputSock) {
            return outputSock < rhs.outputSock;
        } else if (inputSock != rhs.inputSock) {
            return inputSock < rhs.inputSock;
        } else {
            return 0;
        }
    }
};
Q_DECLARE_METATYPE(EdgeInfo)

struct cmpEdge {
    bool operator()(const EdgeInfo &lhs, const EdgeInfo &rhs) const {
        return lhs.outputNode < rhs.outputNode && lhs.inputNode < rhs.inputNode &&
               lhs.outputSock < rhs.outputSock && lhs.inputSock < rhs.inputSock;
    }
};

struct SOCKET_INFO {
    QString nodeid;
    QString name;
    PARAM_CONTROL control;
    QString type;
    QVariant defaultValue;

    SOCKET_INFO() : control(CONTROL_NONE) {}
    SOCKET_INFO(const QString& id, const QString& name)
        : nodeid(id)
        , name(name)
        , control(CONTROL_NONE)
    {}

	bool operator==(const SOCKET_INFO& rhs) const {
		return nodeid == rhs.nodeid && name == rhs.name;
	}
	bool operator<(const SOCKET_INFO& rhs) const {
		if (nodeid != rhs.nodeid) {
			return nodeid < rhs.nodeid;
		}
		else if (name != rhs.name) {
			return name < rhs.name;
		}
		else {
			return 0;
		}
	}
};
typedef QMap<QString, SOCKET_INFO> SOCKETS_INFO;


struct INPUT_SOCKET
{
    SOCKET_INFO info;
    QMap<QString, SOCKETS_INFO> outNodes;      //structure for storing temp link info, cann't use to normal precedure, except copy/paste and io.
    QList<QPersistentModelIndex> linkIndice;
};
typedef FuckQMap<QString, INPUT_SOCKET> INPUT_SOCKETS;
Q_DECLARE_METATYPE(INPUT_SOCKETS)


struct OUTPUT_SOCKET
{
    SOCKET_INFO info;
    QMap<QString, SOCKETS_INFO> inNodes;    //structure for storing temp link info...
    QList<QPersistentModelIndex> linkIndice;
};
typedef FuckQMap<QString, OUTPUT_SOCKET> OUTPUT_SOCKETS;
Q_DECLARE_METATYPE(OUTPUT_SOCKETS)

struct COLOR_RAMP
{
    qreal pos, r, g, b;
    COLOR_RAMP() : pos(0), r(0), g(0), b(0) {}
    COLOR_RAMP(const qreal& pos, const qreal& r, const qreal& g, const qreal& b)
        : pos(pos), r(r), g(g), b(b) {}
};
typedef QVector<COLOR_RAMP> COLOR_RAMPS;
Q_DECLARE_METATYPE(COLOR_RAMPS)

Q_DECLARE_METATYPE(QLinearGradient)

Q_DECLARE_METATYPE(QVector<qreal>);

struct BLACKBOARD_INFO
{
    QSizeF sz;
    QString title;
    QString content;
    //params
    bool special;
    BLACKBOARD_INFO() : special(false) {}
};
Q_DECLARE_METATYPE(BLACKBOARD_INFO)

struct NODE_DESC {
    INPUT_SOCKETS inputs;
    PARAMS_INFO params;
    OUTPUT_SOCKETS outputs;
    QStringList categories;
    bool is_subgraph = false;
};
typedef QMap<QString, NODE_DESC> NODE_DESCS;

struct NODE_CATE {
    QString name;
    QStringList nodes;
};
typedef QMap<QString, NODE_CATE> NODE_CATES;

typedef QMap<int, QVariant> NODE_DATA;
Q_DECLARE_METATYPE(NODE_DATA)


struct PARAM_UPDATE_INFO {
    QString name;
    QVariant oldValue;
    QVariant newValue;
};
Q_DECLARE_METATYPE(PARAM_UPDATE_INFO)

enum SOCKET_UPDATE_WAY {
    SOCKET_INSERT,
    SOCKET_REMOVE,
    SOCKET_UPDATE_NAME,
    SOCKET_UPDATE_TYPE,
    SOCKET_UPDATE_DEFL
};

struct SOCKET_UPDATE_INFO {
    SOCKET_INFO oldInfo;
    SOCKET_INFO newInfo;
    SOCKET_UPDATE_WAY updateWay;
    bool bInput;
};
Q_DECLARE_METATYPE(SOCKET_UPDATE_INFO)

struct STATUS_UPDATE_INFO {
    QVariant oldValue;
    QVariant newValue;
    int role;
};
Q_DECLARE_METATYPE(STATUS_UPDATE_INFO)

struct LINK_UPDATE_INFO {
    EdgeInfo oldEdge;
    EdgeInfo newEdge;
};

typedef QMap<QString, NODE_DATA> NODES_DATA;

struct CURVE_RANGE {
    qreal xFrom;
    qreal xTo;
    qreal yFrom;
    qreal yTo;
};

#endif
