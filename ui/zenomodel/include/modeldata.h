#ifndef __MODEL_DATA_H__
#define __MODEL_DATA_H__

#include <QtWidgets>
#include "fuckqmap.h"

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
    CONTROL_COLOR,
    CONTROL_CURVE,
    CONTROL_HSLIDER,
    CONTROL_HSPINBOX,
    CONTROL_SPINBOX_SLIDER,
    CONTROL_VEC4_INT,
    CONTROL_VEC4_FLOAT,
    CONTROL_VEC3_INT,
    CONTROL_VEC3_FLOAT,
    CONTROL_VEC2_INT,
    CONTROL_VEC2_FLOAT,
    CONTROL_DICTPANEL,      //for socket, this control allow to link multiple sockets.  for panel, this control displays as a table.
    CONTROL_NONVISIBLE,     //for legacy param like _KEYS, _POINTS, _HANDLES.
};

enum NODE_TYPE {
    NORMAL_NODE,
    HEATMAP_NODE,
    BLACKBOARD_NODE,
    SUBINPUT_NODE,
    SUBOUTPUT_NODE,
};

enum VPARAM_TYPE
{
    VPARAM_ROOT,
    VPARAM_TAB,
    VPARAM_GROUP,
    VPARAM_INPUTS,
    VPARAM_PARAMETERS,
    VPARAM_OUTPUTS,
    VPARAM_PARAM,
};

enum PARAM_CLASS
{
    PARAM_UNKNOWN,
    PARAM_INPUT,
    PARAM_PARAM,
    PARAM_OUTPUT,
};

enum NODE_OPTION {
    OPT_ONCE = 1,
    OPT_MUTE = 1 << 1,
    OPT_VIEW = 1 << 2,
    OPT_PREP = 1 << 3
};

enum SOCKET_PROPERTY {
    SOCKPROP_UNKNOWN = 0,
    SOCKPROP_NORMAL = 1,
    SOCKPROP_EDITABLE = 1 << 1,
    SOCKPROP_MULTILINK = 1 << 2,
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

struct SLIDER_INFO {
    qreal step;
    qreal min;
    qreal max;
    SLIDER_INFO() : step(1.), min(0.), max(0.) {}
};
Q_DECLARE_METATYPE(SLIDER_INFO)

typedef QMap<QString, QVariant> CONTROL_PROPERTIES;
Q_DECLARE_METATYPE(CONTROL_PROPERTIES)

inline const char* cPathSeperator = ":";

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

    QVariant defaultValue;  // a native value or a curvemodel.
    QList<EdgeInfo> links;  //structure for storing temp link info, cann't use to normal precedure, except copy/paste and io.

    SOCKET_PROPERTY sockProp;

    SOCKET_INFO() : control(CONTROL_NONE), sockProp(SOCKPROP_NORMAL) {}
    SOCKET_INFO(const QString& id, const QString& name)
        : nodeid(id)
        , name(name)
        , control(CONTROL_NONE)
        , sockProp(SOCKPROP_NORMAL)
    {}
    SOCKET_INFO(const QString& id, const QString& name, PARAM_CONTROL ctrl, const QString& type, const QVariant& defl)
        : nodeid(id), name(name), control(ctrl), type(type), defaultValue(defl), sockProp(SOCKPROP_NORMAL)
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
};
typedef FuckQMap<QString, INPUT_SOCKET> INPUT_SOCKETS;
Q_DECLARE_METATYPE(INPUT_SOCKETS)


struct OUTPUT_SOCKET
{
    SOCKET_INFO info;
};
typedef FuckQMap<QString, OUTPUT_SOCKET> OUTPUT_SOCKETS;
Q_DECLARE_METATYPE(OUTPUT_SOCKETS)


struct VPARAM_INFO
{
    PARAM_INFO m_info;
    VPARAM_TYPE vType;
    QString coreParam;
    PARAM_CLASS m_cls;
    QVector<VPARAM_INFO> children;
    QVariant controlInfos;

    VPARAM_INFO() : vType(VPARAM_PARAM), m_cls(PARAM_UNKNOWN) {}
};
Q_DECLARE_METATYPE(VPARAM_INFO);


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

typedef QVector<qreal> UI_VECTYPE;

Q_DECLARE_METATYPE(UI_VECTYPE);

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
    QString name;
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

struct CURVE_POINT {
    QPointF point;
    QPointF leftHandler;
    QPointF rightHandler;
    int controlType;
};

struct CURVE_DATA {
    QString key;
    QVector<CURVE_POINT> points;
    int cycleType;
    CURVE_RANGE rg;
};

typedef QList<QPersistentModelIndex> PARAM_LINKS;
Q_DECLARE_METATYPE(PARAM_LINKS)

#endif
