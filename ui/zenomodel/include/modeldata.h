#ifndef __MODEL_DATA_H__
#define __MODEL_DATA_H__

#include <QtWidgets>
#include "fuckqmap.h"
#include "zassert.h"

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
    CONTROL_COLOR_NORMAL,
    CONTROL_CURVE,
    CONTROL_HSLIDER,
    CONTROL_HSPINBOX,
    CONTROL_HDOUBLESPINBOX,
    CONTROL_SPINBOX_SLIDER,
    CONTROL_VEC4_INT,
    CONTROL_VEC4_FLOAT,
    CONTROL_VEC3_INT,
    CONTROL_VEC3_FLOAT,
    CONTROL_VEC2_INT,
    CONTROL_VEC2_FLOAT,
    CONTROL_DICTPANEL,      //for socket, this control allow to link multiple sockets.  for panel, this control displays as a table.
    CONTROL_NONVISIBLE,     //for legacy param like _KEYS, _POINTS, _HANDLES.
    CONTROL_GROUP_LINE
};

enum NODE_TYPE {
    NORMAL_NODE,
    HEATMAP_NODE,
    BLACKBOARD_NODE,
    SUBINPUT_NODE,
    SUBOUTPUT_NODE,
    GROUP_NODE
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
    PARAM_INNER_INPUT,      //socket in socket, like key in dict param.
    PARAM_INNER_OUTPUT,
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
    SOCKPROP_DICTLIST_PANEL = 1 << 3,
    SOCKPROP_GROUP_LINE = 1 << 4,
};

struct PARAM_INFO {
    QString name;
    QString toolTip;
    QVariant defaultValue;
    QVariant value;
    PARAM_CONTROL control;
    QString typeDesc;
    QVariant controlProps;
    bool bEnableConnect;     //enable connection with other out socket.

    PARAM_INFO() : control(CONTROL_NONE), bEnableConnect(false) {}
};
Q_DECLARE_METATYPE(PARAM_INFO)

struct SLIDER_INFO {
    qreal step;
    qreal min;
    qreal max;
    SLIDER_INFO() : step(1.), min(0.), max(100.) {}
};
Q_DECLARE_METATYPE(SLIDER_INFO)

typedef QVariantMap CONTROL_PROPERTIES;

inline const char* cPathSeperator = ":";

typedef QMap<QString, PARAM_INFO> PARAMS_INFO;
Q_DECLARE_METATYPE(PARAMS_INFO)

struct EdgeInfo
{
    QString outSockPath;    //option: path for socket.
    QString inSockPath;

    EdgeInfo() = default;
    EdgeInfo(const QString &outpath, const QString &inpath)
        : outSockPath(outpath), inSockPath(inpath) {}
    
    bool operator==(const EdgeInfo &rhs) const {
        return outSockPath == rhs.outSockPath && inSockPath == rhs.inSockPath;
    }
    bool operator<(const EdgeInfo &rhs) const {
        if (outSockPath != rhs.outSockPath) {
            return outSockPath < rhs.outSockPath;
        } else if (inSockPath != rhs.inSockPath) {
            return inSockPath < rhs.inSockPath;
        } else {
            return 0;
        }
    }

    bool isValid() const {
        return !inSockPath.isEmpty() && !outSockPath.isEmpty();
    }

};
Q_DECLARE_METATYPE(EdgeInfo)

struct DICTKEY_INFO
{
    QString key;
    QList<EdgeInfo> links;
};

struct DICTPANEL_INFO
{
    QList<DICTKEY_INFO> keys;
    bool bCollasped;
};

struct SOCKET_INFO {
    QString nodeid;
    QString name;
    QString toolTip;
    PARAM_CONTROL control;
    QString type;

    QVariant defaultValue;  // a native value or a curvemodel.
    QList<EdgeInfo> links;  //structure for storing temp link info, cann't use to normal precedure, except copy/paste and io.

    //QList<DICTKEY_INFO> keys;
    DICTPANEL_INFO dictpanel;
    int sockProp;

    CONTROL_PROPERTIES ctrlProps;

    SOCKET_INFO() : control(CONTROL_NONE), sockProp(SOCKPROP_NORMAL) {}
    SOCKET_INFO(const QString& id, const QString& name)
        : nodeid(id)
        , name(name)
        , control(CONTROL_NONE)
        , sockProp(SOCKPROP_NORMAL)
    {}
    SOCKET_INFO(const QString& id, const QString& name, PARAM_CONTROL ctrl, const QString& type, const QVariant& defl, const QString& tip)
        : nodeid(id), name(name), control(ctrl), type(type), defaultValue(defl), sockProp(SOCKPROP_NORMAL), toolTip(tip)
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
Q_DECLARE_METATYPE(INPUT_SOCKET)
typedef FuckQMap<QString, INPUT_SOCKET> INPUT_SOCKETS;
Q_DECLARE_METATYPE(INPUT_SOCKETS)


struct OUTPUT_SOCKET
{
    SOCKET_INFO info;
};
Q_DECLARE_METATYPE(OUTPUT_SOCKET)
typedef FuckQMap<QString, OUTPUT_SOCKET> OUTPUT_SOCKETS;
Q_DECLARE_METATYPE(OUTPUT_SOCKETS)


struct VPARAM_INFO
{
    PARAM_INFO m_info;
    VPARAM_TYPE vType;
    QString refParamPath;
    PARAM_CLASS m_cls;
    QVector<VPARAM_INFO> children;
    QVariant controlInfos;
    uint m_uuid;

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
    QStringList items;
    QColor background;
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
typedef QList<EdgeInfo> LINKS_DATA;

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
    bool visible;
    bool timeline;
};

typedef QMap<QString, CURVE_DATA> CURVES_DATA;
Q_DECLARE_METATYPE(CURVES_DATA);


typedef QList<QPersistentModelIndex> PARAM_LINKS;
Q_DECLARE_METATYPE(PARAM_LINKS)

#endif
