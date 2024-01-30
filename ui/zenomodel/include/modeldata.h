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
    CONTROL_PYTHON_EDITOR,
    CONTROL_BUTTON,
    CONTROL_COLOR,
    CONTROL_PURE_COLOR,
    CONTROL_COLOR_VEC3F,
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
    GROUP_NODE,
    NO_VERSION_NODE
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
    PARAM_LEGACY_INPUT,     //params described by legacy zsgfile.
    PARAM_LEGACY_PARAM,
    PARAM_LEGACY_OUTPUT
};

enum NODE_OPTION {
    OPT_ONCE = 1,
    OPT_MUTE = 1 << 1,
    OPT_VIEW = 1 << 2,
    OPT_PREP = 1 << 3,
    OPT_CACHE = 1 << 4
};

enum SOCKET_PROPERTY {
    SOCKPROP_UNKNOWN = 0,
    SOCKPROP_NORMAL = 1,
    SOCKPROP_EDITABLE = 1 << 1,
    SOCKPROP_MULTILINK = 1 << 2,
    SOCKPROP_DICTLIST_PANEL = 1 << 3,
    SOCKPROP_GROUP_LINE = 1 << 4,
    SOCKPROP_LEGACY = 1 << 5,
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
    SOCKET_PROPERTY sockProp;
    QString paramPath;

    PARAM_INFO() : control(CONTROL_NONE), bEnableConnect(false), sockProp(SOCKPROP_NORMAL) {}
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
    QStringList outSockPath;    //option: path for socket.
    QStringList inSockPath;

    EdgeInfo() = default;
    EdgeInfo(const QStringList& outpath, const QStringList& inpath)
        : outSockPath(outpath), inSockPath(inpath) {}

    bool operator==(const EdgeInfo& rhs) const {
        if (outSockPath.size() != rhs.outSockPath.size() || inSockPath.size() != rhs.inSockPath.size())
            return false;
        for (int i = 0; i < outSockPath.size(); i++)
            if (outSockPath[i] != rhs.outSockPath[i])
                return false;
        for (int i = 0; i < inSockPath.size(); i++)
            if (inSockPath[i] != rhs.inSockPath[i])
                return false;
        return true;

    }

    bool isValid() const {
        for (auto& out : outSockPath)
            if (out.isEmpty())
                return false;
        for (auto& in : inSockPath)
            if (in.isEmpty())
                return false;
        return true;

    }
};
Q_DECLARE_METATYPE(EdgeInfo)

struct DICTKEY_INFO
{
    QString key;
    QString netLabel;
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
    QString netlabel;

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
    QPersistentModelIndex retIdx;   //return idx by core.
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

typedef QVector<QString> UI_VECSTRING;

Q_DECLARE_METATYPE(UI_VECSTRING);

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

struct CommandParam
{
    QString name;
    QString description;
    QVariant value;
    bool bIsCommand = false;
    bool operator==(const CommandParam& rhs) const {
        return name == rhs.name && description == rhs.description && value == rhs.value;
    }
    QStringList paramPath;
};
Q_DECLARE_METATYPE(CommandParam)

struct CURVE_RANGE {
    qreal xFrom;
    qreal xTo;
    qreal yFrom;
    qreal yTo;
    bool operator==(const CURVE_RANGE &rhs) const {
        return xFrom == rhs.xFrom && xTo == rhs.xTo && yFrom == rhs.yFrom && yTo == rhs.yTo;
    }
};

struct CURVE_POINT {
    QPointF point;
    QPointF leftHandler;
    QPointF rightHandler;
    int controlType;
    bool operator==(const CURVE_POINT &rhs) const {
                return point == rhs.point && leftHandler == rhs.leftHandler && rightHandler == rhs.rightHandler &&
                       controlType == rhs.controlType;
    }
};

struct CURVE_DATA {
    QString key;
    QVector<CURVE_POINT> points;
    int cycleType = 0;
    CURVE_RANGE rg;
    bool visible;
    bool timeline;
    bool operator==(const CURVE_DATA &rhs) const {
        return key == rhs.key && cycleType == rhs.cycleType && visible == rhs.visible &&
                       timeline == rhs.timeline && rg == rhs.rg && points == rhs.points;
    }

    QVector<int> pointBases() 
    {
        QVector<int> cpbases;
        if (visible) {
            for (auto point : points) {
                cpbases << point.point.x();
            }
        }
        return cpbases;
    }
    float eval(float cf) const {
        if (points.isEmpty())
            return 0;
        QVector<qreal> cpbases;
        for (auto point : points) {
            cpbases << point.point.x();
        }
        if (cycleType != 0) {
            auto delta = cpbases.back() - cpbases.front();
            if (cycleType == 2) {
                int cd;
                if (delta != 0) {
                    cd = int(std::floor((cf - cpbases.front()) / delta)) & 1;
                    cf = std::fmod(cf - cpbases.front(), delta);
                } else {
                    cd = 0;
                    cf = 0;
                }
                if (cd != 0) {
                    if (cf < 0) {
                        cf = -cf;
                    } else {
                        cf = delta - cf;
                    }
                }
                if (cf < 0)
                    cf = cpbases.back() + cf;
                else
                    cf = cpbases.front() + cf;
            } else {
                if (delta != 0)
                    cf = std::fmod(cf - cpbases.front(), delta);
                else
                    cf = 0;
                if (cf < 0)
                    cf = cpbases.back() + cf;
                else
                    cf = cpbases.front() + cf;
            }
        }
        auto moreit = std::lower_bound(cpbases.begin(), cpbases.end(), cf);
        auto cp = cpbases.end();
        if (moreit == cpbases.end())
            return points.back().point.y();
        else if (moreit == cpbases.begin())
           return points.front().point.y();
        auto lessit = std::prev(moreit);
        CURVE_POINT p = points[lessit - cpbases.begin()];
        CURVE_POINT n = points[moreit - cpbases.begin()];
        float pf = *lessit;
        float nf = *moreit;
        float t = (cf - pf) / (nf - pf);
        if (p.controlType != 2) {
           QPointF p1 = QPointF(pf, p.point.y());
           QPointF p2 = QPointF(nf, n.point.y());
           QPointF h1 = QPointF(pf, p.point.y()) + p.rightHandler;
           QPointF h2 = QPointF(nf, n.point.y()) + n.leftHandler;
           return eval_bezier_value(p1, p2, h1, h2, cf);
        } else {
           return lerp(p.point.y(), n.point.y(), t);
        }
    }
    static float lerp(float from, float to, float t) {
        return from + (to - from) * t;
    }

    static QPointF lerp(QPointF from, QPointF to, float t) {
        return {lerp(from.x(), to.x(), t), lerp(from.y(), to.y(), t)};
    }

    static QPointF bezier(QPointF p1, QPointF p2, QPointF h1, QPointF h2, float t) {
        QPointF a = lerp(p1, h1, t);
        QPointF b = lerp(h1, h2, t);
        QPointF c = lerp(h2, p2, t);
        QPointF d = lerp(a, b, t);
        QPointF e = lerp(b, c, t);
        QPointF f = lerp(d, e, t);
        return f;
    }

    static float eval_bezier_value(QPointF p1, QPointF p2, QPointF h1, QPointF h2, float x) {
        float lower = 0;
        float upper = 1;
        float t = (lower + upper) / 2;
        QPointF np = bezier(p1, p2, h1, h2, t);
        int left_calc_count = 100;
        while (std::abs(np.x() - x) > 0.00001f && left_calc_count > 0) {
            if (x < np.x()) {
                upper = t;
            } else {
                lower = t;
            }
            t = (lower + upper) / 2;
            np = bezier(p1, p2, h1, h2, t);
            left_calc_count -= 1;
        }
        return np.y();
    }
};

typedef QMap<QString, CURVE_DATA> CURVES_DATA;
Q_DECLARE_METATYPE(CURVES_DATA);
Q_DECLARE_METATYPE(CURVE_DATA);

typedef QList<QPersistentModelIndex> PARAM_LINKS;
Q_DECLARE_METATYPE(PARAM_LINKS)

#endif
