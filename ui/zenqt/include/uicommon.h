#ifndef __ZENO_UI_COMMON_H__
#define __ZENO_UI_COMMON_H__

#include <zeno/core/typeinfo.h>
#include <QModelIndex>
#include <QSize>
#include <QDockWidget>
#include <QString>
#include <QEvent>
#include <map>
#include <unordered_map>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>
#include <memory>
#include <zeno/core/data.h>
#include "qkeylist.h"
#include <reflect/registry.hpp>


#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
#define QtSkipEmptyParts Qt::SkipEmptyParts
#define qt_unordered_map std::unordered_map
#else

#define QtSkipEmptyParts QString::SkipEmptyParts
#define qt_unordered_map std::map
#endif

constexpr size_t ui_gParamType_Bool = 15698046148323980066ULL;
constexpr size_t ui_gParamType_String = 1350625706064273086ULL;
constexpr size_t ui_gParamType_Int = 3143511548502526014ULL;
constexpr size_t ui_gParamType_Float = 11532138274943533413ULL;
constexpr size_t ui_gParamType_Vec2i = 16965886646643039397ULL;
constexpr size_t ui_gParamType_Vec2f = 6471145251105555636ULL;
constexpr size_t ui_gParamType_Vec3i = 8367839167412710198ULL;
constexpr size_t ui_gParamType_Vec3f = 1291108797552895579ULL;
constexpr size_t ui_gParamType_Vec4i = 724601356907542639ULL;
constexpr size_t ui_gParamType_Vec4f = 5986645728587245802ULL;
constexpr size_t ui_gParamType_PrimVariant = 9597672160759649577ULL;
constexpr size_t ui_gParamType_Curve = 16327668114186180410ULL;

typedef rapidjson::PrettyWriter<rapidjson::StringBuffer> RAPIDJSON_WRITER;

struct LayerOutNode;

struct TIMELINE_INFO
{
    int beginFrame;
    int endFrame;
    int currFrame;
    bool bAlways;
    int timelinefps;

    TIMELINE_INFO() : beginFrame(0), endFrame(0), currFrame(0), bAlways(false), timelinefps(24) {}
};

struct RECORD_SETTING
{
    QString record_path;
    QString videoname;
    int fps;
    int bitrate;
    int numMSAA;
    int numOptix;
    int width;
    int height;
    bool bExportVideo;
    bool needDenoise;
    bool bAutoRemoveCache;
    bool bAov;
    bool bExr;

    RECORD_SETTING() : fps(24), bitrate(200000), numMSAA(0), numOptix(1), width(1280), height(720), bExportVideo(false), needDenoise(false), bAutoRemoveCache(true), bAov(false), bExr(false) {}
};

enum REC_RETURN_CODE
{
    REC_NOERROR = 0,
    REC_NO_RECORD_OPTION = 1,
    REC_OPTIX_INTERNAL_FATAL = -1,
    REC_FFMPEG_FATAL = -3,
    REC_NOFFMPEG = -2,
};

struct LAYOUT_SETTING {
    std::shared_ptr<LayerOutNode> layerOutNode;
    QSize size;
    void(*cbDumpTabsToZsg)(QDockWidget*, RAPIDJSON_WRITER&);
};

struct USERDATA_SETTING
{
    bool optix_show_background;
    USERDATA_SETTING() : optix_show_background(false) {}
};

struct APP_SETTINGS
{
    zeno::TimelineInfo timeline;
    RECORD_SETTING recordInfo;
    LAYOUT_SETTING layoutInfo;
    USERDATA_SETTING userdataInfo;
};

struct RoundRectInfo
{
    QRectF rc;
    qreal W = 0.;
    qreal H = 0.;
    qreal ltradius = 0.;
    qreal lbradius = 0.;
    qreal rtradius = 0.;
    qreal rbradius = 0.;
};

struct ViewMouseInfo
{
    QEvent::Type type = QEvent::None;
    Qt::KeyboardModifiers modifiers;
    Qt::MouseButtons buttons;
    QPointF pos;
    QPoint angleDelta;
};

enum SearchType : unsigned int
{
    SEARCH_SUBNET,
    SEARCH_NODECLS,
    SEARCH_NODEID,
    SEARCH_ANNO,
    SEARCH_ARGS,
    SEARCH_CUSTOM_NAME,
    SEARCHALL = SEARCH_NODECLS | SEARCH_NODEID | SEARCH_SUBNET | SEARCH_ANNO | SEARCH_ARGS | SEARCH_CUSTOM_NAME
};
ENUM_FLAGS(SearchType)

/*
enum SearchType
{
    SEARCH_SUBNET = 1 << 0,
    SEARCH_NODECLS = 1 << 1,
    SEARCH_NODEID = 1 << 2,		// search node ident.
    SEARCH_ANNO = 1 << 3,
    SEARCH_ARGS = 1 << 4,       // all args.
    SEARCH_CUSTOM_NAME = 1 << 5, 
    SEARCHALL = SEARCH_NODECLS | SEARCH_NODEID | SEARCH_SUBNET | SEARCH_ANNO | SEARCH_ARGS | SEARCH_CUSTOM_NAME
};
*/

enum SearchOpt : unsigned int
{
    SEARCH_FUZZ,
    SEARCH_MATCH_EXACTLY,
    SEARCH_CASE_SENSITIVE,
};
ENUM_FLAGS(SearchOpt)

enum SUBGRAPH_TYPE
{
    SUBGRAPH_NOR = 0,
    SUBGRAPH_METERIAL,
    SUBGRAPH_PRESET
};

enum PARAM_TYPE {
    ZPARAM_INT,
    ZPARAM_FLOAT,
    ZPARAM_STRING,
    ZPARAM_VEC3F,
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

class QtRole
{
    Q_GADGET
public:
    explicit QtRole() {}

    enum MODEL_ROLE {
        ROLE_NODE_NAME = Qt::UserRole + 1,  //node name, like `box1`, `cube1`...
        ROLE_CLASS_NAME,    //asset name, or node class name, like `CreateCube`.
        ROLE_NODE_UUID_PATH,
        ROLE_NODE_DISPLAY_NAME,     //the name displayed on node ui.
        ROLE_NODE_CATEGORY,
        ROLE_NODE_DISPLAY_ICON,     //the res path, like `:/icons/add.svg`.
        ROLE_NODE_UISTYLE,          //ui issues, like icon, background color and so on.
        ROLE_NODE_LOCKED,           //only for asset instance.
        ROLE_PARAMS,        //paramsmodel
        ROLE_SUBGRAPH,      //get the subgraph by the subgraph node.
        ROLE_GRAPH,         //get the graph which owns the current node index.
        ROLE_PARAM_NAME,
        ROLE_PARAM_NAME_EXIST,  //for customui edition, to store the exist name of param.
        ROLE_PARAM_VALUE,
        ROLE_PARAM_QML_VALUE,   //the paramvalue which can be recognized by qml.
        ROLE_PARAM_TYPE,
        ROLE_PARAM_RTTICODE,
        ROLE_PARAM_CONTROL,
        ROLE_PARAM_SOCKPROP,
        ROLE_PARAM_CTRL_PROPERTIES,
        ROLE_PARAM_CONTROL_PROPS,   //给QML用，都返回List，现在应该只有comboboxitems和slider的range，未来可能会包括代码编辑器的设置
        ROLE_PARAM_TOOLTIP,
        ROLE_PARAM_SOCKET_VISIBLE,
        ROLE_PARAM_PERSISTENT_INDEX,
        ROLE_PARAM_ENABLE,
        ROLE_PARAM_VISIBLE,
        ROLE_PARAM_SOCKET_CLR,
        ROLE_PARAM_GROUP,
        ROLE_PARAM_IS_WILDCARD,
        ROLE_SOCKET_TYPE,
        ROLE_PARAM_INFO,

        ROLE_VPARAM_TYPE,
        ROLE_ISINPUT,
        ROLE_LINKS,
        ROLE_LINKID,        //a uuid for a specific link
        ROLE_LINK_OBJECT,
        ROLE_OBJPOS,
        ROLE_OBJPATH,
        ROLE_COLLASPED,
        ROLE_INPUTS,
        ROLE_OUTPUTS,
        ROLE_NODE_STATUS,
        ROLE_NODE_ISVIEW,
        ROLE_NODE_BYPASS,
        ROLE_NODE_NOCACHE,
        ROLE_NODE_CLEARSUBNET,
        ROLE_NODE_DIRTY,
        ROLE_NODE_IS_LOADED,
        ROLE_NODE_RUN_STATE,
        ROLE_NODEDATA,
        ROLE_OUTPUT_OBJS,
        ROLE_OUTPUT_VIEWOBJ,

        ROLE_NODEIDX,
        ROLE_LINK_FROM_IDX,
        ROLE_LINK_TO_IDX,
        ROLE_LINK_FROMPARAM_INFO,
        ROLE_LINK_TOPARAM_INFO,
        ROLE_LINK_OUTKEY,
        ROLE_LINK_INKEY,
        ROLE_LINK_INFO,
        ROLE_INSOCK_IDX,
        ROLE_OUTSOCK_IDX,
        ROLE_NODE_IDX,
        ROLE_MTLID,
        ROLE_KEYFRAMES,
        ROLE_NODETYPE,

        //only for Dopnetwork
        ROLE_DOPNETWORK_ENABLECACHE,
        ROLE_DOPNETWORK_CACHETODISK,
        ROLE_DOPNETWORK_MAXMEM,
        ROLE_DOPNETWORK_MEM,

        ROLE_PLUGIN_PATH,
        ROLE_PLUGIN_LOADED,
    };
    Q_ENUM(MODEL_ROLE)
};

class QmlParamGroup
{
    Q_GADGET
public:
    explicit QmlParamGroup() {}

    enum Value {    //对应common.h的NodeDataGroup
        InputObject,
        InputPrimitive,
        OutputObject,
        OutputPrimitive
    };
    Q_ENUM(Value)
};

class RunStatus
{
    Q_GADGET
public:
    explicit RunStatus() {}

    enum Value {
        NoRun,
        Running,
        RunDone,
        RunError
    };
    Q_ENUM(Value)
};

class QmlNodeType
{
    Q_GADGET
public:
    explicit QmlNodeType() {}
    enum Value {
        Node_Normal,
        SubInput,
        SubOutput,
        Node_Group,
        Node_Legacy,
        Node_SubgraphNode,
        Node_Solver,
        Node_AssetInstance,     //the asset node which generated by main graph tree.
        Node_AssetReference,    //the asset node which created in another asset graph.
        NoVersionNode
    };
    Q_ENUM(Value)
};

class QmlParamControl
{
    Q_GADGET
public:
    explicit QmlParamControl() {}

    enum Value {
        NullControl,
        Lineedit,
        Multiline,
        ReadPathEdit,
        WritePathEdit,
        DirectoryPathEdit,
        Combobox,
        Checkbox,
        Vec2edit,
        Vec3edit,
        Vec4edit,
        ColorVec,
        Heatmap,
        CurveEditor,
        SpinBox,
        Slider,
        DoubleSpinBox,
        SpinBoxSlider,
        PythonEditor,
        PushButton,
        Seperator,
        CodeEditor,
        NoMultiSockPanel,   //disable dist/list panel
    };
    Q_ENUM(Value)
};

enum LOG_ROLE
{
    ROLE_LOGTYPE = Qt::UserRole + 1,
    ROLE_TIME,
    ROLE_FILENAME,
    ROLE_LINENO,
    ROLE_NODE_IDENT,
    ROLE_RANGE_START,
    ROLE_RANGE_LEN
};

enum PANEL_TYPE
{
    PANEL_EMPTY,
    PANEL_GL_VIEW,
    PANEL_EDITOR,
    PANEL_NODE_PARAMS,
    PANEL_GEOM_DATA,
    PANEL_LOG,
    PANEL_LIGHT,
    PANEL_IMAGE,
    PANEL_OPTIX_VIEW,
    PANEL_COMMAND_PARAMS,
    PANEL_OPEN_PATH,
    PANEL_QMLPANEL,
    PANEL_QML_GLVIEW,
    PANEL_PYTHON_EXECUTOR
};

enum CUSTOM_PARAM_ROLE {
    ROLE_ELEMENT_TYPE = Qt::UserRole + 1,          //VPARAM_TYPE
    ROLE_MAP_TO_PARAMNAME,                         //recording the existing param name of current param editting item.
};

enum VPARAM_TYPE
{
    VPARAM_ROOT,
    VPARAM_TAB,
    VPARAM_GROUP,
    VPARAM_PARAM,
};

struct SOCKET_DESCRIPTOR
{
    QString name;
    QString type;
    zeno::ParamControl control = zeno::NullControl;
};

struct NODE_DESCRIPTOR
{
    QString name;

    QList<SOCKET_DESCRIPTOR> inputs;
    QList<SOCKET_DESCRIPTOR> outputs;
};

struct NODE_CATE {
    QString name;
    QStringList nodes;
};
typedef QMap<QString, NODE_CATE> NODE_CATES;

typedef QKeyList<QString, zeno::ParamPrimitive> PARAMS_INFO;

typedef QList<QPersistentModelIndex> PARAM_LINKS;

using NodeCates = QMap<QString, QVector<zeno::NodeInfo>>;

class GraphModel;
struct SEARCH_RESULT
{
    SearchType type;
    QModelIndex targetIdx;  //node or subgraph index.
    GraphModel* subGraph;
    QString socket;     //the socket/param which contains the result.
};
struct LiveObjectData{
    std::string verSrc = "";
    std::string camSrc = "";
    int verLoadCount = 0;
    int camLoadCount = 0;
};

struct ZENO_RECORD_RUN_INITPARAM {
    QString sZsgPath = "";
    bool bRecord = false;
    bool bOptix = false;    //is optix view.
    bool isExportVideo = false;
    bool needDenoise = false;
    bool export_exr = false;
    int iFrame = 0;
    int iSFrame = 0;
    int iSample = 0;
    int iBitrate = 0;
    int iFps = 0;
    QString sPixel = "";
    QString sPath = "";
    QString audioPath = "";
    QString configFilePath = "";
    QString videoName = "";
    QString subZsg = "";
    QString paramsJson = "";
    bool exitWhenRecordFinish = false;
};

struct CURVE_RANGE {
    qreal xFrom;
    qreal xTo;
    qreal yFrom;
    qreal yTo;
    bool operator==(const CURVE_RANGE& rhs) const {
        return xFrom == rhs.xFrom && xTo == rhs.xTo && yFrom == rhs.yFrom && yTo == rhs.yTo;
    }
};

struct CURVE_POINT {
    QPointF point;
    QPointF leftHandler;
    QPointF rightHandler;
    int controlType;
    bool bLockX = false;
    bool bLockY = false;
    bool operator==(const CURVE_POINT& rhs) const {
        return point == rhs.point && leftHandler == rhs.leftHandler && rightHandler == rhs.rightHandler &&
            controlType == rhs.controlType;
    }
};

struct CURVE_DATA {
    QString key;
    QVector<CURVE_POINT> points;
    int cycleType = 0;
    CURVE_RANGE rg;
    bool visible = false;
    bool timeline = true;
    bool operator==(const CURVE_DATA& rhs) const {
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
                }
                else {
                    cd = 0;
                    cf = 0;
                }
                if (cd != 0) {
                    if (cf < 0) {
                        cf = -cf;
                    }
                    else {
                        cf = delta - cf;
                    }
                }
                if (cf < 0)
                    cf = cpbases.back() + cf;
                else
                    cf = cpbases.front() + cf;
            }
            else {
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
        }
        else {
            return lerp(p.point.y(), n.point.y(), t);
        }
    }
    static float lerp(float from, float to, float t) {
        return from + (to - from) * t;
    }

    static QPointF lerp(QPointF from, QPointF to, float t) {
        return { lerp(from.x(), to.x(), t), lerp(from.y(), to.y(), t) };
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
            }
            else {
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

struct SLIDER_INFO {
    qreal step;
    qreal min;
    qreal max;
    SLIDER_INFO() : step(1.), min(0.), max(100.) {}
};

struct CommandParam
{
    QString name;
    QString description;
    QVariant value;
    bool bIsCommand = false;
    bool operator==(const CommandParam& rhs) const {
        return name == rhs.name && description == rhs.description && value == rhs.value;
    }
};

typedef QVariantMap CONTROL_PROPERTIES;

inline const char* cPathSeperator = ":";

struct COLOR_RAMP
{
    qreal pos, r, g, b;
    COLOR_RAMP() : pos(0), r(0), g(0), b(0) {}
    COLOR_RAMP(const qreal& pos, const qreal& r, const qreal& g, const qreal& b)
        : pos(pos), r(r), g(g), b(b) {}
};

typedef QVector<COLOR_RAMP> COLOR_RAMPS;
typedef QVector<qreal> UI_VECTYPE;
typedef QVector<QString> UI_VECSTRING;

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

class QmlNodeRunStatus
{
    Q_GADGET
public:
    explicit QmlNodeRunStatus() {}

    enum Value {
        Clean,
        DirtyReadyToRun,
        Pending,
        Running,
        RunError,
        RunSucceed,
        ResultTaken
    };
    Q_ENUM(Value)
};

class QmlParamType : public QObject
{
    Q_OBJECT
public:
    enum Value {
        Unknown = 0,
        Int = 1,
        Float,
        String,
        Vec3f,
        Geometry,
        List,
        Dict,
        IObject
    };
    Q_ENUM(Value)
};
Q_DECLARE_METATYPE(QmlParamType::Value)

#endif
