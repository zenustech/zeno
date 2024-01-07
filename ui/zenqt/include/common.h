#ifndef __ZENO_UI_COMMON_H__
#define __ZENO_UI_COMMON_H__

#include <QModelIndex>
#include <QSize>
#include <QDockWidget>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>
#include <memory>
#include <zeno/core/data.h>

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
    //todo: other settings.
};

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

enum SearchOpt
{
    SEARCH_FUZZ = 1 << 0,
    SEARCH_MATCH_EXACTLY = 1 << 1,
    SEARCH_CASE_SENSITIVE= 1 << 2,
};

struct SEARCH_RESULT
{
    SearchType type;
    QModelIndex targetIdx;  //node or subgraph index.
    QModelIndex subgIdx;
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
    bool visible;
    bool timeline;
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
Q_DECLARE_METATYPE(CURVES_DATA);
Q_DECLARE_METATYPE(CURVE_DATA);

struct SLIDER_INFO {
    qreal step;
    qreal min;
    qreal max;
    SLIDER_INFO() : step(1.), min(0.), max(100.) {}
};
Q_DECLARE_METATYPE(SLIDER_INFO)

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
Q_DECLARE_METATYPE(COLOR_RAMPS)

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


#endif
