#include "curveutil.h"
#include "model/curvemodel.h"
#include "zenoapplication.h"
#include "widgets/ztimeline.h"
#include "zenomainwindow.h"
#include "zassert.h"
#include "widgets/zveceditor.h"
#include "util/uihelper.h"


namespace curve_util
{
    QRectF fitInRange(CURVE_RANGE rg, const QMargins& margins)
    {
        if (rg.xFrom == rg.xTo || rg.yFrom == rg.yTo)
            return QRectF();

        qreal ratio = (rg.yTo - rg.yFrom) / (rg.xTo - rg.xFrom);
        int width = 512;    //todo: sizehint
        int height = ratio * width;
        QRectF rc;
        rc.setWidth(width + margins.left() + margins.right());
        rc.setHeight(height + margins.top() + margins.bottom());
        return rc;
    }

    QRectF initGridSize(const QSize& sz, const QMargins& margins)
    {
        QRectF rc;
        rc.setWidth(sz.width() + margins.left() + margins.right());
        rc.setHeight(sz.height() + margins.top() + margins.bottom());
        return rc;
    }

    QModelIndex findUniqueItem(QAbstractItemModel* pModel, int role, QVariant value)
    {
        auto lst = pModel->match(pModel->index(0, 0), role, value);
        if (lst.size() != 1)
            return QModelIndex();
        return lst[0];
    }

    QPair<int, int> numframes(qreal scaleX, qreal scaleY)
    {
        int wtfX = 10 * scaleX * 1;
        int wtfY = 10 * scaleY * 1;
        int nX = 10, nY = 5;
        if (4 <= scaleX && scaleX < 12)
        {
            nX = 20;        
        }
        else if (12 <= scaleX)
        {
            nX = 40;
        }

        if (2 <= scaleY && scaleY < 4)
        {
            nY = 10;
        }
        else if (4 <= scaleY)
        {
            nY = 20;
        } 
        else if (8 <= scaleY)
        {
            nY = 40;
        }
        return {nX, nY};
    }

    CurveModel* deflModel(QObject* parent)
    {
        zeno::CurveData curve;
        zeno::CurveData::Range rg;
        rg.xFrom = 0;
        rg.yFrom = 0;
        rg.xTo = 1;
        rg.yTo = 1;
        curve.rg = rg;

        curve.addPoint(rg.xFrom, rg.yFrom, zeno::CurveData::kBezier, { 0,0 }, { 0,0 }, zeno::CurveData::HDL_VECTOR);
        curve.addPoint(rg.xTo, rg.yTo, zeno::CurveData::kBezier, { 0,0 }, { 0,0 }, zeno::CurveData::HDL_VECTOR);

        CurveModel *pModel = new CurveModel("x", curve.rg, parent);
        pModel->initItems(curve);
        return pModel;
    }

    CURVE_DATA toLegacyCurve(zeno::CurveData curve) {
        CURVE_DATA _curve;
        _curve.cycleType = curve.cycleType;
        _curve.rg.xFrom = curve.rg.xFrom;
        _curve.rg.xTo = curve.rg.xTo;
        _curve.rg.yFrom = curve.rg.yFrom;
        _curve.rg.yTo = curve.rg.yTo;
        _curve.visible = curve.visible;
        _curve.timeline = curve.timeline;
        for (int i = 0; i < curve.cpbases.size(); i++) {
            auto& cp = curve.cpoints[i];
            QPointF pt(curve.cpbases[i], cp.v);
            CURVE_POINT curvept;
            curvept.point = pt;
            curvept.controlType = cp.controlType;
            curvept.leftHandler = { cp.left_handler[0], cp.left_handler[1] };
            curvept.rightHandler = { cp.right_handler[0], cp.right_handler[1] };
            _curve.points.append(curvept);
        }
        return _curve;
    }

    CURVES_DATA toLegacyCurves(zeno::CurvesData curves) {
        CURVES_DATA _curves;
        for (auto& [key, curve] : curves.keys) {
            CURVE_DATA _curve = toLegacyCurve(curve);
            _curve.key = QString::fromStdString(key);
            _curves[QString::fromStdString(key)] = _curve;
        }
        return _curves;
    }

    zeno::CurveData fromLegacyCurve(const CURVE_DATA& _curve) {
        zeno::CurveData curve;
        curve.cycleType = (zeno::CurveData::CycleType)_curve.cycleType;
        curve.rg.xFrom = _curve.rg.xFrom;
        curve.rg.xTo = _curve.rg.xTo;
        curve.rg.yFrom = _curve.rg.yFrom;
        curve.rg.yTo = _curve.rg.yTo;
        curve.visible = _curve.visible;

        for (const CURVE_POINT& cp : _curve.points)
        {
            curve.addPoint(cp.point.x(), cp.point.y(), zeno::CurveData::kBezier,
                { (float)cp.leftHandler.x(), (float)cp.leftHandler.y() },
                { (float)cp.rightHandler.x(), (float)cp.rightHandler.y() }, zeno::CurveData::HDL_VECTOR);
        }
        return curve;
    }

    zeno::CurvesData fromLegacyCurves(const CURVES_DATA& _curves) {
        zeno::CurvesData curves;
        for (QString key : _curves.keys()) {
            std::string sKey = key.toStdString();
            CURVE_DATA _curve = _curves[key];
            zeno::CurveData curve = fromLegacyCurve(_curve);
            curves.keys.insert(std::make_pair(sKey, curve));
        }
        return curves;
    }

    zeno::CurvesData deflCurves() {
        CURVE_DATA curve;
        curve.key = "x";
        curve.cycleType = 0;
        CURVE_RANGE rg;
        rg.xFrom = 0;
        rg.yFrom = 0;
        rg.xTo = 1;
        rg.yTo = 1; 
        curve.rg = rg;
        curve.visible = true;
        curve.points.append({QPointF(rg.xFrom, rg .yFrom), QPointF(0, 0), QPointF(0, 0), 0});
        curve.points.append({QPointF(rg.xTo, rg.yTo), QPointF(0, 0), QPointF(0, 0), 0});

        CURVES_DATA curves;
        curves[curve.key] = curve;
        return fromLegacyCurves(curves);
    }

    void updateRange(zeno::CurvesData& curves)
    {
        qreal xFrom = 0;
        qreal xTo = 0;
        qreal yFrom = 0;
        qreal yTo = 0;
        for (auto& [key, curve] : curves.keys) {
            xFrom = curve.rg.xFrom > xFrom ? xFrom : curve.rg.xFrom;
            xTo = curve.rg.xTo > xTo ? curve.rg.xTo : xTo;
            yFrom = curve.rg.yFrom > yFrom ? yFrom : curve.rg.yFrom;
            yTo = curve.rg.yTo > yTo ? curve.rg.yTo : yTo;
        }
        if (fabs(xFrom - xTo) < 0.00000001)
            xTo = xFrom + 1;
        if (fabs(yFrom - yTo) < 0.00000001)
            yTo = yFrom + 1;
        for (auto& [key, curve] : curves.keys) {
            curve.rg.xFrom = xFrom;
            curve.rg.xTo = xTo;
            curve.rg.yFrom = yFrom;
            curve.rg.yTo = yTo;
        }
    }

    QString getCurveKey(int index)
    {
        switch (index) 
        {
        case 0: return "x";
        case 1: return "y";
        case 2: return "z";
        case 3: return "w";
        default: return "";
        }
    }

    bool updateCurve(const QPointF &newPos, CURVE_DATA& curve)
    {
        bool ret = false;
        for (auto& point : curve.points) 
        {
            if (point.point.y() != newPos.y())
            {
                if (point.point.x() == newPos.x() || curve.points.size() == 1) {
                    point.point.setY(newPos.y());
                    curve.rg.yTo = curve.rg.yTo > newPos.y() ? curve.rg.yTo : newPos.y();
                    curve.rg.yFrom = curve.rg.yFrom > newPos.y() ? newPos.y() : curve.rg.yFrom;
                    ret = true;
                }
            }
        }
        return ret;
    }

    void getDelfCurveData(zeno::CurveData& curve, float y, bool visible, const QString& key) {
        curve.visible = visible;
        zeno::CurveData::Range& rg = curve.rg;
        rg.yFrom = rg.yFrom > y ? y : rg.yFrom;
        rg.yTo = rg.yTo > y ? rg.yTo : y;
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline);
        QPair<int, int> fromTo = timeline->fromTo();
        rg.xFrom = fromTo.first;
        rg.xTo = fromTo.second;
        if (curve.cpoints.empty()) {
            //curve.cycleType = 0;
        }
        float x = timeline->value();
        CURVE_POINT point = { QPointF(x, y), QPointF(0, 0), QPointF(0, 0), HDL_ALIGNED };
        curve.cpbases.push_back(x);

        zeno::CurveData::ControlPoint pt;
        pt.v = y;
        curve.cpoints.push_back(pt);
        updateHandler(curve);
    }

    void updateHandler(zeno::CurveData & _curve)
    {
        CURVE_DATA curve = curve_util::toLegacyCurve(_curve);
        if (curve.points.size() > 1) {
            qSort(curve.points.begin(), curve.points.end(),
                [](const CURVE_POINT& p1, const CURVE_POINT& p2) { return p1.point.x() < p2.point.x(); });
            float preX = curve.points.at(0).point.x();
            for (int i = 1; i < curve.points.size(); i++) {
                QPointF p1 = curve.points.at(i - 1).point;
                QPointF p2 = curve.points.at(i).point;
                float distance = fabs(p1.x() - p2.x());
                float handle = distance * 0.2;
                if (i == 1) {
                    curve.points[i - 1].leftHandler = QPointF(-handle, 0);
                    curve.points[i - 1].rightHandler = QPointF(handle, 0);
                }
                if (p2.y() < p1.y() && (curve.points[i - 1].rightHandler.x() < 0)) {
                    handle = -handle;
                }
                curve.points[i].leftHandler = QPointF(-handle, 0);
                curve.points[i].rightHandler = QPointF(handle, 0);
            }
        }
        _curve = curve_util::fromLegacyCurve(curve);
    }

    void updateTimelineKeys(const QVector<int>& keys)
    {
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline);
        timeline->updateKeyFrames(keys);
    }

    QStringList getKeys(const QObject* obj, QVariant qvar, QWidget* pControl, QLabel* pLabel)
    {
        QStringList keys;
        if (ZLineEdit* lineEdit = qobject_cast<ZLineEdit*>(pControl))     //control float
        {
            keys << "x";
        }
        else if (pLabel == obj)
        {  //control label
            if (qvar.canConvert<UI_VECTYPE>()) {
                UI_VECTYPE vec = qvar.value<UI_VECTYPE>();
                for (int i = 0; i < vec.size(); i++) {
                    QString key = curve_util::getCurveKey(i);
                    if (!key.isEmpty())
                        keys << key;
                }
            }
            else if (qvar.userType() == QMetaTypeId<UI_VECSTRING>::qt_metatype_id())
            {
                bool bValid = false;
                zeno::CurvesData val = UiHelper::getCurvesFromQVar(qvar, &bValid);
                if (val.empty() || !bValid) {
                    return keys;
                }

                QStringList _keys;
                for (auto& [skey, _] : val.keys) {
                    _keys.append(QString::fromStdString(skey));
                }
                keys << _keys;
            }
        }
        else if (ZVecEditor* vecEdit = qobject_cast<ZVecEditor*>(pControl)) //control vec
        {
            int idx = vecEdit->getCurrentEditor();
            QString key = curve_util::getCurveKey(idx);
            if (!key.isEmpty())
                keys << key;
        }
        return keys;
    }

    zeno::CurvesData getCurvesData(const QPersistentModelIndex& perIdx, const QStringList& keys) {
        bool bValid = false;
        const auto& qvar = perIdx.data(ROLE_PARAM_VALUE);
        zeno::CurvesData val = UiHelper::getCurvesFromQVar(qvar, &bValid);
        if (val.empty() || !bValid) {
            return val;
        }


        zeno::CurvesData curves;
        for (auto key : keys) {
            std::string skey = key.toStdString();
            if (val.keys.find(skey) != val.keys.end()) {
                curves.keys[skey] = val.keys[skey];
            }
        }
        return curves;
    }

    int getKeyFrameSize(const zeno::CurvesData& curves)
    {
        int size = 0;
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, false);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, false);
        int frame = timeline->value();
        for (auto& [key, curve] : curves.keys) {
            for (const auto& _x : curve.cpbases) {
                int x = (int)_x;
                if ((x == frame) && curve.visible) {
                    size++;
                    break;
                }
            }
        }
        return size;
    }

    QVector<QString> getKeyFrameProperty(const QVariant& val)
    {
        QVector<QString> ret;
        bool bValid = false;
        zeno::CurvesData curves = UiHelper::getCurvesFromQVar(val, &bValid);
        if (curves.empty() || !bValid)
            return ret;

        ret.resize(curves.size());

        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, ret);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, ret);
        for (int i = 0; i < ret.size(); i++)
        {
            QString property = "null";
            const QString& key = curve_util::getCurveKey(i);
            const std::string& skey = key.toStdString();
            if (curves.contains(skey))
            {
                CURVE_DATA data = curve_util::toLegacyCurve(curves[skey]);
                if (data.visible)
                {
                    property = "false";
                    int x = timeline->value();
                    for (const auto& p : data.points) {
                        int px = p.point.x();
                        if (px == x) {
                            property = "true";
                            break;
                        }
                    }
                }
                curves[skey] = curve_util::fromLegacyCurve(data);
            }
            ret[i] = property;
        }

        if (ret.isEmpty())
            ret << "null";
        return ret;
    }

    bool getCurveValue(QVariant& val)
    {
        bool bValid = false;
        zeno::CurvesData curves = UiHelper::getCurvesFromQVar(val, &bValid);
        if (curves.empty() || !bValid)
            return false;

        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, false);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, false);
        int nFrame = timeline->value();
        if (curves.size() > 1)
        {
            UI_VECTYPE newVal;
            newVal.resize(curves.size());
            for (int i = 0; i < newVal.size(); i++)
            {
                const auto& key = curve_util::getCurveKey(i).toStdString();
                if (curves.contains(key))
                {
                    newVal[i] = curves[key].eval(nFrame);
                }
            }
            val = QVariant::fromValue(newVal);
            return true;
        }
        else if (curves.contains("x") && !curves["x"].cpoints.empty())
        {
            if (curves["x"].cpbases.size() == 1) {
                val = QVariant::fromValue(curves["x"].cpoints[0].v);
            }
            else {
                for (int i = 0; i < curves["x"].cpbases.size(); i++) {
                    if (curves["x"].cpbases[i] == nFrame) {
                        val = QVariant::fromValue(curves["x"].cpoints[i].v);
                        return true;
                    }
                }
                val = QVariant::fromValue(curves["x"].eval(nFrame));
            }
            return true;
        }
        return false;
    }

    bool updateCurve(QVariant oldVal, QVariant& newValue)
    {
        bool bValid = false;
        zeno::CurvesData curves = UiHelper::getCurvesFromQVar(oldVal, &bValid);
        if (curves.empty() || !bValid)
            return true;

        bool bUpdate = false;
        UI_VECTYPE datas;
        //vec
        if (newValue.canConvert<UI_VECTYPE>())
        {
            datas = newValue.value<UI_VECTYPE>();
        }
        //float
        else
        {
            datas << newValue.toFloat();
        }
        UI_VECSTRING oldVec;
        if (oldVal.canConvert<UI_VECSTRING>())
        {
            oldVec = oldVal.value<UI_VECSTRING>();
        }
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, false);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, false);
        int nFrame = timeline->value();
        for (int i = 0; i < datas.size(); i++) {
            auto key = curve_util::getCurveKey(i).toStdString();
            if (curves.contains(key))
            {
                QPointF pos(nFrame, datas.at(i));
                CURVE_DATA _curve = curve_util::toLegacyCurve(curves[key]);
                if (curve_util::updateCurve(pos, _curve)) {
                    curves[key] = curve_util::fromLegacyCurve(_curve);
                    bUpdate = true;
                }
            }
            else if (oldVec.size() > i && oldVec[i].toFloat() != datas[i])
            {
                bUpdate = true;
                oldVec[i] = QString::number(datas[i]);
                oldVal = QVariant::fromValue(oldVec);
            }
        }
        if (bUpdate)
        {
            const auto& anyVal = zeno::reflect::make_any<zeno::CurvesData>(curves);
            newValue = QVariant::fromValue(anyVal);
        }
        return bUpdate;
    }
}