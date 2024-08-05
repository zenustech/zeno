#include "curveutil.h"
#include "model/curvemodel.h"


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
}