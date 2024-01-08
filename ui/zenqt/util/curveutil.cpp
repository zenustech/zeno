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
        CURVE_DATA curve;
        curve.key = "x";
        curve.cycleType = 0;
        CURVE_RANGE rg;
        rg.xFrom = 0;
        rg.yFrom = 0;
        rg.xTo = 1;
        rg.yTo = 1;
        curve.rg = rg;
        curve.points.append({QPointF(rg.xFrom, rg.yFrom), QPointF(0, 0), QPointF(0, 0), 0});
        curve.points.append({QPointF(rg.xTo, rg.yTo), QPointF(0, 0), QPointF(0, 0), 0});

        CurveModel *pModel = new CurveModel(curve.key, curve.rg, parent);
        pModel->initItems(curve);
        return pModel;
    }

    CURVES_DATA deflCurves() {
        CURVE_DATA curve;
        curve.key = "x";
        curve.cycleType = 0;
        CURVE_RANGE rg;
        rg.xFrom = 0;
        rg.yFrom = 0;
        rg.xTo = 1;
        rg.yTo = 1;
        curve.rg = rg;
        curve.points.append({QPointF(rg.xFrom, rg.yFrom), QPointF(0, 0), QPointF(0, 0), 0});
        curve.points.append({QPointF(rg.xTo, rg.yTo), QPointF(0, 0), QPointF(0, 0), 0});

        CURVES_DATA curves;
        curves[curve.key] = curve;
        return curves;
    }

    void updateRange(CURVES_DATA& curves)
    {
        qreal xFrom = 0;
        qreal xTo = 0;
        qreal yFrom = 0;
        qreal yTo = 0;
        for (auto curve : curves) {
            xFrom = curve.rg.xFrom > xFrom ? xFrom : curve.rg.xFrom;
            xTo = curve.rg.xTo > xTo ? curve.rg.xTo : xTo;
            yFrom = curve.rg.yFrom > yFrom ? yFrom : curve.rg.yFrom;
            yTo = curve.rg.yTo > yTo ? curve.rg.yTo : yTo;
        }
        if (fabs(xFrom - xTo) < 0.00000001)
            xTo = xFrom + 1;
        if (fabs(yFrom - yTo) < 0.00000001)
            yTo = yFrom + 1;
        for (auto& curve : curves) {
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