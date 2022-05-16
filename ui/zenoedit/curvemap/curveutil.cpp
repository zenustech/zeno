#include "curveutil.h"
#include <zenoui/model/curvemodel.h>

namespace curve_util
{
	QRectF fitInRange(CURVE_RANGE rg, const QMargins& margins)
	{
		if (rg.xFrom == rg.xTo || rg.yFrom == rg.yTo)
			return QRectF();

		qreal ratio = (rg.yTo - rg.yFrom) / (rg.xTo - rg.xFrom);
		int width = 512;	//todo: sizehint
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
}