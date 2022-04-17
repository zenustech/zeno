#include "curveutil.h"

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
        return {wtfX, wtfY};
	}
}