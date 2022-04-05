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


}