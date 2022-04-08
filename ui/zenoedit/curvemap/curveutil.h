#ifndef __CURVE_UTIL_H__
#define __CURVE_UTIL_H__

#include <zenoui/model/modeldata.h>

namespace curve_util
{
	enum ItemType
	{
		ITEM_NODE,
		ITEM_LEFTHANDLE,
		ITEM_RIGHTHANDLE,
		ITEM_CURVE,
	};

	enum ItemStatus
	{
		ITEM_UNTOGGLED,
		ITEM_TOGGLED,
		ITEM_SELECTED,
	};

	enum ROLE_CURVE
	{
		ROLE_ItemType = Qt::UserRole + 1,
		ROLE_ItemObjId,
		ROLE_ItemBelongTo,
		ROLE_ItemPos,
		ROLE_ItemStatus,
		ROLE_MouseClicked,
		ROLE_CurveLeftNode,
		ROLE_CurveRightNode
	};

	QRectF fitInRange(CURVE_RANGE rg, const QMargins& margins);
	QModelIndex findUniqueItem(QAbstractItemModel* pModel, int role, QVariant value);
	QPair<int, int> numframes(qreal scaleX, qreal scaleY);
}

#endif