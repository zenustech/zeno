#include "subnettreeitemdelegate.h"
#include "style/zenostyle.h"
#include <zenoui/util/uihelper.h>


SubnetItemDelegated::SubnetItemDelegated(QWidget* parent)
	: QStyledItemDelegate(parent)
{

}

void SubnetItemDelegated::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
	_base::setModelData(editor, model, index);
}

void SubnetItemDelegated::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	QStyleOptionViewItem opt = option;
	initStyleOption(&opt, index);

	QTreeView* pTreeview = qobject_cast<QTreeView*>(parent());

	painter->save();

	const QWidget* widget = option.widget;

	painter->setClipRect(opt.rect);

	const QAbstractItemView* view = qobject_cast<const QAbstractItemView*>(widget);

	QPalette palette = opt.palette;
	palette.setColor(QPalette::All, QPalette::HighlightedText, palette.color(QPalette::Active, QPalette::Text));
	opt.palette = palette;

	int icon_center_xoffset = 0;	//todo: need icon?

	int iconSize = opt.decorationSize.height();
	int textMargin = ZenoStyle::dpiScaled(5);
	QTextLayout textLayout2(opt.text, opt.font);
	const int maxLineWidth = 8388607; //参照QCommonStylePrivate::viewItemSize
	QSizeF szText = UiHelper::viewItemTextLayout(textLayout2, maxLineWidth);

	int icon_xoffset = icon_center_xoffset - iconSize / 2;
	int icon_yoffset = (opt.rect.height() - iconSize) / 2;
	int text_xoffset = icon_xoffset + iconSize + textMargin;
	int text_yoffset = (opt.rect.height() - szText.height()) / 2;

	QRect iconRect(opt.rect.x() + icon_xoffset, opt.rect.y() + icon_yoffset, iconSize, iconSize);
	QRect textRect(opt.rect.x() + text_xoffset, opt.rect.y() + text_yoffset, szText.width(), szText.height());

	// draw the background

	QColor bgColor, borderColor, textColor;
	if (opt.state & QStyle::State_Selected)
	{
		bgColor = QColor(44, 73, 98);
		borderColor = QColor(27, 145, 225);
		textColor = QColor(255, 255, 255);
	}
	else if (opt.state & QStyle::State_MouseOver)
	{
		textColor = QColor(255, 255, 255);
		bgColor = QColor(43, 43, 43);
	}
	else
	{
		textColor = QColor("#858280");
		bgColor = QColor(43, 43, 43);
	}

	const QPointF oldBrushOrigin = painter->brushOrigin();
	painter->setBrushOrigin(opt.rect.topLeft());
	painter->fillRect(opt.rect, bgColor);
	painter->setBrushOrigin(oldBrushOrigin);

	painter->setPen(QPen(borderColor));
	painter->drawRect(opt.rect.adjusted(0, 0, -1, -1));

	//TODO: 展开项在收缩时也能呈现被选中的状态。

	//展开收缩箭头。
	if (index.model()->hasChildren(index))
		drawExpandArrow(painter, opt);

	// draw the icon
	QIcon::State state = opt.state & QStyle::State_Open ? QIcon::On : QIcon::Off;
	//opt.icon.paint(painter, iconRect, opt.decorationAlignment, QIcon::Normal, state);

	// 添加按钮
	/*
	if (isExpandable && (opt.state & QStyle::State_MouseOver))
	{
		QIcon icon;

		if (pTreeview->GetHoverObj() == MOUSE_IN_ADD)
			icon.addFile(":/icons/add_hover.png");
		else
			icon.addFile(":/icons/add_normal.png");

		iconSize = ZenoStyle::dpiScaled(16);
		int icon_offset = ZenoStyle::dpiScaled(10);

		QRect addiconRect(opt.rect.width() - icon_offset - iconSize,
			opt.rect.y() + icon_offset, iconSize, iconSize);

		if (opt.state & QStyle::State_MouseOver)
			icon.paint(painter, addiconRect, opt.decorationAlignment, QIcon::Normal, state);
	}
	*/

	// draw the text
	if (!opt.text.isEmpty())
	{
		QPalette::ColorGroup cg = opt.state & QStyle::State_Enabled
			? QPalette::Normal : QPalette::Disabled;
		if (cg == QPalette::Normal && !(opt.state & QStyle::State_Active))
			cg = QPalette::Inactive;

		painter->setPen(textColor);

		const int textMargin = ZenoStyle::dpiScaled(2);
		QRect textRect2 = textRect.adjusted(textMargin, 0, -textMargin, 0); // remove width padding
		QTextOption textOption;
		textOption.setWrapMode(QTextOption::ManualWrap);
		textOption.setTextDirection(opt.direction);
		textOption.setAlignment(QStyle::visualAlignment(opt.direction, opt.displayAlignment));

		QPointF paintPosition = textRect2.topLeft();

		QString displayText = opt.text;
		QVector<QTextLayout::FormatRange> selections;

		QTextLayout textLayout(displayText, opt.font);
		textLayout.setTextOption(textOption);
		UiHelper::viewItemTextLayout(textLayout, textRect.width());
		textLayout.draw(painter, paintPosition, selections);
	}
	painter->restore();
}

QSize SubnetItemDelegated::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
	int w = ((QWidget*)parent())->width();
	return ZenoStyle::dpiScaledSize(QSize(w, 25));
}

void SubnetItemDelegated::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
	_base::initStyleOption(option, index);
}

void SubnetItemDelegated::drawExpandArrow(QPainter* painter, const QStyleOptionViewItem& option) const
{
	QTreeView* treeview = (QTreeView*)parent();
	const QModelIndex& index = option.index;

	bool bExpanded = treeview->isExpanded(index);
	QPoint basePt = option.rect.topLeft();

	qreal leftmargin = ZenoStyle::dpiScaled(10),
		height = ZenoStyle::dpiScaled(36),
		bottommargin = ZenoStyle::dpiScaled(13),
		leg = ZenoStyle::dpiScaled(8),
		base_side = ZenoStyle::dpiScaled(10);

	QPainterPath path;
	if (bExpanded)
	{
		bottommargin = ZenoStyle::dpiScaled(12);

		QPoint lb, rb, rt;
		lb.setX(leftmargin);
		lb.setY(height - bottommargin);
		lb += basePt;

		rb.setX(leftmargin + leg);
		rb.setY(height - bottommargin);
		rb += basePt;

		rt.setX(leftmargin + leg);
		rt.setY(height - bottommargin - leg);
		rt += basePt;

		path.moveTo(lb);
		path.lineTo(rb);
		path.lineTo(rt);
		path.lineTo(lb);

		painter->setPen(Qt::NoPen);
		painter->fillPath(path, QBrush(QColor(212, 220, 226)));
	}
	else
	{
		QPointF lt, lb, rp;

		lb.setX(leftmargin);
		lb.setY(height - bottommargin);
		lb += basePt;

		lt.setX(leftmargin);
		lt.setY(height - bottommargin - base_side);
		lt += basePt;

		qreal yyy = (lb.y() + lt.y());
		rp.setY(yyy / 2.0);
		rp.setX(leftmargin + ZenoStyle::dpiScaled(5.0));	//height

		path.moveTo(lb);
		path.lineTo(rp);
		path.lineTo(lt);
		path.lineTo(lb);

		painter->setPen(QPen(QColor(212, 220, 226)));
		painter->drawPath(path);
	}
}