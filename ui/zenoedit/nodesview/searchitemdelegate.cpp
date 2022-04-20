#include "searchitemdelegate.h"
#include <zenoui/style/zenostyle.h>
#include "zenoapplication.h"


SearchItemDelegate::SearchItemDelegate(const QString& search, QObject* parent)
    : QStyledItemDelegate(parent)
	, m_search(search)
{
}

QSize SearchItemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    return QStyledItemDelegate::sizeHint(option, index);
}

void SearchItemDelegate::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);
}

static QVector<QTextLayout::FormatRange> _getSearchFormatRange(const QString& content, const QString& searchText)
{
	QVector<QTextLayout::FormatRange> selections;
	int idx = 0;
	while (true)
	{
		idx = content.indexOf(searchText, idx, Qt::CaseInsensitive);
		if (idx == -1)
			break;
		QTextLayout::FormatRange frg;
		frg.start = idx;
		frg.length = searchText.length();
		frg.format.setBackground(QColor(41, 76, 45));
		selections.push_back(frg);
		idx++;
	}
	return selections;
}

void SearchItemDelegate::paint(QPainter* p, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    //QStyledItemDelegate::paint(painter, option, index);

	QStyleOptionViewItem opt = option;
	initStyleOption(&opt, index);

	const QWidget* widget = option.widget;

	p->save();
	p->setClipRect(opt.rect);

	QRect checkRect = zenoApp->style()->subElementRect(QStyle::SE_ItemViewItemCheckIndicator, &opt, widget);
	QRect iconRect = zenoApp->style()->subElementRect(QStyle::SE_ItemViewItemDecoration, &opt, widget);
	QRect textRect = zenoApp->style()->subElementRect(QStyle::SE_ItemViewItemText, &opt, widget);

	// draw the background
	zenoApp->style()->drawPrimitive(QStyle::PE_PanelItemViewItem, &opt, p, widget);

	// draw the check mark
	if (opt.features & QStyleOptionViewItem::HasCheckIndicator) {
		QStyleOptionViewItem option(opt);
		option.rect = checkRect;
		option.state = option.state & ~QStyle::State_HasFocus;

		switch (opt.checkState) {
		case Qt::Unchecked:
			option.state |= QStyle::State_Off;
			break;
		case Qt::PartiallyChecked:
			option.state |= QStyle::State_NoChange;
			break;
		case Qt::Checked:
			option.state |= QStyle::State_On;
			break;
		}
		zenoApp->style()->drawPrimitive(QStyle::PE_IndicatorItemViewItemCheck, &option, p, widget);
	}

	// draw the icon
	QIcon::Mode mode = QIcon::Normal;
	if (!(opt.state & QStyle::State_Enabled))
		mode = QIcon::Disabled;
	else if (opt.state & QStyle::State_Selected)
		mode = QIcon::Selected;
	QIcon::State state = opt.state & QStyle::State_Open ? QIcon::On : QIcon::Off;
	opt.icon.paint(p, iconRect, opt.decorationAlignment, mode, state);

	// draw the text
	if (!opt.text.isEmpty()) {
		QPalette::ColorGroup cg = opt.state & QStyle::State_Enabled
			? QPalette::Normal : QPalette::Disabled;
		if (cg == QPalette::Normal && !(opt.state & QStyle::State_Active))
			cg = QPalette::Inactive;

		if (opt.state & QStyle::State_Selected) {
			p->setPen(opt.palette.color(cg, QPalette::HighlightedText));
		}
		else {
			p->setPen(opt.palette.color(cg, QPalette::Text));
		}
		if (opt.state & QStyle::State_Editing) {
			p->setPen(opt.palette.color(cg, QPalette::Text));
			p->drawRect(textRect.adjusted(0, 0, -1, -1));
		}

		QTextLayout textLayout(opt.text, QFont("HarmonyOS Sans", 10));

		textLayout.beginLayout();
		QTextLine line = textLayout.createLine();
		line.setLineWidth(textRect.width());
		line.setPosition(QPointF(0, 0));
		textLayout.endLayout();

		QVector<QTextLayout::FormatRange> selections = _getSearchFormatRange(opt.text, m_search);

		p->setPen(QColor("#858280"));
		textLayout.draw(p, textRect.topLeft(), selections);
	}

	// draw the focus rect
	if (opt.state & QStyle::State_HasFocus) {
		QStyleOptionFocusRect o;
		o.QStyleOption::operator=(opt);
		o.rect = zenoApp->style()->subElementRect(QStyle::SE_ItemViewItemFocusRect, &opt, widget);
		o.state |= QStyle::State_KeyboardFocusChange;
		o.state |= QStyle::State_Item;
		QPalette::ColorGroup cg = (opt.state & QStyle::State_Enabled)
			? QPalette::Normal : QPalette::Disabled;
		o.backgroundColor = opt.palette.color(cg, (opt.state & QStyle::State_Selected)
			? QPalette::Highlight : QPalette::Window);
		zenoApp->style()->drawPrimitive(QStyle::PE_FrameFocusRect, &o, p, widget);
	}

	p->restore();
}
