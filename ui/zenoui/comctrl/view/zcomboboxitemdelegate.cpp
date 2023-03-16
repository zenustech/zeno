#include "zcomboboxitemdelegate.h"
#include "style/zenostyle.h"
#include <QtSvg/QSvgRenderer>


ZComboBoxItemDelegate2::ZComboBoxItemDelegate2(QObject* parent)
    : QStyledItemDelegate(parent)
{
}

// painting
void ZComboBoxItemDelegate2::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    QStyleOptionViewItem opt = option;

    QStyledItemDelegate::initStyleOption(&opt, index);
    opt.icon = QIcon(":/icons/checked.svg");

    QComboBox* pCombobox = qobject_cast<QComboBox*>(parent());
    bool bCurrent = pCombobox->currentIndex() == index.row();

    opt.backgroundBrush.setStyle(Qt::SolidPattern);
    QColor foreground = bCurrent ? QColor("#FFFFFF") : QColor("#C3D2DF");
    if (opt.state & QStyle::State_MouseOver)
    {
        opt.backgroundBrush.setColor(QColor("#3B3E45"));
    }
    else
    {
        opt.backgroundBrush.setColor(QColor("#191D21"));
    }

    painter->fillRect(opt.rect, opt.backgroundBrush);

    if (bCurrent)
    {
        qreal icon_xmargin = ZenoStyle::dpiScaled(8);
        qreal icon_ymargin = ZenoStyle::dpiScaled(4);
        qreal iconSz = ZenoStyle::dpiScaled(16);
        QRect iconRect(opt.rect.x() + icon_xmargin, opt.rect.y() + icon_ymargin, iconSz, iconSz);

        QIcon::State state = opt.state & QStyle::State_Open ? QIcon::On : QIcon::Off;
        opt.icon.paint(painter, iconRect, opt.decorationAlignment, QIcon::Normal, state);
    }

    painter->setPen(QPen(foreground));
    painter->drawText(opt.rect.adjusted(ZenoStyle::dpiScaled(32), 0, 0, 0), opt.text, QTextOption(Qt::AlignVCenter));
}

QSize ZComboBoxItemDelegate2::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    int w = ((QWidget*)parent())->width();
    return QSize(w, ZenoStyle::dpiScaled(24));
}