#include "framework.h"
#include "layertreeitemdelegate.h"

LayerTreeitemDelegate::LayerTreeitemDelegate(QWidget* parent)
    : QStyledItemDelegate(parent)
{

}

void LayerTreeitemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    //QStyledItemDelegate::paint(painter, option, index);

    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);

    QRect rc = option.rect;

    //draw icon
    int icon_xmargin = 5;
    int icon_sz = 32;
    int icon_ymargin = (rc.height() - icon_sz) / 2;
    int icon2text_xoffset = 5;
    int button_rightmargin = 10;
    int button_button = 12;
    int text_yoffset = 12;

    QColor bgColor, borderColor;
    if (opt.state & QStyle::State_Selected)
    {
        bgColor = QColor(193, 222, 236);
    }
    else if (opt.state & QStyle::State_MouseOver)
    {
        bgColor = QColor(239, 248, 254);
    }
    else
    {
        bgColor = QColor(255, 255, 255);
    }

    // draw the background
    QRect rcBg(QPoint(0, rc.top()), QPoint(rc.right(), rc.bottom()));
    painter->fillRect(rc, bgColor);

    if (!option.icon.isNull())
    {
        QRect iconRect(opt.rect.x() + icon_xmargin, opt.rect.y() + icon_ymargin, icon_sz, icon_sz);
        QIcon::State state = opt.state & QStyle::State_Open ? QIcon::On : QIcon::Off;
        opt.icon.paint(painter, iconRect, opt.decorationAlignment, QIcon::Normal, state);
    }
    else
    {
        //icon_sz = 0;
        //icon2text_xoffset = 0;
    }

    //draw text
    QFont font("Microsoft YaHei", 9);
    QFontMetricsF fontMetrics(font);
    int w = fontMetrics.horizontalAdvance(opt.text);
    int h = fontMetrics.height();
    int x = opt.rect.x() + icon_xmargin + icon_sz + icon2text_xoffset;
    QRect textRect(x, opt.rect.y(), w, opt.rect.height());
    if (!opt.text.isEmpty())
    {
        painter->setPen(QColor(0, 0, 0));
        painter->setFont(font);
        painter->drawText(textRect, Qt::AlignVCenter, opt.text);
    }

    //draw button
    {
        x = opt.rect.right() - button_rightmargin - button_button - icon_sz * 2;
        int yoffset = 3;
        //x = std::max(x, textRect.right() + 20);
        QRect iconRect(x, opt.rect.y() + yoffset, icon_sz, icon_sz);

        QIcon icon;
        icon.addFile(":/icons/locked.svg", QSize(), QIcon::Normal, QIcon::On);
        icon.addFile(":/icons/locked_off.svg", QSize(), QIcon::Normal, QIcon::Off);
        bool bLocked = index.data(NODELOCK_ROLE).toBool();
        QIcon::State state = bLocked ? QIcon::On : QIcon::Off;
        icon.paint(painter, iconRect, opt.decorationAlignment, QIcon::Normal, state);

        x += icon_sz + button_button;
        iconRect = QRect(x, opt.rect.y() + yoffset, icon_sz, icon_sz);

        icon = QIcon();
        icon.addFile(":/icons/eye.svg", QSize(), QIcon::Normal, QIcon::On);
        icon.addFile(":/icons/eye_off.svg", QSize(), QIcon::Normal, QIcon::Off);

        bool bVisible = index.data(NODELOCK_VISIBLE).toBool();
        state = bVisible ? QIcon::On : QIcon::Off;
        icon.paint(painter, iconRect, opt.decorationAlignment, QIcon::Normal, state);
    }
}

QSize LayerTreeitemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    int w = ((QWidget*)parent())->width();
    return QSize(w, 36);
    //return QStyledItemDelegate::sizeHint(option, index);
}

void LayerTreeitemDelegate::initStyleOption(QStyleOptionViewItem* option, const QModelIndex& index) const
{
    QStyledItemDelegate::initStyleOption(option, index);
}