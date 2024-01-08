#include "zobjectbutton.h"

ZObjectButton::ZObjectButton(const QIcon& icon, const QString& text, QWidget* parent)
    : ZToolButton(
        ZToolButton::Opt_HasIcon | ZToolButton::Opt_HasText | ZToolButton::Opt_TextUnderIcon,
        icon,
        QSize(24, 24),
        text,
        parent
        )
{
    QPalette palette;
    palette.setColor(QPalette::WindowText, QColor(0, 0, 0));
    setPalette(palette);
}

ZObjectButton::~ZObjectButton()
{

}

QBrush ZObjectButton::backgrondColor(QStyle::State state) const
{
    if (state & QStyle::State_MouseOver)
    {
        if (state & QStyle::State_Sunken)
        {
            return QBrush(QColor(46, 46, 46));
        }
        return QBrush(QColor(62, 62, 62));
    }
    else
    {
        if (state & QStyle::State_On)
        {
            return QBrush(QColor(109, 125, 131));
        }
        else if (state & QStyle::State_Sunken)
        {
            return QBrush(QColor(46, 46, 46));
        }
        else if (state & QStyle::State_Raised)
        {
            return QBrush();	//transparent?
        }
        else
        {
            return QBrush();
        }
    }
}



ZMiniToolButton::ZMiniToolButton(const QIcon& icon, QWidget* parent)
    : ZToolButton(
        ZToolButton::Opt_HasIcon,
        icon,
        QSize(16, 16),
        QString(),
        parent
    )
{
    QPalette palette;
    palette.setColor(QPalette::WindowText, QColor(0, 0, 0));
    setPalette(palette);
}

QBrush ZMiniToolButton::backgrondColor(QStyle::State state) const
{
    if (state & QStyle::State_MouseOver)
    {
        if (state & QStyle::State_Sunken)
        {
            return QBrush(QColor(46, 46, 46));
        }
        return QBrush(QColor(147, 147, 147));
    }
    else
    {
        if (state & QStyle::State_On)
        {
            return QBrush(QColor(109, 125, 131));
        }
        else if (state & QStyle::State_Sunken)
        {
            return QBrush(QColor(46, 46, 46));
        }
        else if (state & QStyle::State_Raised)
        {
            return QBrush();	//transparent?
        }
        else
        {
            return QBrush();
        }
    }
}