#include "ziconbutton.h"
#include "style/zenostyle.h"


ZIconButton::ZIconButton(QIcon icon, const QSize &sz, const QColor &hoverClr, const QColor &selectedClr, bool bCheckable, QWidget *parent)
    : _base(parent)
    , m_icon(icon)
    , m_size(sz)
    , m_bCheckable(bCheckable)
    , m_hover(hoverClr)
    , m_selected(selectedClr)
    , m_state(STATE_NORMAL)
    , m_bChecked(false)
{
    setPixmap(m_icon.pixmap(m_size, QIcon::Normal));
    if (sz.isNull())
        setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    else
        setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

QSize ZIconButton::sizeHint() const
{
    return ZenoStyle::dpiScaledSize(m_size);
}

void ZIconButton::setChecked(bool bChecked)
{
    m_bCheckable = bChecked;
}

void ZIconButton::enterEvent(QEvent* event)
{
    m_state = STATE_HOVERED;
    _base::enterEvent(event);
    update();
}

void ZIconButton::leaveEvent(QEvent* event)
{
    m_state = STATE_NORMAL;
    _base::leaveEvent(event);
    update();
}

void ZIconButton::mousePressEvent(QMouseEvent* event)
{
    _base::mousePressEvent(event);
    m_state = STATE_CLICKED;
    update();
}

void ZIconButton::mouseReleaseEvent(QMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
    if (m_bCheckable)
    {
        m_bChecked = !m_bChecked;
        emit toggled(m_bChecked);
        m_state = STATE_CLICKED;
    }
    else
    {
        if (rect().contains(event->pos()))
        {
            emit clicked();
        }
        m_state = STATE_NORMAL;
    }
    update();
}

void ZIconButton::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);

    QColor clr;
    if (m_state == STATE_HOVERED)
    {
        if (m_hover.isValid())
            painter.fillRect(rect(), m_hover);
    }
    else if (m_state == STATE_CLICKED || m_bChecked)
    {
        if (m_selected.isValid())
            painter.fillRect(rect(), m_selected);
    }
    _base::paintEvent(event);
}