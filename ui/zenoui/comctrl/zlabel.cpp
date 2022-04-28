#include "zlabel.h"


ZIconLabel::ZIconLabel(QWidget* parent)
    : QLabel(parent)
    , m_bToggled(false)
    , m_bHovered(false)
    , m_bClicked(false)
    , m_bToggleable(false)
{
    setStyleSheet("background: transparent");
}

void ZIconLabel::setIcons(const QSize& sz, const QString& iconEnable, const QString& iconHover, const QString& iconNormalOn, const QString& iconHoverOn, const QString& iconDisable)
{
    m_iconSz = sz;
    m_icon.addFile(iconEnable, QSize(), QIcon::Normal, QIcon::Off);
    m_icon.addFile(iconHover, QSize(), QIcon::Active, QIcon::Off);
    m_icon.addFile(iconNormalOn, QSize(), QIcon::Normal, QIcon::On);
    m_icon.addFile(iconHoverOn, QSize(), QIcon::Active, QIcon::On);
    m_icon.addFile(iconDisable, QSize(), QIcon::Disabled, QIcon::Off);

    if (!iconNormalOn.isEmpty())
        m_bToggleable = true;

    setPixmap(m_icon.pixmap(m_iconSz, QIcon::Normal));
    setFixedSize(m_iconSz);
}

void ZIconLabel::toggle(bool bToggle)
{
    if (m_bToggleable)
    {
        m_bToggled = bToggle;
        updateIcon();
    }
}

void ZIconLabel::enterEvent(QEvent* event)
{
    m_bHovered = true;
    updateIcon();
}

void ZIconLabel::leaveEvent(QEvent* event)
{
    m_bHovered = false;
    updateIcon();
}

void ZIconLabel::paintEvent(QPaintEvent* e)
{
    QLabel::paintEvent(e);
}

void ZIconLabel::mouseReleaseEvent(QMouseEvent* event)
{
    QLabel::mouseReleaseEvent(event);
    if (m_bToggleable)
    {
        m_bToggled = !m_bToggled;
    }
    updateIcon();
    if (m_bToggleable)
    {
        emit toggled(m_bToggled);
    }
    emit clicked();
}

void ZIconLabel::onClicked()
{

}

void ZIconLabel::updateIcon()
{
    if (isEnabled())
    {
        setPixmap(m_icon.pixmap(m_iconSz, m_bHovered ? QIcon::Active : QIcon::Normal, m_bToggled ? QIcon::On : QIcon::Off));
    }
    else
    {
        setPixmap(m_icon.pixmap(m_iconSz, QIcon::Disabled));
    }
    update();
}

void ZIconLabel::onToggled()
{
}



ZTextLabel::ZTextLabel(QWidget* parent)
    : QLabel(parent)
{
    setMouseTracking(true);
}

ZTextLabel::ZTextLabel(const QString& text, QWidget* parent)
    : QLabel(text, parent)
{
    setMouseTracking(true);
}

void ZTextLabel::setTextColor(const QColor& clr)
{
    m_normal = clr;
    QPalette pal = palette();
    pal.setColor(QPalette::WindowText, m_normal);
    setPalette(pal);
}

void ZTextLabel::setBackgroundColor(const QColor& clr)
{
    QPalette pal = palette();
    pal.setColor(backgroundRole(), QColor(56, 57, 56));
    setAutoFillBackground(true);
    setPalette(pal);
}

void ZTextLabel::enterEvent(QEvent* event)
{
    QPalette pal = palette();
    pal.setColor(QPalette::WindowText, QColor(255, 255, 255));
    setPalette(pal);
    setCursor(Qt::PointingHandCursor);
}

void ZTextLabel::leaveEvent(QEvent* event)
{
    QPalette pal = palette();
    pal.setColor(QPalette::WindowText, m_normal);
    setPalette(pal);
    setCursor(Qt::ArrowCursor);
}

void ZTextLabel::mouseReleaseEvent(QMouseEvent* event)
{
    QLabel::mouseReleaseEvent(event);
    emit clicked();
}