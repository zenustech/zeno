#include "zlabel.h"


ZLabel::ZLabel(QWidget* parent)
    : QLabel(parent)
    , m_bToggled(false)
    , m_bHovered(false)
    , m_bClicked(false)
    , m_bToggleable(false)
{
}

void ZLabel::setIcons(const QSize& sz, const QString& iconEnable, const QString& iconHover, const QString& iconNormalOn, const QString& iconHoverOn, const QString& iconDisable)
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

void ZLabel::enterEvent(QEvent* event)
{
    m_bHovered = true;
    updateIcon();
}

void ZLabel::leaveEvent(QEvent* event)
{
    m_bHovered = false;
    updateIcon();
}

void ZLabel::paintEvent(QPaintEvent* e)
{
    QLabel::paintEvent(e);
}

void ZLabel::mouseReleaseEvent(QMouseEvent* event)
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
}

void ZLabel::onClicked()
{

}

void ZLabel::updateIcon()
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

void ZLabel::onToggled()
{
}
