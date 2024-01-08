#include "zlabel.h"


ZDebugLabel::ZDebugLabel(QWidget* parent)
    : QLabel(parent)
{
}

ZDebugLabel::ZDebugLabel(const QString& text, QWidget* parent)
    : QLabel(text, parent)
    , m_text(text)
{
}

void ZDebugLabel::adjustText(const QString& text)
{
    m_text = text;
}



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

void ZIconLabel::setIcons(const QString& iconIdle, const QString& iconLight)
{
    QIcon ic(iconIdle);
    QPixmap px(iconIdle);
    m_iconSz = px.size();
    m_icon.addFile(iconIdle, QSize(), QIcon::Normal, QIcon::Off);
    m_icon.addFile(iconLight, QSize(), QIcon::Active, QIcon::Off);
    m_bToggleable = false;
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
    , m_bUnderlineHover(false)
    , m_bUnderline(false)
    , m_hoverCursor(Qt::PointingHandCursor)
{
    setMouseTracking(true);
}

ZTextLabel::ZTextLabel(const QString& text, QWidget* parent)
    : QLabel(text, parent)
    , m_bUnderlineHover(false)
    , m_bUnderline(false)
    , m_hoverCursor(Qt::PointingHandCursor)
{
    setMouseTracking(true);
}

void ZTextLabel::setHoverCursor(Qt::CursorShape shape)
{
    m_hoverCursor = shape;
}

void ZTextLabel::setUnderline(bool bUnderline)
{
    m_bUnderline = bUnderline;
}

void ZTextLabel::setUnderlineOnHover(bool bUnderline)
{
    m_bUnderlineHover = bUnderline;
}

void ZTextLabel::setTransparent(bool btransparent)
{
    //setAttribute(Qt::WA_OpaquePaintEvent);
    //todo: transparent
    //if (btransparent) {
    //    setStyleSheet("background-color: rgba(0,0,0,0%)");
    //}
    //else {
    //    setStyleSheet("background-color: rgba(0,0,0,100%)");
    //}
    //QWidget::setAttribute(Qt::WA_TranslucentBackground, btransparent);
}

void ZTextLabel::setEnterCursor(Qt::CursorShape shape)
{
    m_hoverCursor = shape;
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

    if (m_bUnderlineHover || m_bUnderline)
    {
        QFont fnt = this->font();
        fnt.setUnderline(true);
        setFont(fnt);    
    }

    setCursor(m_hoverCursor);
}

void ZTextLabel::leaveEvent(QEvent* event)
{
    QPalette pal = palette();
    pal.setColor(QPalette::WindowText, m_normal);
    setPalette(pal);

    if (m_bUnderlineHover)
    {
        QFont fnt = this->font();
        fnt.setUnderline(false);
        setFont(fnt);
    }

    setCursor(Qt::ArrowCursor);
}

void ZTextLabel::mouseReleaseEvent(QMouseEvent* event)
{
    QLabel::mouseReleaseEvent(event);
    if (event->button() == Qt::LeftButton)
        emit clicked();
    else if (event->button() == Qt::RightButton)
        emit rightClicked();
}
