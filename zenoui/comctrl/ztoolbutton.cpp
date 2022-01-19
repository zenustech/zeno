#include "../style/zenostyle.h"
#include "../style/zstyleoption.h"
#include "ztoolbutton.h"


ZToolButton::ZToolButton(int option, const QIcon& icon, const QSize& iconSize, const QString& text, QWidget* parent)
    : QWidget(parent)
    , m_bDown(false)
    , m_bChecked(false)
    , m_bPressed(false)
    , m_bHideText(false)
    , m_bHovered(false)
    , m_options(option)
    , m_text(text)
    , m_icon(icon)
    , m_iconSize(iconSize)
    , m_font(QFont("Microsoft YaHei", 9))
{
    setMouseTracking(true);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

ZToolButton::~ZToolButton()
{
}

QString ZToolButton::text() const
{
    return m_text;
}

QIcon ZToolButton::icon() const
{
    return m_icon;
}

bool ZToolButton::isPressed() const
{
    return m_bPressed;
}

bool ZToolButton::isHovered() const
{
    return m_bHovered;
}

QSize ZToolButton::iconSize() const
{
    return m_iconSize;
}

bool ZToolButton::isChecked() const
{
    return m_bChecked;
}

bool ZToolButton::isDown() const
{
    return m_bDown;
}

int ZToolButton::buttonOption() const
{
    return m_options;
}

void ZToolButton::setCheckable(bool bCheckable)
{
    if (bCheckable)
        m_options |= Opt_Checkable;
    else
        m_options &= ~Opt_Checkable;
}

QSize ZToolButton::sizeHint() const
{
    int w = 0, h = 0;

    int marginLeft = style()->pixelMetric(static_cast<QStyle::PixelMetric>(ZenoStyle::PM_ButtonLeftMargin), 0, this);
    int marginRight = style()->pixelMetric(static_cast<QStyle::PixelMetric>(ZenoStyle::PM_ButtonRightMargin), 0, this);
    int marginTop = style()->pixelMetric(static_cast<QStyle::PixelMetric>(ZenoStyle::PM_ButtonTopMargin), 0, this);
    int marginBottom = style()->pixelMetric(static_cast<QStyle::PixelMetric>(ZenoStyle::PM_ButtonBottomMargin), 0, this);

    if (!m_text.isEmpty())
    {
        //todo: multi line
        QFontMetrics fontMetrics(m_font);
        int textWidth = fontMetrics.horizontalAdvance(m_text);
        int textHeight = fontMetrics.height();
        //ignore margin between icon and text.
        if (m_options & Opt_UpRight)
        {
            w = qMax(textHeight, m_iconSize.width());
            h = textWidth + m_iconSize.width();
        }
        else if (m_options & Opt_TextUnderIcon)
        {
            w = qMax(textWidth, m_iconSize.width());
            h = textHeight + m_iconSize.height();
        }
        else if (m_options & Opt_TextRightToIcon)
        {
            w = textWidth + m_iconSize.width();
            h = qMax(textHeight, m_iconSize.height());
        }
        else
        {
            //no icon
            w = textWidth;
        }
    }
    else
    {
        w = m_iconSize.width();
        h = m_iconSize.height();
    }

    w += marginLeft + marginRight;
    h += marginTop + marginBottom;
    return QSize(w, h);
}

void ZToolButton::initStyleOption(ZStyleOptionToolButton* option) const
{
    if (!option)
        return;

    option->initFrom(this);
    if (isChecked())
        option->state |= QStyle::State_On;
    if (isDown() || isPressed())
        option->state |= QStyle::State_Sunken;
    if (!isChecked() && !isDown() && !isPressed())
        option->state |= QStyle::State_Raised;
    if (isHovered())
        option->state |= QStyle::State_MouseOver;

    option->state |= QStyle::State_AutoRaise;

    option->icon = icon();
    option->iconSize = iconSize();
    option->text = text();
    option->buttonOpts = buttonOption();
    option->bDown = isDown();
    option->font = m_font;
    option->palette.setBrush(QPalette::All, QPalette::Window, QBrush(backgrondColor(option->state)));
    option->palette.setBrush(QPalette::All, QPalette::WindowText, QColor(160, 160, 160));

    if (!(buttonOption() & (Opt_DownArrow | Opt_RightArrow)))
        option->subControls = QStyle::SC_ToolButton;

    option->hideText = m_bHideText;
}

QBrush ZToolButton::backgrondColor(QStyle::State state) const
{
    if (state & QStyle::State_MouseOver)
    {
        return m_clrBgHover;
    }
    else
    {
        if (state & QStyle::State_On)
        {
            return m_clrBgChecked;
        }
        else if (state & QStyle::State_Sunken)
        {
            return m_clrBgDown;
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

void ZToolButton::initColors(ZStyleOptionToolButton* option) const
{
    option->hoveredBgColor = QColor();
    option->selectedBgColor = QColor();
    option->ActiveBgColor = QColor();
}

void ZToolButton::setIcon(const QIcon& icon)
{
    m_icon = icon;
}

void ZToolButton::setIconSize(const QSize& size)
{
    m_iconSize = size;
}

void ZToolButton::showToolTip()
{
    QPoint pt = QCursor::pos();
    QHelpEvent* e = new QHelpEvent(QEvent::ToolTip, mapFromGlobal(pt), pt);
    QApplication::postEvent(this, e);
}

void ZToolButton::setChecked(bool bChecked)
{
    if (bChecked == m_bChecked)
        return;
    m_bChecked = bChecked;
    update();
}

void ZToolButton::setShortcut(QString text)
{

}

void ZToolButton::setDown(bool bDown)
{
    if (bDown == m_bDown)
        return;
    m_bDown = bDown;
    update();
}

void ZToolButton::setPressed(bool bPressed)
{
    if (bPressed == m_bPressed)
        return;
    m_bPressed = bPressed;
    update();
}

void ZToolButton::setButtonOptions(int options)
{
    m_options = options;
}

bool ZToolButton::event(QEvent* e)
{
    return QWidget::event(e);
}

void ZToolButton::mousePressEvent(QMouseEvent* e)
{
    if (e->button() == Qt::LeftButton)
    {
        setPressed(true);
        emit LButtonPressed();
    }
    else if (e->button() == Qt::RightButton)
    {
        setPressed(false);
        emit RButtonPressed();
    }
}

void ZToolButton::mouseReleaseEvent(QMouseEvent* e)
{
    setPressed(false);
    if (buttonOption() & Opt_Checkable)
    {
        setChecked(!m_bChecked);
    }
    emit clicked();
}

void ZToolButton::setCustomTip(QString tip)
{
    m_customTip = tip;
}

QString ZToolButton::getCustomTip() const
{
    return m_customTip;
}

void ZToolButton::setText(const QString& text)
{
    m_text = text;
}

void ZToolButton::enterEvent(QEvent* e)
{
    m_bHovered = true;
    update();
}

void ZToolButton::leaveEvent(QEvent* e)
{
    m_bHovered = false;
    update();
}

void ZToolButton::updateIcon()
{
    update();
}

void ZToolButton::paintEvent(QPaintEvent* event)
{
    QStylePainter p(this);
    ZStyleOptionToolButton option;
    initStyleOption(&option);
    p.drawComplexControl(static_cast<QStyle::ComplexControl>(ZenoStyle::CC_ZenoToolButton), option);
}