#include "../style/zenostyle.h"
#include "../style/zstyleoption.h"
#include "ztoolbutton.h"
#include <zenoedit/zenoapplication.h>


ZToolButton::ZToolButton(QWidget* parent)
    : QWidget(parent)
    , m_bDown(false)
    , m_bChecked(false)
    , m_bPressed(false)
    , m_bHideText(false)
    , m_bHovered(false)
    , m_radius(0)
{
    initDefault();
    setMouseTracking(true);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    setCursor(QCursor(Qt::PointingHandCursor));
}

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
    , m_radius(0)
{
    initDefault();
    setMouseTracking(true);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    setCursor(QCursor(Qt::PointingHandCursor));
    QFont font = zenoApp->font();
    font.setPointSize(9);
    m_font = font;
}

ZToolButton::ZToolButton(int option, const QString& icon, const QString& iconOn, const QSize& iconSize, const QString& text, QWidget* parent)
    : QWidget(parent)
    , m_bDown(false)
    , m_bChecked(false)
    , m_bPressed(false)
    , m_bHideText(false)
    , m_bHovered(false)
    , m_options(option)
    , m_text(text)
    , m_icon(QIcon(icon))
    , m_iconSize(iconSize)
    , m_radius(0)
{
    initDefault();
    setIcon(m_iconSize, icon, "", iconOn, "");
    setMouseTracking(true);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    setCursor(QCursor(Qt::PointingHandCursor));
    QFont font = zenoApp->font();
    font.setPointSize(9);
    m_font = font;
}

ZToolButton::~ZToolButton()
{
}

void ZToolButton::initDefault()
{
    m_clrText = m_clrTextOn = m_clrTextHover = m_clrTextOnHover = QColor(160, 160, 160);
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

QMargins ZToolButton::margins() const
{
    return m_margins;
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

    int marginLeft = 0, marginRight = 0, marginTop = 0, marginBottom = 0;

    marginLeft = m_margins.left();
    marginRight = m_margins.right();
    marginTop = m_margins.top();
    marginBottom = m_margins.bottom();

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
        else if (m_options & (Opt_TextRightToIcon | Opt_TextLeftToIcon))
        {
            w = textWidth + m_iconSize.width();
            w += style()->pixelMetric(static_cast<QStyle::PixelMetric>(ZenoStyle::PM_IconTextSpacing), nullptr, this);
            h = qMax(textHeight, m_iconSize.height());
        }
        else
        {
            //no icon
            w = textWidth;
            h = textHeight;
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
    if (m_options & Opt_NoBackground)
        option->bDrawBackground = false;
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
    option->bgRadius = m_radius;
    option->palette.setBrush(QPalette::All, QPalette::Window, QBrush(backgrondColor(option->state)));
    option->palette.setBrush(QPalette::All, QPalette::WindowText, textColor(option->state));

    if (!(buttonOption() & (Opt_DownArrow | Opt_RightArrow)))
        option->subControls = QStyle::SC_ToolButton;

    option->hideText = m_bHideText;
}

QBrush ZToolButton::backgrondColor(QStyle::State state) const
{
    if (state & QStyle::State_MouseOver)
    {
        if (state & QStyle::State_On)
        {
            return m_clrBgOnHovered;
        }
        else
        {
            return m_clrBgNormalHover;
        }
    }
    else
    {
        if (state & QStyle::State_On)
        {
            return m_clrBgOn;
        }
        else
        {
            return m_clrBgNormal;
        }
    }
}

QBrush ZToolButton::textColor(QStyle::State state) const
{
    if (state & (QStyle::State_On | QStyle::State_MouseOver))
    {
        return m_clrTextOnHover;
    }
    else if (state & QStyle::State_MouseOver)
    {
        return m_clrTextHover;
    }
    else if (state & QStyle::State_On)
    {
        return m_clrTextOn;
    }
    else
    {
        return m_clrText;
    }
}

void ZToolButton::setBackgroundClr(const QColor& normalClr, const QColor& hoverClr, const QColor& downClr, const QColor& checkedClr)
{
    m_clrBgNormal = normalClr;
    m_clrBgNormalHover = hoverClr;
    m_clrBgOn = downClr;
    m_clrBgOnHovered = checkedClr;
}

void ZToolButton::setTextClr(const QColor& normal, const QColor& hover, const QColor& normalOn, const QColor& hoverOn)
{
    m_clrText = normal;
    m_clrTextHover = hover;
    m_clrTextOn = normalOn;
    m_clrTextOnHover = hoverOn;
}

void ZToolButton::initColors(ZStyleOptionToolButton* option) const
{
    option->hoveredBgColor = QColor();
    option->selectedBgColor = QColor();
    option->ActiveBgColor = QColor();
}

void ZToolButton::setIcon(const QSize& size, QString icon, QString iconHover, QString iconOn, QString iconOnHover)
{
    if (size.isValid())
    {
        m_iconSize = size;
    }
    else
    {
        QPixmap px(icon);
        m_iconSize = px.size();
    }
    if (iconHover.isEmpty())
        iconHover = icon;
    if (iconOnHover.isEmpty())
        iconOnHover = iconOn;
    m_icon.addFile(icon, m_iconSize, QIcon::Normal, QIcon::Off);
    m_icon.addFile(iconHover, m_iconSize, QIcon::Active, QIcon::Off);
    m_icon.addFile(iconHover, m_iconSize, QIcon::Selected, QIcon::Off);
    m_icon.addFile(iconOn, m_iconSize, QIcon::Normal, QIcon::On);
    m_icon.addFile(iconOnHover, m_iconSize, QIcon::Active, QIcon::On);
    m_icon.addFile(iconOnHover, m_iconSize, QIcon::Selected, QIcon::On);
}

void ZToolButton::setFont(const QFont& font)
{
    m_font = font;
}

void ZToolButton::initAnimation() {
    m_radius = this->height() / 2;
    animInfo.BtnWidth = m_iconSize.width();
    float border = (this->height() - animInfo.BtnWidth) / 2;
    animInfo.mOnOff = false;
    animInfo.m_LeftPos = QPoint(border, border);
    animInfo.m_RightPos = QPoint(this->width() - 4 * border - animInfo.BtnWidth, border);
    animInfo.mButtonRect.setWidth(animInfo.BtnWidth);
    animInfo.mButtonRect.setHeight(animInfo.BtnWidth);
    animInfo.mButtonRect.moveTo(animInfo.mOnOff ? animInfo.m_RightPos : animInfo.m_LeftPos);

    animInfo.mBackColor = animInfo.mOnOff ? m_clrBgOn : m_clrBgNormal;
    animInfo.mAnimationPeriod = 100;
    animInfo.posAnimation = new QVariantAnimation(this);
    animInfo.posAnimation->setDuration(animInfo.mAnimationPeriod);
    connect(animInfo.posAnimation, &QVariantAnimation::valueChanged, [=](const QVariant &value) {
        animInfo.mButtonRect.moveTo(value.toPointF());
        update();
    });
}

void ZToolButton::setIconSize(const QSize& size)
{
    m_iconSize = size;
}

void ZToolButton::setMargins(const QMargins& margins)
{
    m_margins = margins;
}

void ZToolButton::setRadius(int radius)
{
    m_radius = radius;
}

void ZToolButton::setChecked(bool bChecked)
{
    if (bChecked == m_bChecked)
        return;
    m_bChecked = bChecked;
    update();
}

void ZToolButton::setShortcut(QKeySequence text)
{
    QShortcut *shortcut = new QShortcut(text, this);
    connect(shortcut, &QShortcut::activated, this, &ZToolButton::clicked);
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
        emit toggled(m_bChecked);
    }
    else if (m_options & Opt_SwitchAnimation)
    {
        animInfo.posAnimation->setStartValue(animInfo.mOnOff ? animInfo.m_RightPos
                                                                       : animInfo.m_LeftPos);
        animInfo.posAnimation->setEndValue(animInfo.mOnOff ? animInfo.m_LeftPos
                                                                     : animInfo.m_RightPos);
        animInfo.posAnimation->start(QAbstractAnimation::DeletionPolicy::KeepWhenStopped); //Í£Ö¹ºóÉ¾³ý
        animInfo.mOnOff = !animInfo.mOnOff;
        animInfo.mBackColor = animInfo.mOnOff ? m_clrBgOn : m_clrBgNormal;
        emit toggled(animInfo.mOnOff);
    }
    emit clicked();
}

void ZToolButton::toggle(bool bOn)
{
    if (buttonOption() & Opt_Checkable)
    {
        if (m_bChecked == bOn)
            return;
        setChecked(bOn);
        emit toggled(bOn);
    }
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
    if (m_options & Opt_SwitchAnimation)
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(Qt::NoPen);

        QPainterPath path;
        path.addRoundedRect(this->rect(), m_radius, m_radius);
        path.setFillRule(Qt::OddEvenFill);
        painter.drawPath(path); //border

        painter.setBrush(animInfo.mBackColor);
        painter.drawRoundedRect(this->rect(), m_radius, m_radius); //background

        m_icon.paint(&painter, animInfo.mButtonRect.x(), animInfo.mButtonRect.y(), animInfo.mButtonRect.width(), animInfo.mButtonRect.height());
    } else {
        QStylePainter p(this);
        ZStyleOptionToolButton option;
        initStyleOption(&option);
        p.drawComplexControl(static_cast<QStyle::ComplexControl>(ZenoStyle::CC_ZenoToolButton), option);
    }
}