#include "zcheckboxbar.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/style/zstyleoption.h>


ZCheckBoxBar::ZCheckBoxBar(QWidget* parent)
    : QWidget(parent)
    , m_bHover(false)
    , m_checkState(Qt::Unchecked)
{
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    setAutoFillBackground(false);
    setMouseTracking(true);
}

Qt::CheckState ZCheckBoxBar::checkState() const
{
    return m_checkState;
}

void ZCheckBoxBar::setCheckState(Qt::CheckState state)
{
    m_checkState = state;
    update();
}

void ZCheckBoxBar::initStyleOption(ZStyleOptionCheckBoxBar* option)
{
    option->rect = this->rect();
    option->bHovered = m_bHover;
    option->state = m_checkState;
}

void ZCheckBoxBar::paintEvent(QPaintEvent* event)
{
    QStylePainter painter(this);
    painter.setPen(palette().color(QPalette::Text));
    // draw the combobox frame, focusrect and selected etc.
    ZStyleOptionCheckBoxBar opt;
    initStyleOption(&opt);
    painter.drawComplexControl(static_cast<QStyle::ComplexControl>(ZenoStyle::CC_ZenoCheckBoxBar), opt);
}

void ZCheckBoxBar::enterEvent(QEvent* event)
{
    QWidget::enterEvent(event);
    m_bHover = true;
    update();
}

void ZCheckBoxBar::leaveEvent(QEvent* event)
{
    QWidget::leaveEvent(event);
    m_bHover = false;
    update();
}

void ZCheckBoxBar::mousePressEvent(QMouseEvent* event)
{
    m_checkState = (m_checkState == Qt::Unchecked) ? Qt::Checked : Qt::Unchecked;
}

void ZCheckBoxBar::mouseReleaseEvent(QMouseEvent* event)
{
    emit stateChanged(m_checkState);
    update();
}
