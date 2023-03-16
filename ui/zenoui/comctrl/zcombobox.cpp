#include "../style/zenostyle.h"
#include "zcombobox.h"
#include "./view/zcomboboxitemdelegate.h"


ZComboBox::ZComboBox(bool bSysStyle, QWidget *parent)
    : QComboBox(parent)
    , m_bSysStyle(bSysStyle)
{
    setFocusPolicy(Qt::ClickFocus);
    connect(this, SIGNAL(activated(int)), this, SLOT(onComboItemActivated(int)));
    setItemDelegate(new ZComboBoxItemDelegate2(this));
}

ZComboBox::~ZComboBox()
{
}

QSize ZComboBox::sizeHint() const
{
    if (m_bSysStyle)
        return QComboBox::sizeHint();
    else
        return ZenoStyle::dpiScaledSize(QSize(128, 25));
}

void ZComboBox::onComboItemActivated(int index)
{
    // pay attention to the compatiblity of qt!!!
    QString text = itemText(index);
    emit _textActivated(text);
}

void ZComboBox::wheelEvent(QWheelEvent* event)
{
    QComboBox::wheelEvent(event);
}

void ZComboBox::showPopup()
{
    emit beforeShowPopup();
    QComboBox::showPopup();
}

void ZComboBox::hidePopup()
{
    QComboBox::hidePopup();
    emit afterHidePopup();
}

void ZComboBox::initStyleOption(ZStyleOptionComboBox* option)
{
    QStyleOptionComboBox opt;
    QComboBox::initStyleOption(&opt);
    *option = opt;

    option->bdrNormal = QColor(255, 255, 255, 255*0.4);
    option->bdrHoverd = QColor(228, 228, 228);
    option->bdrSelected = QColor(122, 122, 122);

    //option->palette.setColor(QPalette::Active, QPalette::WindowText, QColor(228, 228, 228));
    //option->palette.setColor(QPalette::Inactive, QPalette::WindowText, QColor(158, 158, 158));

    option->clrBackground = QColor(50, 50, 50);
    option->clrBgHovered = QColor(50, 50, 50);
    option->clrText = QColor(255, 255, 255, 255 * 0.4);

    option->btnNormal = QColor(50, 50, 50);
    option->btnHovered = QColor(50, 50, 50);
    option->btnHovered = QColor(50, 50, 50);

    option->textMargin = 5;
    option->palette.setColor(QPalette::ButtonText, option->clrText);
}

void ZComboBox::paintEvent(QPaintEvent* event)
{
    if (m_bSysStyle) {
        QComboBox::paintEvent(event);
    }
    else {
        QStylePainter painter(this);
        painter.setPen(palette().color(QPalette::Text));
        // draw the combobox frame, focusrect and selected etc.
        ZStyleOptionComboBox opt;
        initStyleOption(&opt);
        painter.drawComplexControl(static_cast<QStyle::ComplexControl>(ZenoStyle::CC_ZenoComboBox), opt);
        painter.drawControl(static_cast<QStyle::ControlElement>(ZenoStyle::CE_ZenoComboBoxLabel), opt);
    }
}
