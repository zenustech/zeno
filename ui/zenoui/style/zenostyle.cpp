#include "zenostyle.h"
#include "zstyleoption.h"
#include "../comctrl/ztoolbutton.h"
#include "../comctrl/zobjectbutton.h"
#include "../comctrl/gv/zenoparamwidget.h"
#include <QScreen>
#include <QtSvg/QSvgRenderer>


ZenoStyle::ZenoStyle()
{

}

ZenoStyle::~ZenoStyle()
{

}

qreal ZenoStyle::dpiScaled(qreal value)
{
    static qreal scale = -1;
    if (scale < 0)
    {
        QScreen *screen = qobject_cast<QApplication *>(QApplication::instance())->primaryScreen();
        qreal dpi = screen->logicalDotsPerInch();
        scale = dpi / 96.0;
    }
    return value * scale;
}

QSize ZenoStyle::dpiScaledSize(const QSize &value)
{
    return QSize(ZenoStyle::dpiScaled(value.width()), ZenoStyle::dpiScaled(value.height()));
}

QSize ZenoStyle::sizeFromContents(ContentsType type, const QStyleOption* option, const QSize& size, const QWidget* widget) const
{
    return base::sizeFromContents(type, option, size, widget);
}

void ZenoStyle::drawPrimitive(PrimitiveElement pe, const QStyleOption* option, QPainter* painter, const QWidget* w) const
{
    switch (pe) {
        case PE_ComboBoxLineEdit: {
            if (const QStyleOptionFrame *editOption = qstyleoption_cast<const QStyleOptionFrame *>(option))
            {
                QRect r = option->rect;
                bool hasFocus = option->state & (State_MouseOver | State_HasFocus);

                painter->save();

                QPalette pal = editOption->palette;

                painter->setPen(Qt::NoPen);
                QColor bgClrNormal(36, 37, 36);
                QColor bgClrActive = bgClrNormal;
                if (hasFocus) {
                    painter->fillRect(r, bgClrActive);
                } else {
                    painter->fillRect(r, bgClrNormal);
                }
                painter->restore();
                return;
            }
            return base::drawPrimitive(pe, option, painter, w);
        }
    }
    return base::drawPrimitive(pe, option, painter, w);
}

void ZenoStyle::drawItemText(QPainter* painter, const QRect& rect, int flags, const QPalette& pal, bool enabled,
    const QString& text, QPalette::ColorRole textRole) const
{
    return base::drawItemText(painter, rect, flags, pal, enabled, text, textRole);
}

void ZenoStyle::drawControl(ControlElement element, const QStyleOption* option, QPainter* painter, const QWidget* widget) const
{
    if (CE_ZenoComboBoxLabel == element)
    {
        if (const ZStyleOptionComboBox *cb = qstyleoption_cast<const ZStyleOptionComboBox*>(option))
        {
            QRect editRect = proxy()->subControlRect(CC_ComboBox, cb, SC_ComboBoxEditField, widget);
            painter->save();
            editRect.adjust(cb->textMargin, 0, 0, 0);
            painter->setClipRect(editRect);
            painter->setFont(QFont("HarmonyOS Sans", 12));
            if (!cb->currentIcon.isNull()) {
                //todo
            }
            if (!cb->currentText.isEmpty() && !cb->editable) {
                drawItemText(painter, editRect.adjusted(1, 0, -1, 0),
                             visualAlignment(cb->direction, Qt::AlignLeft | Qt::AlignVCenter),
                             cb->palette, cb->state & State_Enabled, cb->currentText, QPalette::ButtonText);
            }
            painter->restore();
        }
    }
    if (CE_ItemViewItem == element)
    {
		return base::drawControl(element, option, painter, widget);
    }
    return base::drawControl(element, option, painter, widget);
}

QRect ZenoStyle::subControlRect(ComplexControl cc, const QStyleOptionComplex* option, SubControl sc, const QWidget* widget) const
{
    if (cc == CC_ZenoComboBox && sc == SC_ComboBoxArrow)
    {
        if (const QStyleOptionComboBox* cb = qstyleoption_cast<const QStyleOptionComboBox*>(option))
        {
            static const int arrowRcWidth = cb->rect.height();
            const int xpos = cb->rect.x() + cb->rect.width() - cb->rect.height();
            QRect rc(xpos, cb->rect.y(), arrowRcWidth, cb->rect.height());
            return rc;
        }
    }
    else if ((decltype(CC_ZenoToolButton))(std::underlying_type_t<SubControl>)cc == CC_ZenoToolButton)
    {
        const ZStyleOptionToolButton* opt = qstyleoption_cast<const ZStyleOptionToolButton*>(option);
        Q_ASSERT(opt);

        switch (sc)
        {
        case SC_ZenoToolButtonIcon:
        {
            if (opt->buttonOpts & ZToolButton::Opt_UpRight)
            {
                int xleft = opt->rect.width() / 2 - opt->iconSize.width() / 2;
                int ytop = pixelMetric(static_cast<QStyle::PixelMetric>(ZenoStyle::PM_ButtonTopMargin), 0, widget);
                return QRect(xleft, ytop, opt->iconSize.width(), opt->iconSize.height());
            }
            else if (opt->buttonOpts & ZToolButton::Opt_TextUnderIcon)
            {
                int xleft = opt->rect.width() / 2 - opt->iconSize.width() / 2;
                int ytop = pixelMetric(static_cast<QStyle::PixelMetric>(ZenoStyle::PM_ButtonTopMargin), 0, widget);
                return QRect(xleft, ytop, opt->iconSize.width(), opt->iconSize.height());
            }
            else if (opt->buttonOpts & ZToolButton::Opt_TextRightToIcon)
            {
                return QRect(); //todo
            }
            else
            {
                int xpos = opt->rect.width() / 2 - opt->iconSize.width() / 2;
                int ypos = opt->rect.height() / 2 - opt->iconSize.height() / 2;
                return QRect(xpos, ypos, opt->iconSize.width(), opt->iconSize.height());
            }
            break;
        }
        case SC_ZenoToolButtonText:
        {
            if (opt->buttonOpts & ZToolButton::Opt_UpRight)
            {
                QFontMetrics fontMetrics(opt->font);
                int textWidth = fontMetrics.height(); 
                int textHeight = fontMetrics.horizontalAdvance(opt->text);
                int xleft = opt->rect.width() / 2 - textWidth / 2;
                int ypos = opt->rect.height() - textHeight - pixelMetric(static_cast<QStyle::PixelMetric>(ZenoStyle::PM_ButtonBottomMargin), 0, widget);
                QRect rcIcon = subControlRect(cc, option, static_cast<QStyle::SubControl>(SC_ZenoToolButtonIcon), widget);
                return QRect(rcIcon.right(), rcIcon.center().y(), textWidth, textHeight);
            }
            else if (opt->buttonOpts & ZToolButton::Opt_TextUnderIcon)
            {
                QFontMetrics fontMetrics(opt->font);
                int textWidth = fontMetrics.horizontalAdvance(opt->text);
                int textHeight = fontMetrics.height();
                int xleft = opt->rect.width() / 2 - textWidth / 2;
                int ypos = opt->rect.height() - textHeight - pixelMetric(static_cast<QStyle::PixelMetric>(ZenoStyle::PM_ButtonBottomMargin), 0, widget);
                return QRect(xleft, ypos, textWidth, textHeight);
            }
            else if (opt->buttonOpts & ZToolButton::Opt_TextRightToIcon)
            {
                return QRect(); //todo
            }
            else
            {
                return QRect();
            }
        }
        case SC_ZenoToolButtonArrow:
        {
            //todo
            return QRect();
        }
        }
    }
    return base::subControlRect(cc, option, sc, widget);
}

int ZenoStyle::styleHint(StyleHint sh, const QStyleOption* opt, const QWidget* w, QStyleHintReturn* shret) const
{
	//if (QStyle::SH_ItemView_PaintAlternatingRowColorsForEmptyArea == sh)
	//	return 1;
    return QProxyStyle::styleHint(sh, opt, w, shret);
}

int ZenoStyle::pixelMetric(PixelMetric m, const QStyleOption* option, const QWidget* widget) const
{
    if (qobject_cast<const ZMiniToolButton*>(widget))
    {
        switch (m)
        {
        case PM_ButtonLeftMargin:
        case PM_ButtonRightMargin:  return 6;
        case PM_ButtonTopMargin:
        case PM_ButtonBottomMargin: return 6;
        }
    }
    else if (qobject_cast<const ZToolButton*>(widget))
    {
        switch (m)
        {
        case PM_ButtonLeftMargin:
        case PM_ButtonRightMargin:  return 9;
        case PM_ButtonTopMargin:
        case PM_ButtonBottomMargin: return 4;
        }
    }
    switch (m)
    {
        case QStyle::PM_SmallIconSize:
        {
            if (widget && (widget->objectName() == "qt_dockwidget_closebutton" ||
                widget->objectName() == "qt_dockwidget_floatbutton"))
            {
                return dpiScaled(32);
            }
            break;
        }
        case QStyle::PM_DockWidgetTitleBarButtonMargin:
        {
            //only way to customize the height of titlebar.
            return dpiScaled(10);
        }
    }
    return base::pixelMetric(m, option, widget);
}

QRect ZenoStyle::subElementRect(SubElement element, const QStyleOption* option, const QWidget* widget) const
{
    switch (element)
    {
        case QStyle::SE_ItemViewItemText:
        {
            QRect rc = base::subElementRect(element, option, widget);
            rc.adjust(10, 0, 10, 0);
            return rc;
        }
    }
    return base::subElementRect(element, option, widget);
}

void ZenoStyle::drawDropdownArrow(QPainter* painter, QRect downArrowRect, bool isDown) const
{
    QString iconPath = isDown ? ":/icons/ic_pop_down-on.svg" : ":/icons/ic_pop_down.svg";
    QSvgRenderer* render = new QSvgRenderer(QString(iconPath));
    render->render(painter, downArrowRect);
}

void ZenoStyle::drawZenoToolButton(const ZStyleOptionToolButton* option, QPainter* painter, const QWidget* widget) const
{
    QStyle::ComplexControl cc = static_cast<QStyle::ComplexControl>(CC_ZenoToolButton);
    QRect rcIcon = subControlRect(cc, option, static_cast<QStyle::SubControl>(SC_ZenoToolButtonIcon), widget);
    QRect rcText = subControlRect(cc, option, static_cast<QStyle::SubControl>(SC_ZenoToolButtonText), widget);
    QRect rcArrow = subControlRect(cc, option, static_cast<QStyle::SubControl>(SC_ZenoToolButtonArrow), widget);

    //draw the background
    if (option->buttonEnabled && (option->state & (State_MouseOver | State_On)))
    {
        //QRect rect = option->rect.adjusted(0, 0, -1, -1);       //???
        QRect rect = option->rect;
        //todo: round corner
        QBrush bgBrush = option->palette.brush(QPalette::Active, QPalette::Window);
        painter->fillRect(rect, bgBrush);
    }

    //draw icon 
    if (!option->icon.isNull())
    {
        const ZToolButton* pToolButton = qobject_cast<const ZToolButton*>(widget);
        QIcon::Mode mode;
        if (!option->buttonEnabled)
            mode = QIcon::Disabled;
        //else if (pToolButton->isPressed() || pToolButton->isChecked())
        //    mode = QIcon::Selected;
        else if (pToolButton->isHovered())
            mode = QIcon::Active;
        else
            mode = QIcon::Normal;

        option->icon.paint(painter, rcIcon, Qt::AlignCenter, mode);
    }

    //draw text
    if (!option->text.isEmpty())
    {
        QColor text_color = option->buttonEnabled ? option->palette.brush(QPalette::Active, QPalette::WindowText).color() : QColor();
        if (option->buttonOpts & ZToolButton::Opt_TextUnderIcon)
        {
            QStringList textList = option->text.split('\n');
            for (auto iter = textList.begin(); iter != textList.end(); iter++)
            {
                int height = option->fontMetrics.height();
                QString str = *iter;
                painter->save();
                painter->setFont(option->font);
                painter->setPen(text_color);
                painter->drawText(rcText, Qt::AlignHCenter | Qt::TextShowMnemonic, str);
                painter->restore();
            }
        }
        else if (option->buttonOpts & ZToolButton::Opt_UpRight)
        {
            painter->save();
            painter->setFont(option->font);
            painter->setPen(text_color);
            
            painter->restore();
        }
    }
    //draw arrow
    if (option->m_arrowOption != ZStyleOptionToolButton::NO_ARROW)
    {
        if (option->m_arrowOption == ZStyleOptionToolButton::DOWNARROW)
        {
            //todo
        }
        else if (option->m_arrowOption == ZStyleOptionToolButton::RIGHTARROW)
        {
            //todo
        }
    }
}

void ZenoStyle::drawComplexControl(ComplexControl control, const QStyleOptionComplex* option, QPainter* painter, const QWidget* widget) const
{
    switch (control)
    {
    case CC_ZenoComboBox:
        {
            if (const ZStyleOptionComboBox *cmb = qstyleoption_cast<const ZStyleOptionComboBox*>(option))
            {
                QStyleOptionFrame editorOption;
                editorOption.QStyleOption::operator=(*cmb);
                editorOption.rect = option->rect;
                editorOption.state = (cmb->state & (State_Enabled | State_MouseOver | State_HasFocus) |         State_KeyboardFocusChange);

                QPalette palette;
                palette.setColor(QPalette::Active, QPalette::Window, cmb->clrBgHovered);
                palette.setColor(QPalette::Inactive, QPalette::Window, cmb->clrBackground);
                //border
                palette.setColor(QPalette::WindowText, cmb->bdrNormal);

                editorOption.palette = palette;

                painter->save();
                proxy()->drawPrimitive(static_cast<PrimitiveElement>(PE_ComboBoxLineEdit), &editorOption, painter, widget);

                painter->restore();

                QStyleOptionComboBox comboBoxCopy = *cmb;
                QRect downArrowRect = proxy()->subControlRect(
                        static_cast<QStyle::ComplexControl>(ZenoStyle::CC_ZenoComboBox),
                        &comboBoxCopy, SC_ComboBoxArrow, widget);
                painter->setClipRect(downArrowRect);

                QStyleOptionButton buttonOption;
                buttonOption.rect = downArrowRect;
                if (cmb->activeSubControls == SC_ComboBoxArrow) {
                    buttonOption.state = cmb->state;
                }
                bool isDown = (cmb->state & (State_MouseOver | State_Sunken));
                drawDropdownArrow(painter, downArrowRect, isDown);
                return;
            }
            return base::drawComplexControl(control, option, painter, widget);
        }
    case CC_ZenoToolButton:
        if (const ZStyleOptionToolButton* opt = qstyleoption_cast<const ZStyleOptionToolButton*>(option))
        {
            drawZenoToolButton(opt, painter, widget);
            break;
        }
    default:
        return base::drawComplexControl(control, option, painter, widget);
    }
}
