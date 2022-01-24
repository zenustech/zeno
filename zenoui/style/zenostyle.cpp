#include "zenostyle.h"
#include "zstyleoption.h"
#include "../comctrl/ztoolbutton.h"
#include "../comctrl/zobjectbutton.h"
#include "../nodesys/zenoparamwidget.h"
#include <QScreen>
#include "../comctrl/zenodockwidget.h"
#include <QtWidgets/private/qdockwidget_p.h>


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
    switch (type)
    {
        case CT_TabBarTab:
        {
            QSize sz = dpiScaledSize(QSize(120, 37));
            return sz;
        }
    }
    return base::sizeFromContents(type, option, size, widget);
}

void ZenoStyle::drawPrimitive(PrimitiveElement pe, const QStyleOption* option, QPainter* painter, const QWidget* w) const
{
    switch (pe) {
        case PE_FrameTabWidget:
        {
            if (const QStyleOptionTabWidgetFrame* tab = qstyleoption_cast<const QStyleOptionTabWidgetFrame*>(option))
            {
                QStyleOptionTabWidgetFrame frameOpt = *tab;
                frameOpt.rect = w->rect();
                painter->fillRect(frameOpt.rect, QColor(58, 58, 58));
                return;
            }
            break;
        }
        case PE_ComboBoxLineEdit: {
            if (const QStyleOptionFrame *editOption = qstyleoption_cast<const QStyleOptionFrame *>(option))
            {
                QRect r = option->rect;
                bool hasFocus = option->state & (State_MouseOver | State_HasFocus);

                painter->save();
                painter->setRenderHint(QPainter::Antialiasing, true);
                //  ### highdpi painter bug.
                painter->translate(0.5, 0.5);

                QPalette pal = editOption->palette;

                QColor bgClrNormal(37, 37, 37);
                QColor bgClrActive = bgClrNormal;
                if (hasFocus) {
                    painter->fillRect(r.adjusted(0, 0, -1, -1), bgClrActive);
                } else {
                    painter->fillRect(r.adjusted(0, 0, -1, -1), bgClrNormal);
                }

				if (bgClrNormal.isValid() && bgClrActive.isValid())
				{
					// Draw Outline
					QColor bdrClrNormal = pal.color(QPalette::Inactive, QPalette::WindowText);
					QColor bdrClrActive = pal.color(QPalette::Active, QPalette::WindowText);
					if (hasFocus) {
						painter->setPen(bdrClrActive);
					}
					else {
						painter->setPen(bdrClrNormal);
					}
					//painter->drawRect(r.adjusted(0, 0, -1, -1));
				}

                // Draw inner shadow
                //p->setPen(d->topShadow());
                //p->drawLine(QPoint(r.left() + 2, r.top() + 1), QPoint(r.right() - 2, r.top() + 1));
                painter->restore();
                return;
            }
            return base::drawPrimitive(pe, option, painter, w);
        }
        case PE_ComboBoxDropdownButton:
        {
            if (const QStyleOptionButton *btn = qstyleoption_cast<const QStyleOptionButton *>(option))
            {
                State flags = option->state;
                static const qreal margin_offset = dpiScaled(1.0);
                if (flags & (State_Sunken | State_On)) {
                    painter->setPen(QPen(QColor(26, 112, 185), 1));
                    painter->setBrush(QColor(202, 224, 243));
                    //可能要dpiScaled
                    painter->drawRect(option->rect.adjusted(0, 0, -margin_offset, -margin_offset));
                } else if (flags & State_MouseOver) {
                    painter->setPen(QPen(QColor(26, 112, 185), 1));
                    painter->setBrush(QColor(228, 239, 249));
                    painter->drawRect(option->rect.adjusted(0, 0, -margin_offset, -margin_offset));
                }
                return;
            }
            break;
        }
        case PE_FrameMenu:
        {
            painter->fillRect(option->rect, QColor(51, 51, 51));
            return;
        }
        case PE_FrameLineEdit:
        {
            if (qobject_cast<const ZenoGvLineEdit *>(w))
            {
                return;
                painter->save();
                QPalette pal = option->palette;
                State flags = option->state;
                if (flags & (State_HasFocus | State_MouseOver))
                {
                    painter->setPen(QPen(option->palette.color(QPalette::Active, QPalette::WindowText), 1));
                    painter->drawRect(option->rect.adjusted(1, 1, -1, -1));
                }
                else
                {
                    painter->setPen(QPen(option->palette.color(QPalette::Inactive, QPalette::WindowText), 1));
                    painter->drawRect(option->rect.adjusted(1, 1, -1, -1));
                }
                painter->restore();
                return;
            }
            break;
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
    if (CE_MenuBarEmptyArea == element)
    {
        painter->fillRect(option->rect, QColor(58, 58, 58));
        return;
    }
    else if (CE_MenuBarItem == element)
    {
        if (const QStyleOptionMenuItem* mbi = qstyleoption_cast<const QStyleOptionMenuItem*>(option))
        {
            QStyleOptionMenuItem optItem(*mbi);
            bool disabled = !(option->state & State_Enabled);
            int alignment = Qt::AlignCenter | Qt::TextShowMnemonic | Qt::TextDontClip | Qt::TextSingleLine;
            QPalette::ColorRole textRole = disabled ? QPalette::Text : QPalette::ButtonText;

            if (option->state & State_Selected)
            {
                if (option->state & State_Sunken)
                {
                    painter->fillRect(option->rect, QColor(179, 102, 0));
                }
                else
                {
                    painter->fillRect(option->rect, QColor(71, 71, 71));
                }
            }
            else
            {
                painter->fillRect(option->rect, QColor(58, 58, 58));
            }

            optItem.palette.setBrush(QPalette::All, textRole, QColor(190, 190, 190));
            drawItemText(painter, optItem.rect, alignment, optItem.palette, optItem.state & State_Enabled, optItem.text, textRole);
        }
        return;
    }
    else if (CE_TabBarTab == element) {
        if (const QStyleOptionTab* tab = qstyleoption_cast<const QStyleOptionTab*>(option)) {
            proxy()->drawControl(CE_TabBarTabShape, tab, painter, widget);
            proxy()->drawControl(CE_TabBarTabLabel, tab, painter, widget);
            proxy()->drawControl(static_cast<ControlElement>(CE_TabBarTabUnderline), tab, painter, widget);
            proxy()->drawControl(static_cast<ControlElement>(CE_TabBarTabCloseBtn), tab, painter, widget);
            return;
        }
    }
    else if (CE_TabBarTabShape == element)
    {
        if (const QStyleOptionTab* tab = qstyleoption_cast<const QStyleOptionTab*>(option))
        {
            QRect rect(option->rect);
            int rotate = 0;
            bool isDisabled = !(tab->state & State_Enabled);
            bool hasFocus = tab->state & State_HasFocus;
            bool isHot = tab->state & State_MouseOver;
            bool selected = tab->state & State_Selected;
            bool lastTab = tab->position == QStyleOptionTab::End;
            bool firstTab = tab->position == QStyleOptionTab::Beginning;
            bool onlyOne = tab->position == QStyleOptionTab::OnlyOneTab;
            bool leftAligned = proxy()->styleHint(SH_TabBar_Alignment, tab, widget) == Qt::AlignLeft;
            bool centerAligned = proxy()->styleHint(SH_TabBar_Alignment, tab, widget) == Qt::AlignCenter;
            int borderThickness = proxy()->pixelMetric(PM_DefaultFrameWidth, option, widget);
            int tabOverlap = proxy()->pixelMetric(PM_TabBarTabOverlap, option, widget);
            painter->fillRect(rect, selected ? QColor(48, 48, 48) : QColor(58, 58, 58));
            return;
        }
    }
    else if (CE_TabBarTabLabel == element)
    {
        if (const QStyleOptionTab *tab = qstyleoption_cast<const QStyleOptionTab *>(option))
        {
            QStyleOptionTab _tab(*tab);
            bool selected = tab->state & State_Selected;
            QColor textColor;
            painter->save();
            if (selected)
            {
                textColor = QColor(244, 235, 229);
            }
            else
            {
                textColor = QColor(129, 125, 123);
            }
            _tab.palette.setBrush(QPalette::WindowText, textColor);
            QFont font("HarmonyOS Sans", 11);
            font.setBold(false);
            QPen pen(textColor);
            painter->setPen(pen);
            painter->setFont(font);
            
            int alignment = Qt::AlignLeft | Qt::AlignVCenter;
            QRect tr = proxy()->subElementRect(SE_TabBarTabText, &_tab, widget);
            proxy()->drawItemText(painter, tr, alignment, _tab.palette, _tab.state & State_Enabled, _tab.text, QPalette::WindowText);

            painter->restore();
            return;
        }
    }
    else if (CE_TabBarTabUnderline == element)
    {
        if (const QStyleOptionTab* tab = qstyleoption_cast<const QStyleOptionTab*>(option))
        {
            static const int height = 2;
            bool selected = tab->state & State_Selected;
            if (selected)
            {
                QRect rc = QRect(tab->rect.left(), tab->rect.bottom() - height, tab->rect.width(), height);
                painter->fillRect(rc, QColor(23, 160, 252));
            }
            return;
        }
    }
    else if (CE_TabBarTabCloseBtn == element)
    {
        if (const QStyleOptionTab* tab = qstyleoption_cast<const QStyleOptionTab*>(option))
        {
            return;
        }
    }
    else if (CE_MenuItem == element)
    {
        return drawMenuItem(element, option, painter, widget);
    }
    else if (CE_MenuEmptyArea == element)
    {
        if (const QStyleOptionMenuItem* menuitem = qstyleoption_cast<const QStyleOptionMenuItem*>(option))
        {
            painter->fillRect(option->rect, QColor(58, 58, 58));
            return;
        }
    }
    else if (CE_ZenoComboBoxLabel == element)
    {
        if (const ZStyleOptionComboBox *cb = qstyleoption_cast<const ZStyleOptionComboBox*>(option))
        {
            QRect editRect = proxy()->subControlRect(CC_ComboBox, cb, SC_ComboBoxEditField, widget);
            painter->save();
            editRect.adjust(cb->textMargin, 0, 0, 0);
            painter->setClipRect(editRect);
            painter->setFont(QFont("Microsoft YaHei", 10));
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
    else if (CE_DockWidgetTitle == element)
    {
        if (const ZenoDockWidget* pCustomDock = qobject_cast<const ZenoDockWidget*>(widget))
        {
            if (const QStyleOptionDockWidget* dwOpt = qstyleoption_cast<const QStyleOptionDockWidget*>(option))
            {
                QRect rect = option->rect;
                if (pCustomDock && pCustomDock->isFloating())
                {
                    base::drawControl(element, option, painter, widget);
                    return;
                }

                //hide all buttons.
                QDockWidgetLayout *dwLayout = qobject_cast<QDockWidgetLayout*>(pCustomDock->layout());
                QWidget *pCloseBtn = dwLayout->widgetForRole(QDockWidgetLayout::CloseButton);
                QWidget *pFloatBtn = dwLayout->widgetForRole(QDockWidgetLayout::FloatButton);
                if (pCloseBtn)
                    pCloseBtn->setVisible(false);
                if (pFloatBtn)
                    pFloatBtn->setVisible(false);

                const bool verticalTitleBar = dwOpt->verticalTitleBar;

                if (verticalTitleBar)
                {
                    rect = rect.transposed();

                    painter->translate(rect.left() - 1, rect.top() + rect.width());
                    painter->rotate(-90);
                    painter->translate(-rect.left() + 1, -rect.top());
                }

                QColor bgClr = option->palette.window().color();
                bgClr = bgClr.darker(110);
                QColor bdrClr = option->palette.window().color().darker(130);
                painter->setBrush(/*bgClr*/QColor(58, 58, 58));
                painter->setPen(Qt::NoPen);
                painter->drawRect(rect.adjusted(0, 1, -1, -2));     //titlebar margin to inside widget

                int buttonMargin = 4;
                int mw = proxy()->pixelMetric(QStyle::PM_DockWidgetTitleMargin, dwOpt, widget);
                int fw = proxy()->pixelMetric(PM_DockWidgetFrameWidth, dwOpt, widget);
                const QDockWidget *dw = qobject_cast<const QDockWidget *>(widget);
                bool isFloating = dw && dw->isFloating();

                QRect r = option->rect.adjusted(0, 2, -1, -3);
                QRect titleRect = r;

                if (dwOpt->closable)
                {
                    QSize sz = proxy()->standardIcon(QStyle::SP_TitleBarCloseButton, dwOpt, widget).actualSize(QSize(10, 10));
                    titleRect.adjust(0, 0, -sz.width() - mw - buttonMargin, 0);
                }

                if (dwOpt->floatable)
                {
                    //QSize sz = proxy()->standardIcon(QStyle::SP_TitleBarMaxButton, dwOpt, widget).actualSize(QSize(10, 10));
                    //titleRect.adjust(0, 0, -sz.width() - mw - buttonMargin, 0);
                }

                if (isFloating)
                {
                    titleRect.adjust(0, -fw, 0, 0);
                    if (widget && widget->windowIcon().cacheKey() != QApplication::windowIcon().cacheKey())
                        titleRect.adjust(titleRect.height() + mw, 0, 0, 0);
                }
                else
                {
                    titleRect.adjust(mw, 0, 0, 0);
                    if (!dwOpt->floatable && !dwOpt->closable)
                        titleRect.adjust(0, 0, -mw, 0);
                }
                if (!verticalTitleBar)
                    titleRect = visualRect(dwOpt->direction, r, titleRect);

                if (!dwOpt->title.isEmpty())
                {
                    QString titleText = painter->fontMetrics().elidedText(dwOpt->title, Qt::ElideRight,
                                                                          verticalTitleBar ? titleRect.height() : titleRect.width());
                    const int indent = 4;
                    drawItemText(painter, rect.adjusted(indent + 1, 1, -indent - 1, -1),
                                 Qt::AlignLeft | Qt::AlignVCenter | Qt::TextShowMnemonic,
                                 dwOpt->palette,
                                 dwOpt->state & State_Enabled, titleText,
                                 QPalette::WindowText);
                }
                return;
            }
        }
    }
    return base::drawControl(element, option, painter, widget);
}

QRect ZenoStyle::subControlRect(ComplexControl cc, const QStyleOptionComplex* option, SubControl sc, const QWidget* widget) const
{
    if (cc == CC_ZenoComboBox && sc == SC_ComboBoxArrow)
    {
        if (const QStyleOptionComboBox* cb = qstyleoption_cast<const QStyleOptionComboBox*>(option))
        {
            static const int arrowRcWidth = 18;
            const int xpos = cb->rect.x() + cb->rect.width() - dpiScaled(arrowRcWidth);
            QRect rc(xpos, cb->rect.y(), dpiScaled(arrowRcWidth), cb->rect.height());
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
        case QStyle::PM_MenuPanelWidth: return 1;
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

void ZenoStyle::drawZenoLineEdit(PrimitiveElement pe, const QStyleOption* option, QPainter* painter, const QWidget* widget) const
{
    QColor clrBorder, clrBackground, clrForeground;

    //todo
    //clrBorder = DrawerFunc::getColorFromWidget(widget, option->state, "border");
    //clrBackground = DrawerFunc::getColorFromWidget(widget, option->state, "background");
    //clrForeground = DrawerFunc::getColorFromWidget(widget, option->state, "foreground");

    painter->setPen(clrBorder);
    painter->setBrush(clrBackground);
    painter->drawRect(option->rect.adjusted(0, 0, -1, -1));
}

void ZenoStyle::drawDropdownArrow(QPainter* painter, QRect downArrowRect) const
{
    QRectF arrowRect;
    arrowRect.setWidth(dpiScaled(16));
    arrowRect.setHeight(dpiScaled(16));
    arrowRect.moveTo(downArrowRect.x() + (downArrowRect.width() - arrowRect.width()) / 2.0,
                     downArrowRect.y() + (downArrowRect.height() - arrowRect.height()) / 2.0);

    QPointF bottomPoint = QPointF(arrowRect.center().x(), arrowRect.bottom());
    QPixmap px = QIcon(":/icons/downarrow.png").pixmap(ZenoStyle::dpiScaledSize(QSize(16, 16)));
    painter->drawPixmap(arrowRect.topLeft(), px);
}

void ZenoStyle::drawNewItemMenu(const QStyleOptionMenuItem* menuitem, QPainter* p, const QWidget* w) const
{
    //todo
}

void ZenoStyle::drawMenuItem(ControlElement element, const QStyleOption* option, QPainter* painter, const QWidget* widget) const
{
    //base QProxyStyle::drawControl
    if (const QStyleOptionMenuItem* menuitem = qstyleoption_cast<const QStyleOptionMenuItem*>(option)) {
        // windows always has a check column, regardless whether we have an icon or not
        const qreal factor = 1;// QWindowsXPStylePrivate::nativeMetricScaleFactor(widget);
        int checkcol = qRound(qreal(25) * factor);
        const int gutterWidth = qRound(qreal(3) * factor);
        {
            const QSizeF size(16, 16);
            const QMarginsF margins(3,3,3,3);
            checkcol = qMax(menuitem->maxIconWidth, qRound(gutterWidth + size.width() + margins.left() + margins.right()));
        }
        QRect rect = option->rect;

        //draw vertical menu line
        if (option->direction == Qt::LeftToRight)
            checkcol += rect.x();
        QPoint p1 = QStyle::visualPos(option->direction, menuitem->rect, QPoint(checkcol, rect.top()));
        QPoint p2 = QStyle::visualPos(option->direction, menuitem->rect, QPoint(checkcol, rect.bottom()));
        QRect gutterRect(p1.x(), p1.y(), gutterWidth, p2.y() - p1.y() + 1);
        painter->fillRect(gutterRect, QColor(58, 58, 58));

        int x, y, w, h;
        menuitem->rect.getRect(&x, &y, &w, &h);
        int tab = menuitem->tabWidth;
        bool dis = !(menuitem->state & State_Enabled);
        bool checked = menuitem->checkType != QStyleOptionMenuItem::NotCheckable
            ? menuitem->checked : false;
        bool act = menuitem->state & State_Selected;

        if (menuitem->menuItemType == QStyleOptionMenuItem::Separator) {
            int yoff = y - 2 + h / 2;
            const int separatorSize = 0;// qRound(qreal(6) * QWindowsStylePrivate::nativeMetricScaleFactor(widget));
            QPoint p1 = QPoint(x + checkcol, yoff);
            QPoint p2 = QPoint(x + w + separatorSize, yoff);
            
            QPen pen(QColor(148, 148, 148));
            painter->setPen(pen);
            painter->fillRect(option->rect, QColor(58, 58, 58));
            painter->drawLine(p1, p2);
            return;
        }

        QRect vCheckRect = visualRect(option->direction, menuitem->rect, QRect(menuitem->rect.x(),
            menuitem->rect.y(), checkcol - (gutterWidth + menuitem->rect.x()), menuitem->rect.height()));

        if (act)
        {
            painter->fillRect(option->rect, QColor(179, 102, 0));
        }
        else
        {
            painter->fillRect(option->rect, QColor(58, 58, 58));
        }

        if (menuitem->checkType != QStyleOptionMenuItem::NotCheckable)
        {
            const QSizeF size(12, 12);
            const QMarginsF margins(0, 0, 0, 0);
            QRect checkRect(0, 0, qRound(size.width() + margins.left() + margins.right()),
                qRound(size.height() + margins.bottom() + margins.top()));
            checkRect.moveCenter(vCheckRect.center());
            QRect _checkRc = checkRect;

            QPen pen(QColor(148, 148, 148));
            painter->setPen(pen);
            painter->drawRect(checkRect);
            if (checked)
            {
                QIcon iconChecked(":/icons/checked.png");
                painter->drawPixmap(checkRect, iconChecked.pixmap(size.width(), size.height()));
            }
        }

        if (!menuitem->icon.isNull()) {
            QIcon::Mode mode = dis ? QIcon::Disabled : QIcon::Normal;
            if (act && !dis)
                mode = QIcon::Active;
            QPixmap pixmap;
            if (checked)
                pixmap = menuitem->icon.pixmap(proxy()->pixelMetric(PM_SmallIconSize, option, widget), mode, QIcon::On);
            else
                pixmap = menuitem->icon.pixmap(proxy()->pixelMetric(PM_SmallIconSize, option, widget), mode);
            const int pixw = pixmap.width() / pixmap.devicePixelRatio();
            const int pixh = pixmap.height() / pixmap.devicePixelRatio();
            QRect pmr(0, 0, pixw, pixh);
            pmr.moveCenter(vCheckRect.center());
            painter->setPen(menuitem->palette.text().color());
            painter->drawPixmap(pmr.topLeft(), pixmap);
        }

        const QColor textColor = QColor(200, 200, 200);// menuitem->palette.text().color();
        if (dis)
            painter->setPen(textColor);
        else
            painter->setPen(textColor);

        const int windowsItemFrame = 2, windowsItemHMargin = 3, windowsItemVMargin = 4, windowsRightBorder = 15, windowsArrowHMargin = 6;

        int xm = windowsItemFrame + checkcol + windowsItemHMargin + (gutterWidth - menuitem->rect.x()) - 1;
        int xpos = menuitem->rect.x() + xm;
        QRect textRect(xpos, y + windowsItemVMargin, w - xm - windowsRightBorder - tab + 1, h - 2 * windowsItemVMargin);
        QRect vTextRect = visualRect(option->direction, menuitem->rect, textRect);
        QString s = menuitem->text;
        if (!s.isEmpty()) {    // draw text
            painter->save();
            int t = s.indexOf(QLatin1Char('\t'));
            int text_flags = Qt::AlignVCenter | Qt::TextShowMnemonic | Qt::TextDontClip | Qt::TextSingleLine;
            if (!proxy()->styleHint(SH_UnderlineShortcut, menuitem, widget))
                text_flags |= Qt::TextHideMnemonic;
            text_flags |= Qt::AlignLeft;
            if (t >= 0) {
                QRect vShortcutRect = visualRect(option->direction, menuitem->rect,
                    QRect(textRect.topRight(), QPoint(menuitem->rect.right(), textRect.bottom())));
                painter->drawText(vShortcutRect, text_flags, s.mid(t + 1));
                s = s.left(t);
            }
            QFont font = menuitem->font;
            if (menuitem->menuItemType == QStyleOptionMenuItem::DefaultItem)
                font.setBold(true);
            painter->setFont(font);
            painter->drawText(vTextRect, text_flags, s.left(t));
            painter->restore();
        }
        if (menuitem->menuItemType == QStyleOptionMenuItem::SubMenu) {// draw sub menu arrow
            int dim = (h - 2 * windowsItemFrame) / 2;
            PrimitiveElement arrow;
            arrow = (option->direction == Qt::RightToLeft) ? PE_IndicatorArrowLeft : PE_IndicatorArrowRight;
            xpos = x + w - windowsArrowHMargin - windowsItemFrame - dim;
            QRect  vSubMenuRect = visualRect(option->direction, menuitem->rect, QRect(xpos, y + h / 2 - dim / 2, dim, dim));
            QStyleOptionMenuItem newMI = *menuitem;
            newMI.rect = vSubMenuRect;
            newMI.state = dis ? State_None : State_Enabled;
            newMI.palette.setColor(QPalette::ButtonText, QColor(214, 214, 214));    //arrow color
            proxy()->drawPrimitive(arrow, &newMI, painter, widget);
        }
    }
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
        QRect rect = option->rect.adjusted(0, 0, -1, -1);
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
                //drawPrimitive(PE_FrameLineEdit, &editorOption, painter, widget);

                painter->restore();
                painter->save();

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

                //draw arrow button
                State flags = option->state;
                static const qreal margin_offset = 0;//dpiScaled(1.0);
                if (flags & (State_Sunken | State_On))
                {
                    painter->setPen(QPen(QColor(158, 158, 158), 2));
                    painter->setBrush(Qt::NoBrush);
                    //painter->setBrush(QColor(202, 224, 243));
                    //可能要dpiScaled
                    painter->drawRect(downArrowRect.adjusted(0, 0, -margin_offset, -margin_offset));
                }
                else if (flags & State_MouseOver)
                {
                    painter->setPen(QPen(QColor(158, 158, 158), 2));
                    painter->setBrush(Qt::NoBrush);
                    painter->drawRect(downArrowRect.adjusted(0, 0, -margin_offset, -margin_offset));
                }

                painter->restore();

                painter->setPen(QPen(QColor(0, 0, 0), 1));
                drawDropdownArrow(painter, downArrowRect);
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
