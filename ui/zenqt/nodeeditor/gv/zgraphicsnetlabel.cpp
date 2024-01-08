#include "zgraphicsnetlabel.h"
#include "zenosocketitem.h"
#include <zenomodel/include/modelrole.h>


ZGraphicsNetLabel::ZGraphicsNetLabel(bool bInput, const QString& text, QGraphicsItem* parent)
    : ZGraphicsTextItem(parent)
    , m_bInput(bInput)
{
    setText(text);

    setFlags(ItemIsSelectable | ItemIsFocusable);
    setTextInteractionFlags(Qt::NoTextInteraction);

    QFont font = QApplication::font();
    font.setPointSize(13);
    font.setBold(true);
    setFont(font);
    setDefaultTextColor(QColor(203, 158, 39));
    //setBackground(QColor(0, 0, 0));
}

QRectF ZGraphicsNetLabel::boundingRect() const
{
    QRectF rc = ZGraphicsTextItem::boundingRect();
    return rc;
}

void ZGraphicsNetLabel::SetTextInteraction(bool on, bool selectAll)
{
    if (on && textInteractionFlags() == Qt::NoTextInteraction)
    {
        // switch on editor mode:
        setTextInteractionFlags(Qt::TextEditorInteraction);
        // manually do what a mouse click would do else:
        setFocus(Qt::MouseFocusReason); // this gives the item keyboard focus
        setSelected(true); // this ensures that itemChange() gets called when we click out of the item
        if (selectAll) // option to select the whole text (e.g. after creation of the TextItem)
        {
            QTextCursor c = textCursor();
            c.select(QTextCursor::Document);
            setTextCursor(c);
        }
    }
    else if (!on && textInteractionFlags() == Qt::TextEditorInteraction)
    {
        // turn off editor mode:
        setTextInteractionFlags(Qt::NoTextInteraction);
        // deselect text (else it keeps gray shade):
        QTextCursor c = this->textCursor();
        c.clearSelection();
        this->setTextCursor(c);
        clearFocus();
    }
}

void ZGraphicsNetLabel::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
#ifdef ZENO_NODESVIEW_OPTIM
    if (editor_factor < 0.1) {
        return;
    }
#endif
    QStyleOptionGraphicsItem myOption(*option);
    myOption.state &= ~QStyle::State_Selected;
    myOption.state &= ~QStyle::State_HasFocus;

    QRectF rc = boundingRect();

    if (m_bg.isValid())
    {
        painter->setPen(Qt::NoPen);
        painter->setBrush(m_bg);
        painter->drawRect(rc);
    }
    QGraphicsTextItem::paint(painter, &myOption, widget);

    auto itemPenWidth = 1.0;
    const qreal pad = itemPenWidth / 2;
    if (option->state & (QStyle::State_HasFocus))
    {
        const qreal penWidth = 0; // cosmetic pen

        const QColor fgcolor = option->palette.windowText().color();
        const QColor bgcolor( // ensure good contrast against fgcolor
            fgcolor.red() > 127 ? 0 : 255,
            fgcolor.green() > 127 ? 0 : 255,
            fgcolor.blue() > 127 ? 0 : 255);

        painter->setPen(QPen(QColor("#FFFFFF"), 0, Qt::DashLine));
        painter->setBrush(Qt::NoBrush);
        painter->drawRect(boundingRect().adjusted(pad, pad, -pad, -pad));
    }
    else if (option->state & (QStyle::State_Selected))
    {
        painter->setPen(QPen(QColor("#FFFFFF"), 0, Qt::DashLine));
        painter->setBrush(Qt::NoBrush);
        painter->drawRect(boundingRect().adjusted(pad, pad, -pad, -pad));
    }
    //draw a bottom line
    painter->setPen(QPen(QColor(255, 255, 255), 3));
    rc.adjust(pad, 0, 0, 0);
    painter->drawLine(rc.bottomLeft(), rc.bottomRight());
}

QVariant ZGraphicsNetLabel::itemChange(GraphicsItemChange change, const QVariant& value)
{
    if (change == QGraphicsItem::ItemSelectedChange
        && textInteractionFlags() != Qt::NoTextInteraction
        && !value.toBool())
    {
        // item received SelectedChange event AND is in editor mode AND is about to be deselected:
        SetTextInteraction(false); // leave editor mode
    }
    return ZGraphicsTextItem::itemChange(change, value);
}

int ZGraphicsNetLabel::type() const
{
    return Type;
}

QModelIndex ZGraphicsNetLabel::paramIdx() const
{
    if (ZenoSocketItem* pSocketItem = qgraphicsitem_cast<ZenoSocketItem*>(parentItem()))
        return pSocketItem->paramIndex();
    else
        return QModelIndex();
}

void ZGraphicsNetLabel::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    ZGraphicsTextItem::mousePressEvent(event);
}

void ZGraphicsNetLabel::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    ZGraphicsTextItem::mouseReleaseEvent(event);

    if (Qt::LeftButton == event->button() && 
        textInteractionFlags() == Qt::NoTextInteraction)
    {
        emit clicked();
    }
}

void ZGraphicsNetLabel::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    if (textInteractionFlags() == Qt::TextEditorInteraction)
    {
        // if editor mode is already on: pass double click events on to the editor:
        ZGraphicsTextItem::mouseDoubleClickEvent(event);
        return;
    }

    // if editor mode is off:
    // 1. turn editor mode on and set selected and focused:
    SetTextInteraction(true);
    setCursor(QCursor(Qt::ArrowCursor));

    // 2. send a single click to this QGraphicsTextItem (this will set the cursor to the mouse position):
    // create a new mouse event with the same parameters as evt
    QGraphicsSceneMouseEvent* click = new QGraphicsSceneMouseEvent(QEvent::GraphicsSceneMousePress);
    click->setButton(event->button());
    click->setPos(event->pos());
    ZGraphicsTextItem::mousePressEvent(click);
    delete click; // don't forget to delete the event
}

void ZGraphicsNetLabel::focusInEvent(QFocusEvent* event)
{
    ZGraphicsTextItem::focusInEvent(event);
}

void ZGraphicsNetLabel::focusOutEvent(QFocusEvent* event)
{
    ZGraphicsTextItem::focusOutEvent(event);
}

void ZGraphicsNetLabel::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    ZGraphicsTextItem::hoverEnterEvent(event);
    if (textInteractionFlags() != Qt::TextEditorInteraction)
        setCursor(QCursor(Qt::PointingHandCursor));
}

void ZGraphicsNetLabel::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    ZGraphicsTextItem::hoverLeaveEvent(event);
    setCursor(QCursor(Qt::ArrowCursor));
}

void ZGraphicsNetLabel::keyPressEvent(QKeyEvent* event)
{
    if (event->key() != Qt::Key_Space && event->key() != Qt::Key_Tab && event->key() != Qt::Key_Return)
    {
        //if (event->key() == Qt::Key_Delete)
        //{
        //    emit aboutToDelete();
        //}
        ZGraphicsTextItem::keyPressEvent(event);
    }
    else {
        SetTextInteraction(false); // leave editor mode
    }
}

void ZGraphicsNetLabel::keyReleaseEvent(QKeyEvent* event)
{
    ZGraphicsTextItem::keyReleaseEvent(event);
}

void ZGraphicsNetLabel::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{
    setSelected(true);

    QMenu* menu = new QMenu;
    QAction* pDelete = new QAction(tr("Delete Net Label"));
    
    connect(pDelete, &QAction::triggered, this, [=]() {
        emit actionTriggered(pDelete);
    });

    menu->addAction(pDelete);

    auto idx = paramIdx();
    if (idx.isValid() && PARAM_OUTPUT == idx.data(ROLE_PARAM_CLASS)) {
        QAction* pEdit = new QAction(tr("Edit Net Label"));
        connect(pEdit, &QAction::triggered, this, [=]() {
            SetTextInteraction(true);
            setCursor(QCursor(Qt::ArrowCursor));
        });
        menu->addAction(pEdit);
    }

    menu->exec(QCursor::pos());
    menu->deleteLater();
}
