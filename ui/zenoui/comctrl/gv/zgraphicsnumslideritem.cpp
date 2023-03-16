#include "zgraphicsnumslideritem.h"
#include <zenoui/style/zenostyle.h>
#include <zeno/utils/log.h>
#include "zgraphicstextitem.h"
#include <zenoedit/zenoapplication.h>


ZGraphicsNumSliderItem::ZGraphicsNumSliderItem(const QVector<qreal>& steps, QGraphicsItem* parent)
    : _base(parent)
    , m_steps(steps)
{
    qreal maxWidth = 0, maxHeight = 0;
    QFont font = zenoApp->font();
    int padding = ZenoStyle::dpiScaled(5);

    for (int i = 0; i < m_steps.length(); i++)
    {
        QString text = QString::number(m_steps[i]);
        QSizeF sz = ZSimpleTextItem::size(text, font, padding, padding, padding, padding);
        maxWidth = qMax(maxWidth, sz.width());
        maxHeight = qMax(maxHeight, sz.height());
    }

    for (int i = 0; i < m_steps.length(); i++)
    {
        QString text = QString::number(m_steps[i]);
        ZSimpleTextItem* pLabel = new ZSimpleTextItem(text, this);
        pLabel->setFixedWidth(maxWidth);
        pLabel->setAlignment(Qt::AlignCenter);
        pLabel->setBrush(QColor(80, 80, 80));
        pLabel->setBackground(QColor(21, 21, 21));
        pLabel->setFont(font);
        pLabel->setHoverCursor(Qt::SizeHorCursor);
        pLabel->setPadding(padding, padding, padding, padding);

        QRectF rc = pLabel->boundingRect();
        pLabel->setPos(QPointF(0, i * maxHeight));
        m_labels.append(pLabel);
    }

    setFlags(ItemIsMovable | ItemIsFocusable);
}

ZGraphicsNumSliderItem::~ZGraphicsNumSliderItem()
{
}

void ZGraphicsNumSliderItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mousePressEvent(event);
    m_lastPos = event->screenPos();
    //todo:transparent.
}

void ZGraphicsNumSliderItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    //disable move obj.
    //_base::mouseMoveEvent(event);
    for (auto label : m_labels)
    {
#if 0
        bool bHovered = label->sceneBoundingRect().contains(
            QPointF(label->sceneBoundingRect().center().x(), event->scenePos().y()));
#else
        bool bHovered = label->isHovered();
#endif
        if (bHovered)
        {
            QPointF pos = event->screenPos();
            qreal dx = pos.x() - m_lastPos.x();
            static const int speed_factor = 10;
            if (std::abs(dx) > speed_factor)
            {
                qreal scale = label->text().toFloat();
                int pieces = dx / speed_factor;;
                qreal Dx = pieces * scale;
                emit numSlided(Dx);
                m_lastPos = event->screenPos();
            }
        }
    }
}

void ZGraphicsNumSliderItem::keyPressEvent(QKeyEvent* event)
{
    _base::keyPressEvent(event);
}

void ZGraphicsNumSliderItem::keyReleaseEvent(QKeyEvent* event)
{
    _base::keyReleaseEvent(event);
    if (isVisible())
    {
        hide();
        event->accept();
        emit slideFinished();
    }
}

void ZGraphicsNumSliderItem::focusOutEvent(QFocusEvent* event)
{
    _base::focusOutEvent(event);
    hide();
    emit slideFinished();
}

QRectF ZGraphicsNumSliderItem::boundingRect() const
{
    return childrenBoundingRect();
}

QPainterPath ZGraphicsNumSliderItem::shape() const
{
    QPainterPath path;
    path.addRect(boundingRect());
    return path;
}

void ZGraphicsNumSliderItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}
