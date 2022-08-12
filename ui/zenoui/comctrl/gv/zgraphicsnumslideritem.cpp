#include "zgraphicsnumslideritem.h"
#include <zenoui/style/zenostyle.h>
#include <zeno/utils/log.h>
#include "zgraphicstextitem.h"


ZGraphicsNumSliderItem::ZGraphicsNumSliderItem(const QVector<qreal>& steps, QGraphicsItem* parent)
    : _base(parent)
    , m_steps(steps)
{
    qreal maxWidth = 0, maxHeight = 0;
    QFont font("HarmonyOS Sans", 16);
    int padding = ZenoStyle::dpiScaled(15);

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
        pLabel->setBackground(QColor(0, 0, 0));
        pLabel->setFont(QFont("HarmonyOS Sans", 16));
        pLabel->setPadding(padding, padding, padding, padding);

        QRectF rc = pLabel->boundingRect();
        pLabel->setPos(QPointF(0, i * maxHeight));
        m_labels.append(pLabel);
    }

    //setAcceptHoverEvents(true);
    setFlags(ItemIsMovable | ItemIsFocusable);
}

ZGraphicsNumSliderItem::~ZGraphicsNumSliderItem()
{
}

void ZGraphicsNumSliderItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mousePressEvent(event);
    m_lastPos = event->pos();
    //todo:transparent.
}

void ZGraphicsNumSliderItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    //disable move obj.
    //_base::mouseMoveEvent(event);
    for (auto label : m_labels)
    {
        if (label->isHovered())
        {
            QPointF pos = event->pos();
            qreal dx = pos.x() - m_lastPos.x();
            qreal scale = label->text().toFloat();
            int pieces = dx;
            qreal Dx = pieces * scale;
            //zeno::log_critical("Dx: {}", Dx);
            emit numSlided(Dx);
        }
    }
    m_lastPos = event->pos();
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