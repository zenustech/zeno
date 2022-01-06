#include "zenoheatmapitem.h"
#include <QtAlgorithms>


ZenoItemNoDragThrough::ZenoItemNoDragThrough(QGraphicsItem *parent)
    : QGraphicsItem(parent)
{
    setAcceptHoverEvents(true);
}

void ZenoItemNoDragThrough::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    parentItem()->setFlag(QGraphicsItem::ItemIsMovable, false);
    QGraphicsItem::hoverEnterEvent(event);
}

void ZenoItemNoDragThrough::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    parentItem()->setFlag(QGraphicsItem::ItemIsMovable, true);
    QGraphicsItem::hoverLeaveEvent(event);
}


///////////////////////////////////////////////////////////
ZenoRampDraggerItem::ZenoRampDraggerItem(QGraphicsItem *parent)
    : _base(parent)
    , m_parent(parent)
    , m_selected(false)
{
    setFlags(QGraphicsItem::ItemIsMovable | QGraphicsItem::ItemIsSelectable);
}

qreal ZenoRampDraggerItem::width() const
{
    return 10;  //parameterized.
}

QRectF ZenoRampDraggerItem::parentRect() const
{
    if (ZenoColorChannelItem *item = qgraphicsitem_cast<ZenoColorChannelItem *>(m_parent)) {
        return item->rect();
    } else if (ZenoColorRampItem *item = qgraphicsitem_cast<ZenoColorRampItem *>(m_parent)) {
        return item->rect();
    } else {
        return QRectF();
    }
}

qreal ZenoRampDraggerItem::height() const {
    return parentRect().height();
}

QRectF ZenoRampDraggerItem::boundingRect() const
{
    return QRectF(-width() / 2, 0, width(), height());
}

void ZenoRampDraggerItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QPen pen;
    QColor color;
    if (m_selected)
        color = QColor("#EE8844");
    else
        color = QColor("#B0B0B0");
    pen.setColor(color);
    pen.setWidth(2);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    painter->drawRect(-width() / 2, 0, width(), height());
}

qreal ZenoRampDraggerItem::getValue()
{
    qreal f = this->pos().x();
    f = qMax(0., qMin(1., f / parentRect().width()));
    return f;
}

void ZenoRampDraggerItem::setValue(qreal x)
{
    setX(x * parentRect().width());
}

void ZenoRampDraggerItem::setX(qreal x)
{
    x = qMax(0., qMin(parentRect().width(), x));
    setPos(x, 0);
}

void ZenoRampDraggerItem::incX(qreal dx)
{
    setX(pos().x());
    if (ZenoColorChannelItem *item = qgraphicsitem_cast<ZenoColorChannelItem *>(m_parent)) {
        item->updateRamps();
    } else if (ZenoColorRampItem *item = qgraphicsitem_cast<ZenoColorRampItem *>(m_parent)) {
        item->updateRamps();
    }
}

void ZenoRampDraggerItem::setSelected(bool selected)
{
    _base::setSelected(selected);
    m_selected = selected;
}

void ZenoRampDraggerItem::remove()
{
    if (ZenoColorRampItem *item = qgraphicsitem_cast<ZenoColorRampItem *>(m_parent))
    {
        item->removeRamp(this);
    }
}

void ZenoRampDraggerItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    _base::mousePressEvent(event);
    if (ZenoColorRampItem *item = qgraphicsitem_cast<ZenoColorRampItem *>(m_parent))
    {
        item->updateRampSelection(this);
    }
    incX(event->pos().x());
}

void ZenoRampDraggerItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    _base::mouseMoveEvent(event);
    incX(event->pos().x());
}

void ZenoRampDraggerItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    _base::mouseReleaseEvent(event);
    incX(event->pos().x());
}

void ZenoRampDraggerItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
    _base::mouseDoubleClickEvent(event);
    remove();
}


//////////////////////////////////////////////////////////////////////////////////
ZenoColorChannelItem::ZenoColorChannelItem(ZenoHeatMapItem* parent)
    : QGraphicsLayoutItem()
    , ZenoItemNoDragThrough(parent)
    , m_parent(parent)
    , m_dragger(nullptr)
{
}

void ZenoColorChannelItem::setGeometry(const QRectF& rect)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(rect);

    setPos(rect.x(), rect.y());
    m_rect = rect;
    m_dragger = new ZenoRampDraggerItem(this);
}

void ZenoColorChannelItem::setColor(qreal r, qreal g, qreal b) {
    m_color = QColor(r, g, b);
}

QRectF ZenoColorChannelItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

qreal ZenoColorChannelItem::getValue()
{
    return m_dragger->getValue();
}

void ZenoColorChannelItem::setValue(qreal x)
{
    m_dragger->setValue(x);
}

void ZenoColorChannelItem::updateRamps()
{
    m_parent->updateRampColor();
}

void ZenoColorChannelItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    painter->setPen(Qt::NoPen);
    QLinearGradient grad(0, 0, m_rect.width(), 0);
    grad.setColorAt(0.0, QColor(0, 0, 0));
    grad.setColorAt(1.0, QColor(m_color));
    QBrush brush(grad);
    painter->setBrush(brush);
    painter->drawRect(0, 0, m_rect.width(), m_rect.height());
}

QRectF ZenoColorChannelItem::rect() const
{
    return m_rect;
}

QSizeF ZenoColorChannelItem::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
    switch (which) {
        case Qt::PreferredSize:
        case Qt::MinimumSize:
            return QSizeF(256, 24);
        case Qt::MaximumSize:
            return QSizeF(1000, 1000);
    }
    return constraint;
}


////////////////////////////////////////////////////////////////////////////////////////////
ZenoColorRampItem::ZenoColorRampItem(ZenoHeatMapItem *parent)
    : QGraphicsLayoutItem()
    , ZenoItemNoDragThrough(parent)
    , m_parent(parent)
{
}

void ZenoColorRampItem::updateRampSelection(ZenoRampDraggerItem* this_dragger)
{
    for (auto dragger : m_draggers)
    {
        dragger->setSelected(false);
    }
    this_dragger->setSelected(true);
    m_parent->updateRampSelection();
}

int ZenoColorRampItem::currSelectedIndex()
{
    for (int i = 0; i < m_draggers.size(); i++)
    {
        if (m_draggers[i]->IsSelected())
            return i;
    }
    return -1;
}

void ZenoColorRampItem::updateRampColor(qreal r, qreal g, qreal b)
{
    int i = currSelectedIndex();
    if (i != -1) {
        COLOR_RAMPS& ramps = m_parent->ramps();
        ramps[i].r = r;
        ramps[i].g = g;
        ramps[i].b = b;
    }
}

void ZenoColorRampItem::removeRamp(ZenoRampDraggerItem* dragger)
{
    int index = m_draggers.indexOf(dragger);
    auto pDragger = m_draggers[index];
    m_draggers.removeAt(index);
    delete pDragger;
    initDraggers();
}

void ZenoColorRampItem::addRampAt(qreal fac)
{
    COLOR_RAMPS& colorRamps = ramps();
    qSort(colorRamps.begin(), colorRamps.end(), [=](const COLOR_RAMP& lhs, const COLOR_RAMP& rhs) {
        return lhs.pos < rhs.pos;
    });

    int i = 0;
    qreal lf = 0, lr = 0, lg = 0, lb = 0;
    for (i = colorRamps.size() - 1; i >= 0; i--)
    {
        lf = colorRamps[i].pos;
        lr = colorRamps[i].r, lg = colorRamps[i].g, lb = colorRamps[i].b;
        if (fac >= colorRamps[i].pos)
            break;
        else
            return;
    }

    qreal rf = 0, rr = 0, rg = 0, rb = 0;
    if (colorRamps.size() > i + 1) {
        rf = colorRamps[i + 1].pos;
        rr = colorRamps[i + 1].r;
        rg = colorRamps[i + 1].g;
        rb = colorRamps[i + 1].b;
    } else {
        rf = lf;
        rr = lr;
        rg = lg;
        rb = lb;
    }

    qreal intf = (fac - lf) / (rf - lf);
    qreal newr = (1 - intf) * lr + intf * rr;
    qreal newg = (1 - intf) * lg + intf * rg;
    qreal newb = (1 - intf) * lb + intf * rb;

    COLOR_RAMP newRamp;
    newRamp.pos = fac;
    newRamp.r = newr;
    newRamp.g = newg;
    newRamp.b = newb;
    colorRamps.insert(i, newRamp);
    
    initDraggers();
    updateRampSelection(m_draggers[i]);
}

void ZenoColorRampItem::initDraggers()
{
    for (auto dragger : m_draggers) {
        delete dragger;
    }
    m_draggers.clear();
    for (auto ramp : ramps()) {
        ZenoRampDraggerItem *dragger = new ZenoRampDraggerItem(this);
        dragger->setValue(ramp.pos);
        m_draggers.push_back(dragger);
    }
}

void ZenoColorRampItem::updateRamps()
{
    for (int i = 0; i < m_draggers.size(); i++)
    {
        qreal f = m_draggers[i]->getValue();
        ramps()[i].pos = f;
    }
    update();
}

COLOR_RAMPS& ZenoColorRampItem::ramps()
{
    return m_parent->ramps();
}

void ZenoColorRampItem::setGeometry(const QRectF& rect)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(rect);

    setPos(rect.x(), rect.y());
    m_rect = rect;
    initDraggers();
}

QRectF ZenoColorRampItem::rect() const
{
    return m_rect;
}

QRectF ZenoColorRampItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

void ZenoColorRampItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    qreal f = event->pos().x();
    if (0 <= f && f <= m_rect.width())
    {
        f /= m_rect.width();
        addRampAt(f);
    }
}

void ZenoColorRampItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    painter->setPen(Qt::NoPen);
    QLinearGradient grad(0, 0, m_rect.width(), 0);
    for (auto ramp : ramps())
    {
        grad.setColorAt(ramp.pos, QColor(int(ramp.r * 255), int(ramp.g * 255), int(ramp.b * 255)));
    }
    QBrush brush(grad);
    painter->setBrush(brush);
    painter->drawRect(0, 0, m_rect.width(), m_rect.height());
}

QSizeF ZenoColorRampItem::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
    switch (which)
    {
        case Qt::PreferredSize:
        case Qt::MinimumSize:
            return QSizeF(256, 24);
        case Qt::MaximumSize:
            return QSizeF(1000, 1000);
    }
    return constraint;
}
 

////////////////////////////////////////////////////////////////////////////////////
ZenoHeatMapItem::ZenoHeatMapItem(const COLOR_RAMPS& ramps, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
    , m_colorramp(nullptr)
    , m_colorR(nullptr)
    , m_colorG(nullptr)
    , m_colorB(nullptr)
    , m_colorRamps(ramps)
{
    //default value:
    if (m_colorRamps.isEmpty())
    {
        m_colorRamps.push_back(COLOR_RAMP(0.0, 0, 0, 0));
        m_colorRamps.push_back(COLOR_RAMP(0.5, 1, 0, 0));
        m_colorRamps.push_back(COLOR_RAMP(0.9, 1, 1, 0));
        m_colorRamps.push_back(COLOR_RAMP(1.0, 1, 1, 1));
    }
    initWidgets();
}

void ZenoHeatMapItem::initWidgets()
{
    QGraphicsLinearLayout *pMainLayout = new QGraphicsLinearLayout(Qt::Vertical);

    m_colorramp = new ZenoColorRampItem(this);

    m_colorR = new ZenoColorChannelItem(this);
    m_colorR->setColor(255, 0, 0);
    m_colorG = new ZenoColorChannelItem(this);
    m_colorG->setColor(0, 255, 0);
    m_colorB = new ZenoColorChannelItem(this);
    m_colorB->setColor(0, 0, 255);

    pMainLayout->addItem(m_colorramp);
    pMainLayout->addItem(m_colorR);
    pMainLayout->addItem(m_colorG);
    pMainLayout->addItem(m_colorB);

    setLayout(pMainLayout);
}

void ZenoHeatMapItem::updateRampColor()
{
    qreal r = m_colorR->getValue();
    qreal g = m_colorG->getValue();
    qreal b = m_colorB->getValue();
    m_colorramp->updateRampColor(r, g, b);
}

void ZenoHeatMapItem::updateRampSelection() {
    int idx = m_colorramp->currSelectedIndex();
    if (idx == -1) return;

    const COLOR_RAMP &ramp = m_colorRamps[idx];
    m_colorR->setValue(ramp.r);
    m_colorG->setValue(ramp.g);
    m_colorB->setValue(ramp.b);
}

COLOR_RAMPS& ZenoHeatMapItem::ramps()
{
    return m_colorRamps;
}

void ZenoHeatMapItem::setColorRamps(const COLOR_RAMPS &ramps)
{
    m_colorRamps = ramps;
}
