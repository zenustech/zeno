#include "zenobackgrounditem.h"

ZenoBackgroundItem::ZenoBackgroundItem(const BackgroundComponent &comp, QGraphicsItem *parent)
    : _base(parent)
    , m_rect(QRectF(0, 0, comp.rc.width(), comp.rc.height()))
    , lt_radius(comp.lt_radius)
    , rt_radius(comp.rt_radius)
    , lb_radius(comp.lb_radius)
    , rb_radius(comp.rb_radius)
    , m_bFixRadius(true)
    , m_img(nullptr)
    , m_bSelected(false)
{
    if (!comp.imageElem.image.isEmpty()) {
        m_img = new ZenoImageItem(comp.imageElem.image, comp.imageElem.imageHovered, comp.imageElem.imageOn, comp.rc.size(), this);
        m_img->setZValue(100);
        m_img->show();
    }
    setColors(comp.clr_normal, comp.clr_hovered, comp.clr_selected);
}

QRectF ZenoBackgroundItem::boundingRect() const {
    return m_rect;
}

void ZenoBackgroundItem::resize(QSizeF sz) {
    QPointF topLeft = m_rect.topLeft();
    m_rect = QRectF(topLeft.x(), topLeft.y(), sz.width(), sz.height());
    if (m_img)
        m_img->resize(sz);
}

void ZenoBackgroundItem::setColors(const QColor &clrNormal, const QColor &clrHovered, const QColor &clrSelected) {
    m_clrNormal = clrNormal;
    m_clrHovered = clrHovered;
    m_clrSelected = clrSelected;
    update();
}

void ZenoBackgroundItem::setRadius(int lt, int rt, int lb, int rb) {
    lt_radius = lt;
    rt_radius = rt;
    lb_radius = lb;
    rb_radius = rb;
    update();
}

std::pair<qreal, qreal> ZenoBackgroundItem::getRxx2(QRectF r, qreal xRadius, qreal yRadius, bool AbsoluteSize) const {
    if (AbsoluteSize) {
        qreal w = r.width() / 2;
        qreal h = r.height() / 2;

        if (w == 0) {
            xRadius = 0;
        } else {
            xRadius = 100 * qMin(xRadius, w) / w;
        }
        if (h == 0) {
            yRadius = 0;
        } else {
            yRadius = 100 * qMin(yRadius, h) / h;
        }
    } else {
        if (xRadius > 100)// fix ranges
            xRadius = 100;

        if (yRadius > 100)
            yRadius = 100;
    }

    qreal w = r.width();
    qreal h = r.height();
    qreal rxx2 = w * xRadius / 100;
    qreal ryy2 = h * yRadius / 100;
    return std::make_pair(rxx2, ryy2);
}

QPainterPath ZenoBackgroundItem::shape() const
{
    QPainterPath path;
    QRectF r = m_rect.normalized();

    if (r.isNull())
        return path;

    if (lt_radius <= 0 && rt_radius <= 0 && lb_radius <= 0 && rb_radius <= 0) {
        path.addRect(r);
        return path;
    }

    qreal x = r.x();
    qreal y = r.y();
    qreal w = r.width();
    qreal h = r.height();

    auto pair = getRxx2(r, lt_radius, lt_radius, m_bFixRadius);
    qreal rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.moveTo(x, y);
    } else {
        path.arcMoveTo(x, y, rxx2, ryy2, 180);
        path.arcTo(x, y, rxx2, ryy2, 180, -90);
    }

    pair = getRxx2(r, rt_radius, rt_radius, m_bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y);
    } else {
        path.arcTo(x + w - rxx2, y, rxx2, ryy2, 90, -90);
    }

    pair = getRxx2(r, rb_radius, rb_radius, m_bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y + h);
    } else {
        path.arcTo(x + w - rxx2, y + h - rxx2, rxx2, ryy2, 0, -90);
    }

    pair = getRxx2(r, lb_radius, lb_radius, m_bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x, y + h);
    } else {
        path.arcTo(x, y + h - rxx2, rxx2, ryy2, 270, -90);
    }

    path.closeSubpath();
    return path;
}

void ZenoBackgroundItem::toggle(bool bSelected)
{
    if (m_img) {
        m_img->toggle(bSelected);
    } else {
        m_bSelected = bSelected;
    }
}

void ZenoBackgroundItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    QPainterPath path = shape();
    if (m_img) {
        painter->setClipPath(path);
    } else {
        if (m_bSelected) {
            painter->fillPath(path, m_clrSelected);
        } else {
            painter->fillPath(path, m_clrNormal);
        }
    }
}


///////////////////////////////////////////////////////////////////////////
ZenoBackgroundWidget::ZenoBackgroundWidget(QGraphicsItem *parent, Qt::WindowFlags wFlags)
    : QGraphicsWidget(parent, wFlags)
    , lt_radius(0)
    , rt_radius(0)
    , lb_radius(0)
    , rb_radius(0)
    , m_bFixRadius(true)
    , m_bSelected(false)
{
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
}

QRectF ZenoBackgroundWidget::boundingRect() const
{
    return _base::boundingRect();
}

void ZenoBackgroundWidget::setColors(const QColor& clrNormal, const QColor& clrHovered, const QColor& clrSelected)
{
    m_clrNormal = clrNormal;
    m_clrHovered = clrHovered;
    m_clrSelected = clrSelected;
    update();
}

void ZenoBackgroundWidget::setRadius(int lt, int rt, int lb, int rb)
{
    lt_radius = lt;
    rt_radius = rt;
    lb_radius = lb;
    rb_radius = rb;
    update();
}

QSizeF ZenoBackgroundWidget::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    QSizeF sz = layout()->effectiveSizeHint(which, constraint);
    return sz;
}

void ZenoBackgroundWidget::setGeometry(const QRectF& rect)
{
    QGraphicsWidget::setGeometry(rect);
}

void ZenoBackgroundWidget::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    QPainterPath path = shape();
    if (m_bSelected) {
        painter->fillPath(path, m_clrSelected);
    } else {
        painter->fillPath(path, m_clrNormal);
    }
}

void ZenoBackgroundWidget::toggle(bool bSelected)
{
    m_bSelected = bSelected;
}

QPainterPath ZenoBackgroundWidget::shape() const
{
    QGraphicsLayout *pLayout = layout();
    //it's complicated to position the shape... 
    QRectF rcc = pLayout->geometry();

    QPainterPath path;
    QRectF r = rcc.normalized();

    if (r.isNull())
        return path;

    if (lt_radius <= 0 && rt_radius <= 0 && lb_radius <= 0 && rb_radius <= 0) {
        path.addRect(r);
        return path;
    }

    qreal x = r.x();
    qreal y = r.y();
    qreal w = r.width();
    qreal h = r.height();

    auto pair = getRxx2(r, lt_radius, lt_radius, m_bFixRadius);
    qreal rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.moveTo(x, y);
    } else {
        path.arcMoveTo(x, y, rxx2, ryy2, 180);
        path.arcTo(x, y, rxx2, ryy2, 180, -90);
    }

    pair = getRxx2(r, rt_radius, rt_radius, m_bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y);
    } else {
        path.arcTo(x + w - rxx2, y, rxx2, ryy2, 90, -90);
    }

    pair = getRxx2(r, rb_radius, rb_radius, m_bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y + h);
    } else {
        path.arcTo(x + w - rxx2, y + h - rxx2, rxx2, ryy2, 0, -90);
    }

    pair = getRxx2(r, lb_radius, lb_radius, m_bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x, y + h);
    } else {
        path.arcTo(x, y + h - rxx2, rxx2, ryy2, 270, -90);
    }

    path.closeSubpath();
    return path;
}

std::pair<qreal, qreal> ZenoBackgroundWidget::getRxx2(QRectF r, qreal xRadius, qreal yRadius, bool AbsoluteSize) const
{
    if (AbsoluteSize) {
        qreal w = r.width() / 2;
        qreal h = r.height() / 2;

        if (w == 0) {
            xRadius = 0;
        } else {
            xRadius = 100 * qMin(xRadius, w) / w;
        }
        if (h == 0) {
            yRadius = 0;
        } else {
            yRadius = 100 * qMin(yRadius, h) / h;
        }
    } else {
        if (xRadius > 100)// fix ranges
            xRadius = 100;

        if (yRadius > 100)
            yRadius = 100;
    }

    qreal w = r.width();
    qreal h = r.height();
    qreal rxx2 = w * xRadius / 100;
    qreal ryy2 = h * yRadius / 100;
    return std::make_pair(rxx2, ryy2);
}