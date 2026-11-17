#pragma once

#include <QPointF>
#include <QRectF>
#include <QSizeF>
#include <algorithm>

// Display coordinates are logical widget pixels. Render pixels may have a
// different size or device pixel ratio; all interaction uses the same image rect.
class ViewportImageMapping
{
public:
    ViewportImageMapping(QSizeF viewport, QSizeF image) : m_image(image)
    {
        if (viewport.width() <= 0 || viewport.height() <= 0 ||
            image.width() <= 0 || image.height() <= 0)
            return;
        const qreal scale = std::min(viewport.width() / image.width(),
                                     viewport.height() / image.height());
        const QSizeF displayed(image.width() * scale, image.height() * scale);
        m_rect = QRectF(QPointF((viewport.width() - displayed.width()) * .5,
                               (viewport.height() - displayed.height()) * .5), displayed);
    }

    QRectF imageRect() const { return m_rect; }
    bool contains(QPointF point) const
    {
        return !m_rect.isEmpty() && point.x() >= m_rect.left() && point.y() >= m_rect.top() &&
               point.x() < m_rect.right() && point.y() < m_rect.bottom();
    }
    QPointF normalized(QPointF point) const
    {
        if (m_rect.isEmpty()) return QPointF(-1, -1);
        return QPointF((point.x() - m_rect.left()) / m_rect.width(),
                       (point.y() - m_rect.top()) / m_rect.height());
    }
    QPointF normalizedDelta(QPointF delta) const
    {
        if (m_rect.isEmpty()) return QPointF();
        return QPointF(delta.x() / m_rect.width(), delta.y() / m_rect.height());
    }
    QPointF imagePixel(QPointF point) const
    {
        const auto uv = normalized(point);
        return QPointF(uv.x() * m_image.width(), uv.y() * m_image.height());
    }

private:
    QSizeF m_image;
    QRectF m_rect;
};
