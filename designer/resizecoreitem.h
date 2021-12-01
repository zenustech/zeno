#ifndef __RESIZE_COREITEM_H__
#define __RESIZE_COREITEM_H__

#include "framework.h"

class ResizableCoreItem : public QGraphicsItem
{
public:
	ResizableCoreItem(QGraphicsItem* parent = nullptr);
	void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
	virtual void resize(QSizeF sz) = 0;
};

class MySvgItem : public QGraphicsSvgItem {
public:
    MySvgItem(QGraphicsItem *parent = 0);
    MySvgItem(const QString &fileName, QGraphicsItem *parent = 0);

    void setSize(QSizeF size);
    void setSize(qreal width, qreal height) {
        setSize(QSizeF(width, height));
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);
    QRectF boundingRect();

private:
    QSizeF m_size;
};

class ResizableImageItem : public ResizableCoreItem
{
    typedef ResizableCoreItem _base;
public:
	ResizableImageItem(const QString& normal, const QString& hovered, const QString& selected, QSizeF sz, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void resize(QSizeF sz) override;
    bool resetImage(const QString &normal, const QString &hovered, const QString &selected, QSizeF sz);

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

private:
    QPixmap m_normal;
    QPixmap m_hovered;
    QPixmap m_selected;
	QGraphicsPixmapItem *m_pixmap;

	QString m_svgNormal;
    QString m_svgHovered;
    QString m_svgSelected;
	MySvgItem* m_svg;

    QSizeF m_size;
};

class ResizableRectItem : public ResizableCoreItem
{
public:
	ResizableRectItem(QRectF rc, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void resize(QSizeF sz) override;

private:
	QGraphicsRectItem* m_rectItem;
};

class ResizableEclipseItem : public ResizableCoreItem
{
public:
	ResizableEclipseItem(const QRectF& rect, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void resize(QSizeF sz) override;

private:
	QGraphicsEllipseItem* m_ellipseItem;
};

class ResizableTextItem : public ResizableCoreItem
{
public:
	ResizableTextItem(const QString& text, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void resize(QSizeF sz) override;
    void setText(const QString &text);
    void setTextProp(QFont font, QColor color);

private:
	QGraphicsTextItem* m_pTextItem;
};

#endif