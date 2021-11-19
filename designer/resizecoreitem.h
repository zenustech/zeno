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

class ResizablePixmapItem : public ResizableCoreItem
{
public:
	ResizablePixmapItem(const QPixmap& pixmap, QGraphicsItem* parent = nullptr);
	QRectF boundingRect() const override;
	void resize(QSizeF sz) override;

private:
	QGraphicsPixmapItem* m_pixmapitem;
	QPixmap m_original;
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

private:
	QGraphicsTextItem* m_pTextItem;
};

#endif