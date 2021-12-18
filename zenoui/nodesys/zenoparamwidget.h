#ifndef __ZENO_PARAM_WIDGET_H__
#define __ZENO_PARAM_WIDGET_H__

#include <QtWidgets>
#include "../model/modelrole.h"
#include "nodesys_common.h"

class ZenoParamWidget : public QGraphicsProxyWidget
{
	Q_OBJECT
public:
    ZenoParamWidget(QGraphicsItem *parent = nullptr, Qt::WindowFlags wFlags = Qt::WindowFlags());
    ~ZenoParamWidget();

    enum {
        Type = ZTYPE_PARAMWIDGET
    };
    int type() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
};

class ZenoParamLineEdit : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamLineEdit(const QString& text, QGraphicsItem* parent = nullptr);

private:
    QLineEdit* m_pLineEdit;
};

class ZenoParamLabel : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoParamLabel(const QString &text, const QFont& font, const QBrush& fill, QGraphicsItem *parent = nullptr);
    void setAlignment(Qt::Alignment alignment);

private:
    QLabel* m_label;
};

class ZenoTextLayoutItem : public QGraphicsLayoutItem, public QGraphicsTextItem
{
public:
    ZenoTextLayoutItem(const QString &text, const QFont &font, const QColor &color, QGraphicsItem *parent = nullptr);
    void setGeometry(const QRectF &rect) override;
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

//signals:
//    void geometrySetup(const QPointF &pos);

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;

private:
    QString m_text;
};


class ZenoSocketItem;
class ZenoSvgLayoutItem : public QGraphicsLayoutItem
{
public:
    ZenoSvgLayoutItem(ZenoSocketItem* item, QGraphicsLayoutItem *parent = nullptr, bool isLayout = false);
    void updateGeometry() override;
    void setGeometry(const QRectF &rect) override;

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;

private:
    ZenoSocketItem* m_item;
    
};

//Qt offical layout item demo.
class LayoutItem : public QGraphicsLayoutItem, public QGraphicsItem
{
public:
    LayoutItem(QGraphicsItem *parent = nullptr);

    // Inherited from QGraphicsLayoutItem
    void setGeometry(const QRectF &geom) override;
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;

    // Inherited from QGraphicsItem
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

private:
    QPixmap m_pix;
};


#endif