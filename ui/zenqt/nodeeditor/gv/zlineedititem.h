#ifndef __ZLINEEDIT_ITEM_H__
#define __ZLINEEDIT_ITEM_H__

#include <QtWidgets>

#include "zgraphicstextitem.h"

class ZLineEditItem : public ZGraphicsTextItem
{
    Q_OBJECT
    typedef ZGraphicsTextItem _base;
public:
    explicit ZLineEditItem(QGraphicsItem* parent = nullptr);
    explicit ZLineEditItem(const QString& text, QGraphicsItem* parent = nullptr);
    ~ZLineEditItem();
    void setText(const QString& text);
    QRectF boundingRect() const override;

private:
    QString m_text;
};


#endif