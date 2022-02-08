#ifndef __ZENO_PARAMNAME_ITEM_H__
#define __ZENO_PARAMNAME_ITEM_H__

#include <QtGui>
#include <QtWidgets>

class ZenoParamNameItem : public QGraphicsLayoutItem
{
public:
    ZenoParamNameItem(const QString &paramName, QGraphicsLayoutItem *parent = nullptr, bool isLayout = false);

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;
};

#endif