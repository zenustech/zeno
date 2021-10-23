#ifndef QDMGRAPHICSSOCKETOUT_H
#define QDMGRAPHICSSOCKETOUT_H

#include "qdmgraphicssocket.h"
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QWidget>

class QDMGraphicsSocketOut final : public QDMGraphicsSocket
{
public:
    QDMGraphicsSocketOut();

    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;
};

#endif // QDMGRAPHICSSOCKETOUT_H
