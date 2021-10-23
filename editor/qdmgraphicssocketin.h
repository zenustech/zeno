#ifndef QDMGRAPHICSSOCKETIN_H
#define QDMGRAPHICSSOCKETIN_H

#include "qdmgraphicssocket.h"
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QWidget>
#include <QDebug>

class QDMGraphicsSocketIn final : public QDMGraphicsSocket
{
public:
    QDMGraphicsSocketIn();

    virtual void linkAttached(QDMGraphicsLinkFull *link) override;
    virtual void paint(QPainter *painter, QStyleOptionGraphicsItem const *styleOptions, QWidget *widget) override;
};

#endif // QDMGRAPHICSSOCKETIN_H
