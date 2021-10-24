#ifndef QDMGRAPHICSSOCKETIN_H
#define QDMGRAPHICSSOCKETIN_H

#include "qdmgraphicssocket.h"

class QDMGraphicsSocketIn final : public QDMGraphicsSocket
{
public:
    QDMGraphicsSocketIn();

    virtual void linkAttached(QDMGraphicsLinkFull *link) override;
};

#endif // QDMGRAPHICSSOCKETIN_H
