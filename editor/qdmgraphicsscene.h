#ifndef QDMGRAPHICSSCENE_H
#define QDMGRAPHICSSCENE_H

#include <QObject>
#include <QGraphicsScene>
#include <set>
#include "qdmgraphicsnode.h"
#include "qdmgraphicssocket.h"
#include "qdmgraphicslinkhalf.h"
#include "qdmgraphicslinkfull.h"

class QDMGraphicsScene : public QGraphicsScene
{
    std::set<QDMGraphicsNode *> nodes;
    std::set<QDMGraphicsLinkFull *> links;
    QDMGraphicsLinkHalf *pendingLink{nullptr};

public:
    QDMGraphicsScene();
    ~QDMGraphicsScene();

    QDMGraphicsNode *addNode();
    QDMGraphicsLinkFull *addLink(QDMGraphicsSocket *srcSocket, QDMGraphicsSocket *dstSocket);
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void removeLink(QDMGraphicsLinkFull *link);
    void socketClicked(QDMGraphicsSocket *socket);
    void cursorMoved();
};

#endif // QDMGRAPHICSSCENE_H
