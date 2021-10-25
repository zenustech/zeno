#ifndef QDMGRAPHICSSCENE_H
#define QDMGRAPHICSSCENE_H

#include <QGraphicsScene>
#include <set>
#include "qdmgraphicsnode.h"
#include "qdmgraphicssocket.h"
#include "qdmgraphicslinkhalf.h"
#include "qdmgraphicslinkfull.h"
#include "qdmgraphicsbackground.h"
#include <QString>

class QDMGraphicsScene : public QGraphicsScene
{
    std::set<std::unique_ptr<QDMGraphicsNode>> nodes;
    std::set<std::unique_ptr<QDMGraphicsLinkFull>> links;
    std::unique_ptr<QDMGraphicsLinkHalf> pendingLink;
    std::unique_ptr<QDMGraphicsBackground> background;
    QDMGraphicsNode *floatingNode{};

public:
    QDMGraphicsScene();
    ~QDMGraphicsScene();

    QDMGraphicsNode *addNode();
    QDMGraphicsLinkFull *addLink(QDMGraphicsSocket *srcSocket, QDMGraphicsSocket *dstSocket);
    void removeNode(QDMGraphicsNode *node);
    void removeLink(QDMGraphicsLinkFull *link);
    void socketClicked(QDMGraphicsSocket *socket);
    // TODO: duplicatePressed as well...
    void deletePressed();
    void blankClicked();
    void cursorMoved();
    QPointF getCursorPos() const;

public slots:
    void addNodeByName(QString name);
};

#endif // QDMGRAPHICSSCENE_H
