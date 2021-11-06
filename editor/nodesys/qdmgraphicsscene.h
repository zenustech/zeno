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

ZENO_NAMESPACE_BEGIN

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
    // TODO: duplicatePressed as well... (Ctrl-D)
    void deletePressed();
    void blankClicked();
    void cursorMoved();
    QPointF getCursorPos() const;

public slots:
    void addNodeByName(QString name);

signals:
    void nodeAdded(QDMGraphicsNode *node);
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSSCENE_H
