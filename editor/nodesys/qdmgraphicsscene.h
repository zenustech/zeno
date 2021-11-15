#ifndef QDMGRAPHICSSCENE_H
#define QDMGRAPHICSSCENE_H

#include <QGraphicsScene>
#include <zeno/ztd/property.h>
#include "qdmgraphicsnode.h"
#include "qdmgraphicssocket.h"
#include "qdmgraphicslinkhalf.h"
#include "qdmgraphicslinkfull.h"
#include "qdmgraphicsbackground.h"
#include <QString>
#include <vector>
#include <set>

ZENO_NAMESPACE_BEGIN

class QDMGraphicsScene : public QGraphicsScene
{
    Q_OBJECT

    std::set<std::unique_ptr<QDMGraphicsNode>> nodes;
    std::set<std::unique_ptr<QDMGraphicsLinkFull>> links;
    QDMGraphicsNode *subnetNode{};

    std::unique_ptr<QDMGraphicsBackground> background;
    std::unique_ptr<QDMGraphicsLinkHalf> pendingLink;
    QDMGraphicsNode *floatingNode{};
    QDMGraphicsNode *currentNode{};

public:
    QDMGraphicsScene();
    ~QDMGraphicsScene();

    QDMGraphicsNode *addNode();
    QDMGraphicsLinkFull *addLink(QDMGraphicsSocket *srcSocket, QDMGraphicsSocket *dstSocket);
    void removeNode(QDMGraphicsNode *node);
    void removeLink(QDMGraphicsLinkFull *link);

    void setCurrentNode(QDMGraphicsNode *node);
    std::vector<QDMGraphicsNode *> getVisibleNodes() const;
    std::vector<QDMGraphicsScene *> getChildScenes() const;
    std::string allocateNodeName(std::string const &prefix) const;
    std::string getName() const;

    QPointF getCursorPos() const;
    void socketClicked(QDMGraphicsSocket *socket);
    void deletePressed();
    void copyPressed();
    void pastePressed();
    void blankClicked();
    void doubleClicked();
    void cursorMoved();

public slots:
    void addSubNetNode();
    void addNodeByType(QString type);
    void updateSceneSelection();

signals:
    void sceneUpdated();
    void sceneCreatedOrRemoved();
    void currentNodeChanged(QDMGraphicsNode *node);
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSSCENE_H
