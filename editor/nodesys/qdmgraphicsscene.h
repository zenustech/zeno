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
    std::unique_ptr<QDMGraphicsLinkHalf> pendingLink;
    std::unique_ptr<QDMGraphicsBackground> background;
    QDMGraphicsNode *floatingNode{};
    QDMGraphicsNode *currentNode{};

public:
    QDMGraphicsScene();
    ~QDMGraphicsScene();

    QDMGraphicsNode *addNode();
    QDMGraphicsLinkFull *addLink(QDMGraphicsSocket *srcSocket, QDMGraphicsSocket *dstSocket);
    void removeNode(QDMGraphicsNode *node);
    void removeLink(QDMGraphicsLinkFull *link);

    ztd::property<std::string> name;
    ztd::prop_list<std::unique_ptr<QDMGraphicsScene>> childScenes;

    void setCurrentNode(QDMGraphicsNode *node);
    std::vector<QDMGraphicsNode *> getVisibleNodes() const;

    QPointF getCursorPos() const;
    void socketClicked(QDMGraphicsSocket *socket);
    void deletePressed();
    void copyPressed();
    void pastePressed();
    void blankClicked();
    void cursorMoved();

public slots:
    void addNodeByType(QString type);
    void updateSceneSelection();

signals:
    void sceneUpdated();
    void currentNodeChanged(QDMGraphicsNode *node);
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSSCENE_H
