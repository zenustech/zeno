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

struct Interceptor;

class QDMGraphicsScene : public QGraphicsScene
{
    Q_OBJECT

    friend Interceptor;

    std::set<std::unique_ptr<QDMGraphicsNode>> nodes;
    std::set<std::unique_ptr<QDMGraphicsLinkFull>> links;
    QDMGraphicsNode *subnetNode{};

    std::unique_ptr<QDMGraphicsBackground> background;
    std::unique_ptr<QDMGraphicsLinkHalf> pendingLink;
    QDMGraphicsNode *floatingNode{};
    QDMGraphicsNode *currentNode{};

    QDMGraphicsNode *addNode();
    void addSubnetNode();
    void addSubnetInput();
    void addSubnetOutput();
    void addNormalNode(std::string const &type);
    void updateFloatingNode();

public:
    QDMGraphicsScene();
    ~QDMGraphicsScene();

    QDMGraphicsLinkFull *addLink(QDMGraphicsSocket *srcSocket, QDMGraphicsSocket *dstSocket);
    void removeNode(QDMGraphicsNode *node);
    void removeLink(QDMGraphicsLinkFull *link);

    void setSubnetNode(QDMGraphicsNode *node);
    void setCurrentNode(QDMGraphicsNode *node);
    //std::vector<QDMGraphicsNode *> getVisibleNodes() const;
    std::vector<QDMGraphicsScene *> getChildScenes() const;
    std::string allocateNodeName(std::string const &prefix) const;
    QDMGraphicsScene *getParentScene() const;
    std::string getFullPath() const;
    std::string getName() const;

    QPointF getCursorPos() const;
    void socketClicked(QDMGraphicsSocket *socket);
    void deletePressed();
    void copyPressed();
    void pastePressed();
    void blankClicked();
    void cursorMoved();

public slots:
    void updateSceneSelection();
    void addNodeByType(QString type);

signals:
    void sceneUpdated();
    void sceneCreatedOrRemoved();
    void currentNodeChanged(QDMGraphicsNode *node);
    void subnetSceneEntered(QDMGraphicsScene *subScene);
};

ZENO_NAMESPACE_END

#endif // QDMGRAPHICSSCENE_H
