//
// Created by bate on 11/21/21.
//

#ifndef ZENO_QDMGRAPHICSNODESUBNET_H
#define ZENO_QDMGRAPHICSNODESUBNET_H

#include "qdmgraphicsnode.h"

class QDMGraphicsNodeSubnet final : public QDMGraphicsNode {
    std::unique_ptr<QDMGraphicsScene> subnetScene;
    std::unique_ptr<dop::Descriptor> subnetDescStorage;

public:
    QDMGraphicsNodeSubnet();
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    [[nodiscard]] QDMGraphicsScene *getSubnetScene() const override;
    void setupParamEdit(QDMNodeParamEdit *paredit) override;
    void initialize();
};

class QDMGraphicsNodeSubnetIn final : public QDMGraphicsNode {
    std::unique_ptr<dop::Descriptor> subnetDescStorage;

public:
    QDMGraphicsNodeSubnetIn();
    void initialize();
};

class QDMGraphicsNodeSubnetOut final : public QDMGraphicsNode {
    std::unique_ptr<dop::Descriptor> subnetDescStorage;

public:
    QDMGraphicsNodeSubnetOut();
    void initialize();
};


#endif //ZENO_QDMGRAPHICSNODESUBNET_H
