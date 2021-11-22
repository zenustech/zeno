//
// Created by bate on 11/21/21.
//

#ifndef ZENO_QDMGRAPHICSNODESUBNET_H
#define ZENO_QDMGRAPHICSNODESUBNET_H

#include "qdmgraphicsnode.h"

ZENO_NAMESPACE_BEGIN

class QDMGraphicsNodeSubnetIn;
class QDMGraphicsNodeSubnetOut;

class QDMGraphicsNodeSubnet final : public QDMGraphicsNode {
    friend Interceptor;

    std::unique_ptr<QDMGraphicsScene> subnetScene;
    std::unique_ptr<dop::Descriptor> subnetDescStorage;
    QDMGraphicsNodeSubnetIn *subnetInNode;
    QDMGraphicsNodeSubnetOut *subnetOutNode;

public:
    QDMGraphicsNodeSubnet();
    ~QDMGraphicsNodeSubnet() override;

    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
    [[nodiscard]] QDMGraphicsScene *getSubnetScene() const override;
    void setupParamEdit(QDMNodeParamEdit *paredit) override;
    void initialize();
    void addSubnetInput(std::string const &name);
    void addSubnetOutput(std::string const &name);
};

class QDMGraphicsNodeSubnetIn final : public QDMGraphicsNode {
    std::unique_ptr<dop::Descriptor> subnetDescStorage;

public:
    QDMGraphicsNodeSubnetIn();
    ~QDMGraphicsNodeSubnetIn() override;

    void initialize();
    QDMGraphicsSocketOut *addSocket();
    QDMGraphicsNode *underlyingNode() override;
};

class QDMGraphicsNodeSubnetOut final : public QDMGraphicsNode {
    std::unique_ptr<dop::Descriptor> subnetDescStorage;

public:
    QDMGraphicsNodeSubnetOut();
    ~QDMGraphicsNodeSubnetOut() override;

    void initialize();
    QDMGraphicsSocketIn *addSocket();
    QDMGraphicsNode *underlyingNode() override;
};

ZENO_NAMESPACE_END

#endif //ZENO_QDMGRAPHICSNODESUBNET_H
