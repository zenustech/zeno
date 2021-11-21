//
// Created by bate on 11/21/21.
//

#include "qdmgraphicsnodesubnet.h"
#include "qdmnodeparamedit.h"
#include "utilities.h"
#include "qdmgraphicsscene.h"
#include <zeno/dop/SubnetNode.h>
#include <QPushButton>

ZENO_NAMESPACE_BEGIN

QDMGraphicsNodeSubnet::QDMGraphicsNodeSubnet() = default;

void QDMGraphicsNodeSubnet::initialize()
{
    subnetDescStorage = std::make_unique<dop::Descriptor>();
    dop::Descriptor &desc = *subnetDescStorage;
    desc.name = "SubnetNode";
    desc.factory = std::make_unique<dop::SubnetNode>;
    initByDescriptor(desc);
    subnetScene = std::make_unique<QDMGraphicsScene>();
    subnetScene->initAsSubnet(this);

    {
        auto node = new QDMGraphicsNodeSubnetIn;
        subnetScene->addNode(node);
        node->setPos(QPointF(-200, -100));
        node->initialize();
        subnetInNode = node;
    }

    {
        auto node = new QDMGraphicsNodeSubnetOut;
        subnetScene->addNode(node);
        node->setPos(QPointF(200, -100));
        node->initialize();
        subnetOutNode = node;
    }
}

void QDMGraphicsNodeSubnet::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        if (this->getSubnetScene()) {
            emit getScene()->subnetSceneEntered(subnetScene.get());
        }
    }

    QGraphicsItem::mouseDoubleClickEvent(event);
}

QDMGraphicsScene *QDMGraphicsNodeSubnet::getSubnetScene() const
{
    return subnetScene.get();
}

void QDMGraphicsNodeSubnet::setupParamEdit(QDMNodeParamEdit *paredit) {
    paredit->clearRows();

    {
        auto btnNew = new QPushButton;
        btnNew->setText("(+I)");
        paredit->addRow("New Input", btnNew);
        QObject::connect(btnNew, &QPushButton::clicked, [this] {
            auto name = find_unique_name(getInputNames(), "in");
            this->addSubnetInput(QString::fromStdString(name));
        });
    }

    {
        auto btnNew = new QPushButton;
        btnNew->setText("(+O)");
        paredit->addRow("New Output", btnNew);
        QObject::connect(btnNew, &QPushButton::clicked, [this] {
            auto name = find_unique_name(getOutputNames(), "out");
            this->addSubnetOutput(QString::fromStdString(name));
        });
    }

    QDMGraphicsNode::setupParamEdit(paredit);
}

void QDMGraphicsNodeSubnet::addSubnetInput(QString name) {
    auto sockExt = addSocketIn();
    sockExt->setName(name);
    auto sockInt = subnetInNode->addSocket();
    sockInt->setName(name);
}

void QDMGraphicsNodeSubnet::addSubnetOutput(QString name) {
    auto sockExt = addSocketOut();
    sockExt->setName(name);
    auto sockInt = subnetOutNode->addSocket();
    sockInt->setName(name);
}


QDMGraphicsNodeSubnetIn::QDMGraphicsNodeSubnetIn() = default;

void QDMGraphicsNodeSubnetIn::initialize()
{
    subnetDescStorage = std::make_unique<dop::Descriptor>();
    dop::Descriptor &desc = *subnetDescStorage;
    desc.name = "SubnetIn";
    desc.factory = std::make_unique<dop::SubnetIn>;
    initByDescriptor(desc);
}

QDMGraphicsSocketOut *QDMGraphicsNodeSubnetIn::addSocket()
{
    auto sock = addSocketOut();
    sock->setName("dummy");
    return sock;
}


QDMGraphicsNodeSubnetOut::QDMGraphicsNodeSubnetOut() = default;

void QDMGraphicsNodeSubnetOut::initialize()
{
    subnetDescStorage = std::make_unique<dop::Descriptor>();
    dop::Descriptor &desc = *subnetDescStorage;
    desc.name = "SubnetOut";
    desc.factory = std::make_unique<dop::SubnetOut>;
    initByDescriptor(desc);
}

QDMGraphicsSocketIn *QDMGraphicsNodeSubnetOut::addSocket()
{
    auto sock = addSocketIn();
    sock->setName("dummy");
    return sock;
}

ZENO_NAMESPACE_END
