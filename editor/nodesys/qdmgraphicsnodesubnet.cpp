//
// Created by bate on 11/21/21.
//

#include "qdmgraphicsnodesubnet.h"
#include "qdmnodeparamedit.h"
#include "qdmgraphicsscene.h"
#include <zeno/dop/SubnetNode.h>
#include <QPushButton>

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
             subnetScene->addSubnetInput();
        });
    }

    {
        auto btnNew = new QPushButton;
        btnNew->setText("(+O)");
        paredit->addRow("New Output", btnNew);
        QObject::connect(btnNew, &QPushButton::clicked, [this] {
            subnetScene->addSubnetOutput();
        });
    }

    QDMGraphicsNode::setupParamEdit(paredit);
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
