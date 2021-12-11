#include "interceptor.h"
#include "qdmgraphicsnodesubnet.h"
#include <zeno/dop/Node.h>
#include <zeno/dop/SubnetNode.h>


ZENO_NAMESPACE_BEGIN

void Interceptor::toDopGraph
    ( QDMGraphicsScene *scene
    , dop::SubnetNode *d_scene
    , std::map<QDMGraphicsNode *, dop::Node *> &nodes
    )
{
    std::map<QDMGraphicsSocketIn *, QDMGraphicsSocketOut *> links;
    for (auto *link: scene->links) {
        links.emplace(link->dstSocket, link->srcSocket);
    }

    std::map<QDMGraphicsSocketOut *, std::pair<dop::Node *, int>> sockets;
    for (auto *node: scene->nodes) {
        auto desc = node->getDescriptor();
        auto d_node = d_scene->addNode(*desc);

        d_node->name = node->getName();
        auto numIn = node->socketIns.size();
        auto numOut = node->socketOuts.size();
        d_node->inputs.resize(numIn);
        d_node->outputs.resize(numOut);

        if (auto d_subnet = dynamic_cast<dop::SubnetNode *>(d_node)) {
            auto subnet = dynamic_cast<QDMGraphicsNodeSubnet *>(node);
            [[unlikely]] if (!subnet)
                throw std::runtime_error("got subnet but qt node not subnet");
            std::map<QDMGraphicsNode *, dop::Node *> lut;
            toDopGraph(subnet->subnetScene.get(), d_subnet, lut);
        }

        nodes.emplace(node, d_node);
        for (size_t i = 0; i < node->socketOuts.size(); i++) {
            sockets.try_emplace(node->socketOuts[i], d_node, i);
        }
    }

    for (auto *node: scene->nodes) {
        for (size_t i = 0; i < node->socketIns.size(); i++) {
            dop::Input input{};
            auto *sockIn = node->socketIns[i];
            input.value = sockIn->value;

            if (auto it = links.find(sockIn); it != links.end()) {
                auto *sockOut = it->second;
                auto srcNode = static_cast<QDMGraphicsNode *>(sockOut->parentItem());
                input.sockid = srcNode->socketOutIndex(sockOut);
                input.node = nodes.at(srcNode);
            }

            auto d_node = nodes.at(node);
            d_node->inputs[i] = input;
        }
    }
}

void Interceptor::toDopGraph
    ( QDMGraphicsScene *scene
    , dop::SubnetNode *d_scene
    )
{
    std::map<QDMGraphicsNode *, dop::Node *> nodes;
    return toDopGraph(scene, d_scene, nodes);
}

ZENO_NAMESPACE_END
