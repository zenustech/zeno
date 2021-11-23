#include "interceptor.h"
#include "qdmgraphicsnodesubnet.h"
#include <zeno/dop/Node.h>
#include <zeno/dop/SubnetNode.h>


ZENO_NAMESPACE_BEGIN

void Interceptor::toDopGraph
    ( QDMGraphicsScene *scene
    , dop::SceneGraph *d_scene
    , std::map<QDMGraphicsNode *, dop::Node *> &nodes
    )
{
    std::map<QDMGraphicsSocketIn *, QDMGraphicsSocketOut *> links;
    for (auto const &link: scene->links) {
        links.emplace(link->dstSocket, link->srcSocket);
    }

    std::map<QDMGraphicsSocketOut *, std::pair<dop::Node *, int>> sockets;
    for (auto const &node: scene->nodes) {
        auto desc = node->getDescriptor();
        auto d_node = desc->create();

        d_node->name = node->getName();
        auto numIn = node->socketIns.size();
        auto numOut = node->socketOuts.size();
        d_node->inputs.resize(numIn);
        d_node->outputs.resize(numOut);

        if (auto subnet = dynamic_cast<QDMGraphicsNodeSubnet *>(node.get())) {
            auto d_subnet = std::make_unique<dop::SceneGraph>();
            std::map<QDMGraphicsNode *, dop::Node *> lut;
            toDopGraph(subnet->subnetScene.get(), d_subnet.get(), lut);

            auto d_sub_in = static_cast<dop::SubnetIn *>(lut.at(subnet->subnetInNode));
            auto d_sub_out = static_cast<dop::SubnetOut *>(lut.at(subnet->subnetOutNode));
            auto d_sub = static_cast<dop::SubnetNode *>(d_node.get());
            d_sub->subnet = std::move(d_subnet);
            d_sub->subnetIn = d_sub_in;
            d_sub->subnetOut = d_sub_out;
            d_sub->inputs.resize(numIn);
            d_sub->inputs.resize(numOut);
        }

        nodes.emplace(node.get(), d_node.get());
        for (size_t i = 0; i < node->socketOuts.size(); i++) {
            sockets.try_emplace(node->socketOuts[i].get(), d_node.get(), i);
        }

        d_scene->nodes.insert(std::move(d_node));
    }

    for (auto const &node: scene->nodes) {
        for (size_t i = 0; i < node->socketIns.size(); i++) {
            dop::Input input{};
            auto sockIn = node->socketIns[i].get();
            input.value = sockIn->value;

            if (auto it = links.find(sockIn); it != links.end()) {
                auto *sockOut = it->second;
                auto srcNode = static_cast<QDMGraphicsNode *>(sockOut->parentItem());
                input.sockid = srcNode->socketOutIndex(sockOut);
                input.node = nodes.at(srcNode);
            }

            auto d_node = nodes.at(node.get());
            d_node->inputs[i] = input;
        }
    }
}

void Interceptor::toDopGraph
    ( QDMGraphicsScene *scene
    , dop::SceneGraph *d_scene
    )
{
    std::map<QDMGraphicsNode *, dop::Node *> nodes;
    return toDopGraph(scene, d_scene, nodes);
}

ZENO_NAMESPACE_END
