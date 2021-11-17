#include "interceptor.h"


void Interceptor::toDopGraph
    ( QDMGraphicsScene *scene
    , dop::SceneGraph *d_scene
    )
{
    std::map<QDMGraphicsSocketIn *, QDMGraphicsSocketOut *> links;
    for (auto const &link: scene->links) {
        links.try_emplace(link->dstSocket, link->srcSocket);
    }

    for (auto const &node: scene->nodes) {
        auto d_node = std::make_unique<dop::Node>();
        d_node->desc = node->getDescriptor();
        d_node->name = node->getName();

        d_scene->nodes.insert(std::move(d_node));
    }
}
