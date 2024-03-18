#ifndef ZENO_LIVEHTTPSERVER_H
#define ZENO_LIVEHTTPSERVER_H

#ifdef ZENO_LIVESYNC
#include "crow.h"
#endif

#include "tinygltf/json.hpp"

using json = nlohmann::json;

class LiveHttpServer
{
public:
    LiveHttpServer();
#ifdef ZENO_LIVESYNC
    crow::SimpleApp app;
#endif
    std::vector<std::pair<std::string, int>> clients;

    int frameMeshDataCount(int frame);

    json d_frame_mesh{};
};

#endif //ZENO_LIVEHTTPSERVER_H
