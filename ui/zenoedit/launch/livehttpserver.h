#ifndef ZENO_LIVEHTTPSERVER_H
#define ZENO_LIVEHTTPSERVER_H

#include "crow.h"
#include "json.hpp"

using json = nlohmann::json;

class LiveHttpServer
{
public:
    LiveHttpServer();

    crow::SimpleApp app;
    std::vector<std::pair<std::string, int>> clients;

    int frameMeshDataCount(int frame);

    json d_frame_mesh{};
};

#endif //ZENO_LIVEHTTPSERVER_H
