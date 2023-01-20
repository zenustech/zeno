#ifndef ZENO_LIVEHTTPSERVER_H
#define ZENO_LIVEHTTPSERVER_H

#include "crow.h"

class LiveHttpServer {
public:
    LiveHttpServer();

    crow::SimpleApp app;
    std::vector<std::pair<std::string, int>> clients;
};

#endif //ZENO_LIVEHTTPSERVER_H
