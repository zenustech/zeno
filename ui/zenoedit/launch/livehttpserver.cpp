#include "livehttpserver.h"

LiveHttpServer::LiveHttpServer() {
    CROW_ROUTE(app, "/hello")([&](){
        std::cout << "****** Client Num " << clients.size() << "\n";
        return "<h1>LiveHttpServer: Hello world - Zeno<h1>";
    });

    CROW_ROUTE(app, "/sync_info")
        .methods("POST"_method)
            ([&](const crow::request& req){
                auto x = crow::json::load(req.body);
                if (! x)
                    return crow::response(crow::status::BAD_REQUEST);
                auto f = x["FRAME"].i();
                return crow::response{"<h1>LiveHttpServer: SyncInfo - Zeno<h1>"};
            });

    CROW_ROUTE(app, "/set_client_info")
    .methods("POST"_method)
    ([&](const crow::request& req){
        auto x = crow::json::load(req.body);
        if (! x)
            return crow::response(crow::status::BAD_REQUEST);

        auto h = x["Host"].s();
        auto p = x["Port"].i();
        auto r = x["Remove"].b();
        std::cout << "****** Host " << h << "\n";
        std::cout << "****** Port " << p << "\n";
        std::cout << "****** Remove " << r << "\n";

        int index = -1;
        for(int i=0; i<clients.size(); i++){
            if (clients[i].first == h && clients[i].second == p){
                index = i;
                break;
            }
        }

        if(r){
            if (index != -1) {
                clients.erase(clients.begin() + index);
                return crow::response{"<h1>LiveHttpServer: The client data will remove - " + std::string(h) + ":" +
                                      std::to_string(p) + " Zeno<h1>"};
            }
            return crow::response{"<h1>LiveHttpServer: The client data not exists - " + std::string(h) + ":" +
                                  std::to_string(p) + " Zeno<h1>"};
        }else {
            if (index != -1)
                return crow::response{"<h1>LiveHttpServer: The client info has been set - Zeno<h1>"};

            clients.emplace_back(h, p);
            return crow::response{"<h1>LiveHttpServer: The client data is set - " + std::string(h) + ":" +
                                  std::to_string(p) + " Zeno<h1>"};
        }
    });

    std::thread([&]() {
        try
        {
            std::cout << "LiveHttp Server Running On 18080.\n";
            app.port(18080).multithreaded().run();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }).detach();


}
