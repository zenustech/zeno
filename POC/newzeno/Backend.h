#pragma once

#include "Container.h"


namespace zeno::v2::backend {

struct Context {
    std::vector<container::any> inputs;
    std::vector<container::any> outputs;
};


struct Session;
struct Scope {
    Session *session;
    std::map<int, container::any> objects;

    Scope(Session *session) : session(session) {
    }
};


struct Session {
    std::map<std::string, std::function<void(Context *)>> nodes;
    std::map<std::string, std::string> nodeinfo;

    static Session &get() {
        static std::unique_ptr<Session> session;
        if (!session) {
            session = std::make_unique<Session>();
        }
        return *session;
    }

    template <class F>
    int defineNode
        ( std::string const &name
        , F func
        , std::string const &info
        ) {
        nodes[name] = func;
        nodeinfo[name] = info;
        return 1;
    }

    std::unique_ptr<Scope> makeScope() const {
        return std::make_unique<Scope>(const_cast<Session *>(this));
    }
};

}
