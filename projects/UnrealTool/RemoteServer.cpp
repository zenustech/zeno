#include "httplib/httplib.h"
#include <zeno/core/Session.h>
#include <zeno/extra/EventCallbacks.h>
#include <vector>
#include <string>

namespace zeno {

namespace remote {
struct SubjectCommit {
    std::set<std::string> ChangedSubjects;

    SubjectCommit(std::initializer_list<std::string> Changes) {
        for (const std::string& SubjectName : Changes) {
            ChangedSubjects.insert(SubjectName);
        }
    }
};

class SubjectHistory {
    std::vector<SubjectCommit> Commits;

public:
    void Commit(std::initializer_list<std::string> Changes) {
        Commits.emplace_back(Changes);
    }

    std::set<std::string> Diff(const size_t StartIdx) {
        std::set<std::string> Result;
        if (Commits.size() > StartIdx) {
            for (size_t Idx = StartIdx; Idx < Commits.size(); ++Idx) {
                for (const std::string& SubjectName : Commits[Idx].ChangedSubjects) {
                    Result.insert(SubjectName);
                }
            }
        }
        return Result;
    }
};
}

class ZenoRemoteServer {
    httplib::Server Srv;
    remote::SubjectHistory History;

private:
    static void IndexPage(const httplib::Request& Req, httplib::Response& Res);

public:
    void Run() {
        Srv.Get("/", &ZenoRemoteServer::IndexPage);
        Srv.listen("127.0.0.1", 23343);
    }
};

}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "processthreadsapi.h"

DWORD WINAPI RunServerWrapper(LPVOID lpParam) {
    zeno::ZenoRemoteServer Server;
    Server.Run();
    return 0;
}

void StartServerThread() {
    DWORD ThreadID;
    HANDLE hServerThread = CreateThread(nullptr, 0, RunServerWrapper, (LPVOID)nullptr, 0, &ThreadID);
}
#else // Not Windows
// TODO [darc] : support linux and unix :
void StartServerThread() { static_assert(false); }
#endif // defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

namespace zeno {

static int defUnrealToolInit = getSession().eventCallbacks->hookEvent("init", [] {
    StartServerThread();
});

void ZenoRemoteServer::IndexPage(const httplib::Request& Req, httplib::Response& Res) {
    Res.set_content("{\"api_version\": 1}", "application/json");
}
}

