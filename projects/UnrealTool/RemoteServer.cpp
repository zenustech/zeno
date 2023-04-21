#include "httplib/httplib.h"
#include "msgpack/msgpack.h"
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/unreal/ZenoUnrealTypes.h>

#if !defined(UT_MAYBE_UNUSED) && defined(__has_cpp_attribute)
    #if __has_cpp_attribute(maybe_unused)
        #define UT_MAYBE_UNUSED [[maybe_unused]]
    #endif
#endif // !defined(UT_MAYBE_UNUSED) && defined(__has_cpp_attribute)
#ifndef UT_MAYBE_UNUSED
    #define UT_MAYBE_UNUSED
#endif

#if !defined(UT_NODISCARD) && defined(__has_cpp_attribute)
    #if __has_cpp_attribute(nodiscard)
        #define UT_NODISCARD [[nodiscard]]
    #endif
#endif // !defined(UT_NODISCARD) && defined(__has_cpp_attribute)
#ifndef UT_NODISCARD
    #define UT_NODISCARD
#endif // UT_NODISCARD

namespace zeno {

namespace remote {

static struct Flags {
    bool IsMainProcess;

    Flags()
        : IsMainProcess(false)
    {}
} StaticFlags;

static struct SubjectRegistry {
    std::map<std::string, zeno::unreal::SubjectContainer> Elements;
    std::function<void(const std::set<std::string>&)> Callback;

    void Push(const std::vector<zeno::unreal::SubjectContainer>& InitList) {
        if (StaticFlags.IsMainProcess) {
            std::set<std::string> ChangeList;
            for (const zeno::unreal::SubjectContainer& Value : InitList) {
                ChangeList.emplace(Value.Name);
                Elements.try_emplace(Value.Name, Value);
            }
            if (Callback) {
                Callback(ChangeList);
            }
        } else {
            // In child process, transfer data with http
            httplib::Client Cli { "http://localhost:23343" };
            zeno::unreal::SubjectContainerList List { InitList };
            std::vector<uint8_t> Data = msgpack::pack(List);
            Cli.Post("/subject/push", reinterpret_cast<const char*>(Data.data()), Data.size(), "application/binary");
        }
    }
} StaticRegistry;

struct SubjectCommit {
    std::set<std::string> ChangedSubjects;

    explicit SubjectCommit(const std::set<std::string>& Changes) {
        for (const std::string& SubjectName : Changes) {
            ChangedSubjects.insert(SubjectName);
        }
    }
};

class SubjectHistory {
    std::vector<SubjectCommit> Commits;

public:
    void Commit(const std::set<std::string>& Changes) {
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

    UT_NODISCARD size_t GetTopIndex() const {
        return Commits.size() - 1;
    }
};
}

#define SERVER_HANDLER_WRAPPER(FUNC) [this] (const httplib::Request& Req, httplib::Response& Res) { FUNC(Req, Res); }

class ZenoRemoteServer {
    httplib::Server Srv;
    remote::SubjectHistory History;

private:
    static void IndexPage(const httplib::Request& Req, httplib::Response& Res);
    void FetchDataDiff(const httplib::Request& Req, httplib::Response& Res);
    static void PushData(const httplib::Request& Req, httplib::Response& Res);
    static void FetchData(const httplib::Request& Req, httplib::Response& Res);

public:
    void Run() {
        zeno::remote::StaticRegistry.Callback = [this] (const std::set<std::string>& Changes) {
            History.Commit(Changes);
        };
        Srv.Get("/", &ZenoRemoteServer::IndexPage);
        Srv.Get("/subject/diff", SERVER_HANDLER_WRAPPER(FetchDataDiff));
        Srv.Post("/subject/push", PushData);
        Srv.Get("/subject/fetch", FetchData);
        zeno::remote::StaticFlags.IsMainProcess = true;
        Srv.listen("127.0.0.1", 23343);
        // Listen failed or server exited, set flag to false
        zeno::remote::StaticFlags.IsMainProcess = false;
    }
};

#undef SERVER_HANDLER_WRAPPER

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

UT_MAYBE_UNUSED static int defUnrealToolInit = getSession().eventCallbacks->hookEvent("init", [] {
    StartServerThread();
});

void ZenoRemoteServer::IndexPage(const httplib::Request& Req, httplib::Response& Res) {
    Res.set_content(R"({"api_version": 1, "protocol": "msgpack"})", "application/json");
}

void ZenoRemoteServer::FetchDataDiff(const httplib::Request &Req, httplib::Response &Res) {
    int32_t client_version = std::atoi(Req.get_param_value("client_version").c_str());
    std::set<std::string> Changes = History.Diff(client_version);
    zeno::unreal::Diff Diff { std::move(Changes) };
    std::vector<uint8_t> Data = msgpack::pack(Diff);
    Res.set_content(reinterpret_cast<const char*>(Data.data()), Data.size(), "application/binary");
}

void ZenoRemoteServer::PushData(const httplib::Request& Req, httplib::Response &Res) {
    const std::string& Body = Req.body;
    std::error_code Err;
    auto Container = msgpack::unpack<zeno::unreal::SubjectContainerList>(reinterpret_cast<const uint8_t*>(Body.data()), Body.size(), Err);
    if (!Err) {
        zeno::remote::StaticRegistry.Push(Container.Data);
        Res.status = 204;
    } else {
        Res.status = 400;
    }
}

void ZenoRemoteServer::FetchData(const httplib::Request &Req, httplib::Response &Res) {
    size_t Num = Req.get_param_value_count("key");
    zeno::unreal::SubjectContainerList List;
    List.Data.reserve(Num);
    for (size_t Idx = 0; Idx < Num; ++Idx) {
        std::string Key = Req.get_param_value("key", Idx);
        auto Value = zeno::remote::StaticRegistry.Elements.find(Key);
        if (Value != zeno::remote::StaticRegistry.Elements.end()) {
            List.Data.emplace_back( Value->second );
        }
    }
    std::vector<uint8_t> Data = msgpack::pack(List);
    Res.set_content(reinterpret_cast<const char*>(Data.data()), Data.size(), "application/binary");
}
}

namespace zeno {

struct TransferPrimitiveToUnreal : public INode {
    void apply() override {
        zeno::remote::StaticFlags.IsMainProcess = false;
        std::string processor_type = get_input2<std::string>("type");
        std::string subject_name = get_input2<std::string>("name");
        std::shared_ptr<PrimitiveObject> prim = get_input2<PrimitiveObject>("prim");
        if (processor_type == "StaticMeshNoUV") {
            zeno::unreal::Mesh Mesh { prim->verts, prim->tris };
            std::vector<uint8_t> Data = msgpack::pack(Mesh);
            zeno::remote::StaticRegistry.Push({ zeno::unreal::SubjectContainer{ subject_name, static_cast<int16_t>(zeno::unreal::ESubjectType::Mesh), std::move(Data) }, });
        }
        set_output2("primRef", prim);
    }
};

ZENO_DEFNODE(TransferPrimitiveToUnreal)({
      {
              {"enum StaticMeshNoUV", "type", "StaticMeshNoUV"},
              {"string", "name", "SubjectFromZeno"},
              {"prim"},
          },
          { "primRef" },
          {},
          {"Unreal"},
      }
 );


}
