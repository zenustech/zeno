#ifndef ZENO_REMOTE_TOKEN
#define ZENO_REMOTE_TOKEN "ZENO_DEFAULT_TOKEN"
#endif

#ifndef ZENO_LOCAL_TOKEN
#define ZENO_LOCAL_TOKEN "ZENO_LOCAL_TOKEN"
#endif

#include "httplib/httplib.h"
#include "msgpack/msgpack.h"
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <limits>
#include <cassert>
#include <random>
#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/unreal/ZenoRemoteTypes.h>
#include <zeno/logger.h>

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

/**
 * Copy from https://stackoverflow.com/questions/440133/how-do-i-create-a-random-alpha-numeric-string-in-c
 * @param length string length
 * @return Random string
 */
std::string RandomString( size_t Length )
{
    auto randchar = []() -> char
    {
        const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(Length,0);
    std::generate_n( str.begin(), Length, randchar );
    return str;
}

namespace zeno {

namespace remote {

static struct Flags {
    bool IsMainProcess;

    Flags()
        : IsMainProcess(false)
    {}
} StaticFlags;

/**
 * Subject container
 */
static struct SubjectRegistry {
    std::map<std::string, SubjectContainer> Elements;
    std::map<std::string, std::map<std::string, zeno::remote::ParamValue>> SessionalParameters;
    std::function<void(const std::set<std::string>&)> Callback;

    /**
     * Push subject container to registry
     * @param InitList Subject container list
     */
    void Push(const std::vector<SubjectContainer>& InitList) {
        if (StaticFlags.IsMainProcess) {
            std::set<std::string> ChangeList;
            for (const SubjectContainer& Value : InitList) {
                ChangeList.emplace(Value.Name);
                Elements.try_emplace(Value.Name, Value);
            }
            if (Callback) {
                Callback(ChangeList);
            }
        } else {
            // In child process, transfer data with http
            httplib::Client Cli { "http://localhost:23343" };
            Cli.set_default_headers({ {"X-Zeno-SessionKey", ZENO_LOCAL_TOKEN} });
            SubjectContainerList List { InitList };
            std::vector<uint8_t> Data = msgpack::pack(List);
            Cli.Post("/subject/push", reinterpret_cast<const char*>(Data.data()), Data.size(), "application/binary");
        }
    }

    /**
     * Get subject container
     * @param Key Subject key
     * @return Subject container
     */
    template <typename T>
    std::optional<T> Get(const std::string& Key) {
        CONSTEXPR ESubjectType RequiredSubjectType = TGetClassSubjectType<T>::Value;
        if (StaticFlags.IsMainProcess) {
            auto TargetIter = Elements.find(Key);
            if (TargetIter == Elements.end()) return std::nullopt;
            if (TargetIter->second.Type != static_cast<int16_t>(RequiredSubjectType)) return std::nullopt;
            std::error_code Err;
            T Result = msgpack::unpack<T>(TargetIter->second.Data, Err);
            if (!Err) {
                return std::make_optional(Result);
            }
        } else {
            // In child process, transfer data with http
            httplib::Client Cli { "http://localhost:23343" };
            Cli.set_default_headers({ {"X-Zeno-SessionKey", ZENO_LOCAL_TOKEN} });
            httplib::Params Param;
            Param.insert(std::make_pair("key", Key));
            const httplib::Result Response = Cli.Get("/subject/fetch", Param, httplib::Headers {}, httplib::Progress {});
            if (Response) {
                const std::string& Body = Response->body;
                std::error_code Err;
                auto List = msgpack::unpack<struct SubjectContainerList>(reinterpret_cast<uint8_t*>(const_cast<char*>(Body.data())), Body.size(), Err);
                if (!Err) {
                    for (const auto& Subject : List.Data) {
                        if (Subject.Type == static_cast<int16_t>(RequiredSubjectType) && Key == Subject.Name) {
                            T Result = msgpack::unpack<T>(Subject.Data, Err);
                            if (!Err) {
                                return std::make_optional(Result);
                            }
                        }
                    }
                }
            }
        }
        return std::nullopt;
    }

    /**
     * Set parameter to session
     * @param SessionKey Session key
     * @param Key Parameter key
     * @param Value Parameter value
     */
    void SetParameter(const std::string& SessionKey, const std::string& Key, zeno::remote::ParamValue& Value) {
        if (StaticFlags.IsMainProcess) {
            auto ParamMapIter = SessionalParameters.find(SessionKey);
            if (ParamMapIter == SessionalParameters.end()) {
                auto [Iter, Success] = SessionalParameters.insert(std::make_pair(SessionKey, std::map<std::string, zeno::remote::ParamValue> {}));
                if (Success) {
                    ParamMapIter = Iter;
                }
            }

            assert(ParamMapIter != SessionalParameters.end());
            ParamMapIter->second.insert(std::make_pair(Key, Value));
        } else {
            // In child process, transfer data with http
            httplib::Client Cli { "http://localhost:23343" };
            Cli.set_default_headers({ {"X-Zeno-SessionKey", ZENO_LOCAL_TOKEN} });
            httplib::Params Param;
            Param.insert(std::make_pair("session_key", SessionKey));
            Param.insert(std::make_pair("key", Key));
            std::vector<uint8_t> Data = msgpack::pack(Value);
            const httplib::Result Response = Cli.Post("/graph/param/push", reinterpret_cast<const char*>(Data.data()), Data.size(), "application/binary");
        }
    }

    /**
     * Get parameter from session
     * @param SessionKey Session key
     * @param Key Parameter key
     * @return Parameter value
     */
    [[nodiscard]]
    const zeno::remote::ParamValue* GetParameter(const std::string& SessionKey, const std::string& Key) const {
        if (StaticFlags.IsMainProcess) {
            auto ParamMapIter = SessionalParameters.find(SessionKey);
            if (ParamMapIter != SessionalParameters.end()) {
                auto ParamIter = ParamMapIter->second.find(Key);
                if (ParamIter != ParamMapIter->second.end()) {
                    return &ParamIter->second;
                }
            }
            return nullptr;
        } else {
            // In child process, transfer data with http
            httplib::Client Cli { "http://localhost:23343" };
            Cli.set_default_headers({ {"X-Zeno-SessionKey", ZENO_LOCAL_TOKEN} });
            httplib::Params Param;
            Param.insert(std::make_pair("key", Key));
            const httplib::Result Response = Cli.Get("/graph/param/fetch", Param, httplib::Headers {}, httplib::Progress {});
            if (Response) {
                const std::string& Body = Response->body;
                std::error_code Err;
                auto List = msgpack::unpack<struct ParamValueBatch>(reinterpret_cast<uint8_t*>(const_cast<char*>(Body.data())), Body.size(), Err);
                if (!Err) {
                    for (const auto& Subject : List.Values) {
                        // Alloc new memory and copy it
                        // TODO [darc] : fix memory leak(might be) :
                        auto* NewValue = new zeno::remote::ParamValue;
                        *NewValue = Subject;
                        return NewValue;
                    }
                }
            }
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

ESubjectType NameToSubjectType(const std::string& InStr) {
    if (InStr == "StaticMeshNoUV") {
        return ESubjectType::Mesh;
    } else if (InStr == "HeightField") {
        return ESubjectType::HeightField;
    }
    return ESubjectType::Invalid;
};

}

#define SERVER_HANDLER_WRAPPER(FUNC) [this] (const httplib::Request& Req, httplib::Response& Res) { FUNC(Req, Res); }
#define HANDLER_SESSION_CHECK(FUNC) [this] (const httplib::Request& Req, httplib::Response& Res) { if ((Req.remote_addr == "127.0.0.1" && Req.has_header(ZENO_SESSION_HEADER_NAME) && Req.get_header_value(ZENO_SESSION_HEADER_NAME) == ZENO_LOCAL_TOKEN) || (Req.has_header(ZENO_SESSION_HEADER_NAME) && IsValidSession(Req.get_header_value(ZENO_SESSION_HEADER_NAME)))) { FUNC(Req, Res); } else { Res.status = 403; } }

class ZenoRemoteServer {

    inline const static std::string ZENO_SESSION_HEADER_NAME = "X-Zeno-SessionKey";

    httplib::Server Srv;
    remote::SubjectHistory History;
    std::vector<std::string> Sessions;

private:
    bool IsValidSession(const std::string& SessionKey) const;

    static void OnError(const httplib::Request& Req, httplib::Response& Res);

    static void IndexPage(const httplib::Request& Req, httplib::Response& Res);
    void NewSession(const httplib::Request& Req, httplib::Response& Res);

    void FetchDataDiff(const httplib::Request& Req, httplib::Response& Res);
    static void PushData(const httplib::Request& Req, httplib::Response& Res);
    static void FetchData(const httplib::Request& Req, httplib::Response& Res);
    static void ParseGraphInfo(const httplib::Request& Req, httplib::Response& Res);

    static void PushParameter(const httplib::Request& Req, httplib::Response& Res);
    static void FetchParameter(const httplib::Request& Req, httplib::Response& Res);

public:
    void Run() {
        zeno::remote::StaticRegistry.Callback = [this] (const std::set<std::string>& Changes) {
            History.Commit(Changes);
        };
        Srv.set_payload_max_length(1024 * 1024 * 1024); // 1 GB
        Srv.set_error_handler(OnError);
        Srv.Get("/", HANDLER_SESSION_CHECK(IndexPage));
        Srv.Get("/auth", SERVER_HANDLER_WRAPPER(NewSession));
        Srv.Get("/subject/diff", HANDLER_SESSION_CHECK(FetchDataDiff));
        Srv.Post("/subject/push", HANDLER_SESSION_CHECK(PushData));
        Srv.Get("/subject/fetch", HANDLER_SESSION_CHECK(FetchData));
        Srv.Post("/graph/parse", HANDLER_SESSION_CHECK(ParseGraphInfo));
        Srv.Post("/graph/param/push", HANDLER_SESSION_CHECK(PushParameter));
        Srv.Get("/graph/param/fetch", HANDLER_SESSION_CHECK(FetchParameter));
        zeno::remote::StaticFlags.IsMainProcess = true;
        Srv.listen("127.0.0.1", 23343);
        // Listen failed or server exited, set flag to false
        zeno::remote::StaticFlags.IsMainProcess = false;
    }
};

#undef HANDLER_SESSION_CHECK
#undef SERVER_HANDLER_WRAPPER

}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include "processthreadsapi.h"
#include "zeno/core/defNode.h"

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

bool ZenoRemoteServer::IsValidSession(const std::string &SessionKey) const {
    return std::find(Sessions.begin(), Sessions.end(), SessionKey) != Sessions.end();
}

void ZenoRemoteServer::IndexPage(const httplib::Request& Req, httplib::Response& Res) {
    Res.set_content(R"({"api_version": 1, "protocol": "msgpack"})", "application/json");
}

/**
 * GET /auth
 * HEADER X-Zeno-Token {string}
 * return session key in plain text with 204 if success
 * return status code 401 if failed
 */
void ZenoRemoteServer::NewSession(const httplib::Request &Req, httplib::Response &Res) {
    const static std::string HeaderKey = "X-Zeno-Token";
    if (Req.has_header(HeaderKey)) {
        std::string Token = Req.get_header_value(HeaderKey);
        if (Token == ZENO_REMOTE_TOKEN) {
            Res.status = 201;
            std::string SessionKey = RandomString(32);
            Res.set_content(SessionKey, "text/plain");
            Sessions.emplace_back(std::move(SessionKey));
            return;
        }
    }
    Res.status = 401;
}

/**
 * GET /subject/diff?client_version={int}
 * return zeno::remote::Diff
 */
void ZenoRemoteServer::FetchDataDiff(const httplib::Request &Req, httplib::Response &Res) {
    int32_t client_version = std::atoi(Req.get_param_value("client_version").c_str());
    std::set<std::string> Changes = History.Diff(client_version);
    std::vector<std::string> VChanges;
    VChanges.assign(Changes.begin(), Changes.end());
    remote::Diff Diff { std::move(VChanges), static_cast<int32_t>(History.GetTopIndex()) };
    std::vector<uint8_t> Data = msgpack::pack(Diff);
    Res.set_content(reinterpret_cast<const char*>(Data.data()), Data.size(), "application/binary");
}

/**
 * POST /subject/push
 * BODY msgpack packed data of zeno::remote::SubjectContainerList
 * return status code 204 if success, 400 if failed
 */
void ZenoRemoteServer::PushData(const httplib::Request& Req, httplib::Response &Res) {
    const std::string& Body = Req.body;
    std::error_code Err;
    auto Container = msgpack::unpack<remote::SubjectContainerList>(reinterpret_cast<const uint8_t*>(Body.data()), Body.size(), Err);
    if (!Err) {
        zeno::remote::StaticRegistry.Push(Container.Data);
        Res.status = 204;
    } else {
        Res.status = 400;
    }
}

/**
 * GET /subject/fetch?key={string}&key={string}  ...
 * return msgpack packed data of zeno::remote::SubjectContainerList, list will be empty if nothing found
 */
void ZenoRemoteServer::FetchData(const httplib::Request &Req, httplib::Response &Res) {
    size_t Num = Req.get_param_value_count("key");
    remote::SubjectContainerList List;
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

/**
 * POST /graph/parse
 * BODY a string of zsl file (json).
 * return msgpack packed data of zeno::remote::GraphInfo
 */
void ZenoRemoteServer::ParseGraphInfo(const httplib::Request &Req, httplib::Response &Res) {
    const std::string& BodyStr = Req.body;
    try {
        std::shared_ptr<zeno::Graph> NewGraph = zeno::getSession().createGraph();
        NewGraph->loadGraph(BodyStr.c_str());
    } catch (...) {
        Res.status = 400;
    }
}

/**
 * GET /graph/param/push
 * BODY msgpack packed data of zeno::remote::ParamValueBatch
 * return 204 if success, 400 if failed
 */
void ZenoRemoteServer::PushParameter(const httplib::Request &Req, httplib::Response &Res) {
    // Get session key
    const static std::string HeaderKey = "X-Zeno-Session";
    const std::string& SessionKey = Req.get_header_value(HeaderKey);
    // Fetch post body from request and parse it as msgpack
    const std::string& Body = Req.body;
    std::error_code Err;
    auto Container = msgpack::unpack<remote::ParamValueBatch>(reinterpret_cast<const uint8_t*>(Body.data()), Body.size(), Err);
    if (!Err) {
        for (auto& Param : Container.Values) {
            zeno::remote::StaticRegistry.SetParameter(SessionKey, Param.Name, Param);
        }
        Res.status = 204;
    } else {
        Res.status = 400;
    }
}

/**
 * GET /graph/param/fetch?key={string}&key={string}  ...
 * return msgpack packed data of zeno::remote::ParamValueBatch, batch will be empty if nothing found
 */
void ZenoRemoteServer::FetchParameter(const httplib::Request &Req, httplib::Response &Res) {
    // Get session key
    const static std::string HeaderKey = "X-Zeno-Session";
    const std::string& SessionKey = Req.get_header_value(HeaderKey);
    // Get requested parameter names
    size_t Num = Req.get_param_value_count("key");
    remote::ParamValueBatch Batch;
    Batch.Values.reserve(Num);
    for (size_t Idx = 0; Idx < Num; ++Idx) {
        std::string Key = Req.get_param_value("key", Idx);
        auto Value = zeno::remote::StaticRegistry.GetParameter(SessionKey, Key);
        if (Value) {
            Batch.Values.emplace_back( *Value );
        }
    }
    // Pack with msgpack and send
    std::vector<uint8_t> Data = msgpack::pack(Batch);
    Res.set_content(reinterpret_cast<const char*>(Data.data()), Data.size(), "application/binary");
}

void ZenoRemoteServer::OnError(const httplib::Request &Req, httplib::Response &Res) {
    Res.set_content(R"({"msg": "Opps, who am I, where am I, what am I doing?"})", "application/json");
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
            std::vector<std::array<remote::AnyNumeric, 3>> verts;
            std::vector<std::array<int32_t, 3>> tris;
            for (const std::array<float, 3>& data : prim->verts) {
                verts.push_back( { data.at(0), data.at(2), data.at(1) });
            }
            for (const std::array<int32_t, 3>& data : prim->tris) {
                tris.emplace_back(data);
            }
            remote::Mesh Mesh { std::move(verts), std::move(tris) };
            std::vector<uint8_t> Data = msgpack::pack(Mesh);
            zeno::remote::StaticRegistry.Push({ remote::SubjectContainer{ subject_name, static_cast<int16_t>(remote::ESubjectType::Mesh), std::move(Data) }, });
        } else if (processor_type == "HeightField") {
            if (prim->verts.has_attr("height")) {
                auto& HeightAttrs = prim->verts.attr<float>("height");
                // Currently height field are always square.
                const auto N = static_cast<int32_t>(std::round(std::sqrt(prim->verts.size())));
                std::vector<uint16_t> RemappedHeightFieldData;
                RemappedHeightFieldData.reserve(prim->verts.size());
                for (float Height : HeightAttrs) {
                    // Map height [-255, 255] in R to [0, UINT16_MAX] in Z
                    auto NewValue = static_cast<uint16_t>(((Height + 255.f) / (255.f * 2.f)) * std::numeric_limits<uint16_t>::max());
                    RemappedHeightFieldData.push_back(NewValue);
                }
                remote::HeightField HeightField { N, N, RemappedHeightFieldData };
                std::vector<uint8_t> Data = msgpack::pack(HeightField);
                zeno::remote::StaticRegistry.Push({ remote::SubjectContainer{ subject_name, static_cast<int16_t>(remote::ESubjectType::HeightField), std::move(Data) }, });
            } else {
                log_error(R"(Primitive type HeightField must have attribute "float")");
            }
        }
        set_output2("primRef", prim);
    }
};

ZENO_DEFNODE(TransferPrimitiveToUnreal)({
      {
              {"enum StaticMeshNoUV HeightField", "type", "StaticMeshNoUV"},
              {"string", "name", "SubjectFromZeno"},
              {"prim"},
          },
          { "primRef" },
          {},
          {"Unreal"},
      }
 );

struct ReadPrimitiveFromRegistry : public INode {
    template <typename T>
    std::shared_ptr<zeno::PrimitiveObject> ToPrimitiveObject(T& Data) {
        assert(false);
        return nullptr;
    }

    template <>
    std::shared_ptr<zeno::PrimitiveObject> ToPrimitiveObject(zeno::remote::Mesh& Data) {
        std::shared_ptr<zeno::PrimitiveObject> Prim = std::make_shared<zeno::PrimitiveObject>();
        // Triangles
        Prim->tris.reserve(Data.triangles.size());
        for (const auto& [x, z, y] : Data.triangles) {
            Prim->tris.emplace_back(x, y, z);
        }
        // Vertices
        for (const auto& [a, b, c] : Data.vertices) {
            Prim->verts.emplace_back(a.data(), c.data(), b.data());
        }

        return Prim;
    }

    template <>
    std::shared_ptr<zeno::PrimitiveObject> ToPrimitiveObject(zeno::remote::HeightField& Data) {
        std::shared_ptr<zeno::PrimitiveObject> Prim = std::make_shared<zeno::PrimitiveObject>();
        size_t Nx = std::max(get_input2<int>("nx"), Data.Nx);
        size_t Ny = std::max(get_input2<int>("ny"), Data.Ny);
        float Dx = 1.f / std::max((float)Nx - 1.f, 1.f);
        float Dy = 1.f / std::max((float)Ny - 1.f, 1.f);
        vec3f ax {1, 0, 0};
        vec3f ay {0, 0, 1};
        float Scale = get_input2<float>("scale");
        vec3f o = (ax + ay) / 2;
        ax *= Dx; ay *= Dy;
        ax *= Scale;
        ay *= Scale;
        Prim->resize(Nx * Ny);

        auto &pos = Prim->add_attr<vec3f>("pos");
#pragma omp parallel for collapse(2)
        for (intptr_t y = 0; y < Ny; y++)
            for (intptr_t x = 0; x < Nx; x++) {
                vec3f p = o + x * ax + y * ay;
                size_t i = x + y * Nx;
                pos[i] = p;
            }
        Prim->tris.resize((Nx - 1) * (Ny - 1) * 2);
#pragma omp parallel for collapse(2)
        for (intptr_t y = 0; y < Ny - 1; y++)
            for (intptr_t x = 0; x < Nx - 1; x++) {
                intptr_t index = y * (Nx - 1) + x;
                Prim->tris[index * 2][2] = y * Nx + x;
                Prim->tris[index * 2][1] = y * Nx + x + 1;
                Prim->tris[index * 2][0] = (y + 1) * Nx + x + 1;
                Prim->tris[index * 2 + 1][2] = (y + 1) * Nx + x + 1;
                Prim->tris[index * 2 + 1][1] = (y + 1) * Nx + x;
                Prim->tris[index * 2 + 1][0] = y * Nx + x;
            }

        auto& Arr = Prim->verts.add_attr<float>("height");
        size_t Idx = 0;
        for (const auto& Row : Data.Data) {
            for (const uint16_t Height : Row) {
                Arr[Idx] = ((float)Height / std::numeric_limits<uint16_t>::max()) * (255.f * 2) - 255.f;
                Prim->verts[Idx] = { Prim->verts[Idx].at(0), Arr[Idx], Prim->verts[Idx].at(2) };
                Idx++;
            }
        }

        return Prim;
    }

    void apply() override {
        zeno::remote::StaticFlags.IsMainProcess = false;
        std::string subject_name = get_input2<std::string>("name");
        remote::ESubjectType Type = remote::NameToSubjectType(get_input2<std::string>("type"));
        std::shared_ptr<zeno::PrimitiveObject> OutPrim;
        if (Type == remote::ESubjectType::Mesh) {
            std::optional<remote::Mesh> Data = remote::StaticRegistry.Get<remote::Mesh>(subject_name);
            if (Data.has_value()) {
                OutPrim = ToPrimitiveObject(Data.value());
            }
        } else if (Type == remote::ESubjectType::HeightField) {
            std::optional<remote::HeightField> Data = remote::StaticRegistry.Get<remote::HeightField>(subject_name);
            if (Data.has_value()) {
                OutPrim = ToPrimitiveObject(Data.value());
            }
        }
        if (!OutPrim) {
            zeno::log_error("Prim data not found.");
            return;
        }
        set_output2("prim", OutPrim);
    }
};

ZENO_DEFNODE(ReadPrimitiveFromRegistry)({
    {
        {"string", "name", "SubjectFromZeno"},
        {"enum StaticMeshNoUV HeightField", "type", "StaticMeshNoUV"},
        {"int", "nx", "0"},
        {"int", "ny", "0"},
        {"float", "scale", "250"},
    },
    { "prim" },
    {},
    { "Unreal" }
});

}
