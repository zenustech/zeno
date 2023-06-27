#ifndef ZENO_REMOTE_TOKEN
#define ZENO_REMOTE_TOKEN "ZENO_DEFAULT_TOKEN"
#endif // ZENO_REMOTE_TOKEN

#ifndef ZENO_LOCAL_TOKEN
#define ZENO_LOCAL_TOKEN "ZENO_LOCAL_TOKEN"
#endif // ZENO_LOCAL_TOKEN

#ifndef ZENO_TOOL_SERVER_ADDRESS
#define ZENO_TOOL_SERVER_ADDRESS "http://localhost:23343"
#endif // ZENO_TOOL_SERVER_ADDRESS

#ifndef ZENO_SESSION_HEADER_KEY
#define ZENO_SESSION_HEADER_KEY "X-Zeno-SessionKey"
#endif // ZENO_SESSION_HEADER_KEY

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
#include <memory>
#include <stdexcept>
#include <zeno/core/INode.h>
#include <zeno/core/Session.h>
#include <zeno/core/Graph.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/GlobalComm.h>
#include <zeno/extra/GraphException.h>
#include <zeno/extra/EventCallbacks.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/unreal/ZenoRemoteTypes.h>
#include <zeno/unreal/UnrealTool.h>
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

namespace zeno {

namespace remote {

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
}

}

#define SERVER_HANDLER_WRAPPER(FUNC) [this] (const httplib::Request& Req, httplib::Response& Res) { FUNC(Req, Res); }
#define HANDLER_SESSION_CHECK(FUNC) [this] (const httplib::Request& Req, httplib::Response& Res) { if ((Req.remote_addr == "127.0.0.1" && Req.has_header(ZENO_SESSION_HEADER_NAME) && Req.get_header_value(ZENO_SESSION_HEADER_NAME) == ZENO_LOCAL_TOKEN) || (Req.has_header(ZENO_SESSION_HEADER_NAME) && IsValidSession(Req.get_header_value(ZENO_SESSION_HEADER_NAME)))) { FUNC(Req, Res); } else { Res.status = 403; } }

class ZenoRemoteServer {

    inline const static std::string ZENO_SESSION_HEADER_NAME = ZENO_SESSION_HEADER_KEY;

    httplib::Server Srv;
    std::map<zeno::remote::SessionKeyType, remote::SubjectHistory> History;
    std::vector<std::string> Sessions;

private:
    bool IsValidSession(const std::string& SessionKey) const;
    remote::SubjectHistory& GetGlobalHistory();

    static void OnError(const httplib::Request& Req, httplib::Response& Res);

    static void IndexPage(const httplib::Request& Req, httplib::Response& Res);
    void NewSession(const httplib::Request& Req, httplib::Response& Res);

    void FetchDataDiff(const httplib::Request& Req, httplib::Response& Res);
    static void PushData(const httplib::Request& Req, httplib::Response& Res);
    void FetchData(const httplib::Request& Req, httplib::Response& Res);

    static void ParseGraphInfo(const httplib::Request& Req, httplib::Response& Res);
    static void RunGraph(const httplib::Request& Req, httplib::Response& Res);

    static void PushParameter(const httplib::Request& Req, httplib::Response& Res);
    static void FetchParameter(const httplib::Request& Req, httplib::Response& Res);

    static void SetCurrentSession(const httplib::Request& Req, httplib::Response& Res);
    static void GetCurrentSession(const httplib::Request& Req, httplib::Response& Res);

    static std::string ParseSessionKey(const zeno::remote::SessionKeyType& SessionKey);
    static std::string ParseSessionKey(const httplib::Request &Req);

public:
    void Run() {
        zeno::remote::StaticRegistry.Callback = [this] (const std::set<std::string>& Changes, const zeno::remote::SessionKeyType& SessionKey) {
            auto HistoryIter = History.find(SessionKey);
            if (HistoryIter == History.end()) {
                HistoryIter = History.emplace(SessionKey, zeno::remote::SubjectHistory{}).first;
            }
            auto& HistoryObj = HistoryIter->second;

            HistoryObj.Commit(Changes);
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
        Srv.Post("/graph/run", HANDLER_SESSION_CHECK(RunGraph));
        Srv.Get("/graph/param/fetch", HANDLER_SESSION_CHECK(FetchParameter));
        Srv.Get("/session/set", HANDLER_SESSION_CHECK(SetCurrentSession));
        Srv.Get("/session/current", HANDLER_SESSION_CHECK(GetCurrentSession));
        Srv.set_write_timeout(120);
        Srv.listen("127.0.0.1", 23343);
        // Listen failed or server exited, set flag to false
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

// Those function will be called after static initialization, it is safe. Just ignore CTidy.
[[maybe_unused]] static int defUnrealToolInit =
    getSession().eventCallbacks->hookEvent("init", [] {
        zeno::remote::StaticFlags.SetIsMainProcess(true);
        StartServerThread();
    });

[[maybe_unused]] static int defUnrealToolRunnerInit =
    getSession().eventCallbacks->hookEvent("preRunnerStart", [] {
        zeno::remote::StaticFlags.SetIsMainProcess(false);
    });

bool ZenoRemoteServer::IsValidSession(const std::string &SessionKey) const {
    return std::find(Sessions.begin(), Sessions.end(), SessionKey) != Sessions.end();
}

void ZenoRemoteServer::IndexPage(const httplib::Request &Req, httplib::Response &Res) {
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
        // TODO [darc] : refactor session mechanism
        if (Token == ZENO_REMOTE_TOKEN) {
            Res.status = 201;
            // Use token as session for now
            // std::string SessionKey = zeno::remote::RandomString2(32);
            std::string SessionKey = Token;
            Res.set_content(SessionKey, "text/plain");
            if (std::find(Sessions.begin(), Sessions.end(), SessionKey) == Sessions.end()) {
                Sessions.emplace_back(std::move(SessionKey));
            }
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
    const std::string& SessionKey = ParseSessionKey(Req);
    auto HistoryIter = History.find(SessionKey);
    if (HistoryIter == History.end()) {
        History.emplace(SessionKey, zeno::remote::SubjectHistory{});
        HistoryIter = History.find(SessionKey);
    }
    auto& HistoryObj = HistoryIter->second;
    auto& HistoryGlobal = GetGlobalHistory();

    int32_t client_version = std::atoi(Req.get_param_value("client_version").c_str());
    std::set<std::string> Changes = HistoryObj.Diff(client_version);
    std::set<std::string> GlobalSubjects = HistoryGlobal.Diff(0);
    std::vector<std::string> VChanges;
    VChanges.assign(Changes.begin(), Changes.end());
    VChanges.insert(VChanges.end(), GlobalSubjects.begin(), GlobalSubjects.end());
    remote::Diff Diff{std::move(VChanges), static_cast<int32_t>(HistoryObj.GetTopIndex())};
    std::vector<uint8_t> Data = msgpack::pack(Diff);
    Res.set_content(reinterpret_cast<const char *>(Data.data()), Data.size(), "application/binary");
}

/**
 * POST /subject/push
 * BODY msgpack packed data of zeno::remote::SubjectContainerList
 * return status code 204 if success, 400 if failed
 */
void ZenoRemoteServer::PushData(const httplib::Request &Req, httplib::Response &Res) {
    const std::string& SessionKey = ParseSessionKey(Req);
    if (SessionKey.empty() && Req.has_param("session_key")) {
        std::string ParamSessionKey = Req.get_param_value("session_key");
    }

    const std::string &Body = Req.body;
    std::error_code Err;
    auto Container =
        msgpack::unpack<remote::SubjectContainerList>(reinterpret_cast<const uint8_t *>(Body.data()), Body.size(), Err);
    if (!Err) {
        zeno::remote::StaticRegistry.Push(Container.Data, SessionKey);
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
    std::string SessionKey = Req.has_param("session_key") ? Req.get_param_value("session_key") : ParseSessionKey(Req);
    if (SessionKey.empty() && !zeno::remote::StaticFlags.IsMainProcess()) {
        SessionKey = zeno::remote::StaticFlags.GetCurrentSession();
    }
    bool bSearchAllSession = false;
    if (Req.has_param("search_all_session")) {
        bSearchAllSession = Req.get_param_value("search_all_session") == "true";
    }

    auto& Elements = zeno::remote::StaticRegistry.GetOrCreateSessionElement(SessionKey);
    auto& GlobalElements = GetOrCreate(remote::StaticRegistry.SessionalElements, std::string{ "" });

    size_t Num = Req.get_param_value_count("key");
    remote::SubjectContainerList List;
    List.Data.reserve(Num);
    for (size_t Idx = 0; Idx < Num; ++Idx) {
        std::string Key = Req.get_param_value("key", Idx);
        auto Value = Elements.find(Key);
        if (Value != Elements.end()) {
            List.Data.emplace_back(Value->second);
            break;
        }
        Value = GlobalElements.find(Key);
        if (Value != GlobalElements.end()) {
            List.Data.emplace_back(Value->second);
            break;
        }
        // Search all session if specified
        if (bSearchAllSession) {
            for (auto& Session : Sessions) {
                auto& Elements = zeno::remote::StaticRegistry.GetOrCreateSessionElement(Session);
                auto Value = Elements.find(Key);
                if (Value != Elements.end()) {
                    List.Data.emplace_back(Value->second);
                    break;
                }
            }
        }
    }
    std::vector<uint8_t> Data = msgpack::pack(List);
    Res.set_content(reinterpret_cast<const char *>(Data.data()), Data.size(), "application/binary");
}

/**
 * POST /graph/parse
 * BODY a string of zsl file (json).
 * return msgpack packed data of zeno::remote::GraphInfo
 */
void ZenoRemoteServer::ParseGraphInfo(const httplib::Request &Req, httplib::Response &Res) {
    const std::string &BodyStr = Req.body;
    zeno::remote::GraphInfo GraphInfo;
    GraphInfo.bIsValid = false;
    try {
        auto &Session = zeno::getSession();
        std::shared_ptr<zeno::Graph> NewGraph = Session.createGraph();
        NewGraph->loadGraph(BodyStr.c_str());
        // Find nodes with class name "DeclareRemoteParameter"
        std::unique_ptr<INodeClass> &DeclareNodeClass =
            zeno::safe_at(Session.nodeClasses, "DeclareRemoteParameter", "node class not found");
        std::unique_ptr<INodeClass> &OutputNodeClass =
            zeno::safe_at(Session.nodeClasses, "SetExecutionResult", "node class not found");
        for (const auto &[NodeName, NodeClass] : NewGraph->nodes) {
            if (NodeClass->nodeClass == DeclareNodeClass.get()) {
                zeno::StringObject *InputName = dynamic_cast<zeno::StringObject *>(
                    zeno::safe_at(NodeClass->inputs, "name", "name not found").get());
                zeno::StringObject *InputType = dynamic_cast<zeno::StringObject *>(
                    zeno::safe_at(NodeClass->inputs, "type", "type not found").get());
                if (InputName == nullptr || InputType == nullptr) {
                    throw std::runtime_error("name or type not found");
                }
                // TODO [darc] : Support passing default value :
                GraphInfo.InputParameters.insert(
                    std::make_pair(InputName->value,
                                   zeno::remote::ParamDescriptor{
                                       InputName->value,
                                       static_cast<int8_t>(zeno::remote::GetParamTypeFromString(InputType->value))}));
            }
            if (NodeClass->nodeClass == OutputNodeClass.get()) {
                zeno::StringObject *InputName = dynamic_cast<zeno::StringObject *>(
                    zeno::safe_at(NodeClass->inputs, "name", "name not found").get());
                zeno::StringObject *InputType = dynamic_cast<zeno::StringObject *>(
                    zeno::safe_at(NodeClass->inputs, "type", "type not found").get());
                if (InputName == nullptr || InputType == nullptr) {
                    throw std::runtime_error("name or type not found");
                }
                GraphInfo.OutputParameters.insert(std::make_pair(
                    InputName->value,
                    zeno::remote::ParamDescriptor{InputName->value,
                                                  static_cast<int16_t>(remote::NameToSubjectType(InputType->value))}));
            }
        }
        GraphInfo.bIsValid = true;
        Res.status = 200;
    } catch (...) {
        Res.status = 400;
    }
    std::vector<uint8_t> Data = msgpack::pack(GraphInfo);
    Res.set_content(reinterpret_cast<const char *>(Data.data()), Data.size(), "application/binary");
}

/**
 * POST /graph/run
 * BODY msgpack packed data of zeno::remote::GraphRunInfo
 * return msgpack packed data of zeno::remote::GraphRunResult
 */
void ZenoRemoteServer::RunGraph(const httplib::Request &Req, httplib::Response &Res) {
    const std::string& SessionKey = ParseSessionKey(Req);
    const std::string& BodyStr = Req.body;
    std::error_code Err;
    auto RunInfo = msgpack::unpack<zeno::remote::GraphRunInfo>(reinterpret_cast<const uint8_t *>(BodyStr.data()), BodyStr.size(), Err);
    if (!Err) {
        try {
            // Initialize graph
            auto& Session = zeno::getSession();
            Session.globalState->clearState();
            Session.globalComm->clearState();
            Session.globalStatus->clearState();
            auto Graph = Session.createGraph();
            Graph->loadGraph(RunInfo.GraphDefinition.c_str());
            // Set input parameters
            for (auto& Param : RunInfo.Values.Values) {
                zeno::remote::StaticRegistry.SetParameter(SessionKey, Param.Name, Param);
            }
            remote::StaticFlags.CurrentSession = SessionKey;
            Session.globalState->frameid = 0;
            Session.globalComm->newFrame();
            Session.globalState->frameBegin();
            // Run graph
            while (Session.globalState->substepBegin()) {
                GraphException::catched([&] {Graph->applyNodesToExec();}, *Session.globalStatus);
                Session.globalState->substepEnd();
            }
            Session.globalComm->finishFrame();
            remote::StaticFlags.CurrentSession = "";
            Res.status = 204;
        } catch (...) {
            Res.status = 400;
        }
    }
    else {
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
    const std::string &SessionKey = ParseSessionKey(Req);
    // Fetch post body from request and parse it as msgpack
    const std::string &Body = Req.body;
    std::error_code Err;
    auto Container =
        msgpack::unpack<remote::ParamValueBatch>(reinterpret_cast<const uint8_t *>(Body.data()), Body.size(), Err);
    if (!Err) {
        for (auto &Param : Container.Values) {
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
    const std::string &SessionKey = ParseSessionKey(Req);
    // Get requested parameter names
    size_t Num = Req.get_param_value_count("key");
    remote::ParamValueBatch Batch;
    Batch.Values.reserve(Num);
    for (size_t Idx = 0; Idx < Num; ++Idx) {
        std::string Key = Req.get_param_value("key", Idx);
        auto Value = zeno::remote::StaticRegistry.GetParameter(SessionKey, Key);
        if (Value) {
            Batch.Values.emplace_back(*Value);
        }
    }
    // Pack with msgpack and send
    std::vector<uint8_t> Data = msgpack::pack(Batch);
    Res.set_content(reinterpret_cast<const char *>(Data.data()), Data.size(), "application/binary");
}

/**
 * GET /session/set
 * @param Req
 * @param Res
 */
void ZenoRemoteServer::SetCurrentSession(const httplib::Request &Req, httplib::Response &Res) {
    // Get session key
    const static std::string HeaderKey = ZenoRemoteServer::ZENO_SESSION_HEADER_NAME;
    const std::string &SessionKey = Req.get_header_value(HeaderKey);

    // TODO [darc] : fix race condition here :
    if (!remote::StaticFlags.CurrentSession.empty()) {
        Res.status = 409;
    } else {
        remote::StaticFlags.CurrentSession = SessionKey;
        Res.status = 204;
    }
}

/**
 * GET /session/current
 * return current session key
 */
void ZenoRemoteServer::GetCurrentSession(const httplib::Request &Req, httplib::Response &Res) {
    Res.set_content(remote::StaticFlags.CurrentSession, "text/plain");
}

void ZenoRemoteServer::OnError(const httplib::Request &Req, httplib::Response &Res) {
    Res.set_content(R"({"msg": "Oops, who am I, where am I, what am I doing?"})", "application/json");
}

std::string ZenoRemoteServer::ParseSessionKey(const remote::SessionKeyType &SessionKey) {
    if (SessionKey == ZENO_LOCAL_TOKEN) {
        return "";
    }
    return SessionKey;
}

std::string ZenoRemoteServer::ParseSessionKey(const httplib::Request &Req) {
    return ParseSessionKey(Req.get_header_value(ZENO_SESSION_HEADER_NAME));
}

remote::SubjectHistory &ZenoRemoteServer::GetGlobalHistory() {
    auto HistoryIter = History.find("");
    if (HistoryIter == History.end()) {
        HistoryIter = History.emplace("", remote::SubjectHistory{}).first;
    }
    return HistoryIter->second;
}

}

namespace zeno {

template <remote::ESubjectType Type>
struct IObjectExtractor {
    remote::SubjectContainer operator()(IObject* Node, const std::string& InName = "", const std::map<std::string, std::string>& InMeta = {}) {
        return { std::string{}, static_cast<int16_t>(remote::ESubjectType::Invalid), std::vector<uint8_t>{} };
    }
};

template <>
struct IObjectExtractor<remote::ESubjectType::Mesh> {
    remote::SubjectContainer operator()(IObject* Node, const std::string& InName = "", const std::map<std::string, std::string>& InMeta = {}) {
        auto* PrimObj = safe_dynamic_cast<PrimitiveObject>(Node);
        std::vector<std::array<remote::AnyNumeric, 3>> verts;
        std::vector<std::array<int32_t, 3>> tris;
        for (const std::array<float, 3>& data : PrimObj->verts) {
            verts.push_back( { data.at(0), data.at(2), data.at(1) });
        }
        for (const std::array<int32_t, 3>& data : PrimObj->tris) {
            tris.emplace_back(data);
        }
        remote::Mesh Mesh { std::move(verts), std::move(tris) };
        Mesh.Meta = InMeta;
        std::vector<uint8_t> Data = msgpack::pack(Mesh);
        return remote::SubjectContainer{ InName, static_cast<int16_t>(remote::ESubjectType::Mesh), std::move(Data) };
    }
};

template <>
struct IObjectExtractor<remote::ESubjectType::HeightField> {
    remote::SubjectContainer operator()(IObject* Node, const std::string& InName = "", const std::map<std::string, std::string>& InMeta = {}) {
        auto* PrimObj = safe_dynamic_cast<PrimitiveObject>(Node);
        if (PrimObj->verts.has_attr("height")) {
            auto& HeightAttrs = PrimObj->verts.attr<float>("height");
            // Currently height field are always square.
            const auto N = static_cast<int32_t>(std::round(std::sqrt(PrimObj->verts.size())));
            std::vector<uint16_t> RemappedHeightFieldData;
            RemappedHeightFieldData.reserve(PrimObj->verts.size());
            for (float Height : HeightAttrs) {
                // Map height [-255, 255] in R to [0, UINT16_MAX] in Z
                constexpr uint16_t uint16Max = std::numeric_limits<uint16_t>::max();
                // LandscapeDataAccess.h: static_cast<uint16>(FMath::RoundToInt(FMath::Clamp<float>(Height * LANDSCAPE_INV_ZSCALE + MidValue, 0.f, MaxValue)))
                auto NewValue = static_cast<uint16_t>(std::round(zeno::clamp(Height * UE_LANDSCAPE_ZSCALE + 0x8000, 0.f, static_cast<float>(uint16Max))));
                RemappedHeightFieldData.push_back(NewValue);
            }
            remote::HeightField HeightField { N, N, RemappedHeightFieldData };
            HeightField.Meta = InMeta;
            std::vector<uint8_t> Data = msgpack::pack(HeightField);
            return remote::SubjectContainer{ InName, static_cast<int16_t>(remote::ESubjectType::HeightField), std::move(Data) };
        } else {
            log_error(R"(Primitive type HeightField must have attribute "float")");
            return { std::string{}, static_cast<int16_t>(remote::ESubjectType::Invalid), std::vector<uint8_t>{} };
        }
    }
};

template <>
struct IObjectExtractor<remote::ESubjectType::PointSet> {
    remote::SubjectContainer operator()(IObject* Node, const std::string& InName = "", const std::map<std::string, std::string>& InMeta = {}) {
        auto* PrimObj = safe_dynamic_cast<PrimitiveObject>(Node);
        if (PrimObj->size() == 0) {
            log_error(R"(Primitive type HeightField must have attribute "float")");
            return { std::string{}, static_cast<int16_t>(remote::ESubjectType::Invalid), std::vector<uint8_t>{} };
        }
        zeno::remote::PointSet Result;
        Result.Points.reserve(PrimObj->size());
        for (const auto& Point : PrimObj->verts) {
            // Inverse Z and Y
            Result.Points.push_back({ Point.at(0), Point.at(2), Point.at(1) });
        }
        Result.Meta = InMeta;
        std::vector<uint8_t> Data = msgpack::pack(Result);
        return remote::SubjectContainer{ InName, static_cast<int16_t>(remote::ESubjectType::PointSet), std::move(Data) };
    }
};

struct TransferPrimitiveToUnreal : public INode {
    void apply() override {
        std::string processor_type = get_input2<std::string>("type");
        std::string subject_name = get_input2<std::string>("name");
        std::shared_ptr<PrimitiveObject> prim = get_input2<PrimitiveObject>("prim");
        if (processor_type == "StaticMeshNoUV") {
            remote::SubjectContainer NewSubject = IObjectExtractor<remote::ESubjectType::Mesh>{}(prim.get(), subject_name);
            zeno::remote::StaticRegistry.Push({ std::move(NewSubject), });
        } else if (processor_type == "HeightField") {
            remote::SubjectContainer NewSubject = IObjectExtractor<remote::ESubjectType::HeightField>{}(prim.get(), subject_name);
            zeno::remote::StaticRegistry.Push({ std::move(NewSubject), });
        } else if (processor_type == "Points") {
            remote::SubjectContainer NewSubject = IObjectExtractor<remote::ESubjectType::PointSet>{}(prim.get(), subject_name);
            zeno::remote::StaticRegistry.Push({ std::move(NewSubject), });
        } else {
            log_error("Unknown processor type: " + processor_type);
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

using SavePrimitiveToGlobalRegistry = TransferPrimitiveToUnreal;
ZENO_DEFNODE(SavePrimitiveToGlobalRegistry)({
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
        int32_t Nx = std::max(get_input2<int32_t>("nx"), Data.Nx);
        int32_t Ny = std::max(get_input2<int32_t>("ny"), Data.Ny);
        float Scale = get_input2<float>("scale");
        return zeno::remote::ConvertHeightDataToPrimitiveObject(Data, Nx, Ny, { Scale, Scale, Scale });
    }

    void apply() override {
        std::string subject_name = get_input2<std::string>("name");
        remote::ESubjectType Type = remote::NameToSubjectType(get_input2<std::string>("type"));
        std::shared_ptr<zeno::PrimitiveObject> OutPrim;
        if (Type == remote::ESubjectType::Mesh) {
            std::optional<remote::Mesh> Data = remote::StaticRegistry.Get<remote::Mesh>(subject_name, zeno::remote::StaticFlags.GetCurrentSession(), true);
            if (Data.has_value()) {
                OutPrim = ToPrimitiveObject(Data.value());
            }
        } else if (Type == remote::ESubjectType::HeightField) {
            std::optional<remote::HeightField> Data = remote::StaticRegistry.Get<remote::HeightField>(subject_name, zeno::remote::StaticFlags.GetCurrentSession(), true);
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

struct DeclareRemoteParameter : public INode {
    void apply() override {
        const std::string& Name = get_input2<std::string>("name");
        const std::string& Type = get_input2<std::string>("type");
        const remote::EParamType ParamType = remote::GetParamTypeFromString(Type);
        if (ParamType == remote::EParamType::Invalid) {
            zeno::log_error("Invalid parameter type: {}", Type);
            return;
        }
        const std::string& SessionKey = zeno::remote::StaticFlags.GetCurrentSession();
        if (SessionKey.empty()) {
            zeno::log_warn("No session set in main process.");
        }

        const zeno::remote::ParamValue* Value = zeno::remote::StaticRegistry.GetParameter(SessionKey, Name);
        if (Value) {
            if (ParamType == remote::EParamType::Float) {
                const float Data = Value->Cast<float>();
                set_output2("ParamValue", std::make_shared<zeno::NumericObject>(Data));
            } else if (ParamType == remote::EParamType::Integer) {
                const int32_t Data = Value->Cast<int32_t>();
                set_output2("ParamValue", std::make_shared<zeno::NumericObject>(Data));
            }
            return;
        }

        // If there is no parameter with this name, return default value
        if (has_input("DefaultValue")) {
            set_output2("ParamValue", get_input("DefaultValue"));
        } else {
            zeno::log_error("Parameter {} not found.", Name);
        }
    }
};

ZENO_DEFNODE(DeclareRemoteParameter) ({
    {
        {"string", "name", "ParamA"},
        {"enum Integer Float", "type", "Integer"},
        { "DefaultValue" },
    },
    { "ParamValue" },
    {},
    { "Unreal" }
});

struct SetExecutionResult : public INode {
    void apply() override {
        const std::string ProcessorType = get_input2<std::string>("type");
        const remote::ESubjectType Type = remote::NameToSubjectType(ProcessorType);
        const std::string SubjectName = get_input2<std::string>("name");
        std::shared_ptr<zeno::remote::MetaData> MetaData = nullptr;
        if (has_input("meta")) {
            MetaData = get_input2<zeno::remote::MetaData>("meta");
        } else {
            MetaData = std::make_shared<zeno::remote::MetaData>();
        }
        std::shared_ptr<zeno::IObject> Value = get_input<zeno::IObject>("value");
        if (!Value) {
            zeno::log_error("No value provided.");
            return;
        }
        const std::string SessionKey = zeno::remote::StaticFlags.GetCurrentSession();
        std::map<std::string, std::string> Meta;
        if (MetaData) {
            Meta = MetaData->Data;
        }
        if (Type == remote::ESubjectType::Mesh) {
            remote::SubjectContainer NewSubject = IObjectExtractor<remote::ESubjectType::Mesh>{}(Value.get(), SubjectName, Meta);
            remote::StaticRegistry.Push( { NewSubject }, SessionKey);
        } else if (Type == remote::ESubjectType::HeightField) {
            remote::SubjectContainer NewSubject = IObjectExtractor<remote::ESubjectType::HeightField>{}(Value.get(), SubjectName, Meta);
            remote::StaticRegistry.Push( { NewSubject }, SessionKey);
        } else if (Type == remote::ESubjectType::PointSet) {
            remote::SubjectContainer NewSubject = IObjectExtractor<remote::ESubjectType::PointSet>{}(Value.get(), SubjectName, Meta);
            remote::StaticRegistry.Push( { NewSubject }, SessionKey);
        } else {
            log_error("Unknown processor type: " + ProcessorType);
        }
    }
};

ZENO_DEFNODE(SetExecutionResult) ({
{
        {"string", "name", "OutputA"},
        { "value" },
        {"enum StaticMeshNoUV HeightField", "type", "StaticMeshNoUV"},
        { "meta" },
    },
    {
    },
    {},
    { "Unreal" }
});

}
