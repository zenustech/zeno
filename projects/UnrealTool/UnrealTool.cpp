#include "zeno/unreal/UnrealTool.h"
#include "zeno/types/PrimitiveObject.h"

std::string zeno::remote::RandomString2(size_t Length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 61);
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    std::string str(Length,0);
    std::generate_n( str.begin(), Length, [&](){ return charset[ dis(gen) ]; } );
    return str;
}

std::string zeno::remote::RandomString(size_t Length) {
    auto randchar = []() -> char
    {
      const char charset[] =
          "0123456789"
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
          "abcdefghijklmnopqrstuvwxyz";
      const size_t max_index = (sizeof(charset) - 1);
      // Use C++ 11 random
      return charset[ rand() % max_index ];
    };
    std::string str(Length,0);
    std::generate_n( str.begin(), Length, randchar );
    return str;
}
std::shared_ptr<zeno::PrimitiveObject>
zeno::remote::ConvertHeightDataToPrimitiveObject(const zeno::remote::HeightField &InHeightData, int Nx, int Ny, const float Scale) {
    std::shared_ptr<zeno::PrimitiveObject> Prim = std::make_shared<zeno::PrimitiveObject>();
    Nx = std::max(Nx, InHeightData.Nx);
    Ny = std::max(Ny, InHeightData.Ny);
    float Dx = 1.f / std::max((float)Nx - 1.f, 1.f);
    float Dy = 1.f / std::max((float)Ny - 1.f, 1.f);
    vec3f ax {1, 0, 0};
    vec3f ay {0, 0, 1};
    vec3f o = -((ax + ay) / 2);
    ax *= Dx; ay *= Dy;
    ax *= Scale;
    ay *= Scale;
    Prim->resize(Nx * Ny);

    auto &pos = Prim->add_attr<vec3f>("pos");
#pragma omp parallel for collapse(2)
    for (int32_t y = 0; y < Ny; y++)
        for (int32_t x = 0; x < Nx; x++) {
            vec3f p = o + x * ax + y * ay;
            size_t i = x + y * Nx;
            pos[i] = p;
        }
    Prim->tris.resize((Nx - 1) * (Ny - 1) * 2);
#pragma omp parallel for collapse(2)
    for (int32_t y = 0; y < Ny - 1; y++)
        for (int32_t x = 0; x < Nx - 1; x++) {
            int32_t index = y * (Nx - 1) + x;
            Prim->tris[index * 2][2] = y * Nx + x;
            Prim->tris[index * 2][1] = y * Nx + x + 1;
            Prim->tris[index * 2][0] = (y + 1) * Nx + x + 1;
            Prim->tris[index * 2 + 1][2] = (y + 1) * Nx + x + 1;
            Prim->tris[index * 2 + 1][1] = (y + 1) * Nx + x;
            Prim->tris[index * 2 + 1][0] = y * Nx + x;
        }

    auto& Arr = Prim->verts.add_attr<float>("height");
    size_t Idx = 0;
    for (const auto& Row : InHeightData.Data) {
        for (const uint16_t Height : Row) {
            Arr[Idx] = ((float)Height / std::numeric_limits<uint16_t>::max()) * (255.f * 2) - 255.f;
            Prim->verts[Idx] = { Prim->verts[Idx].at(0), Arr[Idx], Prim->verts[Idx].at(2) };
            Idx++;
        }
    }

    return Prim;
}

std::string zeno::remote::Flags::GetCurrentSession() {
    if (IsMainProcess()) {
        std::string Result = CurrentSession;
        return Result;
    } else if (!CurrentSession.empty()) {
        return CurrentSession;
    } else {
        // Request session key from main process
        httplib::Client Cli { ZENO_TOOL_SERVER_ADDRESS };
        auto Res = Cli.Get("/session/current", httplib::Headers { { ZENO_SESSION_HEADER_KEY, ZENO_LOCAL_TOKEN } });
        if (Res && Res->status == 200) {
            CurrentSession = Res->body;
            return Res->body;
        } else {
            return "";
        }
    }
}
bool zeno::remote::Flags::IsMainProcess() const {
    return IsMainProcess_;
}

void zeno::remote::Flags::SetIsMainProcess(bool isMainProcess) {
    IsMainProcess_ = isMainProcess;
}
zeno::remote::Flags::Flags() : IsMainProcess_(false)
{}

void zeno::remote::SubjectRegistry::Push(const std::vector<SubjectContainer> &InitList, const std::string &SessionKey) {
    if (StaticFlags.IsMainProcess()) {
        std::set<std::string> ChangeList;
        auto& ElementMap = GetOrCreateSessionElement(SessionKey);

        for (const SubjectContainer& Value : InitList) {
            ChangeList.emplace(Value.Name);
            ElementMap.insert_or_assign(Value.Name, Value);
        }
        if (Callback) {
            Callback(ChangeList, SessionKey);
        }
    } else {
        // In child process, transfer data with http
        httplib::Client Cli { ZENO_TOOL_SERVER_ADDRESS };
        Cli.set_default_headers({ {ZENO_SESSION_HEADER_KEY, ZENO_LOCAL_TOKEN} });
        SubjectContainerList List { InitList };
        std::vector<uint8_t> Data = msgpack::pack(List);
        const std::string& Url = StringFormat("/subject/push?session_key=%s", SessionKey);
        Cli.Post(Url, reinterpret_cast<const char*>(Data.data()), Data.size(), "application/binary");
    }
}

void zeno::remote::SubjectRegistry::SetParameter(const std::string &SessionKey, const std::string &Key,
                                                 zeno::remote::ParamValue &Value) {
    if (StaticFlags.IsMainProcess()) {
        auto& ParamMapIter = GetOrCreate(SessionalParameters, SessionKey);

        ParamMapIter.insert_or_assign(Key, Value);
    } else {
        // In child process, transfer data with http
        httplib::Client Cli { ZENO_TOOL_SERVER_ADDRESS };
        Cli.set_default_headers({ {ZENO_SESSION_HEADER_KEY, ZENO_LOCAL_TOKEN} });
        httplib::Params Param;
        Param.insert(std::make_pair("session_key", SessionKey));
        Param.insert(std::make_pair("key", Key));
        std::vector<uint8_t> Data = msgpack::pack(Value);
        const httplib::Result Response = Cli.Post("/graph/param/push", reinterpret_cast<const char*>(Data.data()), Data.size(), "application/binary");
    }
}

const zeno::remote::ParamValue *zeno::remote::SubjectRegistry::GetParameter(const std::string &SessionKey,
                                                                            const std::string &Key) const {
    static zeno::remote::ParamValue TempValue;
    if (StaticFlags.IsMainProcess()) {
        auto ParamMapIter = SessionalParameters.find(SessionKey);
        if (ParamMapIter != SessionalParameters.end()) {
            auto ParamIter = ParamMapIter->second.find(Key);
            if (ParamIter != ParamMapIter->second.end()) {
                TempValue = ParamIter->second;
                return &TempValue;
            }
        }
        return nullptr;
    } else {
        // In child process, transfer data with http
        httplib::Client Cli{ZENO_TOOL_SERVER_ADDRESS};
        Cli.set_default_headers({{ZENO_SESSION_HEADER_KEY, ZENO_LOCAL_TOKEN}});
        httplib::Params Param;
        Param.insert(std::make_pair("key", Key));
        const httplib::Result Response =
            Cli.Get("/graph/param/fetch", Param, httplib::Headers{}, httplib::Progress{});
        if (Response) {
            const std::string &Body = Response->body;
            std::error_code Err;
            auto List = msgpack::unpack<struct ParamValueBatch>(
                reinterpret_cast<uint8_t *>(const_cast<char *>(Body.data())), Body.size(), Err);
            if (!Err) {
                for (const auto &Subject : List.Values) {
                    // Alloc new memory and copy it
                    // TODO [darc] : fix race condition(might be) :
                    TempValue = Subject;
                    return &TempValue;
                }
            }
        }
        return nullptr;
    }
}
