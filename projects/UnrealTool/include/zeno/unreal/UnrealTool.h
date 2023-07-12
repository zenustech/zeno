#pragma once

#include "StaticDefinition.h"
#include "ZenoRemoteTypes.h"
#include "httplib/httplib.h"
#include "msgpack/msgpack.h"
#include "zeno/core/IObject.h"
#include <optional>
#include <string>

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

namespace zeno {
struct PrimitiveObject;
}

namespace zeno::remote {

/**
 * @brief Get or create element in a map
 * @tparam KeyType Key type
 * @tparam ValueType Value type
 * @param Map Map
 */
template<typename KeyType, typename ValueType>
ValueType& GetOrCreate(std::map<KeyType, ValueType>& Map, const KeyType& Key) {
    auto It = Map.find(Key);
    if (It == Map.end()) {
        It = Map.emplace(Key, ValueType()).first;
    }
    return It->second;
}

/**
 * Copy from https://stackoverflow.com/questions/440133/how-do-i-create-a-random-alpha-numeric-string-in-c
 * @deprecated Use RandomString2 instead
 * @param length string length
 * @return Random string
 */
std::string RandomString( size_t Length );

// Get a random string with given length using C++ 11 random library
std::string RandomString2( size_t Length );

// Copy from https://gist.github.com/Zitrax/a2e0040d301bf4b8ef8101c0b1e3f1d5
// And
// Copy from https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

/**
 * @brief Convert string to const char*
 */
template<typename T>
auto StringConvert_Internal(T&& t) {
    if constexpr (std::is_same<std::remove_cv_t<std::remove_reference_t<T>>, std::string>::value) {
        return std::forward<T>(t).c_str();
    }
    else {
        return std::forward<T>(t);
    }
}

/**
 * @tparam Args Arguments type
 * @param format Format string
 * @param args Arguments
 * @return Formatted string
 */
template<typename ... Args>
std::string StringFormat_Internal( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return { buf.get(), buf.get() + size - 1 }; // We don't want the '\0' inside
}

/**
 * @tparam Args Arguments type
 * @param Fmt Format string
 * @param args Arguments
 * @return Formatted string
 */
template<typename ... Args>
std::string StringFormat(const std::string& Fmt, Args&& ... args) {
    return StringFormat_Internal(Fmt, StringConvert_Internal(std::forward<Args>(args))...);
}

using SessionKeyType = std::string;

struct Flags {
    bool IsMainProcess() const;
    void SetIsMainProcess(bool isMainProcess);
    std::string GetCurrentSession();

    std::string CurrentSession;

    Flags();

private:
    bool IsMainProcess_;


};

/**
 * Subject container
 */
struct SubjectRegistry {
    [[deprecated("Use SessionalElements instead.")]]
    std::map<std::string, SubjectContainer> Elements;
    std::map<std::string, std::map<std::string, zeno::remote::ParamValue>> SessionalParameters;
    std::map<std::string, std::map<std::string, zeno::remote::SubjectContainer>> SessionalElements;
    std::function<void(const std::set<std::string>&, const SessionKeyType&)> Callback;

    /**
     * Get or create session element
     * @param SessionKey Session key
     * @return Session element
     */
    [[nodiscard]]
    inline auto& GetOrCreateSessionElement(const std::string& SessionKey) {
        return GetOrCreate(SessionalElements, SessionKey);
    }

    /**
     * Push subject container to registry
     * @param InitList Subject container list
     * @param SessionKey Session key, empty for global elements
     */
    void Push(const std::vector<SubjectContainer>& InitList, const std::string& SessionKey = "");

    /**
     * Get subject container
     * @param Key Subject key
     * @param SessionKey Session key, empty for global elements
     * @return Subject container
     */
    template <typename T>
    std::optional<T> Get(const std::string& Key, const std::string& SessionKey = "", bool bSearchAllSession = false) {
        CONSTEXPR ESubjectType RequiredSubjectType = T::SubjectType;
        if (StaticFlags.IsMainProcess()) {
            auto& ElementMap = GetOrCreateSessionElement(SessionKey);
            auto& GlobalElementMap = GetOrCreateSessionElement("");

            // Try to find in sessional elements
            auto TargetIter = ElementMap.find(Key);
            if (TargetIter == ElementMap.end()) {
                // If not found, try to find in global elements
                TargetIter = GlobalElementMap.find(Key);
                if (TargetIter == GlobalElementMap.end()) {
                    // Flag to indicate whether found. Because iterator to compare must from same container.
                    bool bFound = false;
                    if (bSearchAllSession) {
                        // Search all sessions
                        for (auto& [SessionKey, SessionElementMap] : SessionalElements) {
                            TargetIter = SessionElementMap.find(Key);
                            if (TargetIter != SessionElementMap.end()) {
                                bFound = true;
                                break;
                            }
                        }
                    }
                    // If still not found, return empty
                    if (!bFound) {
                        return std::nullopt;
                    }
                }
            }
            if (TargetIter->second.Type != static_cast<int16_t>(RequiredSubjectType)) return std::nullopt;
            std::error_code Err;
            T Result = msgpack::unpack<T>(TargetIter->second.Data, Err);
            if (!Err) {
                return std::make_optional(Result);
            }
        } else {
            // In child process, transfer data with http
            httplib::Client Cli { ZENO_TOOL_SERVER_ADDRESS };
            Cli.set_default_headers({ {ZENO_SESSION_HEADER_KEY, ZENO_LOCAL_TOKEN} });
            httplib::Params Param;
            Param.insert(std::make_pair("key", Key));
            Param.insert(std::make_pair("session_key", SessionKey));
            Param.insert(std::make_pair("search_all_session", bSearchAllSession ? "true" : "false"));
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
    void SetParameter(const std::string& SessionKey, const std::string& Key, zeno::remote::ParamValue& Value);

    /**
     * Get parameter from session
     * @param SessionKey Session key
     * @param Key Parameter key
     * @return Parameter value
     */
    [[nodiscard]] const zeno::remote::ParamValue *GetParameter(const std::string &SessionKey,
                                                               const std::string &Key) const;
};

/** Convert zeno::remote::HeightData to PrimitiveObject
 * @param HeightData Height data
 * @return PrimitiveObject
 */
std::shared_ptr<zeno::PrimitiveObject> ConvertHeightDataToPrimitiveObject(const zeno::remote::HeightField& InHeightData, int Nx = 0, int Ny = 0, std::array<float, 3> Scale = { 100.0f, 100.0f, 100.0f});

struct MetaData : IObjectClone<MetaData> {
    std::map<std::string, std::string> Data;
};

}
