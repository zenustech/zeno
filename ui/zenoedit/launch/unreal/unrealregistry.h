#ifndef ZENO_UNREALREGISTRY_H
#define ZENO_UNREALREGISTRY_H

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <optional>
#include <random>
#include "model/subject.h"

/**
 * maintain subjects send to unreal live link
 */
class UnrealSubjectRegistry {

public:
    template <class T = IUnrealSubject>
    void put(const std::string& key, const T& data) {
        m_subjects.push_back(key);
    }

    template <>
    void put<UnrealHeightFieldSubject>(const std::string& key, const UnrealHeightFieldSubject& data) {
        m_subjects.push_back(key);
        m_height_field_subjects.push_back(data);
    }

    void markDirty(bool flag);

    bool isDirty() const;

    [[nodiscard]]
    const std::vector<std::string>& subjects() const {
        return m_subjects;
    }

    [[nodiscard]]
    const std::vector<UnrealHeightFieldSubject>& height_fields() const {
        return m_height_field_subjects;
    }

private:
    std::vector<std::string> m_subjects;
    std::vector<UnrealHeightFieldSubject> m_height_field_subjects;

    bool m_bIsDirty = true;

public:
    static UnrealSubjectRegistry& getStatic() {
        static UnrealSubjectRegistry sUnrealSubjectRegistry;

        return sUnrealSubjectRegistry;
    }
};

struct UnrealSessionInfo {
    std::optional<std::string> udp_address;
    std::optional<uint16_t> udp_port;
};

struct UnrealSessionRegistry {

public:
    std::string newSession();

    bool updateSession(const std::string& sessionName, const UnrealSessionInfo& info);

    const UnrealSessionInfo& getSessionInfo(const std::string& sessionName);

    bool hasSession(const std::string& sessionName);

    void removeSession(const std::string& sessionName);

    std::vector<UnrealSessionInfo> all();

private:
    std::unordered_map<std::string, UnrealSessionInfo> m_session_info;

    static std::string newSessionName() {
        std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

        std::random_device rd;
        std::mt19937 generator(rd());

        std::shuffle(str.begin(), str.end(), generator);

        return str.substr(0, 32);    // assumes 32 < number of characters in str
    }

public:
    static UnrealSessionRegistry& getStatic() {
        static UnrealSessionRegistry sUnrealSessionRegistry;

        return sUnrealSessionRegistry;
    }
};

#endif //ZENO_UNREALREGISTRY_H
