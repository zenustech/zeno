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
        m_subjects.insert(key, data);
    }

    template <>
    void put<UnrealHeightFieldSubject>(const std::string& key, const UnrealHeightFieldSubject& data) {
        m_subjects.insert_or_assign(key, data);
        m_height_field_subjects.push_back(key);
    }

    void markDirty(bool flag);

    bool isDirty() const;

    [[nodiscard]]
    const std::unordered_map<std::string, IUnrealSubject>& subjects() const {
        return m_subjects;
    }

    [[nodiscard]]
    std::vector<UnrealHeightFieldSubject> height_fields() const {
        std::vector<UnrealHeightFieldSubject> subs;
        for (const std::string& name : m_height_field_subjects) {
            subs.push_back((UnrealHeightFieldSubject&)m_subjects.at(name));
        }
        return subs;
    }

private:
    std::unordered_map<std::string, IUnrealSubject> m_subjects;
    std::vector<std::string> m_height_field_subjects;

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
    std::string newSession() {
        std::string name = newSessionName();
        UnrealSessionInfo info {};
        m_session_info.insert(std::make_pair(name, std::move(info)));
        return name;
    }

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
