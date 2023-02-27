#ifndef ZENO_UNREALREGISTRY_H
#define ZENO_UNREALREGISTRY_H

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
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

#endif //ZENO_UNREALREGISTRY_H
