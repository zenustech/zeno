#ifndef ZENO_UNREALREGISTRY_H
#define ZENO_UNREALREGISTRY_H

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

struct IUnrealSubject {
    std::string m_name;
};

struct UnrealHeightFieldSubject : public IUnrealSubject {
    int64_t m_resolution;
    std::vector<float> m_height;

    template <class T>
    inline void pack(T& pack) {
        pack(m_name, m_resolution, m_height);
    }
};

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
