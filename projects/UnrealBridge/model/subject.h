#ifndef ZENO_SUBJECT_H
#define ZENO_SUBJECT_H
#include <string>
#include <vector>

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

enum class EZenoSubjectType : int16_t {
    Invalid = -1,
    HeightField = 0,
    End,
};

#endif //ZENO_SUBJECT_H
