#ifndef ZENO_UNREALREGISTRY_H
#define ZENO_UNREALREGISTRY_H

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <optional>
#include <random>
#include "model/subject.h"
#include "zeno/core/INode.h"
#include "zeno/types/PrimitiveObject.h"

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
        m_height_field_subjects.insert_or_assign(key, data);
    }

    void markDirty(bool flag);

    bool isDirty() const;

    [[nodiscard]]
    const std::vector<std::string>& subjects() const {
        return m_subjects;
    }

    [[nodiscard]]
    const std::unordered_map<std::string, UnrealHeightFieldSubject>& height_fields() const {
        return m_height_field_subjects;
    }

private:
    std::vector<std::string> m_subjects;
    std::unordered_map<std::string, UnrealHeightFieldSubject> m_height_field_subjects;

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

// Subjects used by zeno node

#define REGISTER_UNREAL_ZENO_SUBJECT(SubjectType, ApplyFuncBody) static struct STATIC_REGISTER_FOR_UNREAL_ZENO_SUBJECT_##SubjectType : ISubjectApplyable {  \
        STATIC_REGISTER_FOR_UNREAL_ZENO_SUBJECT_##SubjectType() {                                                                                           \
            IUnrealZenoSubject::SubjectApplyDispatchMap.insert_or_assign(EZenoSubjectType::SubjectType, this);                                              \
        }                                                                                                                                                   \
        void apply(std::shared_ptr<IUnrealZenoSubject> data, INode& context) override ApplyFuncBody                                                         \
    } Instance_STATIC_REGISTER_FOR_UNREAL_ZENO_SUBJECT_##SubjectType;

namespace zeno {

struct IUnrealZenoSubject {
    virtual EZenoSubjectType type() = 0;

    inline static std::unordered_map<EZenoSubjectType, struct ISubjectApplyable*> SubjectApplyDispatchMap {};
};

struct ISubjectApplyable {
    virtual void apply(std::shared_ptr<IUnrealZenoSubject> data, INode& context) = 0;

    template <typename T>
    static T* cast_checked(const std::shared_ptr<IUnrealZenoSubject>& data) {
        static_assert(std::is_base_of_v<IUnrealZenoSubject, T>, "T is not a subclass of IUnrealZenoSubject.");
        if (data) {
            IUnrealZenoSubject* subject = data.get();
            T* target = dynamic_cast<T*>(subject);
            return target;
        }
        throw std::exception("Bad cast.");
    }
};

// HeightField data subject
struct UnrealZenoHeightFieldSubject : IUnrealZenoSubject {
    std::vector<float> heights;

    EZenoSubjectType type() override { return EZenoSubjectType::HeightField; }
};
REGISTER_UNREAL_ZENO_SUBJECT(HeightField, {
    auto* target = cast_checked<UnrealZenoHeightFieldSubject>(data);
    std::shared_ptr<PrimitiveObject> prim = std::make_shared<PrimitiveObject>();
    prim->verts.resize(target->heights.size());
    prim->verts.add_attr<float>("height");
    auto& heights = prim->verts.attr<float>("height");
    for (size_t i = 0; i < target->heights.size(); ++i) {
        heights[i] = target->heights[i];
    }
    context.outputs["prim"] = std::move(prim);
});

}

#undef REGISTER_UNREAL_ZENO_SUBJECT

enum class EZenoSubjectApplyResult : uint8_t {
    Success = 0,
    NotFound,
    InvalidType,
    UnregisterType,
    TypeNotMatch,
};

class ZenoSubjectRegistry {

public:
    EZenoSubjectApplyResult apply(const std::string& subjectName, const EZenoSubjectType inSubjectType, zeno::INode& context);

    void put(const std::string& subjectName, const std::shared_ptr<zeno::IUnrealZenoSubject>& subject);

private:
    std::unordered_map<std::string, std::shared_ptr<zeno::IUnrealZenoSubject>> subjects;

public:
    static ZenoSubjectRegistry& getStatic() {
        static ZenoSubjectRegistry sZenoSubjectRegistry;

        return sZenoSubjectRegistry;
    }
};

#endif //ZENO_UNREALREGISTRY_H
