#pragma once

#include <zeno/core/coredata.h>
#include <zeno/utils/api.h>
#include <memory>   //abi problem

namespace zeno {

    struct ZENO_API IUserData {
        virtual std::unique_ptr<IUserData> clone() = 0;
        virtual ~IUserData() = default;

        virtual bool has(const String& key) = 0;
        virtual size_t size() const = 0;
        virtual Vector<String> keys() const = 0;

        virtual String get_string(const String& key, String defl = "") const = 0;
        virtual void set_string(const String& key, const String& sval) = 0;
        virtual bool has_string(const String& key) const = 0;

        virtual int get_int(const String& key, int defl = 0) const = 0;
        virtual void set_int(const String& key, int iVal) = 0;
        virtual bool has_int(const String& key) const = 0;

        virtual float get_float(const String& key, float defl = 0.f) const = 0;
        virtual void set_float(const String& key, float fVal) = 0;
        virtual bool has_float(const String& key) const = 0;

        virtual Vec2f get_vec2f(const String& key, Vec2f defl = Vec2f()) const = 0;
        virtual void set_vec2f(const String& key, const Vec2f& vec) = 0;
        virtual bool has_vec2f(const String& key) const = 0;

        virtual Vec2i get_vec2i(const String& key) const = 0;
        virtual void set_vec2i(const String& key, const Vec2i& vec) = 0;
        virtual bool has_vec2i(const String& key) const = 0;

        virtual Vec3f get_vec3f(const String& key, Vec3f defl = Vec3f()) const = 0;
        virtual void set_vec3f(const String& key, const Vec3f& vec) = 0;
        virtual bool has_vec3f(const String& key) const = 0;

        virtual Vec3i get_vec3i(const String& key) const = 0;
        virtual void set_vec3i(const String& key, const Vec3i& vec) = 0;
        virtual bool has_vec3i(const String& key) const = 0;

        virtual Vec4f get_vec4f(const String& key) const = 0;
        virtual void set_vec4f(const String& key, const Vec4f& vec) = 0;
        virtual bool has_vec4f(const String& key) const = 0;

        virtual Vec4i get_vec4i(const String& key) const = 0;
        virtual void set_vec4i(const String& key, const Vec4i& vec) = 0;
        virtual bool has_vec4i(const String& key) const = 0;

        virtual bool get_bool(const String& key, bool defl = false) const = 0;
        virtual void set_bool(const String& key, bool val = false) = 0;
        virtual bool has_bool(const String& key) const = 0;

        virtual void del(String const& name) = 0;
    };

    struct ZENO_API IObject {
        IObject();
        IObject(const IObject& rhs);
        IObject& operator=(const IObject& rhs);
        virtual ~IObject();     // don't consider abi problem right now.
        virtual std::unique_ptr<zeno::IObject> clone() const = 0; //TODO：abi compatible for shared_ptr
        virtual String key() const;
        virtual void update_key(const String& key);
        IUserData* userData();
        virtual void Delete();  //TODO: for abi compatiblity when dtor cann't be mark virutal.

        String m_key;
        std::unique_ptr<IUserData> m_usrData;   //TODO: abi unique_ptr
    };

    template <class Derived, class CustomBase = IObject>
    struct IObjectClone : CustomBase {

        virtual std::unique_ptr<IObject> clone() const override {
            auto spClonedObj = std::make_unique<Derived>(static_cast<Derived const&>(*this));
            spClonedObj->m_usrData = std::move(m_usrData->clone());
            return spClonedObj;
        }

        void Delete() override {
            //CustomBase::Delete();
            //delete this;
        }
    };

    using zany = std::unique_ptr<zeno::IObject>;
}