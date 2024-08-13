#pragma once

#include <reflect/container/any>
#include <zeno/types/ObjectDef.h>
#include <memory>
#include "zeno_types/reflect/reflection.generated.hpp"


using namespace zeno::reflect;

namespace zeno
{
    template<typename T>
    class ObjectPtrContainer : public zeno::reflect::any::IContainer
    {
    public:
        using ValueType = T;

        std::shared_ptr<ValueType> m_value;

        ObjectPtrContainer(std::shared_ptr<T> spValue) {
            m_value = spValue;
        }

        virtual const RTTITypeInfo& type() const override {
            return zeno::reflect::type_info<std::shared_ptr<zeno::PrimitiveObject>>();
        }

        virtual bool is_type(const RTTITypeInfo& other_type) const override {
            return other_type == this->type();
        }

        virtual zeno::reflect::any::IContainer* clone() const override {
            return new ObjectPtrContainer<ValueType>(m_value);
        }

        virtual IContainer* deep_clone() const override {
            if (!m_value)
                return nullptr;

            std::shared_ptr<ValueType> clone_obj = std::dynamic_pointer_cast<ValueType>(m_value->clone());
            return new ObjectPtrContainer<ValueType>(clone_obj);
        }

        virtual IContainer* deep_move_clone() const override {
            if (!m_value)
                return nullptr;

            std::shared_ptr<ValueType> clone_obj = std::dynamic_pointer_cast<ValueType>(m_value->move_clone());
            return new ObjectPtrContainer<ValueType>(clone_obj);
        }

        virtual AnyConversionMethod is_convertible_to(const RTTITypeInfo& other_type) const override {
            auto self_type = type();
            if (other_type == self_type) {
                return AnyConversionMethod::AsIs;
            }
            else {
                if (other_type.hash_code() == zeno::types::gParamType_sharedIObject)
                    return AnyConversionMethod::AsIs;
                return AnyConversionMethod::Impossible;
            }
        }

        virtual void* get_data_ptr_unsafe() const override {
            return nullptr;
        }

        virtual bool is_enable_from_this() override {
            return false;
        }
    };


}