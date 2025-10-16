#pragma once

#include <zeno/core/coredata.h>
#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>
#include <zeno/core/data.h>

namespace zeno
{
    struct NodeImpl;
    struct ListObject;
    struct DictObject;
    struct PrimitiveObject;
    struct INode;
    struct GlobalState;

    #define IMPL(p) (p)->m_pImpl

    struct ZENO_API INodeImpl
    {
        INodeImpl(INode* pNode);
        ~INodeImpl();

        zany get_input_object(const zeno::String& param);
        CustomUI export_customui() const;

        zeno::SharedPtr<PrimitiveObject> get_input_PrimitiveObject(const zeno::String& param);
        zeno::SharedPtr<ListObject> get_input_ListObject(const zeno::String& param);
        zeno::SharedPtr<DictObject> get_input_DictObject(const zeno::String& param);
        //TODO: VDB, NumericObject

        void set_output_object(const zeno::String& param, zany pObject);

        int get_input2_int(const zeno::String& param);
        float get_input2_float(const zeno::String& param);
        zeno::String get_input2_string(const zeno::String& param);
        zeno::Vec2i get_input2_vec2i(const zeno::String& param);
        zeno::Vec2f get_input2_vec2f(const zeno::String& param);
        zeno::Vec3i get_input2_vec3i(const zeno::String& param);
        zeno::Vec3f get_input2_vec3f(const zeno::String& param);
        zeno::Vec4f get_input2_vec4i(const zeno::String& param);
        zeno::Vec4i get_input2_vec4f(const zeno::String& param);
        bool get_input2_bool(const zeno::String& param);
        bool has_input(const zeno::String& param);

        int get_param_int(const zeno::String& param);
        float get_param_float(const zeno::String& param);
        String get_param_string(const zeno::String& param);
        bool get_param_bool(const zeno::String& param);
        zany get_output(const zeno::String& param);

        void set_output_int(const zeno::String& param, int val);
        void set_output_float(const zeno::String& param, float val);
        void set_output_vec2f(const zeno::String& param, Vec2f val);
        void set_output_vec2i(const zeno::String& param, Vec2i val);
        void set_output_vec3f(const zeno::String& param, Vec3f val);
        void set_output_vec3i(const zeno::String& param, Vec3i val);

        int GetFrameId() const;
        GlobalState* getGlobalState();

        NodeImpl* m_pImpl;
    };


}
