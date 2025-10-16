#pragma once

#include <zeno/core/IObject.h>
#include <zeno/core/data.h>

namespace zeno
{
    struct NodeImpl;
    struct ListObject;
    struct DictObject;
    struct PrimitiveObject;
    struct GeometryObject_Adapter;
    struct GlobalState;
    struct CalcContext;

    struct ZENO_API INode {
        virtual void apply() = 0;
        virtual CustomUI export_customui() const;
        virtual NodeType type() const;
        virtual zeno::String uuid() const;
        virtual ~INode() = default;     //暂时不考虑abi问题
        virtual void clearCalcResults();
        virtual float time() const;

        IObject* get_input(const zeno::String& param);
        zany clone_input(const zeno::String& param);
        std::unique_ptr<PrimitiveObject> clone_input_PrimitiveObject(const zeno::String& param);
        PrimitiveObject* get_input_PrimitiveObject(const zeno::String& param);
        std::unique_ptr<GeometryObject_Adapter> clone_input_Geometry(const zeno::String& param);
        GeometryObject_Adapter* get_input_Geometry(const zeno::String& param);
        ListObject* get_input_ListObject(const zeno::String& param);
        DictObject* get_input_DictObject(const zeno::String& param);
        int get_input2_int(const zeno::String& param);
        float get_input2_float(const zeno::String& param);
        String get_input2_string(const zeno::String& param);
        bool get_input2_bool(const zeno::String& param);
        bool has_input(const zeno::String& param);
        bool has_link_input(const zeno::String& param);

        bool is_upstream_dirty(const zeno::String& param) const;
        void check_break();

        zeno::Vec2i get_input2_vec2i(const zeno::String& param);
        zeno::Vec2f get_input2_vec2f(const zeno::String& param);
        zeno::Vec3i get_input2_vec3i(const zeno::String& param);
        zeno::Vec3f get_input2_vec3f(const zeno::String& param);
        zeno::Vec4i get_input2_vec4i(const zeno::String& param);
        zeno::Vec4f get_input2_vec4f(const zeno::String& param);

        int get_param_int(const zeno::String& param);
        float get_param_float(const zeno::String& param);
        String get_param_string(const zeno::String& param);
        bool get_param_bool(const zeno::String& param);

        bool set_output(const zeno::String& param, zany&& pObject);
        bool set_output_int(const zeno::String& param, int val);
        bool set_output_float(const zeno::String& param, float val);
        bool set_output_string(const zeno::String& param, zeno::String val);
        bool set_output_vec2f(const zeno::String& param, Vec2f val);
        bool set_output_vec2i(const zeno::String& param, Vec2i val);
        bool set_output_vec3f(const zeno::String& param, Vec3f val);
        bool set_output_vec3i(const zeno::String& param, Vec3i val);

        int GetFrameId() const;
        GlobalState* getGlobalState();

        NodeImpl* m_pAdapter = nullptr;
    };
}

