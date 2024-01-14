#ifndef __ZENO_COMMON_H__
#define __ZENO_COMMON_H__

#include <variant>
#include <string>
#include <vector>
#include <zeno/utils/vec.h>


#define ENUM_FLAGS(enum_class) \
constexpr enum_class operator|(enum_class X, enum_class Y) {\
    return static_cast<enum_class>(\
        static_cast<unsigned int>(X) | static_cast<unsigned int>(Y));\
}\
\
constexpr enum_class operator&(enum_class X, enum_class Y) {\
    return static_cast<enum_class>(\
        static_cast<unsigned int>(X) & static_cast<unsigned int>(Y));\
}\
\
constexpr enum_class operator^(enum_class X, enum_class Y) {\
    return static_cast<enum_class>(\
        static_cast<unsigned int>(X) ^ static_cast<unsigned int>(Y));\
}\

namespace zeno {

    enum ParamType
    {
        Param_Null,
        Param_Bool,
        Param_Int,
        Param_String,
        Param_Float,
        Param_Vec2i,
        Param_Vec3i,
        Param_Vec4i,
        Param_Vec2f,
        Param_Vec3f,
        Param_Vec4f,
        Param_Prim,
        Param_Dict,
        Param_List,
        //Param_Color,  //need this?
        Param_Curve,
        Param_SrcDst,
    };

    enum NodeStatus : unsigned int
    {
        None = 0,
        Mute = 1,
        View = 1<<1,
    };
    ENUM_FLAGS(NodeStatus)

    enum NodeType
    {
        Node_Normal,
        SubInput,
        SubOutput,
        Node_Group,
        Node_Legacy,
        Node_SubgraphNode,
        NoVersionNode
    };

    enum SubnetType
    {

    };

    enum SocketProperty : unsigned int
    {
        Socket_Normal,
        Socket_Editable,
        Socket_MultiInput,
        Socket_Legacy,
    };
    ENUM_FLAGS(SocketProperty)

    //ui issues:
    enum VParamType
    {
        Param_Root,
        Param_Tab,
        Param_Group,
        Param_Param
    };

    enum ParamControl
    {
        NullControl,
        Lineedit,
        Multiline,
        Pathedit,
        Combobox,
        Checkbox,
        Vec2edit,
        Vec3edit,
        Vec4edit,
        Color,
        ColorVec,
        Heatmap,
        CurveEditor,
        SpinBox,
        Slider,
        DoubleSpinBox,
        SpinBoxSlider,
        PythonEditor,
        PushButton,
        Seperator,
    };

    enum LinkFunction
    {
        Link_Copy,
        Link_Ref
    };

    enum ZSG_VERSION
    {
        VER_2,          //old version io
        VER_2_5,        //new version io
        VER_3,          //the final io format, supporting tree layout.
    };

    using zvariant = std::variant<
        int, zeno::vec2i, zeno::vec3i, zeno::vec4i,
        float, zeno::vec2f, zeno::vec3f, zeno::vec4f,
        zeno::vec2s, zeno::vec3s, zeno::vec4s, std::string>;

    using ctrlpropvalue = std::variant<
        std::vector<std::string>,
        int,
        float,
        std::string>;
}


#endif