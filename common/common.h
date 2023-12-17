#ifndef __ZENO_COMMON_H__
#define __ZENO_COMMON_H__


#define ENUM_FLAGS(enum_class) \
constexpr enum_class operator|(enum_class X, enum_class Y) {\
    return static_cast<enum_class>(\
        static_cast<unsigned int>(X) | static_cast<unsigned int>(Y));\
}\
\
enum_class& operator|=(enum_class& X, enum_class Y) {\
    X = X | Y; return X;\
}\
\
constexpr enum_class operator&(enum_class X, enum_class Y) {\
    return static_cast<enum_class>(\
        static_cast<unsigned int>(X) & static_cast<unsigned int>(Y));\
}\
\
enum_class& operator&=(enum_class& X, enum_class Y) {\
    X = X & Y; return X;\
}\
\
constexpr enum_class operator^(enum_class X, enum_class Y) {\
    return static_cast<enum_class>(\
        static_cast<unsigned int>(X) ^ static_cast<unsigned int>(Y));\
}\
\
enum_class& operator^=(enum_class& X, enum_class Y) {\
    X = X ^ Y; return X;\
}\

namespace zeno {

    enum ParamType
    {
        Param_Null,
        Param_Int,
        Param_String,
        Param_Float,
        Param_Vec3f,
        Param_Vec4f,
        Param_Prim,
        Param_Dict,
        Param_List,
        Param_SrcDst,
    };

    enum NodeStatus : unsigned int
    {
        Null = 0,
        Cached = 1,
        Mute = 1<<1,
        View = 1<<2,
    };
    ENUM_FLAGS(NodeStatus)

    enum NodeType
    {
        Normal,
        SubInput,
        SubOutput,
        Group,
        Legacy,
        SubgraphNode
    };

    enum SubnetType
    {

    };

    enum SocketProperty : unsigned int
    {
        Normal,
        Editable,
        MultiInput,
        Legacy,
    };
    ENUM_FLAGS(SocketProperty)

    //ui issues:
    enum VParamType
    {
        Root,
        Tab,
        Group,
        Param
    };

    enum ParamControl
    {
        Null,
        Lineedit,
        Multiline,
        Pathedit,
        Combobox,
        Checkbox,
        Vec2edit,
        Vec3edit,
        Vec4edit,
        Color,
        Heatmap
    };

    enum LinkFunction
    {
        Link_Copy,
        Link_Ref
    };

    using zvariant = std::variant<
        std::string,
        int,
        float,
        double,
        bool,
        zeno::vec2i,
        zeno::vec2f,
        zeno::vec3i,
        zeno::vec3f,
        zeno::vec4i,
        zeno::vec4f>;

    using ctrlpropvalue = std::variant<
        std::vector<std::string>,
        int,
        float,
        std::string>;
}


#endif