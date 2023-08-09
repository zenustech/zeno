#ifndef ZENO_PROPERTYVISITOR_H
#define ZENO_PROPERTYVISITOR_H

#include "Timer.h"
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <zeno/PrimitiveObject.h>
#include <zeno/types/AttrVector.h>
#include <zeno/core/Descriptor.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/logger.h>
#include <zeno/zeno.h>

#if 0
// Examples

// 1. Split Parameter class and Node class
struct CalcPathCostParameter : public zeno::reflect::INodeParameterObject<CalcPathCostParameter> {
    GENERATE_PARAMETER_BODY(CalcPathCostParameter);

    std::shared_ptr<zeno::PrimitiveObject> Primitive;
    DECLARE_INPUT_FIELD(Primitive, "prim");

    std::string OutputChannel;
    DECLARE_INPUT_FIELD(OutputChannel, "output_channel");

    std::string OutputTest;
    DECLARE_OUTPUT_FIELD(OutputTest, "test");
};

struct CalcPathCost_Simple : public zeno::reflect::IAutoNode<CalcPathCostParameter> {
    GENERATE_AUTONODE_BODY(CalcPathCost_Simple);

    void apply() override;
};

// 2. Merge parameter with node
struct CalcPathCost_Simple : public zeno::reflect::IParameterAutoNode<CalcPathCost_Simple> {
    GENERATE_NODE_BODY(CalcPathCost_Simple);

    std::shared_ptr<zeno::PrimitiveObject> Primitive;
    DECLARE_INPUT_FIELD(Primitive, "prim");

    std::string OutputChannel;
    DECLARE_INPUT_FIELD(OutputChannel, "output_channel");

    std::string OutputTest;
    DECLARE_OUTPUT_FIELD(OutputTest, "test");

    void apply() override;
};

#endif

namespace zeno {

    using namespace zeno;

    // using friend struct to break INode package!
    struct reflect {
        static constexpr unsigned int crc_table[256] = {
            0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
            0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
            0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
            0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
            0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
            0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
            0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
            0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
            0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
            0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
            0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
            0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
            0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
            0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
            0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
            0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
            0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
            0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
            0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
            0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
            0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
            0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
            0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
            0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
            0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
            0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
            0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
            0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
            0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
            0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
            0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
            0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
            0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
            0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
            0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
            0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
            0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
            0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
            0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
            0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
            0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
            0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
            0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d};

        template<int size, int idx = 0, class dummy = void>
        struct MM {
            static constexpr unsigned int crc32(const char *str, unsigned int prev_crc = 0xFFFFFFFF) {
                return MM<size, idx + 1>::crc32(str, (prev_crc >> 8) ^ crc_table[(prev_crc ^ str[idx]) & 0xFF]);
            }
        };

        // This is the stop-recursion function
        template<int size, class dummy>
        struct MM<size, size, dummy> {
            static constexpr unsigned int crc32(const char *str, unsigned int prev_crc = 0xFFFFFFFF) {
                return prev_crc ^ 0xFFFFFFFF;
            }
        };

        template<typename>
        struct IsSharedPtr : std::false_type {};

        template<typename T>
        struct IsSharedPtr<std::shared_ptr<T>> : std::true_type {};

        template<typename T>
        struct RawType {
            using Type = T;
        };

        template<typename T>
        struct RawType<std::shared_ptr<T>> {
            using Type = T;
        };

        template<typename T>
        using RawType_t = typename RawType<T>::Type;

        template<typename>
        struct ValueTypeToString {
            inline static std::string TypeName;
        };

        template<>
        struct ValueTypeToString<zeno::vec3f> {
            inline static std::string TypeName = "vec3f";
        };

        template<>
        struct ValueTypeToString<std::string> {
            inline static std::string TypeName = "string";
        };

        template<>
        struct ValueTypeToString<int> {
            inline static std::string TypeName = "int";
        };

        template<>
        struct ValueTypeToString<float> {
            inline static std::string TypeName = "float";
        };

        template<>
        struct ValueTypeToString<bool> {
            inline static std::string TypeName = "bool";
        };

        struct TypeAutoCallbackList {
            std::vector<std::function<void(INode *)>> InputHook;
            std::vector<std::function<void(INode *)>> OutputHook;

            std::vector<std::function<void()>> BindingHook;
        };

        struct NodeParameterBase {
            TypeAutoCallbackList HookList;
            INode *Target = nullptr;

            using Super = NodeParameterBase;

            explicit NodeParameterBase(INode *Node);
            NodeParameterBase(NodeParameterBase &&RhsToMove) noexcept;
            NodeParameterBase(const NodeParameterBase &) = delete;

            void RunInputHooks() const;
            void RunOutputHooks() const;
            void RunBindingHooks() const;

            virtual ~NodeParameterBase();
        };

        template<typename T>
        struct INodeParameterObject : public NodeParameterBase {
            using Super = INodeParameterObject<T>;
            using ThisType = T;

            static Descriptor &GetDescriptor() {
                static Descriptor SNDescriptor{};
                return SNDescriptor;
            }

            explicit INodeParameterObject(INode *Node);

            static T &GetDefaultObject() {
                static bool bHasInitializedDefaultObject = false;
                static T ST{nullptr};

                if (!bHasInitializedDefaultObject) {
                    GetDescriptor().inputs.emplace_back("SRC");
                    GetDescriptor().outputs.emplace_back("DST");
                    bHasInitializedDefaultObject = true;
                }

                return ST;
            }
        };

        template<typename ParentType, typename InputType, size_t Hash>
        struct Field {
            using Type = std::remove_cv_t<InputType>;
            using Parent = std::remove_cv_t<ParentType>;

            inline constexpr static size_t TypeHash = Hash;

            Type &ValueRef;
            std::string KeyName;
            std::string DisplayName;
            std::string DefaultValue;
            bool IsOptional = false;

            Field(Parent &ParentRef, Type &InValueRef, std::string InKeyName, bool InIsOptional = false, const std::optional<std::string> &InDisplayName = std::nullopt, const std::optional<std::string> &InDefaultValue = std::nullopt)
                : ValueRef(InValueRef), KeyName(std::move(InKeyName)), IsOptional(InIsOptional) {
                DisplayName = InDisplayName.value_or(KeyName);
                DefaultValue = InDefaultValue.value_or("");
            }

            Field(const Field &Other) = default;

            /**
             * Note: don't call lambda returned without ownership
             * @return
             */
            virtual std::function<void(INode *)> ToCaptured() = 0;
        };

        template<typename ParentType, typename InputType, size_t Hash>
        struct InputField : public Field<ParentType, InputType, Hash> {
            InputField(Parent &ParentRef, Type &InValueRef, std::string InKeyName, bool InIsOptional = false, const std::optional<std::string> &InDisplayName = std::nullopt, const std::optional<std::string> &InDefaultValue = std::nullopt)
                : Field<ParentType, InputType, Hash>(ParentRef, InValueRef, InKeyName, InIsOptional, InDisplayName, InDefaultValue) {
                ParentRef.HookList.InputHook.push_back(ToCaptured());

                static bool bHasInitialized = false;
                static SocketDescriptor SDescriptor = SocketDescriptor{ValueTypeToString<Type>::TypeName, KeyName, DefaultValue, DisplayName};
                if (!bHasInitialized) {
                    bHasInitialized = true;
                    ParentType::GetDescriptor().inputs.push_back(SDescriptor);
                }
            }

            inline void ReadObject(INode *Node) {
                zeno::log_debug("[AutoNode] Reading zany '{}'", KeyName);
                if (!IsOptional || Node->has_input(KeyName)) {
                    ValueRef = Node->get_input<RawType_t<Type>>(KeyName);
                }
            }

            inline void ReadPrimitiveValue(INode *Node) {
                zeno::log_debug("[AutoNode] Reading primitive value '{}'", KeyName);
                if (!IsOptional || Node->has_input(KeyName)) {
                    ValueRef = Node->get_input2<RawType_t<Type>>(KeyName);
                }
            }

            void Read(INode *Node) {
                if (nullptr == Node) {
                    zeno::log_error("Trying to read value from a nullptr Node.");
                    return;
                }
                if constexpr (IsSharedPtr<Type>()) {
                    ReadObject(Node);
                } else {
                    ReadPrimitiveValue(Node);
                }
            }

            std::function<void(INode *)> ToCaptured() override {
                return [this](INode *Node) {
                    if (nullptr != Node) {
                        Read(Node);
                    }
                };
            }
        };

        template<typename ParentType, typename InputType, size_t Hash>
        struct OutputField : public Field<ParentType, InputType, Hash> {
            OutputField(Parent &ParentRef, Type &InValueRef, std::string InKeyName, bool InIsOptional = false, const std::optional<std::string> &InDisplayName = std::nullopt, const std::optional<std::string> &InDefaultValue = std::nullopt)
                : Field<ParentType, InputType, Hash>(ParentRef, InValueRef, InKeyName, InIsOptional, InDisplayName) {
                ParentRef.HookList.OutputHook.push_back(ToCaptured());

                static bool bHasInitialized = false;
                static SocketDescriptor SDescriptor = SocketDescriptor{ValueTypeToString<Type>::TypeName, KeyName, DefaultValue, DisplayName};

                if (!bHasInitialized) {
                    bHasInitialized = true;
                    ParentType::GetDescriptor().outputs.push_back(SDescriptor);
                }
            }

            inline void WriteObject(INode *Node) {
                if (ValueRef) {
                    Node->set_output(KeyName, ValueRef);
                } else if (!IsOptional) {
                    zeno::log_error("Node '{}' has invalid output '{}'.", Node->myname, KeyName);
                }
            }

            inline void WritePrimitiveValue(INode *Node) {
                Node->set_output2(KeyName, ValueRef);
            }

            void Write(INode *Node) {
                if (nullptr == Node) {
                    zeno::log_error("Trying to read value from a nullptr Node.");
                    return;
                }
                if constexpr (IsSharedPtr<Type>()) {
                    WriteObject(Node);
                } else {
                    WritePrimitiveValue(Node);
                }
            }

            std::function<void(INode *)> ToCaptured() override {
                return [this](INode *Node) {
                    if (nullptr != Node) {
                        Write(Node);
                    }
                };
            }
        };

        template<typename NodeParameterType>
        struct IAutoNode : public INode {
            // Can't check incomplete type
            //static_assert(std::is_base_of_v<NodeParameterBase, NodeParameterType>);

            using ParamType = NodeParameterType;

            std::unique_ptr<NodeParameterType> AutoParameter = nullptr;

            void preApply() override {
                for (auto const &[ds, bound]: inputBounds) {
                    requireInput(ds);
                }

                AutoParameter = std::make_unique<NodeParameterType>(this);

                log_debug("==> enter {}", myname);
                {
#ifdef ZENO_BENCHMARKING
                    Timer _(myname);
#endif
                    apply();
                }

                AutoParameter->RunOutputHooks();
                AutoParameter.reset();
                log_debug("==> leave {}", myname);
            }
        };

        template<typename NodeType>
        struct IParameterAutoNode : public IAutoNode<NodeType>, public INodeParameterObject<NodeType> {
            IParameterAutoNode(INode *Node = nullptr) : IAutoNode<NodeType>(), INodeParameterObject<NodeType>(Node) {}
        };

        struct IPrimitiveBindingField {
            const std::shared_ptr<PrimitiveObject> &Primitive;
            const std::string& KeyName;
            const bool bIsOptional;

            IPrimitiveBindingField(std::shared_ptr<PrimitiveObject> &InPrimitive, const std::string& InKeyName, bool bInIsOptional) : Primitive(InPrimitive), KeyName(InKeyName), bIsOptional(bInIsOptional) {}
        };

        template<typename ParentType, typename ValueType>
        struct PrimitiveUserDataBindingField : public IPrimitiveBindingField {
            using Type = std::remove_cv_t<ValueType>;
            using Parent = std::remove_cv_t<ParentType>;

            Type &ValueRef;

            PrimitiveUserDataBindingField(Parent &ParentRef, std::shared_ptr<PrimitiveObject> &InPrimitive, ValueType &InValueRef, const std::string &InKeyName, bool bIsOptional = false) : IPrimitiveBindingField(InPrimitive, InKeyName, bIsOptional), ValueRef(InValueRef) {
                ParentRef.HookList.BindingHook.push_back(ToCaptured());
            }

            std::function<void()> ToCaptured() {
                return [this]() {
                    if (Primitive) {
                        if constexpr (IsSharedPtr<ValueType>()) {
                            if (!bIsOptional || Primitive->userData().has<ValueType>(KeyName)) {
                                ValueRef = Primitive->userData().get<ValueType>(KeyName);
                            }
                        } else {
                            if (!bIsOptional || Primitive->userData().has<ValueType>(KeyName)) {
                                ValueRef = Primitive->userData().get2<ValueType>(KeyName);
                            }
                        }
                    }
                };
            }
        };

        enum class EZenoPrimitiveAttr {
            VERT = 0,
            POINT = 1,
            LINE = 2,
            TRIANGLE = 3,
            QUAD = 4,
            LOOP = 5,
            POLY = 6,
            EDGE = 7,
            UV = 8,
        };

        template<typename ParentType, typename ValueType, EZenoPrimitiveAttr AttrType>
        struct PrimitiveAttributeBindingField : public IPrimitiveBindingField { /** TODO [darc] : not implemented yet : */ };

        template<typename ParentType, typename ValueType, EZenoPrimitiveAttr AttrType>
        struct PrimitiveAttributeBindingField<ParentType, zeno::AttrVector<ValueType>, AttrType> : public IPrimitiveBindingField {
            using Type = ValueType;
            using ArrayType = zeno::AttrVector<ValueType>;
            using Parent = std::remove_cv_t<ParentType>;

            ArrayType& ArrayRef;

            PrimitiveAttributeBindingField(Parent &ParentRef, std::shared_ptr<PrimitiveObject> &InPrimitive, ArrayType &InArrayRef, const std::string &InKeyName, bool bIsOptional = false) : IPrimitiveBindingField(InPrimitive, InKeyName, bIsOptional), ArrayRef(InArrayRef) {
                ParentRef.HookList.BindingHook.push_back(ToCaptured());
            }

            std::function<void()> ToCaptured() {
                return [this] () {
                    if (!Primitive) {
                        zeno::log_error("Invalid primitive binding.");
                        return;
                    }

                    if constexpr (AttrType == EZenoPrimitiveAttr::VERT)
                    {
                        if (!bIsOptional || Primitive->verts.has_attr(KeyName)) {
                            ArrayRef = Primitive->verts.attr<Type>(KeyName);
                        }
                    }
                    else if constexpr (AttrType == EZenoPrimitiveAttr::POINT)
                    {
                        if (!bIsOptional || Primitive->points.has_attr(KeyName)) {
                            ArrayRef = Primitive->points.attr<Type>(KeyName);
                        }
                    }
                    else if constexpr (AttrType == EZenoPrimitiveAttr::LINE)
                    {
                        if (!bIsOptional || Primitive->lines.has_attr(KeyName)) {
                            ArrayRef = Primitive->lines.attr<Type>(KeyName);
                        }
                    }
                    else if constexpr (AttrType == EZenoPrimitiveAttr::TRIANGLE)
                    {
                        if (!bIsOptional || Primitive->tris.has_attr(KeyName)) {
                            ArrayRef = Primitive->tris.attr<Type>(KeyName);
                        }
                    }
                    else if constexpr (AttrType == EZenoPrimitiveAttr::QUAD)
                    {
                        if (!bIsOptional || Primitive->quads.has_attr(KeyName)) {
                            ArrayRef = Primitive->quads.attr<Type>(KeyName);
                        }
                    }
                    else if constexpr (AttrType == EZenoPrimitiveAttr::LOOP)
                    {
                        if (!bIsOptional || Primitive->loops.has_attr(KeyName)) {
                            ArrayRef = Primitive->loops.attr<Type>(KeyName);
                        }
                    }
                    else if constexpr (AttrType == EZenoPrimitiveAttr::POLY)
                    {
                        if (!bIsOptional || Primitive->polys.has_attr(KeyName)) {
                            ArrayRef = Primitive->polys.attr<Type>(KeyName);
                        }
                    }
                    else if constexpr (AttrType == EZenoPrimitiveAttr::EDGE)
                    {
                        if (!bIsOptional || Primitive->edges.has_attr(KeyName)) {
                            ArrayRef = Primitive->edges.attr<Type>(KeyName);
                        }
                    }
                    else if constexpr (AttrType == EZenoPrimitiveAttr::UV)
                    {
                        if (!bIsOptional || Primitive->uvs.has_attr(KeyName)) {
                            ArrayRef = Primitive->uvs.attr<Type>(KeyName);
                        }
                    }
                };
            }
        };

    };

    template<typename T>
    reflect::INodeParameterObject<T>::INodeParameterObject(INode *Node) : NodeParameterBase(Node) {
    }
}// namespace zeno

#define GENERATE_AUTONODE_BODY(CLS)                                            \
    inline static struct R_Do_not_use {                                        \
        R_Do_not_use() {                                                       \
            const ParamType &PDO = ParamType::GetDefaultObject();              \
            zeno::getSession().defNodeClass([]() -> std::unique_ptr<INode> {   \
                return std::make_unique<CLS>();                                \
            },                                                                 \
                                            #CLS, ParamType::GetDescriptor()); \
        }                                                                      \
    } AutoStaticRegisterInstance_Do_not_use;

#define GENERATE_PARAMETER_BODY(CLS)          \
    explicit CLS(INode *Node) : Super(Node) { \
        if (nullptr != Node) {                \
            RunInputHooks();                  \
            RunBindingHooks();                \
        }                                     \
    }

#define GENERATE_NODE_BODY(CLS)                                                \
    CLS() : zeno::reflect::IParameterAutoNode<CLS>(nullptr) {}                 \
    explicit CLS(INode *Node) : zeno::reflect::IParameterAutoNode<CLS>(Node) { \
        if (nullptr != Node) {                                                 \
            RunInputHooks();                                                   \
            RunBindingHooks();                                                 \
        }                                                                      \
    }                                                                          \
    inline static struct R_Do_not_use {                                        \
        R_Do_not_use() {                                                       \
            const ParamType &PDO = ParamType::GetDefaultObject();              \
            zeno::getSession().defNodeClass([]() -> std::unique_ptr<INode> {   \
                return std::make_unique<CLS>();                                \
            },                                                                 \
                                            #CLS, ParamType::GetDescriptor()); \
        }                                                                      \
    } AutoStaticRegisterInstance_Do_not_use;

// This don't take into account the nul char
#define COMPILE_TIME_CRC32_STR(x) (zeno::reflect::MM<sizeof(x) - 1>::crc32(x))

#define DECLARE_FIELD(Type, FieldName, ...) zeno::reflect::Type<ThisType, decltype(FieldName), COMPILE_TIME_CRC32_STR(#FieldName)> FieldName##Type##_Do_not_use{*this, FieldName, __VA_ARGS__};
#define DECLARE_INPUT_FIELD(FieldName, KeyName, ...) DECLARE_FIELD(InputField, FieldName, KeyName, __VA_ARGS__)
#define DECLARE_OUTPUT_FIELD(FieldName, KeyName, ...) DECLARE_FIELD(OutputField, FieldName, KeyName, __VA_ARGS__)

#define BINDING_PRIMITIVE_USERDATA(PrimitiveName, FieldName, ChannelName, ...) zeno::reflect::PrimitiveUserDataBindingField<ThisType, decltype(FieldName)> Internal##FieldName##BindingWith##PrimitiveName##ChannelName##_Do_not_use { *this, PrimitiveName, FieldName, ChannelName, __VA_ARGS__ };
#define BINDING_PRIMITIVE_ATTRIBUTE(PrimitiveName, FieldName, ChannelName, Type, ...) zeno::reflect::PrimitiveAttributeBindingField<ThisType, decltype(FieldName), Type> Internal##FieldName##Attr##In##PrimitiveName##_Do_not_use { *this, PrimitiveName, FieldName, ChannelName, __VA_ARGS__ };

#endif//ZENO_PROPERTYVISITOR_H
