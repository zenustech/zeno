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
#include <zeno/core/Descriptor.h>
#include <zeno/utils/logger.h>
#include <zeno/zeno.h>

namespace zeno {

    using namespace zeno;

    // using friend struct to break INode package!
    struct reflect {
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

        struct TypeAutoCallbackList {
            std::vector<std::function<void(INode *)>> InputHook;
            std::vector<std::function<void(INode *)>> OutputHook;
        };

        struct NodeParameterBase {
            TypeAutoCallbackList HookList;
            INode *Target = nullptr;

            using Super = NodeParameterBase;

            explicit NodeParameterBase(INode *Node);
            NodeParameterBase(NodeParameterBase &&RhsToMove) noexcept;
            NodeParameterBase(const NodeParameterBase &) = delete;

            void RunInputHooks() const;

            virtual ~NodeParameterBase();
        };

        template<typename T>
        struct INodeParameterObject : public NodeParameterBase {
            using Super = INodeParameterObject<T>;
            using ThisType = T;

            inline static Descriptor SNDescriptor{};

            explicit INodeParameterObject(INode *Node);

            static T &GetDefaultObject() {
                static T ST{nullptr};
                SNDescriptor.inputs.emplace_back("SRC");
                SNDescriptor.outputs.emplace_back("DST");
                return ST;
            }
        };

        template<typename ParentType, typename InputType>
        struct Field {
            using Type = std::remove_cv_t<InputType>;
            using Parent = std::remove_cv_t<ParentType>;

            Type &ValueRef;
            std::string KeyName;
            std::string DisplayName;
            std::string DefaultValue;
            bool IsOptional = false;

            inline static bool bHasInitialized = false;
            inline static SocketDescriptor SDescriptor = {""};

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

        template<typename ParentType, typename InputType>
        struct InputField : public Field<ParentType, InputType> {
            InputField(Parent &ParentRef, Type &InValueRef, std::string InKeyName, bool InIsOptional = false, const std::optional<std::string> &InDisplayName = std::nullopt, const std::optional<std::string> &InDefaultValue = std::nullopt)
                : Field<ParentType, InputType>(ParentRef, InValueRef, InKeyName, InIsOptional, InDisplayName, InDefaultValue) {
                ParentRef.HookList.InputHook.push_back(ToCaptured());

                if (!bHasInitialized) {
                    SDescriptor = SocketDescriptor{ValueTypeToString<Type>::TypeName, KeyName, DefaultValue, DisplayName};
                    bHasInitialized = true;
                    ParentType::SNDescriptor.inputs.push_back(SDescriptor);
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
                    Read(Node);
                };
            }
        };

        template<typename ParentType, typename InputType>
        struct OutputField : public Field<ParentType, InputType> {
            OutputField(Parent &ParentRef, Type &InValueRef, std::string InKeyName, bool InIsOptional = false, const std::optional<std::string> &InDisplayName = std::nullopt, const std::optional<std::string> &InDefaultValue = std::nullopt)
                : Field<ParentType, InputType>(ParentRef, InValueRef, InKeyName, InIsOptional, InDisplayName) {
                ParentRef.HookList.OutputHook.push_back(ToCaptured());

                if (!bHasInitialized) {
                    SDescriptor = SocketDescriptor{ValueTypeToString<Type>::TypeName, KeyName, DefaultValue, DisplayName};
                    bHasInitialized = true;
                    ParentType::SNDescriptor.outputs.push_back(SDescriptor);
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
                    Write(Node);
                };
            }
        };

        template<typename NodeType, typename NodeParameterType>
        struct IAutoNode : public INode {
            static_assert(std::is_base_of_v<NodeParameterBase, NodeParameterType>);

            using ParamType = NodeParameterType;

            std::unique_ptr<NodeParameterType> AutoParameter = nullptr;
            inline static bool bHasInitialized = false;

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
                log_debug("==> leave {}", myname);
            }

            void complete() override {
                INode::complete();
                AutoParameter.reset();
            }
        };
    };

    template<typename T>
    reflect::INodeParameterObject<T>::INodeParameterObject(INode *Node) : NodeParameterBase(Node) {
    }
}// namespace zeno

#define GENERATE_AUTONODE_BODY(CLS)                                          \
    inline static struct R {                                                 \
        R() {                                                                \
            const ParamType &PDO = ParamType::GetDefaultObject();            \
            zeno::getSession().defNodeClass([]() -> std::unique_ptr<INode> { \
                return std::make_unique<CLS>();                              \
            },                                                               \
                                            #CLS, ParamType::SNDescriptor);  \
        }                                                                    \
    } AutoStaticRegisterInstance;

#define GENERATE_PARAMETER_BODY(CLS)                            \
    explicit CLS(INode *Node) : Super(Node) { \
        if (nullptr != Node) {                                  \
            RunInputHooks();                                    \
        }                                                       \
    }

#define DECLARE_FIELD(Type, FieldName, ...) zeno::reflect::Type<ThisType, decltype(FieldName)> FieldName##Field { *this, FieldName, __VA_ARGS__ };
#define DECLARE_INPUT_FIELD(FieldName, ...) DECLARE_FIELD(InputField, FieldName, __VA_ARGS__)
#define DECLARE_OUTPUT_FIELD(FieldName, ...) DECLARE_FIELD(OutputField, FieldName, __VA_ARGS__)

#endif//ZENO_PROPERTYVISITOR_H
