#ifndef ZENO_PROPERTYVISITOR_H
#define ZENO_PROPERTYVISITOR_H

#include <functional>
#include <type_traits>
#include <tuple>
#include <optional>
#include <utility>
#include <string>
#include <map>
#include <zeno/zeno.h>
#include <zeno/utils/logger.h>

namespace zeno {

    using namespace zeno;

    // using friend struct to break INode package!
    struct reflect {
        template <typename>
        struct IsSharedPtr : std::false_type {};

        template <typename T>
        struct IsSharedPtr<std::shared_ptr<T>> : std::true_type {};

        template <typename T>
        struct RawType {
            using Type = T;
        };

        template <typename T>
        struct RawType<std::shared_ptr<T>> {
            using Type = T;
        };

        template <typename T>
        using RawType_t = typename RawType<T>::Type;

        struct TypeAutoCallbackList {
            std::vector<std::function<void(INode*)>> InputHook;
            std::vector<std::function<void(INode*)>> OutputHook;
        };

        struct NodeParameterBase {
            TypeAutoCallbackList HookList;
            INode* Target = nullptr;

            explicit NodeParameterBase(INode* Node);
            NodeParameterBase(NodeParameterBase&& RhsToMove) noexcept;
            NodeParameterBase(const NodeParameterBase&) = delete;

            virtual ~NodeParameterBase();
        };

        template <typename ParentType, typename InputType>
        struct Field {
            using Type = std::remove_cv_t<InputType>;
            using Parent = std::remove_cv_t<ParentType>;

            Type& ValueRef;
            std::string KeyName;
            std::string DisplayName;
            bool IsOptional = false;

            Field(Parent& ParentRef, Type& InValueRef, std::string InKeyName, bool InIsOptional = false, const std::optional<std::string>& InDisplayName = std::nullopt)
            : ValueRef(InValueRef)
            , KeyName(std::move(InKeyName))
            , IsOptional(InIsOptional)
            {
                DisplayName = InDisplayName.value_or(KeyName);
            }

            Field(const Field& Other) = default;

            virtual std::function<void(INode*)> ToCaptured() = 0;
        };

        template <typename ParentType, typename InputType>
        struct InputField : public Field<ParentType, InputType> {
            InputField(Parent& ParentRef, Type& InValueRef, std::string InKeyName, bool InIsOptional = false, const std::optional<std::string>& InDisplayName = std::nullopt)
                : Field<ParentType, InputType>(ParentRef, InValueRef, InKeyName, InIsOptional, InDisplayName)
            {
                ParentRef.HookList.InputHook.push_back(ToCaptured());
            }

            inline void ReadObject(INode* Node) {
                if (!IsOptional || Node->has_input(KeyName)) {
                    ValueRef = Node->get_input<RawType_t<Type>>(KeyName);
                }
            }

            inline void ReadPrimitiveValue(INode* Node) {
                if (!IsOptional || Node->has_input(KeyName)) {
                    ValueRef = Node->get_input2<RawType_t<Type>>(KeyName);
                }
            }

            void Read(INode* Node) {
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

            /**
             * Note: don't call lambda returned without ownership
             * @return
             */
            std::function<void(INode*)> ToCaptured() override {
                return [this] (INode* Node) {
                    Read(Node);
                };
            }
        };

        template <typename ParentType, typename InputType>
        struct OutputField : public Field<ParentType, InputType> {
            OutputField(Parent& ParentRef, Type& InValueRef, std::string InKeyName, bool InIsOptional = false, const std::optional<std::string>& InDisplayName = std::nullopt)
                : Field<ParentType, InputType>(ParentRef, InValueRef, InKeyName, InIsOptional, InDisplayName)
            {
                ParentRef.HookList.OutputHook.push_back(ToCaptured());
            }

            inline void WriteObject(INode* Node) {
                if (ValueRef) {
                    Node->set_output(KeyName, ValueRef);
                } else if (!IsOptional) {
                    zeno::log_error("Node '{}' has invalid output '{}'.", Node->myname, KeyName);
                }
            }

            inline void WritePrimitiveValue(INode* Node) {
                Node->set_output2(KeyName, ValueRef);
            }

            void Write(INode* Node) {
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

            std::function<void(INode*)> ToCaptured() override {
                return [this] (INode* Node) {
                    Write(Node);
                };
            }
        };

        template <typename NodeParameterType>
        struct IAutoNode : public INode {
            static_assert(std::is_base_of_v<NodeParameterBase, NodeParameterType>);

            std::unique_ptr<NodeParameterType> AutoParameter = nullptr;

            ZENO_API IAutoNode() = default;

            void preApply() override {
                INode::preApply();
                AutoParameter = std::make_unique<NodeParameterType>( this );
            }

            void complete() override {
                INode::complete();
                AutoParameter.reset();
            }
        };
    };

}

#endif//ZENO_PROPERTYVISITOR_H
