#pragma once

#include <string>
#include <concepts>
#include <utility>
#include "clang/AST/Type.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/AST/DeclCXX.h"
#include "inja/inja.hpp"
#include "template/template_literal"
#include "utils.hpp"
#include "args.hpp"

class ReflectionASTConsumer;

namespace zeno::reflect
{
    struct CodeCompilerState {
        std::unordered_map<size_t, uint8_t> type_hash_flag;
        inja::json types_register_data;
        ReflectionASTConsumer* m_consumer;

        CodeCompilerState(ReflectionASTConsumer* in_consumer);
    };
    
    template <typename T>
    concept ICodeCompiler = requires(T v, CodeCompilerState& state) {
        { v.compile(state) } -> std::convertible_to<std::string>;
    };

    template <typename T, typename TypeToHash>
    concept IHashImpl = requires(T v, const TypeToHash& s) {
        { v(s) } -> std::convertible_to<size_t>;
    };

    class ForwardDeclarationGenerator {
        clang::QualType m_qual_type;

    public:
        ForwardDeclarationGenerator(const clang::QualType& qual_type);

        std::string compile(CodeCompilerState& state);
    };

    template <typename HashImpl = zeno::reflect::FNV1aHash, ICodeCompiler ForwordDeclGenerator = ForwardDeclarationGenerator>
    requires IHashImpl<HashImpl, std::string> && std::default_initializable<HashImpl>
    class RTTITypeGenerator {
        clang::QualType m_qual_type;

    public:
        RTTITypeGenerator(
            clang::QualType qual_type
        ) : m_qual_type(qual_type) {}

        std::string compile(CodeCompilerState& state) {
            const size_t hash_value = HashImpl{}(m_qual_type.getCanonicalType().getAsString());
            const std::string cppType = m_qual_type.getAsString();
            const std::string name = m_qual_type.getCanonicalType().getAsString();
            if (is_blacklisted_keyword(cppType)) {
                if (GLOBAL_CONTROL_FLAGS->verbose) {
                    llvm::outs() << "[debug] Skipping compiler internal type \"" << cppType << "\"\n";
                }
                return "";
            }
            if (state.type_hash_flag.contains(hash_value)) {
                return "";
            }
            state.type_hash_flag.insert_or_assign(hash_value, 1);

            inja::json data;

            const std::string forward_decl = ForwordDeclGenerator(m_qual_type).compile(state);
            data["forwordDecl"] = forward_decl;
            data["cppType"] = cppType;
            data["name"] = name;
            data["hash"] = hash_value;
            data["isPointer"] = m_qual_type->isPointerType();
            
            return inja::render(text::RTTI, data);
        }

        static inline bool is_blacklisted_keyword(std::string_view keyword) {
            return keyword.starts_with("__") 
            || keyword.find("__int128") != std::string::npos
            || keyword.find("__va_list_tag") != std::string::npos
            || keyword.find("__NSConstantString") != std::string::npos
            ;
        }
    };

    class TemplateHeaderGenerator {
        CodeCompilerState& m_compiler_state;
        std::stringstream m_rtti_block{};

    public:
        TemplateHeaderGenerator(CodeCompilerState& state);

        std::string compile();
        std::string compile(CodeCompilerState& state);

        template <ICodeCompiler T>
        void add_rtti_block(T generator) {
            m_rtti_block << generator.compile(m_compiler_state);
        }

        template <ICodeCompiler T = RTTITypeGenerator<>>
        void add_rtti_type(clang::QualType type) {
            if (type.getAsString() == "void" || type.getAsString() == "void*" || type.getAsString() == "std::nullptr_t") {
                return;
            }
            m_rtti_block << RTTITypeGenerator(type).compile(m_compiler_state);
        }
    };
}
