#pragma once

#include <string>
#include <unordered_map>
#include <set>
#include "metadata.hpp"
#include "codegen.hpp"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

enum class TranslationUnitType {
    Header,
    Standalone,
};

enum class ParserErrorCode {
    Success = 0,
    InternalError = 1,
    TUCreationFailure = 2,
};

struct TranslationUnit {
    std::string identity_name;
    std::string source;
    TranslationUnitType type = TranslationUnitType::Standalone;
};

struct ReflectionFieldDecl {
    std::string name;
    uint32_t is_ptr: 1 = false;
    uint32_t is_ref: 1 = false;
};

struct ReflectionFunctionDecl {
    std::string name;
    uint32_t is_pure: 1 = false;
    uint32_t is_virtual: 1 = false;
    uint32_t is_pure_virtual: 1 = false;
};

struct BaseStructDecl {
    std::string name;
    clang::AccessSpecifier access_spec;
};

struct ReflectionStruct {
    std::string name;
    std::vector<BaseStructDecl> bases;
    std::unordered_map<std::string, ReflectionFieldDecl> fields;
    std::unordered_map<std::string, ReflectionFunctionDecl> functions;
    MetadataContainer metadata;
};

struct ReflectionModel {
    std::string debug_name;
    std::unordered_map<std::string, ReflectionStruct> structs;
    std::set<std::string> generated_headers;
};

ParserErrorCode generate_reflection_model(const TranslationUnit& unit, ReflectionModel& out_model, zeno::reflect::CodeCompilerState& root_state);
ParserErrorCode post_generate_reflection_model(const ReflectionModel& model, const zeno::reflect::CodeCompilerState& state);
ParserErrorCode pre_generate_reflection_model();

struct ASTLabels {
    inline static const char* RECORD_LABEL = "cxxRecord";
    inline static const char* TYPEDEF_LABEL = "typedef";
    inline static const char* TYPE_ALIAS_LABEL = "typeAlias";
};

class ReflectionASTConsumer;

struct TypeAliasMatchCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
    TypeAliasMatchCallback(ReflectionASTConsumer* context);

    void run(const clang::ast_matchers::MatchFinder::MatchResult &result) override;

private:
    ReflectionASTConsumer* m_context;
};

struct RecordTypeMatchCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
    RecordTypeMatchCallback(ReflectionASTConsumer* context);
    
    void run(const clang::ast_matchers::MatchFinder::MatchResult &result) override;

private:
    ReflectionASTConsumer* m_context;
};

class ReflectionASTConsumer : public clang::ASTConsumer {
public:
    ReflectionASTConsumer(zeno::reflect::CodeCompilerState& state, std::string header_path);

    void HandleTranslationUnit(clang::ASTContext &context) override;

    void add_type_mapping(const std::string& alias_name, clang::QualType real_name);

    zeno::reflect::TemplateHeaderGenerator template_header_generator;

    clang::ASTContext* scoped_context = nullptr;
private:
    std::unique_ptr<RecordTypeMatchCallback> record_type_handler = std::make_unique<RecordTypeMatchCallback>(this);
    std::unique_ptr<TypeAliasMatchCallback> type_alias_handler = std::make_unique<TypeAliasMatchCallback>(this);

    std::unordered_map<std::string, clang::QualType> type_name_mapping;
    zeno::reflect::CodeCompilerState& m_compiler_state;
    std::string m_header_path;

    friend struct RecordTypeMatchCallback;
};
