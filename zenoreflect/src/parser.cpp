#include <fstream>
#include "args.hpp"
#include "log.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "serialize.hpp"
#include "codegen.hpp"
#include "template/template_literal"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

template <class T>
inline void add_type_to_generator(T* m_context, clang::QualType type) {
    m_context->template_header_generator->add_rtti_type(type);
    auto ut1 = type.getNonReferenceType();
    ut1.removeLocalConst();
    auto ut = ut1->getCanonicalTypeUnqualified();
    m_context->template_header_generator->add_rtti_type(m_context->scoped_context->getConstType(ut));
    m_context->template_header_generator->add_rtti_type(m_context->scoped_context->getLValueReferenceType(ut));
    m_context->template_header_generator->add_rtti_type(m_context->scoped_context->getRValueReferenceType(ut));
    m_context->template_header_generator->add_rtti_type(m_context->scoped_context->getLValueReferenceType(m_context->scoped_context->getConstType(ut)));
    m_context->template_header_generator->add_rtti_type(m_context->scoped_context->getPointerType(m_context->scoped_context->getConstType(ut)));
    m_context->template_header_generator->add_rtti_type(m_context->scoped_context->getPointerType(ut));
}

class ReflectionGeneratorAction : public ASTFrontendAction {
public:
    ReflectionGeneratorAction(zeno::reflect::CodeCompilerState& compielr_state, std::string header_path): m_compiler_state(compielr_state), m_header_path(std::move(header_path)) {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &compiler, StringRef code) override {
        return std::make_unique<ReflectionASTConsumer>(m_compiler_state, m_header_path);
    }

private:
    zeno::reflect::CodeCompilerState& m_compiler_state;
    std::string m_header_path;
};

ParserErrorCode generate_reflection_model(const TranslationUnit &unit, ReflectionModel &out_model, zeno::reflect::CodeCompilerState& root_state) {
    out_model.debug_name = unit.identity_name;
    std::vector<std::string> args = zeno::reflect::get_parser_command_args(GLOBAL_CONTROL_FLAGS->cpp_version, GLOBAL_CONTROL_FLAGS->include_dirs, GLOBAL_CONTROL_FLAGS->pre_include_headers, GLOBAL_CONTROL_FLAGS->verbose);

    const std::string template_header_dir = zeno::reflect::get_file_path_in_header_output(std::format("reflect/{}", GLOBAL_CONTROL_FLAGS->target_name));
    const std::string gen_template_header_path = std::format("{}/{}.generated.hpp", template_header_dir, zeno::reflect::normalize_filename(unit.identity_name));
    zeno::reflect::mkdirs(template_header_dir);
    zeno::reflect::truncate_file(gen_template_header_path);
    out_model.generated_headers.insert(gen_template_header_path);

    if (!clang::tooling::runToolOnCodeWithArgs(
        std::make_unique<ReflectionGeneratorAction>(root_state, gen_template_header_path),
        unit.source.c_str(),
        args,
        unit.identity_name.c_str()
    )) {
        return ParserErrorCode::InternalError;
    }

    return ParserErrorCode::Success;
}

ParserErrorCode post_generate_reflection_model(const ReflectionModel &model, const zeno::reflect::CodeCompilerState& state)
{
    const std::string generated_header_dir = zeno::reflect::get_file_path_in_header_output("reflect");
    const std::string generated_header_path = zeno::reflect::get_file_path_in_header_output("reflect/reflection.generated.hpp");
    std::ofstream ghp_stream(generated_header_path, std::ios::out | std::ios::trunc);
    ghp_stream << "#pragma once" << "\r\n";
    auto header_list = zeno::reflect::find_files_with_extension(generated_header_dir, ".hpp");
    for (const std::string& s : header_list) {
        const auto relative_path = zeno::reflect::relative_path_to_header_output(s);
        if (zeno::reflect::relative_path_to_header_output(generated_header_path) != relative_path) {
            ghp_stream << std::format("#include \"{}\"", relative_path) << "\r\n";
        }
    }

    const std::string generated_target_source = GLOBAL_CONTROL_FLAGS->target_type_register_source_path;
    std::ofstream gts_stream(generated_target_source, std::ios::out | std::ios::trunc);
    gts_stream << inja::render(zeno::reflect::text::REFLECTED_TYPE_REGISTER, state.types_register_data);

    return ParserErrorCode::Success;
}

ParserErrorCode pre_generate_reflection_model()
{
    const std::string generated_header_path = zeno::reflect::get_file_path_in_header_output("reflect/reflection.generated.hpp");
    zeno::reflect::truncate_file(generated_header_path);

    return ParserErrorCode::Success;
}

TemplateSpecializationMatchCallback::TemplateSpecializationMatchCallback(ReflectionASTConsumer *context) : m_context(context) {}

void TemplateSpecializationMatchCallback::run(const MatchFinder::MatchResult &result)
{
    if (const ClassTemplateSpecializationDecl* spec_decl = result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>(ASTLabels::TEMPLATE_SPECIALIZATION)) {
        if (spec_decl->getSpecializedTemplate() && spec_decl->getSpecializedTemplate()->getNameAsString() == "_manual_register_rtti_type_internal") {
            if (spec_decl->getTemplateArgs().size() == 1 && m_context) {
                QualType type = spec_decl->getTemplateArgs().get(0).getAsType();
                add_type_to_generator(m_context, type);
            }
        }
    }
}

RecordTypeMatchCallback::RecordTypeMatchCallback(ReflectionASTConsumer *context) : m_context(context) {}

void RecordTypeMatchCallback::run(const MatchFinder::MatchResult &result)
{
    if (const CXXRecordDecl* record_decl = result.Nodes.getNodeAs<CXXRecordDecl>(ASTLabels::RECORD_LABEL)) {
        if (!record_decl->hasDefinition()) {
            return;
        }

        inja::json metadata;

        if (!zeno::reflect::parse_metadata(metadata, record_decl)) {
            return;
        }
        
        // Generate rtti information
        const clang::Type* record_type = record_decl->getTypeForDecl();
        QualType record_qual_type(record_type, 0);
        add_type_to_generator(m_context, record_qual_type);
        if (GLOBAL_CONTROL_FLAGS->verbose) {
            m_context->scoped_context->DumpRecordLayout(record_decl, llvm::outs());
        }

        {
            const std::string normalized_name = zeno::reflect::convert_to_valid_cpp_var_name(record_qual_type.getCanonicalType().getAsString());
            bool found = false;
            for (auto type_info : m_context->m_compiler_state.types_register_data["types"]) {
                if (type_info["normal_name"] == normalized_name) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                inja::json type_data;
                type_data["normal_name"] = normalized_name;
                type_data["qualified_name"] = zeno::reflect::clang_type_name_no_tag(record_qual_type);
                type_data["canonical_typename"] = record_qual_type.getCanonicalType().getAsString();
                type_data["ctors"] = inja::json::array();
                type_data["funcs"] = inja::json::array();
                type_data["fields"] = inja::json::array();
                type_data["base_classes"] = inja::json::array();

                // Metadata
                {
                    std::string metadata_interface = inja::render(zeno::reflect::text::REFLECTED_METADATA, metadata);
                    type_data["metadata"] = metadata_interface;
                }

                // Processing methods
                {
                    for (auto it = record_decl->method_begin(); it != record_decl->method_end(); ++it) {
                        // Register all param types used
                        if (const CXXMethodDecl* method_decl = dyn_cast<CXXMethodDecl>(*it)) {
                            for (unsigned int i = 0; i < method_decl->getNumParams(); ++i) {
                                const ParmVarDecl* param_decl = method_decl->getParamDecl(i);
                                if (param_decl) {
                                    QualType type = param_decl->getType();
                                    add_type_to_generator(m_context, type);
                                }
                            }
                            QualType type = method_decl->getReturnType().getCanonicalType();
                            add_type_to_generator(m_context, type);
                        }

                        // If is aggregate type then add list initialization as a constructor
                        // NOTE: Empty base class optimization might lead to, a class with empty base class is a aggregate class
                        // But if you try list initialization on it, it will be a compiler error there.
                        if (record_decl->isAggregate() && record_decl->getNumBases() == 0) {
                            inja::json ctor_data;
                            ctor_data["is_aggregate_initialize"] = true;
                            ctor_data["params"] = inja::json::array();
                            for (const FieldDecl* field : record_decl->fields())  {
                                QualType type = field->getType();
                                ctor_data["params"].push_back(zeno::reflect::parse_param_data(field));
                            }
                            type_data["ctors"].push_back(ctor_data);
                        }

                        if (const CXXConstructorDecl* constructor_decl = dyn_cast<CXXConstructorDecl>(*it); constructor_decl && constructor_decl->getAccess() == clang::AS_public) {
                            if (!constructor_decl->isDeleted()) {
                                inja::json ctor_data;
                                ctor_data["params"] = inja::json::array();
                                for (unsigned int i = 0; i < constructor_decl->getNumParams(); ++i) {
                                    const ParmVarDecl* param_decl = constructor_decl->getParamDecl(i);
                                    inja::json param_data = zeno::reflect::parse_param_data(param_decl);
                                    ctor_data["params"].push_back(param_data);
                                }
                                type_data["ctors"].push_back(ctor_data);
                            }
                        } else if (const CXXDestructorDecl* destructor_decl = dyn_cast<CXXDestructorDecl>(*it)) {
                        } else if (const CXXConversionDecl* conversion_decl = dyn_cast<CXXConversionDecl>(*it)) {
                        } else if (const CXXMethodDecl* method_decl = dyn_cast<CXXMethodDecl>(*it); method_decl && method_decl->getAccess() == clang::AS_public && !method_decl->isOverloadedOperator()) {

                            inja::json func_data;
                            func_data["name"] = zeno::reflect::convert_to_valid_cpp_var_name(method_decl->getNameAsString());
                            func_data["ret"] = method_decl->getReturnType().getCanonicalType().getAsString();
                            func_data["params"] = inja::json::array();
                            for (unsigned int i = 0; i < method_decl->getNumParams(); ++i) {
                                const ParmVarDecl* param_decl = method_decl->getParamDecl(i);
                                inja::json param_data = zeno::reflect::parse_param_data(param_decl);
                                func_data["params"].push_back(param_data);
                            }
                            func_data["static"] = method_decl->isStatic();
                            func_data["const"] = method_decl->isConst();
                            func_data["noexcept"] = method_decl->getExceptionSpecType() == clang::EST_BasicNoexcept || method_decl->getExceptionSpecType() == clang::EST_NoexceptTrue;

                            std::string method_metadata;
                            if (zeno::reflect::parse_metadata(method_metadata, method_decl)) {
                                func_data["metadata"] = method_metadata;
                            }

                            type_data["funcs"].push_back(func_data);
                        }
                    }
                }

                {
                    for (auto it = record_decl->field_begin(); it != record_decl->field_end(); ++it) {
                        if (const FieldDecl* field_decl = dyn_cast<FieldDecl>(*it); field_decl && field_decl->getAccess() == clang::AS_public) {
                            QualType type = field_decl->getType();
                            m_context->template_header_generator->add_rtti_type(type);
                            m_context->template_header_generator->add_rtti_type(type.getUnqualifiedType());

                            inja::json field_data;
                            field_data["name"] = field_decl->getNameAsString();
                            field_data["type"] = type.getCanonicalType().getAsString();
                            field_data["normal_type"] = zeno::reflect::convert_to_valid_cpp_var_name(type.getCanonicalType().getAsString());

                            std::string field_metadata;
                            if (zeno::reflect::parse_metadata(field_metadata, field_decl)) {
                                field_data["metadata"] = field_metadata;
                            }

                            type_data["fields"].push_back(field_data);
                        }
                    }
                }

                {
                    for (auto it = record_decl->bases_begin(); it != record_decl->bases_end(); ++it) {
                        if (const CXXBaseSpecifier* base_decl = it) {
                            inja::json base_data;
                            QualType type = base_decl->getType().getCanonicalType();
                            base_data["type"] = zeno::reflect::clang_type_name_no_tag(type);

                            add_type_to_generator(m_context, type);

                            type_data["base_classes"].push_back(base_data);
                        }
                    }
                }

                m_context->m_compiler_state.types_register_data["types"].push_back(type_data);
            }
        }

        if (record_decl->getNumBases() > 0) {
            for (const auto& base : record_decl->bases()) {
            }
        }
    }
}

ReflectionASTConsumer::ReflectionASTConsumer(zeno::reflect::CodeCompilerState &state, std::string header_path)
    : m_compiler_state(state)
    , m_header_path(header_path)
    , template_header_generator(std::make_unique<zeno::reflect::TemplateHeaderGenerator>(state))
{
    state.m_consumer = this;
}

void ReflectionASTConsumer::HandleTranslationUnit(ASTContext &context)
{
    scoped_context = &context;
    // template_header_generator.add_rtti_type(context.VoidTy);

    const std::string& gen_template_header_path = m_header_path;

    MatchFinder manual_rtti_register_finder{};
    DeclarationMatcher template_spec_matcher = classTemplateSpecializationDecl().bind(ASTLabels::TEMPLATE_SPECIALIZATION);
    manual_rtti_register_finder.addMatcher(template_spec_matcher, template_specialization_handler.get());
    manual_rtti_register_finder.matchAST(context);

    DeclarationMatcher record_type_matcher = cxxRecordDecl().bind(ASTLabels::RECORD_LABEL);
    MatchFinder record_finder{};
    record_finder.addMatcher(record_type_matcher, record_type_handler.get());
    record_finder.matchAST(context);

    // generate header
    const std::string generated_templates = template_header_generator->compile();
    
    std::ofstream generated_templates_stream(gen_template_header_path, std::ios::out | std::ios::trunc);
    generated_templates_stream << generated_templates;

    scoped_context = nullptr;
}

void ReflectionASTConsumer::add_type_mapping(const std::string &alias_name, QualType real_name)
{
    if (GLOBAL_CONTROL_FLAGS->verbose) {
        llvm::outs() << "[debug] " << "Added type alias: «" << alias_name << "» → «" << real_name << "»\n";
    }
    type_name_mapping.insert_or_assign(alias_name, real_name);
}
