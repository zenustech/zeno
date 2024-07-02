#include "codegen.hpp"
#include "clang/AST/DeclTemplate.h"
#include "parser.hpp"

zeno::reflect::TemplateHeaderGenerator::TemplateHeaderGenerator(CodeCompilerState &state)
    : m_compiler_state(state)
{
}

std::string zeno::reflect::TemplateHeaderGenerator::compile()
{
    return compile(m_compiler_state);
}

std::string zeno::reflect::TemplateHeaderGenerator::compile(CodeCompilerState &state)
{
    inja::json template_data;
    template_data["rttiBlock"] = m_rtti_block.str();
    return inja::render(text::GENERATED_TEMPLATE_HEADER_TEMPLATE, template_data);
}

zeno::reflect::ForwardDeclarationGenerator::ForwardDeclarationGenerator(const clang::QualType &qual_type)
    : m_qual_type(qual_type)
{
}

std::string zeno::reflect::ForwardDeclarationGenerator::compile(CodeCompilerState &state)
{
    std::stringstream buf;
    const clang::Type *type = m_qual_type.getTypePtr();
    const std::string clazz_name = m_qual_type.getAsString();
    
    if (const clang::TemplateSpecializationType* spec_type = type->getAs<clang::TemplateSpecializationType>(); spec_type || clazz_name.ends_with('>')) {
        buf << "// !!! importance: This is a template specialization \"" << clazz_name << "\", doesn't generate forward declaration.\r\n";
        return buf.str();
    }

    if (type->isRecordType()) {
        // struct / class / union ?
        const clang::RecordType *record_type = type->getAs<clang::RecordType>();
        const clang::RecordDecl *record_decl = record_type->getDecl();

        // typename
        std::string type_name = record_decl->getNameAsString();

        // namespace
        const clang::DeclContext *dc = record_decl->getDeclContext();
        std::vector<std::string> namespaces;
        while (dc && !dc->isTranslationUnit()) {
            if (const clang::NamespaceDecl *nd = clang::dyn_cast<clang::NamespaceDecl>(dc)) {
                namespaces.push_back(nd->getNameAsString());
            }
            dc = dc->getParent();
        }

        // print
        std::string declaration;
        if (record_decl->isClass()) {
            declaration = "class " + type_name + ";";
        } else if (record_decl->isStruct()) {
            declaration = "struct " + type_name + ";";
        } else if (record_decl->isUnion()) {
            declaration = "union " + type_name + ";";
        }

        for (auto it = namespaces.rbegin(); it != namespaces.rend(); ++it) {
            buf << "namespace " << *it << " {\r\n";
        }

        buf << declaration << std::endl;

        for (size_t i = 0; i < namespaces.size(); ++i) {
            buf << "}\r\n";
        }
    }

    return buf.str();
}

zeno::reflect::CodeCompilerState::CodeCompilerState(ReflectionASTConsumer* in_consumer)
    : m_consumer(in_consumer)
{
    types_register_data["types"] = std::vector<inja::json>{};
    types_register_data["prefix"] = "";
    types_register_data["headers"] = GLOBAL_CONTROL_FLAGS->input_sources;
}
