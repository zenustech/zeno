#include <string>
#include <cctype>
#include <filesystem>
#include <cassert>
#include "utils.hpp"
#include "args.hpp"
#include "template/template_literal"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

namespace zeno {

namespace reflect {

std::optional<std::string> read_file(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return std::nullopt;
    }
    std::stringstream buf;
    buf << file.rdbuf();
    return buf.str();
}

std::vector<std::string> get_parser_command_args(const std::string &cpp_version, std::vector<std::string> &include_dirs, std::vector<std::string> &pre_include_headers, bool verbose)
{
    std::vector<std::string> result;

    // Set Language to c++ (libclang assume a header file which suffix equal to .h is a c source code)
    result.push_back("-x");
    result.push_back("c++");

    // Ignore wanrings
    result.push_back("-Wno-pragma-once-outside-header");

    // Set standrad version
    result.push_back(std::format("-std=c++{}", cpp_version));
    // Set definitions
    // ZENO_REFLECT_PROCESSING will only set when parsing by this generator
    result.push_back("-DZENO_REFLECT_PROCESSING=1");
    result.push_back("-DWITH_REFLECT=1");
    // Add include directories
    for (const std::string& dir : include_dirs) {
        result.push_back(
            std::format("-I{}", dir)
        );
    }
    // Add pre include headers
    for (const std::string& header : pre_include_headers) {
        result.push_back("-include");
        result.push_back(header);
    }

    // if (verbose) {
    //     result.push_back("-v");
    // }

    // if (verbose) {
    //     std::cout << "[debug] Arguments passing to Clang:\t\"";
    //     for (const std::string& str : result) {
    //         std::cout << str << " ";
    //     }
    //     std::cout << "\"" << std::endl;
    // }

    return result;
}

bool is_vaild_char(char c)
{
    return c >= -1;
}

std::string trim_start(const std::string &str)
{
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }
    return str.substr(start);
}

std::string trim_end(const std::string &str)
{
    size_t end = str.length();
    while (end > 0 && std::isspace(static_cast<unsigned char>(str[end - 1]))) {
        end--;
    }
    return str.substr(0, end);
}

std::string trim(const std::string &str)
{
    return trim_end(trim_start(str));
}

std::vector<std::string_view> split(std::string_view str, std::string_view delimiter)
{
    std::vector<std::string_view> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string_view::npos) {
        if (end > start) {
            parts.push_back(str.substr(start, end - start));
        }
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    if (start < str.length()) {
        parts.push_back(str.substr(start));
    }

    return parts;
}

std::string get_file_path_in_header_output(std::string_view filename)
{
    return (std::filesystem::path(GLOBAL_CONTROL_FLAGS->output_dir) / std::filesystem::path(filename)).string();
}

std::string relative_path_to_header_output(std::string_view abs_path)
{
    const std::filesystem::path header_output_dir(GLOBAL_CONTROL_FLAGS->output_dir);
    const std::filesystem::path input_path(abs_path);

    return std::filesystem::relative(input_path, header_output_dir).string();
}

void truncate_file(const std::string &path)
{
    std::ofstream s(path, std::ios::out | std::ios::trunc);
    s.close();
}

bool mkdirs(std::string_view path)
{
    std::filesystem::path dir_path(path);
    try {
        if (std::filesystem::create_directories(dir_path)) {
            return true;
        }
    } catch (const std::filesystem::filesystem_error& err) {
        std::cout << "error: " << err.what() << "\n";
    }

    return false;
}

std::vector<std::string> find_files_with_extension(std::string_view root, std::string_view extension)
{
    std::vector<std::string> matching_files;

    try {
        if (std::filesystem::exists(root) && std::filesystem::is_directory(root)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
                if (entry.is_regular_file() && entry.path().extension() == extension) {
                    matching_files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& err) {
        std::cout << "error: " << err.what() << "\n";
    }

    return matching_files;
}

std::string normalize_filename(std::string_view input)
{
    return std::filesystem::path(input).lexically_normal().filename().string();
}

std::string convert_to_valid_cpp_var_name(std::string_view type_name)
{
    std::string var_name;
    bool last_was_colon = false;

    for (std::size_t i = 0; i < type_name.size(); ++i) {
        char ch = type_name[i];

        // RecordTypes => struct, class, union
        if (type_name.substr(i, 6) == "struct" && (i + 6 == type_name.size() || type_name[i + 6] == ' ')) {
            var_name += "struct_";
            i += 6; // "struct"
        } else if (type_name.substr(i, 5) == "class" && (i + 5 == type_name.size() || type_name[i + 5] == ' ')) {
            var_name += "class_";
            i += 5; // "class"
        } else if (type_name.substr(i, 5) == "union" && (i + 5 == type_name.size() || type_name[i + 5] == ' ')) {
            var_name += "union_";
            i += 5; // "union"
        } else if (std::isalnum(static_cast<unsigned char>(ch))) { // Valid chars
            var_name += ch;
            last_was_colon = false;
        } else if (ch == ':' || ch == ' ') {
            if (!last_was_colon && !var_name.empty() && var_name.back() != '_') {
                var_name += '_';  // Replace ":" or " "
            }
            last_was_colon = (ch == ':');
        } else if (ch == '+') {
            var_name += "Plus";
        } else if (ch == '-') {
            var_name += "Minus";
        } else if (ch == '/') {
            var_name += "Div";
        } else if (ch == '*') {
            var_name += "Mul";
        } else if (ch == '<') {
            var_name += "LAB";
        } else if (ch == '>') {
            var_name += "RAB";
        }
    }

    if (!var_name.empty() && var_name.back() == '_') {
        var_name.pop_back();
    }

    // Ensure not started with number
    if (!var_name.empty() && std::isdigit(static_cast<unsigned char>(var_name[0]))) {
        var_name = "_" + var_name;
    }

    return var_name;
}

std::string clang_expr_to_string(const clang::Expr *expr)
{
    if (!expr) {
        return "nullptr";
    }
    
    clang::LangOptions lang_opts;
    lang_opts.CPlusPlus = true;
    clang::PrintingPolicy policy(lang_opts);

    std::string str;
    llvm::raw_string_ostream stream(str);
    expr->printPretty(stream, nullptr, policy);

    return str;
}

std::string clang_type_name_no_tag(const clang::QualType& type)
{
    clang::LangOptions lang_opts;
    lang_opts.CPlusPlus = true;
    clang::PrintingPolicy policy(lang_opts);
    policy.SuppressTagKeyword = true;
    return type.getAsString(policy);
}

inja::json parse_param_data(const clang::ParmVarDecl * param_decl)
{

    inja::json param_data;
    clang::QualType type = param_decl->getType();
    param_data["type"] = type.getCanonicalType().getAsString();
    param_data["has_default_arg"] = param_decl->hasDefaultArg();
    if (param_decl->hasDefaultArg()) {
        param_data["default_arg"] = zeno::reflect::clang_expr_to_string(param_decl->getDefaultArg());
    }

    return param_data;

}

inja::json parse_param_data(const clang::FieldDecl *param_decl)
{
    inja::json param_data;
    clang::QualType type = param_decl->getType();
    param_data["type"] = type.getCanonicalType().getAsString();
    param_data["has_default_arg"] = param_decl->hasInClassInitializer();
    if (param_decl->hasInClassInitializer()) {
        param_data["default_arg"] = zeno::reflect::clang_expr_to_string(param_decl->getInClassInitializer());
    }

    return param_data;
}

inja::json parse_metadata(const MetadataContainer &metadata)
{
    inja::json data;

    inja::json properties_json;
    properties_json["MetadataType"] = metadata_type_to_string(metadata.type);

    for (const auto& [key, value] : metadata.properties) {
        if (key == "") {
            continue;
        }
        std::visit([&properties_json, &key](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            inja::json value_data;
            if constexpr (std::is_same_v<T, std::string>) {
                value_data["value"] = arg;
                value_data["is_string"] = true;
                properties_json[key] = value_data;
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                inja::json value_data;
                inja::json array;
                for (const auto& item : arg) {
                    array.push_back(item);
                }
                value_data["value"] = array;
                value_data["is_array"] = true;
                properties_json[key] = value_data;
            }
        }, value);
    }

    data["properties"] = properties_json;

    return data;
}

bool parse_metadata(inja::json &out, const clang::Decl *decl)
{
    if (clang::AnnotateAttr* attr = decl->getAttr<clang::AnnotateAttr>()) {
        MetadataContainer container = parse_metadata_dsl(attr->getAnnotation().str());

        out = parse_metadata(container);

        return true;
    }

    return false;
}

bool parse_metadata(std::string &out, const clang::Decl *decl)
{
    inja::json data;
    if (parse_metadata(data, decl)) {
        out = inja::render(zeno::reflect::text::REFLECTED_METADATA, data);
        return true;
    }

    return false;
}

const clang::Type *get_underlying_type(const clang::Type *type)
{
    while (type) {
        if (const clang::ElaboratedType* et = clang::dyn_cast<clang::ElaboratedType>(type)) {
            type = et->getNamedType().getTypePtr();
        } else if (const clang::TypedefType* tt = clang::dyn_cast<clang::TypedefType>(type)) {
            type = tt->getDecl()->getUnderlyingType().getTypePtr();
        } else {
            break;
        }
    }
    return type;
}

constexpr uint32_t FNV1aHash::hash_32_fnv1a(std::string_view str) const noexcept
{
    uint32_t hash = internal::FNV1aInternal<uint32_t>::val;
    for (const unsigned char c : str) {
        hash = hash ^ c;
        hash *= internal::FNV1aInternal<uint32_t>::prime;
    }
    return hash;
}

constexpr uint64_t FNV1aHash::hash_64_fnv1a(std::string_view str) const noexcept
{
    uint64_t hash = internal::FNV1aInternal<uint64_t>::val;
    for (const unsigned char c : str) {
        hash = hash ^ c;
        hash *= internal::FNV1aInternal<uint64_t>::prime;
    }
    return hash;
}

size_t FNV1aHash::operator()(std::string_view str) const noexcept
{
    if constexpr (sizeof(size_t) == sizeof(uint32_t)) {
        return hash_32_fnv1a(str);
    } else if constexpr (sizeof(size_t) == sizeof(uint64_t)) {
        return hash_64_fnv1a(str);
    } else {
        assert(false);
    }
}

}
}

