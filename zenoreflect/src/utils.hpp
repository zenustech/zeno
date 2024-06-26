#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <ostream>
#include <iostream>
#include <optional>
#include <vector>
#include <format>
#include <string_view>
#include "metadata.hpp"
#include "inja/inja.hpp"

namespace clang {
    class Expr;
    class ParmVarDecl;
    class FieldDecl;
    class CXXRecordDecl;
    class QualType;
    class Decl;
}

namespace zeno {

namespace reflect {

std::optional<std::string> read_file(const std::string& filepath);

std::vector<std::string> get_parser_command_args(
    const std::string& cpp_version,
    std::vector<std::string>& include_dirs,
    std::vector<std::string>& pre_include_headers,
    bool verbose = false
);

bool is_vaild_char(char c);
std::string trim_start(const std::string& str);
std::string trim_end(const std::string& str);
std::string trim(const std::string& str);

std::vector<std::string_view> split(std::string_view str, std::string_view delimiter);

std::string get_file_path_in_header_output(std::string_view filename);
std::string relative_path_to_header_output(std::string_view abs_path);
void truncate_file(const std::string& path);

std::string normalize_filename(std::string_view input);
std::string convert_to_valid_cpp_var_name(std::string_view type_name);

std::string clang_expr_to_string(const clang::Expr* expr);
std::string clang_type_name_no_tag(const clang::QualType& type);
inja::json parse_param_data(const clang::ParmVarDecl* param_decl);
inja::json parse_param_data(const clang::FieldDecl* param_decl);

inja::json parse_metadata(const MetadataContainer& metadata);
bool parse_metadata(inja::json& out, const clang::Decl* decl);
bool parse_metadata(std::string& out, const clang::Decl* decl);

namespace internal {
    template <typename T>
    struct FNV1aInternal {
        static constexpr uint32_t val = 0x811c9dc5U;
        static constexpr uint32_t prime = 0x1000193U;
    };

    template <>
    struct FNV1aInternal<uint64_t> {
        static constexpr uint64_t val = 0xcbf29ce484222325ULL;
        static constexpr uint64_t prime = 0x100000001b3ULL;
    };
}

struct FNV1aHash {
    
    constexpr uint32_t hash_32_fnv1a(std::string_view str) const noexcept;
    constexpr uint64_t hash_64_fnv1a(std::string_view str) const noexcept;

    size_t operator()(std::string_view str) const noexcept;
};

}
}
