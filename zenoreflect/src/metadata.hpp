#pragma once
#include <stdint.h>
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>
#include <optional>
#include <map>
#include <cctype>
#include <stdexcept>

enum class MetadataType : uint8_t {
    None = 0,
    Struct,
    Enum,
    Function,
    EnumValue,
    StructField,
    FunctionParameter,
    Trait,
};

MetadataType string_to_metadata_type(const std::string& str);

std::string metadata_type_to_string(MetadataType type);

struct MetadataContainer {
    using Value = std::variant<std::string, std::vector<std::string>>;

    MetadataType type = MetadataType::None;
    std::unordered_map<std::string, Value> properties;
};

MetadataContainer parse_metadata_dsl(const std::string& in_dsl);

enum class TokenType {
    KEY, STRING, NUMBER, LIST_START, LIST_END, EQUAL, COMMA, END
};

std::string token_type_to_string(TokenType type);

struct Token {
    TokenType type;
    std::string value;
};

class Tokenizer {
private:
    std::string m_origin_string;
    std::istringstream ss;
    char current_char;

public:
    Tokenizer(const std::string& input);

    Token next_token();

    const std::string& origin_string() const;

private:
    void next_char();
    void consume_whitespace();

    Token number();
    Token key();
    Token string();
};

class Parser {
private:
    Tokenizer tokenizer;
    Token current_token;
public:
    Parser(const std::string& input);

    std::map<std::string, std::string> parse();

private:
    void next_token();

    std::string expect(TokenType token_expected);
    std::string parse_value();
    std::string parse_list();
};
