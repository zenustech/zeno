#pragma once
#include <stdint.h>
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>
#include <optional>

enum class MetadataType : uint8_t {
    None = 0,
    Struct,
    Enum,
    Function,
    EnumValue,
    StructField,
    FunctionParameter,
};

MetadataType string_to_metadata_type(const std::string& str);

std::string metadata_type_to_string(MetadataType type);

struct MetadataContainer {
    using Value = std::variant<std::string, std::vector<std::string>>;

    MetadataType type = MetadataType::None;
    std::unordered_map<std::string, Value> properties;
};

/**
 * A CFG DSL garmmar parser
*/
class MetadataParser {
public:
    static MetadataContainer parse(const std::string &in_dsl);

protected:
    enum class TokenType : uint8_t {
        Unknown = 0,
        EndOfFile,
        LeftBracket,
        RightBracket,
        Equal,
        Comma,
        Word,
    };

    struct Token {
        TokenType type;
        // inclusive
        size_t start_range;
        // exclusive
        size_t end_range;
        // Word value
        std::optional<std::string> word_value = std::nullopt;
    };

    std::string current_text;

    size_t m_pos = 0;
    size_t m_end;

    std::optional<Token> current_token;

    struct {
        int32_t is_slate: 1;
        int32_t inside_quote: 1;
    } m_lexer_state;

    struct {
        int32_t found_type: 1;
        int32_t aborted: 1;
        int32_t key_value: 1; // 0 => key, 1 => value
        int32_t inside_bracket: 1;
        std::stringstream key_buffer;
        std::stringstream value_buffer;
    } m_parser_state;

protected:
    MetadataParser(std::string in_dsl);
    // lexer
    Token next_token();
    MetadataContainer run();

    bool is_aborted() const;

    static bool is_operator(char c);
};

MetadataContainer parse_metadata_dsl(const std::string& in_dsl);
