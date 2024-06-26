#include <limits>
#include <sstream>
#include <iostream>
#include "metadata.hpp"
#include "args.hpp"
#include "utils.hpp"

MetadataType string_to_metadata_type(const std::string &str)
{
    if (str == "struct") {
        return MetadataType::Struct;
    } else if (str == "enum") {
        return MetadataType::Enum;
    } else if (str == "function" || str == "method") {
        return MetadataType::Function;
    } else if (str == "enum_value") {
        return MetadataType::EnumValue;
    } else if (str == "field") {
        return MetadataType::StructField;
    } else if (str == "param") {
        return MetadataType::FunctionParameter;
    }
    return MetadataType::None;
}

std::string metadata_type_to_string(MetadataType type)
{
    if (type == MetadataType::Struct) {
        return "struct";
    } else if (type == MetadataType::Enum) {
        return "enum";
    } else if (type == MetadataType::Function) {
        return "function";
    } else if (type == MetadataType::EnumValue) {
        return "enum_value";
    } else if (type == MetadataType::StructField) {
        return "field";
    } else if (type == MetadataType::FunctionParameter) {
        return "param";
    }
    return "none";
}

MetadataContainer parse_metadata_dsl(const std::string &in_dsl)
{
    return MetadataParser::parse(in_dsl);
}

MetadataContainer MetadataParser::parse(const std::string &in_dsl)
{
    auto parser = MetadataParser(in_dsl);
    return parser.run();
}

MetadataParser::MetadataParser(std::string in_dsl)
    : current_text(std::move(in_dsl))
    , m_end(current_text.size())
{
    m_lexer_state.is_slate = 0;
    m_lexer_state.inside_quote = 0;

    m_parser_state.found_type = 0;
    m_parser_state.aborted = 0;
    m_parser_state.key_value = 0;
    m_parser_state.inside_bracket = 0;
}

MetadataParser::Token MetadataParser::next_token()
{
    std::stringstream word;
    while (true) {
        if (m_pos >= m_end) {
            return Token {
                .type = TokenType::EndOfFile,
                .start_range = m_pos,
                .end_range = m_pos,
            };
        }

        const char& c = current_text.at(m_pos);
        char next_char = '\0';
        if (current_text.size() > m_pos + 1) {
            next_char = current_text.at(m_pos + 1);
        }
        size_t length = 1;

        switch (c) {
            case '(':
                if (!m_lexer_state.inside_quote) {
                    return Token {
                        .type = TokenType::LeftBracket,
                        .start_range = m_pos,
                        .end_range = ++m_pos,
                    };
                }
            case ')':
                if (!m_lexer_state.inside_quote) {
                    return Token {
                        .type = TokenType::RightBracket,
                        .start_range = m_pos,
                        .end_range = ++m_pos,
                    };
                }
            case '=':
                if (!m_lexer_state.inside_quote) {
                    return Token {
                        .type = TokenType::Equal,
                        .start_range = m_pos,
                        .end_range = ++m_pos,
                    };
                }
            case ',':
                if (!m_lexer_state.inside_quote) {
                    return Token {
                        .type = TokenType::Comma,
                        .start_range = m_pos,
                        .end_range = ++m_pos,
                    };
                }
            case '\\':
                switch (next_char) {
                    case '\\':
                        word << "\\";
                        break;
                    case 'n':
                        word << "\n";
                        break;
                    case 't':
                        word << "\t";
                        break;
                    case '"':
                        word << "\"";
                        break;
                    case '(':
                        word << "(";
                        break;
                    case ')':
                        word << ')';
                        break;
                    case '=':
                        word << '=';
                        break;
                    case ',':
                        word << ',';
                        break;
                    default:
                        if (GLOBAL_CONTROL_FLAGS->verbose) {
                            std::cout << "[debug] Unknown escape character \"" << "\\" << next_char << "\"";
                        }
                }
                m_pos += 2;
                length += 2;
                break;
            case '\"':
                if (m_lexer_state.inside_quote) {
                    m_lexer_state.inside_quote = false;
                    m_pos++;
                    return Token {
                        .type = TokenType::Word,
                        .start_range = m_pos - length,
                        .end_range = m_pos,
                        .word_value = std::make_optional<std::string>(word.str()),
                    };
                } else {
                    m_lexer_state.inside_quote = true;
                    m_pos++;
                    length++;
                }
                break;
            default:
                word << c;
                if ((is_operator(next_char) || next_char == '\0') && !m_lexer_state.inside_quote) {
                    m_pos ++;
                    return Token {
                        .type = TokenType::Word,
                        .start_range = m_pos - length,
                        .end_range = m_pos,
                        .word_value = std::make_optional<std::string>(word.str()),
                    };
                } else {
                    m_pos++;
                    length++;
                }
        }
    }

    return Token {
        .type = TokenType::Unknown,
        .start_range = std::numeric_limits<size_t>::infinity(),
        .end_range = std::numeric_limits<size_t>::infinity(),
    };
}

bool MetadataParser::is_operator(char c)
{
    return c == '(' || c == ')' || c == '=' || c == ',';
}

MetadataContainer MetadataParser::run()
{
    MetadataContainer container{};

    std::vector<std::string> list;
    std::string temp_key;
    do {
        current_token = std::make_optional(next_token());
        if (current_token->type == TokenType::Word) {
            const std::string trimed_word = zeno::reflect::trim(current_token->word_value.value());
            if (trimed_word.starts_with("#")) {
                if (container.type == MetadataType::None) {
                    container.type = string_to_metadata_type(trimed_word.substr(1));
                } else {
                    std::cout << "[Reflect] Metadata parse aborted." << std::endl << "\t\"" << trimed_word << "\" duplicated type declaration."; 
                    m_parser_state.aborted = true;
                    return container;
                }
            } else {
                if (!m_parser_state.key_value) {
                    m_parser_state.key_buffer << trimed_word;
                } else {
                    m_parser_state.value_buffer << trimed_word;
                }
            }
        } else if (current_token->type == TokenType::Equal) {
            m_parser_state.key_value = true;
            temp_key = m_parser_state.key_buffer.str();
        } else if (current_token->type == TokenType::Comma) {
            if (!m_parser_state.inside_bracket) {
                if (list.empty()) {
                    container.properties.insert_or_assign(temp_key, m_parser_state.value_buffer.str());
                } else {
                    container.properties.insert_or_assign(temp_key, std::move(list));
                }
                list = std::vector<std::string>();
                m_parser_state.key_value = false;
            } else {
                list.push_back(m_parser_state.value_buffer.str());
            }
            m_parser_state.key_buffer.str("");
            m_parser_state.key_buffer.clear();
            m_parser_state.value_buffer.str("");
            m_parser_state.value_buffer.clear();
        } else if (current_token->type == TokenType::LeftBracket) {
            if (m_parser_state.inside_bracket) {
                std::cout << "[Reflect] Metadata parse aborted." << std::endl << "\tNested set isn't allowed" << std::endl;
                m_parser_state.aborted = true;
                return container;
            }
            m_parser_state.inside_bracket = true;
        } else if (current_token->type == TokenType::RightBracket) {
            if (!m_parser_state.inside_bracket) {
                std::cout << "[Reflect] Metadata parse aborted." << std::endl << "\tBrackets not matched" << std::endl;
                m_parser_state.aborted = true;
                return container;
            }
            m_parser_state.inside_bracket = false;
            list.push_back(m_parser_state.value_buffer.str());
            m_parser_state.value_buffer.str("");
            m_parser_state.value_buffer.clear();
        } else if (current_token->type == TokenType::EndOfFile) {
            if (m_parser_state.inside_bracket) {
                std::cout << "[Reflect] Metadata parse aborted." << std::endl << "\tBrackets not matched" << std::endl;
                m_parser_state.aborted = true;
                return container;
            }
            if (!m_parser_state.key_buffer.str().empty()) {
                container.properties.insert_or_assign(m_parser_state.key_buffer.str(), m_parser_state.value_buffer.str());
            }
        }
    } while (current_token->type != TokenType::EndOfFile);
    

    if (GLOBAL_CONTROL_FLAGS->verbose) {
        for (const auto& [k, v] : container.properties) {
            std::visit([&k, &v] (auto& val) {
                using ValType = decltype(val);
                std::cout << "[debug] ";
                if constexpr (std::is_convertible_v<ValType, std::string>) {
                    std::cout << k << " = " << std::get<std::string>(v) << std::endl;
                } else if constexpr (std::is_convertible_v<ValType, std::vector<std::string>>) {
                    std::cout << k << " = ";
                    const std::vector<std::string>& l = std::get<std::vector<std::string>>(v);
                    for (const auto& s : l) {
                        std::cout << s << ",";
                    }
                    std::cout << std::endl;
                }
            }, v);
        }
    }

    return container;
}

bool MetadataParser::is_aborted() const
{
    return m_parser_state.aborted;
}
