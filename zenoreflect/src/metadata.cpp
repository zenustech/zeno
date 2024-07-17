#include <limits>
#include <sstream>
#include <iostream>
#include <vector>
#include <cctype>
#include <cstdio>
#include <format>
#include <stdexcept>
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
    } else if (str == "trait") {
        return MetadataType::Trait;
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
    } else if (type == MetadataType::Trait) {
        return "trait";
    }
    return "none";
}

MetadataContainer parse_metadata_dsl(const std::string &in_dsl)
{
    std::string metadata;
    MetadataContainer container;
    if (in_dsl.starts_with("#struct")) {
        metadata = in_dsl.substr(9);
        container.type = MetadataType::Struct;
    } else if (in_dsl.starts_with("#enum")) {
        metadata = in_dsl.substr(7);
        container.type = MetadataType::Enum;
    } else if (in_dsl.starts_with("#function")) {
        metadata = in_dsl.substr(11);
        container.type = MetadataType::Function;
    } else if (in_dsl.starts_with("#enum_value")) {
        metadata = in_dsl.substr(13);
        container.type = MetadataType::EnumValue;
    } else if (in_dsl.starts_with("#field")) {
        metadata = in_dsl.substr(8);
        container.type = MetadataType::StructField;
    } else if (in_dsl.starts_with("#param")) {
        metadata = in_dsl.substr(8);
        container.type = MetadataType::FunctionParameter;
    } else if (in_dsl.starts_with("#trait")) {
        metadata = in_dsl.substr(8);
        container.type = MetadataType::Trait;
    } else if (in_dsl.starts_with("#method")) {
        metadata = in_dsl.substr(9);
        container.type = MetadataType::Function;
    } else if (in_dsl.starts_with("#property")) {
        metadata = in_dsl.substr(11);
        container.type = MetadataType::StructField;
    }

    Parser parser(metadata);
    auto ast = parser.parse();

    for (const auto& kv : ast) {
        container.properties.insert_or_assign(kv.first, kv.second);
    }

    return container;
}

std::string token_type_to_string(TokenType type)
{
    switch (type) {
        case TokenType::KEY: return "key";
        case TokenType::STRING: return "string";
        case TokenType::NUMBER: return "number";
        case TokenType::LIST_START: return "list start";
        case TokenType::LIST_END: return "list end";
        case TokenType::EQUAL: return "equal";
        case TokenType::COMMA: return "comma";
        case TokenType::END: return "end";
        default: return "unknown";
    }
}

Tokenizer::Tokenizer(const std::string& input)
    : m_origin_string(input)
    , ss(input)
    , current_char(' ')
{
    next_char();
}

Token Tokenizer::next_token()
{
    consume_whitespace();

    if (current_char == EOF) {
        return { TokenType::END, "" };
    }

    if (current_char == '=') {
        next_char();
        return { TokenType::EQUAL, "=" };
    } else if (current_char == ',') {
        next_char();
        return { TokenType::COMMA, "," };
    } else if (current_char == '(') {
        next_char();
        return { TokenType::LIST_START, "(" };
    } else if (current_char == ')') {
        next_char();
        return { TokenType::LIST_END, ")" };
    }else if (std::isdigit(static_cast<unsigned char>(current_char))) {
        return number();
    }else if (current_char == '"') {
        return string();
    }

    return key();
}

const std::string &Tokenizer::origin_string() const
{
    return m_origin_string;
}

void Tokenizer::next_char()
{
    current_char = ss.get();
}

void Tokenizer::consume_whitespace()
{
    while (std::isspace(static_cast<unsigned char>(current_char))) {
        next_char();
    }
}

Token Tokenizer::number()
{
    std::string result;
    while (std::isdigit(static_cast<unsigned char>(current_char))) {
        result.push_back(current_char);
        next_char();
    }
    return { TokenType::NUMBER, result };
}

Token Tokenizer::key()
{
    std::string result;
    while (current_char != EOF && current_char != '=' && current_char != ',' && current_char != '(' && current_char != ')' && !std::isspace(static_cast<unsigned char>(current_char))) {
        result.push_back(current_char);
        next_char();
    }
    return { TokenType::KEY, result };
}

Token Tokenizer::string()
{
    std::string result;

    // Skip initial quote
    if (current_char == '"') {
        next_char();
    }

    while (current_char != '"' && current_char != EOF) {
        if (current_char == '\\') {
            next_char();
            if (current_char == '"') result.push_back('"');
            else result.push_back('\\');
        } else {
            result.push_back(current_char);
        }
        next_char();
    }

    next_char(); // Skip closing quote

    return { TokenType::STRING, result };
}

Parser::Parser(const std::string &input)
    : tokenizer(input)
{
    next_token();
}

std::map<std::string, std::string> Parser::parse()
{
    std::map<std::string, std::string> ast;

    while (current_token.type != TokenType::END)
    {
        std::string key = expect(TokenType::KEY);
        expect(TokenType::EQUAL);
        std::string value = parse_value();
        ast[key] = value;
        if (current_token.type == TokenType::COMMA) {
            next_token();
        }
    }
    
    return ast;
}

void Parser::next_token()
{
    current_token = tokenizer.next_token();
}

std::string Parser::expect(TokenType token_expected)
{
    if (current_token.type != token_expected) {
        throw std::runtime_error(std::format("Unexcepted token when expecting {}, found {}. Origin metadata: \n{}", token_type_to_string(token_expected), token_type_to_string(current_token.type), tokenizer.origin_string()));
    }
    std::string value = current_token.value;
    next_token();
    return value;
}

std::string Parser::parse_value()
{
    if (current_token.type == TokenType::STRING) {
        std::string result = expect(TokenType::STRING);
        // Handle consecutive strings
        while (current_token.type == TokenType::STRING) {
            result += expect(TokenType::STRING);
        }
        return result;
    } else if (current_token.type == TokenType::NUMBER) {
        return expect(TokenType::NUMBER);
    } else if (current_token.type == TokenType::LIST_START) {
        return parse_list();
    }
    
    throw std::runtime_error(std::format("Unexpected value type {}: '{}'. Origin metadata: \n{}", token_type_to_string(current_token.type), current_token.value, tokenizer.origin_string()));
}

std::string Parser::parse_list()
{
    std::string list;
    next_token(); // Skip '('
    while (current_token.type != TokenType::LIST_END) {
        list += parse_value();
        if (current_token.type == TokenType::COMMA) {
            list += ", ";
            next_token();
        }
    }
    next_token(); // Skip ')'
    return list;
}
