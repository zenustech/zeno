#ifndef LEXER_H
#define LEXER_H

#include <iostream>
#include <string>

enum TokenType {TOKEN_EOF, TOKEN_ERROR, TOKEN_IDENTIFIER, TOKEN_NUMBER, TOKEN_STRING,
    TOKEN_LEFT_PAREN, TOKEN_RIGHT_PAREN,
    TOKEN_LEFT_BRACKET, TOKEN_RIGHT_BRACKET};

struct Token
{
    TokenType type;
    // typically only one of the following has a meaningful value
    double number_value;
    std::string string_value;
    
    Token() : type(TOKEN_ERROR), number_value(1e+30), string_value("ERROR") {}
    ~Token() { clear(); }
    
    Token(const Token &source);
    Token &operator=(const Token &source);
    
    void set(TokenType type_);
    void set(TokenType type_, const std::string &value);
    void set(TokenType type_, double value);
    void clear(); // free up string storage after finished with a TOKEN_ERROR or TOKEN_IDENTIFIER
};

std::ostream& operator<<(std::ostream& out, const Token& t);

struct Lexer
{
    std::istream& input;
    
    Lexer(std::istream& input_) : input(input_) {}
    
    void read(Token &tok);
};

#endif
