//
// Created by admin on 2022/5/7.
//
#include "Error.h"
#include <stdint.h>
#include <iostream>
#include <unordered_map>
#pragma once

namespace zfx {

enum class TokenKind {
    Op,
    Seprator,
    KeywordKind,
    Decl,
    Eof
};
enum class Op {
    Plus,           //+
    Minus,          //-
    Multiply,       // *
    Divide,         // /
    Modules,        // %
    Assign,         // =
    L,              //  <
    G,              // >
    LE,             // <=
    GE,             // >=
    MultiplyAssign, // *=
    DivideAssign,   // /=
    ModulesAssign,  // %=
    PlusAssign,     //+=
    MinusAssign,    //-=
    BitNot,         //~
    BitAnd,         //&
    BitXor,         //^
    BitOr,          // |
    At,             //@
    Comma,          //,
    Dot,            //.
    Not,            // !
    And,            // &&
    Or,             // ||
    QuesstionMark   // ?
};

enum class Seprator {
    OpenBracket = 0, //[
    CloseBracket,    //]
    OpenParen,       //(
    CloseParen,      //)
    OpenBrace,       //{
    CloseBrace,      //}
    Colon,           //:
    SemiColon,       //;
};

enum class KeywordKind {
    //for instance if while for ... to be added later as appropriate
    Pos,
    data,
    frame,
    rad,
    //
};

enum class Decl {
    Para,
    Symbol
};

std::string toString(TokenKind kind);
std::string toString(Op op);

struct Token {
    TokenKind kind;
    std::string text;
    Position pos;
    Token(TokenKind kind, const std::string &text, const Position &pos) : kind(kind), text(text), pos(pos) {
    }
    Token(TokenKind kind, char c, const Position &pos) : kind(kind), text(std::string(1, c)), pos(pos) {
    }
    std::string toString() {
        return std::string("Token") + ":" + this->pos.toString() + this->text;
    }
};

class CharStream {
  public:
    std::string data;//rep source code
    uint32_t pos = 0;
    uint32_t line = 1;
    uint32_t col = 0;
    CharStream(const std::string &data) : data(data) {
    }

    char peak() {
        return this->data[this->pos];
    }

    char next() {
        char ch = this->data[this->pos++];
        if (ch == '\n') {
            this->line++;
            this->col = 0;
        } else {
            this->col++;
        }
        return ch;
    }

    bool eof() {
        return this->peak() == '\0';
    }

    Position getPosition() {
        //return the line and column numbers of the current charactor
        return Position(this->pos+1, this->pos + 1,this->line, this->col);
    }
};

    class Scanner {
      private:
        std::list<Token> tokens;
        CharStream stream;
        Position lastPos{0, 0, 0, 0};
        static std::unordered_map<std::string, KeywordKind> KeywordMap;

      public:
        Scanner(CharStream &stream) : stream(stream) {

        }
        Token next() {
            if (this->tokens.empty()) {
                auto t = this->getAToken();
                //set pos
                this->lastPos = t.pos;
                return t;
            } else {
                auto t = this->tokens.front();
                this->lastPos = t.pos;
                this->tokens.pop_front();
                return t;
            }
        }

        Token peek() {
            if (this->tokens.empty()) {
                auto t = this->getAToken();
                this->tokens->push_back(t);
                return t;
            } else {
                auto t = this->tokens.front();
                return t;
            }
        }

        Token peek2() {
            while (this->tokens.size() < 2) {
                auto t = this->getAToken();
                this->tokens.push_back(t);
            }

            if (this->tokens.size() < 2) {
                return Token{};//EofToken
            }

            auto it = this->tokens.begin();
            std::advance(it, 1);
            return *it1;
        }

      private:
        //getAToken主要还是由Token peek Token next这几个函数驱动
        Token getAToken() {
            this->skipWhiteSpace();
            auto Pos = this->stream.getPosition();
            if (this->stream.eof()) {
                return Token(TokenKind::Eof, "Eof", pos);
            } else {
                auto ch = this->stream.peak();
                if (this->) {

                } else if (ch == '"') {
                    return this->parseStringLiteral();
                } else if (ch == '$') {
                        this->stream.next();
                        return Token(TokenKind::Decl, '$', pos);
                } else if (ch == '@') {
                    this->stream.next();
                    return Token(TokenKind::Decl, '@', pos);
                } else if (this->Digit(ch)) {

                }
            } else {
                std::cout << "unexpected character : " + std::string(1,ch) << std::endl;
                this->stream.next();
                return this->getAToken();
            }
        }

        void skipSingleComment() {
            this->stream.next();
            while (this->stream.peek() != '\n' && !this->stream.eof()) {
                this->stream.next();
            }
        }

        void skipWhiteSpace() {
            while (this->isWhiteSpace(this->stream.peak())) {
                this->stream.next();
            }
        };

        bool isWhiteSpace(char ch) {
            return (ch == ' ' || ch == '\n', || ch == '\t');
        }

        bool isLetter(char ch) const {
            return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' || ch <= 'z');
        }

        bool isLetterDigitOurUnderScore(char ch) {
            return (ch >= 'A' && ch <= 'Z' ||
                    ch >= 'a' && ch <= 'z' ||
                    ch >= '0' && ch <= '9' ||
                    ch == '-');
        }

        bool isDigit(char ch) {
            return (ch >= '0' && ch <= '9');
        }
    };
}

