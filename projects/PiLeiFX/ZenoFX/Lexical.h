//
// Created by admin on 2022/5/7.
//
#include "Location.h"
#include <stdint.h>
#include <iostream>
#include <unordered_map>
#pragma once
//inline char opchars[] = "+-*/%=(),.;<>!&|^?:";
/*inline std::set<std::string> opstrs = {
    "(", ")", ",", ".", ";",
    "+", "-", "*", "/", "%", "=",
    "+=", "-=", "*=", "/=", "%=",
    "==", "!=", "<", "<=", ">", ">=",
    "&", "&!", "|", "^", "!", "?", ":",
    };
*/
namespace zfx {

    enum class TokenKind {Op, Seprator, KeywordKind, Eof};
    enum class Op{
        Plus,     //+
        Minus,    //-
        Multiply, // *
        Divide,   // /
        Modules,  // %
        Assign,   // =
        L,        //  <
        G,        // >
        LE,       // <=
        GE,       // >=
        MultiplyAssign,  // *=
        DivideAssign,  // /=
        ModulesAssign,  // %=
        PlusAssign, //+=
        MinusAssign, //-=
        BitNot,   //~
        BitAnd,   //&
        BitXor,   //^
        BitOr,    // |
        At,       //@
        Comma,    //,
        Dot,      //.
        Not,      // !
        And,      // &&
        Or,       // ||
        QuesstionMark  // ?
    };

    enum class Seprator {
        OpenBracket = 0,                //[
        CloseBracket,                   //]
        OpenParen,                      //(
        CloseParen,                     //)
        OpenBrace,                      //{
        CloseBrace,                     //}
        Colon,                          //:
        SemiColon,                      //;
    };

    enum class KeywordKind {
//for instance if while for ... to be added later as appropriate
        Pos,
        data,
        frame,
        rad,
        $//
    };

    std::string toString(TokenKind kind);
    std::string toString(Op op);

    struct Token {
        TokenKind kind;
        std::string text;
        Position pos;
        Token(TokenKind kind, const std::string& text, const Position& pos) : kind(kind), text(text), pos(pos) {}
        Token(TokenKind kind, char c, const Position& pos) : kind(kind), text(std::string(1, c)), pos(pos) {}
        std::string toString() {
            return std::string("Token") + ":" + this->pos.toString() + this->text;
        }
    };


    Class CharStream {
        public:
            std::string data;
            uint32_t pos = 0;
            uint32_t line = 1;
            uint32_t col = 0;
            CharStream(const std::string& data) : data(data){

            }

            char peak() {
                return this->data[this->pos];
            }

            char next() {
                char ch =  this->data[this->pos++];
                if (ch == '\n') {
                    this->line++;
                    this->col = 0;
                } else {
                    this->col++;
                }
              return ch;
            }

            bool eof () {
                return this->peak() == '\0';
            }

            Position getPosition() {
               //return the line and column numbers of the current charactor
               return Position(this->line, this->col);
            }
    };

/*
    class Scanner {
      private:
        std::vector<Token> tokens;
        CharStream& stream;
        Position lastPos;
        static std::unordered_map<std::string, KeywordKind> KeywordMap;
      public:
        Token next() {
            if (this->tokens.empty()) {
                auto t = this->getAToken();
                return t;
                //
            } else {
                auto t = thi->tokens.front();
                //
            }
        }

        Token peek() {
            if(this->tokens.empty()) {
                auto t = this->getAToken();
                this->tokens.push_back(t);
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
        }

        Position getNextPos() {

        }

        Position getLastPos() {

        }
      private:
        Token getAToken() {
            this->skipWhiteSpaces();
            auto pos = this->stream.getPosition();
            if (this->stream.eof() == '\n') {
                return Token(TokenKind::Eof, "EOF", pos);
            } else {
                auto ch = this->stream.peek();
                if (ch == '#') {
                    this->skipSingleComment();
                    return this->getAToken();
                } else if (this->isDigit(ch)) {
                    this->stream.next();
                    auto ch1 = this->stream.peek();
                    std::string literal = "";
                    if (ch == '0') {

                    }
                } else if () {

                }

            }
        }



        void skipWhiteSpaces() {
            while (this->isWhiteSpace()) {
                this->stream.next();
            }
        }

        void skipSingleLineComment() {
            this->stream.next();
            while (this->stream.peek() != '\n' && this->stream.eof()) {
                this->stream.next();
            }
        }

        bool isWhiteSpace(char ch) {
            return (ch == ' ' || ch == '\n' || ch == '\t');
        }

        inline bool isDigit(char ch) {
            return (ch >= '0' && ch <= '9');
        }

        inline bool is_symbolic_atom(char ch) {
            if () return false;
            if (isspace(ch) ||) {
                return true;
            }
            return false;
        }

        inline int swizzle_from_char(char ch) {
            if ('x' <= c && c <= 'z') {
                return c - 'z';
            } else if (c == 'w') {
                return 3;
            } else if('0' <= c && c <= '9') {
                return c - '0';
            } else if () {

            } else {
                return -1;
            }

        }
        Token parseIdentifer() {
            Token token;
            if (this->KeywordMap.find(token.text)) {

            }
            return token;
        }
    };
*/

    class Scanner {
      private:
        std::list<Token> tokens;
        std::string data;
        Position lastPos{0, 0, 0, 0};
      public:
        Scanner(const std::string &data) : data(data) {

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
        Token getAToken() {

        }
    };
}

