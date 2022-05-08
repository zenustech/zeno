//
// Created by admin on 2022/5/7.
//
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

    enum TokenKind {Op, Seprator, KeywordKind, Eof};
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
//for instance if while for ...
    };

    std::string toString(TokenKind kind);
    std::string toString(Op op);

    struct Token {
        TokenKind kind;
        std::string text;
        Token(TokenKind kind, const std::string& text) : kind(kind), text(text) {}
        Token(TokenKind kind, char c) : kind(kind), text(std::string(1, c)) {}

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


    };


    class Scanner {
      public:
        std::vector<Token> tokens;
        CharStream stream;
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

        }

        Token peek2() {
            //
        }


      private:
        Token getAToken() {

        }



        void skipWhiteSpaces() {
            while (this->isWhiteSpace()) {

            }

        }

        bool isWhiteSpace(char ch) {
            return (ch == ' ' || ch == '\n' || ch == '\t');
        }

        bool isDigit(char ch) {
            return (ch >= '0' && ch <= '9');
        }

        bool isLetter() {}
        

    };

}

