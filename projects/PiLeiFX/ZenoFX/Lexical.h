//
// Created by admin on 2022/5/7.
//

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

    };

    std::string toString(TokenKind kind);
    std::string toString(Op op);

    struct Token {
        TokenKind kind;
        std::string text;

    };


    Class CharStream {
        public:
            std::string data;
            CharStream(const std::string& data) : data(data){

            }


    };


    class Scanner {
      public:
        std::vector<Token> tokens;
        CharStream stream;
    };

}

