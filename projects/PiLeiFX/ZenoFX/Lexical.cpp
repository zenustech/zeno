//
// Created by admin on 2022/5/8.
//
#include "Lexical.h"
#include <unordered_map>
#include <string>

namespace zfx {
    std::unordered_map<TokenKind, std::string> tokenToString {
        {TokenKind::Op, "Op"},
        {TokenKind::Seprator, "Seprator"},
            {TokenKind::KeywordKind, "KeywordKind"},
            {TokenKind::Eof, "Eof"}
    }
    std::unordered_map<Op, std::string> OpToString {
        {Op::Plus, "Plus"} ,
        {Op::Minus,  "Minus"},
        {Op::Multiply,"Multiply"},
        {Op::Divide, "Divide"},
        {Op::Modules, "Modules"},
        {Op::L,  "L"},
        {Op::G,  "G"},      // >
        {Op::LE, "LE"},      // <=
        {Op::GE, "GE"},      // >=
        {Op::MultiplyAssign, "MultiplyAssign"}, // *=
        {Op::DivideAssign, "DivideAssign"},  // /=
        {Op::ModulesAssign, "ModulesAssign"}, // %=
        {Op::PlusAssign,"PlusAssign"}, //+=
        {Op::MinusAssign, "MinusAssign"}, //-=
        {Op::BitNot, "BitNot"},   //~
        {Op::BitAnd, "BitAnd"},  //&{    BitXor,   //^
        {Op::BitOr, "BitOr"},   // |
        {Op::At, "At"},      //@
        {Op::Comma, "Comma"},   //,
        {Op::Dot, "Dot"},     //.
        {Op::Not, "Not"},      // !
        {Op::And, "And"},      // &&
        {Op::Or, "Or"},      // ||
        {Op::QuesstionMark, "?"}  // ?
    }
    std::unordered_map<std::string, KeywordKind> Scanner::KeywordMap = {
        {"Pos",KeywordKind::Pos},
        {"data", KeywordKind::data},
        {"frame", KeywordKind::frame}
    };
    std::string toString(TokenKind kind) {
        auto it = tokenToString.find(kind);
        if (it != tokenToString.end()) {
            return it->second;
        }
        return "UnKnown";
    }

    std::string toString(Op op) {
        auto it = OpToString.find(op);
        if (it != OpToString.end()) {
            return it->second;
        }
        return "UnKnown";
    }
}