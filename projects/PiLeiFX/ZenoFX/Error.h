//
// Created by admin on 2022/5/30.
//
#pragma once

#include <string>
namespace zfx {
    struct Position{
        uint32_t begin;
        uint32_t end;
        uint32_t line;
        uint32_t col;

        Position(){}
        Position(uint32_t begin, uint32_t end, uint32_t line, uint32_t col) : begin(begin),
        end(end), line(line), col(col) {

        }

        Position(const Position &rhs) {
            this->begin = rhs.begin;
            this->end = rhs.end;
            this->line = rhs.line;
            this->col = rhs.col;
        }

        std::string ToString() {
            return "ln" + std::to_string(this->line) + ", col" + std::to_string(this->col) +
            ", Pos:" + std::to_string(this->pos);
        }
    };
}
