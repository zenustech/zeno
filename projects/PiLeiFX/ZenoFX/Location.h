//
// Created by admin on 2022/5/8.
//

#pragma once

#include <iostream>
#include <string>
namespace zfx {
    class Position {
      public:
        uint32_t line;
        uint32_t col;
        Position(uint32_t line, uint32_t col) : line(line), col(col){

        }

        Position(const Position& rhs) {
            this->line = rhs.line;
            this->col = rhs.col;
        }
         bool operator==(const Position& rhs) const {
            return this->col == rhs.col && this->line == rhs.line;
         }

         bool operator!=(const Position& rhs) const {
                return !(*this == rhs);
         }

         bool operator<(const Position& rhs) const {
                if (this->line == rhs.line) {
                    return this->col < rhs.col;
                } else {
                    return line < rhs.line;
                }
         }

        bool operator>(const Position& rhs) const {
                 if (this->line == rhs.line) {
                     return this->col > rhs.col;
                 } else {
                     return this->line > rhs.line;
                 }
        }

        bool operator<=(const Position& rhs) const {
            return *this == rhs || *this < rhs;
        }

        bool operator>=(const Position& rhs) const {
            return *this == rhs || *this > rhs;
        }

        std::string toString() {
            return ("(ln" + std::to_string(this->line) +
                    ", col : " + std::to_string(this->col) + ")");
        }
    };

    class Location {
      public:
        Position begin, end;
        Location() : begin(0, 0), end(0, 0){

        }

        Location(const Position& begin, uint32_t length) : begin(begin), end(begin.line, begin.col + length) {

        }

        Location(const Location& begin, const Location& end) : begin(begin), end(end){

        }

        bool operator==(const Location& rhs) {

        }

        bool operator!=(const Location& rhs) {

        }

        bool operator<(const Location& rhs) {

        }

        bool operator>(const Location& rhs) {

        }
        std::string toString() {

        }
    };
}
