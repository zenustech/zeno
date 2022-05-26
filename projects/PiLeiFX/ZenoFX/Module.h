//
// Created by admin on 2022/5/19.
//

#pragma once

namespace zfx {
    class Module {
      public:
        std::string name;
        explicit Module(const std::string name) : name(name) {

        }
    };
}