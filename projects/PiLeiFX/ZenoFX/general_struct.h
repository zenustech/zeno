//
// Created by admin on 2022/5/19.
//
#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace zfx {
    class Module {
      public:
        std::string name;
        //module name actually file name
        Module(std::string name) : name(name) {}
        Module() : name("Module") {}
        virtual ~Module() {}
        virtual void EmitCode() = 0;
    };
}
