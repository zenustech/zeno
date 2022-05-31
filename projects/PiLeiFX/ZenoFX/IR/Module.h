//
// Created by admin on 2022/5/19.
//

#pragma once

namespace zfx {
    class Module {
      public:
        explicit Module(const std::string name) : name(name) {

        }
      //
      virtual std::string print();
      private:
      std::string name;//Module name
      //need to include global variables, constants, function
      std::list<std::string> functions;
    };
}