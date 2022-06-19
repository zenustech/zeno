//
// Created by admin on 2022/6/10.
//

#pragma once

#include "./IR.h"
#include <optional>

namespace zfx {
    //所有的操作语句都继承自Stmt
    class  UnaryOpStmt : public Stmt {
      public:
        //一元操作符的数据类型
        //操作数
        //
    };

    class BinaryOpStmt : public Stmt {
      public:
        //数据类型
        //左右操作数

    };

    class TernaryOpStmt : public Stmt {
      public:
        //三元操作数类型
        //三个stmt

    };
}