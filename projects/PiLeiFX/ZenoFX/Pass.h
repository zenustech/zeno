//
// Created by admin on 2022/5/25.
//

#pragma once

namespace  zfx {
    class Pass {
      public:
        Pass() = default;
       virtual ~Pass() = default;
       virtual void run() = 0;
    };

    class BasicBlockPass : public Pass{
      public:
        void run() override;
    };

    class PassManger {
      public:

    };
}
