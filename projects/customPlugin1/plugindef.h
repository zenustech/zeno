#pragma once
#include <zeno/core/data.h>
#include <reflect/core.hpp>
#include <zeno_types/reflect/reflection.generated.hpp>


namespace zeno {

    struct ZPRIMITIVE() CustomDefPrimitive1
    { 
        vec3f m1; 
        std::vector<std::string> m2;
        int m3;
        float m4 = 2;
    };
}

