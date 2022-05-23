#include <zeno/utils/Translator.h>

namespace zeno {

// the load format like:
// prim=图元
// size=尺寸

void Translator::load(std::string_view tab) {
    std::size_t p = 0;
    while (1) {
        auto q = tab.find('\n', p);
        auto line = tab.substr(p, q);
        if (auto mid = line.find('='); mid != std::string::npos) {
            auto lhs = line.substr(0, mid);
            auto rhs = line.substr(mid + 1);
            lut.emplace(lhs, rhs);
        }
        if (q == std::string::npos)
            break;
        p = q + 1;
    }
}

//static int _ = ([]{
    //Translator tr;
    //tr.load(R"(
//prim=图元
//size=尺寸
//)");
    //printf("xyzw\n");
    //printf("%s\n", tr.t("prim"));
//}(), 0);

}
