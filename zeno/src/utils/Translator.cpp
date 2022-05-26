#include <zeno/utils/Translator.h>
//#include <zeno/utils/zeno_p.h>

namespace zeno {

void Translator::load(std::string_view tab) {
    while (1) {
        auto q = tab.find('\n');
        auto line = tab.substr(0, q);
        //ZENO_P(line);
        if (auto mid = line.find('='); mid != std::string::npos) {
            auto lhs = line.substr(0, mid);
            auto rhs = line.substr(mid + 1);
            while (!rhs.empty() && rhs.back() == '\n')
                rhs.remove_suffix(1);
            lut.emplace(lhs, rhs);
        }
        if (q != std::string::npos)
            tab.remove_prefix(q + 1);
        else
            break;
    }
    for (auto const &[k, v]: lut) {
        m_untrans.lut.emplace(v, k);
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
