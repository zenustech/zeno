#include <zeno/utils/rapidobject_parse.h>
#include <zeno/types/CurveObject.h>
#include <zeno/utils/curveparser.h>

namespace zeno {

    zany parseObject(Value const& x)
    {
        bool bSucceed = false;
        CurveData dat = parseCurve(x, bSucceed);
        if (bSucceed) {
            auto curve = std::make_shared<zeno::CurveObject>();
            curve->keys.insert(std::make_pair("x", dat));
            return curve;
        }
        return zany();
    }

}