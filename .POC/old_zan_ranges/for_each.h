#pragma once


#include <zeno/zan/transform.h>
#include <functional>


ZENO_NAMESPACE_BEGIN
namespace zan {
inline namespace ns_for_each {


static constexpr auto for_each(auto f)
{
    return transform([f] (auto r) {
        for (auto it = r.begin(); it != r.end(); ++it) {
            f(*it);
        }
    });
}


static constexpr auto for_each_apply(auto f)
{
    return transform([f] (auto r) {
        for (auto it = r.begin(); it != r.end(); ++it) {
            std::apply(f, *it);
        }
    });
}


static constexpr auto evaluate =
transform([] (auto r) {
    for (auto it = r.begin(); it != r.end(); ++it);
});


}
}
ZENO_NAMESPACE_END
