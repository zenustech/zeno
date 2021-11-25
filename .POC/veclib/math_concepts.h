#pragma once


#include <cmath>


namespace zeno::ztd {
inline namespace concepts {

template <class T1>
concept has_std_min = requires(T1 t1) {
    std::min(t1, t1);
};

template <class T1>
concept has_std_max = requires(T1 t1) {
    std::max(t1, t1);
};

}
}
