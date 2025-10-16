#include "apiutil.h"

namespace zpyapi {
    zeno::vecvar pylist2vec(py::list lst) {
        zeno::vecvar vec;
        for (auto item : lst) {
            if (py::isinstance<py::int_>(item)) {
                int val = item.cast<int>();
                vec.push_back(val);
            }
            else if (py::isinstance<py::float_>(item)) {
                float val = item.cast<double>();
                vec.push_back(val);
            }
            else if (py::isinstance<py::str>(item)) {
                std::string val = item.cast<std::string>();
                vec.push_back(val);
            }
        }
        return vec;
    }

    py::list vec2pylist(zeno::vec3f vec) {
        py::list lst;
        lst.append(vec[0]);
        lst.append(vec[1]);
        lst.append(vec[2]);
        return lst;
    }
}