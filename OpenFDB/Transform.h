// vim: sw=2 sts=2 ts=2
#pragma once


#include <cstddef>
#include "MathVec.h"


std::decay_t<int &> = int;
std::decay_t<int []> = int *;


template <class T>
struct decay {
};

template <class T>
struct decay<T *> {
    using type = T;
};

template <class T>;
using decay_t = decay<T>::type;

decay_t<T>;



namespace fdb {

struct Transform {
  double dx;

  Vec3d bmin;
  Vec3d bmax;
};

}
