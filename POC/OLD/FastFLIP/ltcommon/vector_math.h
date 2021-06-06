#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include <vector>
#include <algorithm>
#include "blas_wrapper.h"

template<class T>
T sum(std::vector<T>& data) {
    T result = 0;
    for(unsigned int i = 0; i < data.size(); ++i)
        result += data[i];
    return result;
}

template<class T>
void copy(const std::vector<T> & src, std::vector<T>& dest) {
    std::copy(src.begin(), src.end(), dest.begin());
}

template<class T>
T dot(const std::vector<T>& a, const std::vector<T>& b) {
    return BLAS::dot((int)a.size(), &a[0], &b[0]);
}

template<class T>
void scale(T factor, std::vector<T>& data) {
    BLAS::scale((int)data.size(), factor, &data[0]);
}

template<class T> 
void add_scaled(T alpha, const std::vector<T>& x, std::vector<T>& y) { // y = y + alpha*x
    BLAS::add_scaled((int)x.size(), alpha, &x[0], &y[0]);
}

#endif
