#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <tuple>
#include <vector>
#include <array>
#include <cassert>
#include <omp.h>
//#include "SPGrid.h"

using std::cout;
using std::endl;
#define show(x) (cout << #x "=" << (x) << endl)

//using namespace spgrid;


template <class T>
struct range {
    T m_begin;
    T m_end;

    range
        ( T const &begin_
        , T const &end_
        )
    : m_begin(begin_)
    , m_end(end_)
    {}

    struct iterator {
        T m_now;
        T m_end;

        iterator
            ( T const &now_
            , T const &end_
            )
        : m_now(now_)
        , m_end(end_)
        {}

        iterator &operator++() {
            m_now++;
            return *this;
        }

        operator bool() {
            return m_now != m_end;
        }

        T const &operator*() const {
            return m_now;
        }
    };

    auto begin() {
        return iterator(m_begin, m_end);
    }
};


int main(void)
{
    auto r = range(2, 4);
    for (auto it = r.begin(); it; ++it) {
        show(*it);
    }
}
