/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
  ##
  ##   This file is part of the GCE-Math C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*
 * compile-time square-root function
 */

#ifndef _gcem_sqrt_HPP
#define _gcem_sqrt_HPP

namespace internal
{

template<typename T>
constexpr
T
sqrt_recur(const T x, const T xn, const int count)
noexcept
{
    return( abs(xn - x/xn) / (T(1) + xn) < GCLIM<T>::epsilon() ? \
            // if
                xn :
            count < GCEM_SQRT_MAX_ITER ? \
            // else
                sqrt_recur(x,T(0.5)*(xn + x/xn), count+1) :
                xn );
}

template<typename T>
constexpr
T
sqrt_check(const T x, const T m_val)
noexcept
{
    return( is_nan(x) ? \
                GCLIM<T>::quiet_NaN() :
            //
            x < T(0) ? \
                GCLIM<T>::quiet_NaN() :
            //
            is_posinf(x) ? \
                x :
            // indistinguishable from zero or one
            GCLIM<T>::epsilon() > abs(x) ? \
                T(0) :
            GCLIM<T>::epsilon() > abs(T(1)-x) ? \
                x :
            // else
            x > T(4) ? \
                sqrt_check(x/T(4),T(2)*m_val) :
                m_val*sqrt_recur(x,x/T(2),0) );
}

}


/**
 * Compile-time square-root function
 *
 * @param x a real-valued input.
 * @return Computes \f$ \sqrt{x} \f$ using a Newton-Raphson approach.
 */

template<typename T>
constexpr
return_t<T>
sqrt(const T x)
noexcept
{
    return internal::sqrt_check( static_cast<return_t<T>>(x), return_t<T>(1) );
}

#endif
