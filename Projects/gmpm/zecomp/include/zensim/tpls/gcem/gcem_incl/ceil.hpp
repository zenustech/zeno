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

#ifndef _gcem_ceil_HPP
#define _gcem_ceil_HPP

namespace internal
{

template<typename T>
constexpr
int
ceil_resid(const T x, const T x_whole)
noexcept
{
    return( (x > T(0)) && (x > x_whole) );
}

template<typename T>
constexpr
T
ceil_int(const T x, const T x_whole)
noexcept
{
    return( x_whole + static_cast<T>(ceil_resid(x,x_whole)) );
}

template<typename T>
constexpr
T
ceil_check(const T x)
noexcept
{
    return( // NaN check
            is_nan(x) ? \
                GCLIM<T>::quiet_NaN() :
            // +/- infinite
            !is_finite(x) ? \
                x :
            // signed-zero cases
            GCLIM<T>::epsilon() > abs(x) ? \
                x :
            // else
                ceil_int(x, T(static_cast<llint_t>(x))) );
}

}

/**
 * Compile-time ceil function
 *
 * @param x a real-valued input.
 * @return computes the ceiling-value of the input.
 */

template<typename T>
constexpr
return_t<T>
ceil(const T x)
noexcept
{
    return internal::ceil_check( static_cast<return_t<T>>(x) );
}

#endif
