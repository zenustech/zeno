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
 * compile-time binary logarithm function
 */

#ifndef _gcem_log2_HPP
#define _gcem_log2_HPP

namespace internal
{

template<typename T>
constexpr
T
log2_check(const T x)
noexcept
{
    return( is_nan(x) ? \
                GCLIM<T>::quiet_NaN() :
            // x < 0
            x < T(0) ? \
                GCLIM<T>::quiet_NaN() :
            // x ~= 0
            GCLIM<T>::epsilon() > x ? \
                - GCLIM<T>::infinity() :
            // indistinguishable from 1
            GCLIM<T>::epsilon() > abs(x - T(1)) ? \
                T(0) : 
            // 
            x == GCLIM<T>::infinity() ? \
                GCLIM<T>::infinity() :
            // else: log_2(x) = ln(x) / ln(2)
                T(log(x) / GCEM_LOG_2) );
}

}

/**
 * Compile-time binary logarithm function
 *
 * @param x a real-valued input.
 * @return \f$ \log_2(x) \f$ using \f[ \log_{2}(x) = \frac{\log_e(x)}{\log_e(2)} \f] 
 */

template<typename T>
constexpr
return_t<T>
log2(const T x)
noexcept
{
    return internal::log2_check( static_cast<return_t<T>>(x) );
}

#endif
