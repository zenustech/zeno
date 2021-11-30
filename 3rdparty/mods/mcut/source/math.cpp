/**
 * Copyright (c) 2020-2021 CutDigital Ltd.
 * All rights reserved.
 * 
 * NOTE: This file is licensed under GPL-3.0-or-later (default). 
 * A commercial license can be purchased from CutDigital Ltd. 
 *  
 * License details:
 * 
 * (A)  GNU General Public License ("GPL"); a copy of which you should have 
 *      recieved with this file.
 * 	    - see also: <http://www.gnu.org/licenses/>
 * (B)  Commercial license.
 *      - email: contact@cut-digital.com
 * 
 * The commercial license options is for users that wish to use MCUT in 
 * their products for comercial purposes but do not wish to release their 
 * software products under the GPL license. 
 * 
 * Author(s)     : Floyd M. Chitalu
 */

#include "mcut/internal/math.h"
#include <cstdlib>

namespace mcut {
namespace math {

    real_number_t square_root(const real_number_t& number)
    {
#if !defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)
        return std::sqrt(number);
#else
        arbitrary_precision_number_t out(number);
        mpfr_sqrt(out.get_mpfr_handle(), number.get_mpfr_handle(), arbitrary_precision_number_t::get_default_rounding_mode());
        return out;
#endif // #if !defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)
    }

    real_number_t absolute_value(const real_number_t& number)
    {
#if !defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)
        return std::fabs(number);
#else
        real_number_t out(number);
        mpfr_abs(out.get_mpfr_handle(), number.get_mpfr_handle(), arbitrary_precision_number_t::get_default_rounding_mode());
        return out;
#endif // #if defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)
    }

    sign_t sign(const real_number_t& number)
    {
#if !defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)
        int s = (real_number_t(0) < number) - (number < real_number_t(0));
        sign_t result = sign_t::ZERO;
        if (s > 0) {
            result = sign_t::POSITIVE;
        } else if (s < 0) {
            result = sign_t::NEGATIVE;
        }
        return result;
#else
        real_number_t out(number);
        int s = mpfr_sgn(number.get_mpfr_handle());
        sign_t result = sign_t::ZERO;
        if (s > 0) {
            result = sign_t::POSITIVE;
        } else if (s < 0) {
            result = sign_t::NEGATIVE;
        }
        return result;
#endif // #if defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)
    }

    std::ostream& operator<<(std::ostream& os, const vec3& v)
    {
        return os << static_cast<double>(v.x()) << ", " << static_cast<double>(v.y()) << ", " << static_cast<double>(v.z());
    }

    std::ostream& operator<<(std::ostream& os, const matrix_t& m)
    {
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                os << m(i, j) << ", ";
            }
            os << "\n";
        }
        return os;
    }

    bool operator==(const vec3& a, const vec3& b)
    {
        return (a.x() == b.x()) && (a.y() == b.y()) && (a.z() == b.z());
    }

    vec3 cross_product(const vec3& a, const vec3& b)
    {
        return vec3(
            a.y() * b.z() - a.z() * b.y(),
            a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x());
    }

} // namespace math
} // namespace mcut {
