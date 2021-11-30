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

#include "mcut/internal/number.h"

namespace mcut {
namespace math {

#if defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)
    std::ostream& operator<<(std::ostream& os, arbitrary_precision_number_t const& m)
    {
        return os << static_cast<long double>(m);
    }

    arbitrary_precision_number_t operator*(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        arbitrary_precision_number_t c(0.0);
        mpfr_mul(c.get_mpfr_handle(), a.get_mpfr_handle(), b.get_mpfr_handle(), arbitrary_precision_number_t::get_default_rounding_mode());
        return c;
    }

    arbitrary_precision_number_t operator+(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        arbitrary_precision_number_t c(0.0);
        mpfr_add(c.get_mpfr_handle(), a.get_mpfr_handle(), b.get_mpfr_handle(), arbitrary_precision_number_t::get_default_rounding_mode());
        return c;
    }

    arbitrary_precision_number_t operator-(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        arbitrary_precision_number_t c(0.0);
        mpfr_sub(c.get_mpfr_handle(), a.get_mpfr_handle(), b.get_mpfr_handle(), arbitrary_precision_number_t::get_default_rounding_mode());
        return c;
    }

    arbitrary_precision_number_t operator/(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        arbitrary_precision_number_t c(0.0);
        mpfr_div(c.get_mpfr_handle(), a.get_mpfr_handle(), b.get_mpfr_handle(), arbitrary_precision_number_t::get_default_rounding_mode());
        return c;
    }

    arbitrary_precision_number_t operator/(const long double b, const arbitrary_precision_number_t& a)
    {
        arbitrary_precision_number_t x(0.0);
        // NOTE: mpfr_ld_div not availale
        mpfr_d_div(x.get_mpfr_handle(), b, a.get_mpfr_handle(), arbitrary_precision_number_t::get_default_rounding_mode());
        return x;
    }

    //////////////////////////////////////////////////////////////////////////
    //Relational operators

    // WARNING:
    //
    // Please note that following checks for double-NaN are guaranteed to work only in IEEE math mode:
    //
    // is_nan(b) =  (b != b)
    // is_nan(b) = !(b == b)  (we use in code below)
    //
    // Be cautions if you use compiler options which break strict IEEE compliance (e.g. -ffast-math in GCC).
    // Use std::is_nan instead (C++11).

    bool operator>(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        return (mpfr_greater_p(a.get_mpfr_handle(), b.get_mpfr_handle()) != 0);
    }

    bool operator>(const arbitrary_precision_number_t& a, const long double b)
    {
        return !arbitrary_precision_number_t::is_nan(a) && (b == b) && (mpfr_cmp_ld(a.get_mpfr_handle(), b) > 0);
    }

    bool operator>=(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        return (mpfr_greaterequal_p(a.get_mpfr_handle(), b.get_mpfr_handle()) != 0);
    }

    bool operator>=(const arbitrary_precision_number_t& a, const long double b)
    {
        return !arbitrary_precision_number_t::is_nan(a) && (b == b) && (mpfr_cmp_ld(a.get_mpfr_handle(), b) >= 0);
    }

    bool operator<(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        return (mpfr_less_p(a.get_mpfr_handle(), b.get_mpfr_handle()) != 0);
    }

    bool operator<(const arbitrary_precision_number_t& a, const long double b)
    {
        return !arbitrary_precision_number_t::is_nan(a) && (b == b) && (mpfr_cmp_ld(a.get_mpfr_handle(), b) < 0);
    }

    bool operator<=(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        return (mpfr_lessequal_p(a.get_mpfr_handle(), b.get_mpfr_handle()) != 0);
    }

    bool operator<=(const arbitrary_precision_number_t& a, const long double b)
    {
        return !arbitrary_precision_number_t::is_nan(a) && (b == b) && (mpfr_cmp_ld(a.get_mpfr_handle(), b) <= 0);
    }

    bool operator==(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        return (mpfr_equal_p(a.get_mpfr_handle(), b.get_mpfr_handle()) != 0);
    }

    bool operator==(const arbitrary_precision_number_t& a, const long double b)
    {
        return !arbitrary_precision_number_t::is_nan(a) && (b == b) && (mpfr_cmp_ld(a.get_mpfr_handle(), b) == 0);
    }

    bool operator!=(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b)
    {
        return !(a == b);
    }

    bool operator!=(const arbitrary_precision_number_t& a, const long double b)
    {
        return !(a == b);
    }
#endif // #if defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)

} // namespace math {
} // namespace mcut {