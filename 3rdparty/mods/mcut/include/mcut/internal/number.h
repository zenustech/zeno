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

#ifndef NUMBER_H_
#define NUMBER_H_

#if defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)
#include <mpfr.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace mcut {
namespace math {

    typedef double fixed_precision_number_t;

#if defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)

    class arbitrary_precision_number_t {
    public:
        static mp_rnd_t get_default_rounding_mode()
        {
            return (mp_rnd_t)(mpfr_get_default_rounding_mode());
        }

        static mp_prec_t get_default_precision()
        {
            return (mpfr_get_default_prec)(); // 53 bits (from the spec)
        }

        static void set_default_precision(mp_prec_t prec)
        {
            mpfr_set_default_prec(prec);
        }

        static void set_default_rounding_mode(mp_rnd_t rnd_mode)
        {
            mpfr_set_default_rounding_mode(rnd_mode);
        }

        arbitrary_precision_number_t()
        {
            mpfr_init2(get_mpfr_handle(), arbitrary_precision_number_t::get_default_precision());
            mpfr_set_ld(get_mpfr_handle(), 0.0, arbitrary_precision_number_t::get_default_rounding_mode());
        }

        arbitrary_precision_number_t(const long double& value)
        {
            mpfr_init2(get_mpfr_handle(), arbitrary_precision_number_t::get_default_precision());
            mpfr_set_ld(get_mpfr_handle(), value, arbitrary_precision_number_t::get_default_rounding_mode());
        }

        arbitrary_precision_number_t(const char* value)
        {
            mpfr_init2(get_mpfr_handle(), arbitrary_precision_number_t::get_default_precision());
            int ret = mpfr_set_str(get_mpfr_handle(), value, 10, arbitrary_precision_number_t::get_default_rounding_mode());
            if (ret != 0) {
                std::fprintf(stderr, "mpfr_set_str failed\n");
                std::abort();
            }
        }

        // Construct arbitrary_precision_number_t from mpfr_t structure.
        // shared = true allows to avoid deep copy, so that arbitrary_precision_number_t and 'u' share the same data & pointers.
        arbitrary_precision_number_t(const arbitrary_precision_number_t& u, bool shared = false)
        {
            if (shared) {
                std::memcpy(this->get_mpfr_handle(), u.get_mpfr_handle(), sizeof(mpfr_t));
            } else {
                mpfr_init2(this->get_mpfr_handle(), arbitrary_precision_number_t::get_default_precision());
                mpfr_set(this->get_mpfr_handle(), u.get_mpfr_handle(), arbitrary_precision_number_t::get_default_rounding_mode());
            }
        }

        ~arbitrary_precision_number_t()
        {
            clear();
        }

        void clear()
        {
            if ((nullptr != (get_mpfr_handle())->_mpfr_d)) {
                mpfr_clear(get_mpfr_handle());
            }
        }

        arbitrary_precision_number_t(arbitrary_precision_number_t&& other)
        {
            // make sure "other" holds null-pointer (in uninitialized state)
            ((get_mpfr_handle())->_mpfr_d = 0);
            mpfr_swap(get_mpfr_handle(), other.get_mpfr_handle());
        }

        arbitrary_precision_number_t& operator=(arbitrary_precision_number_t&& other)
        {
            if (this != &other) {
                mpfr_swap(get_mpfr_handle(), other.get_mpfr_handle()); // destructor for "other" will be called just afterwards
            }
            return *this;
        }

        const mpfr_t& get_mpfr_handle() const
        {
            return m_mpfr_val;
        }

        mpfr_t& get_mpfr_handle()
        {
            return m_mpfr_val;
        }

        operator long double() const { return static_cast<long double>(to_double()); }

        operator double() const { return static_cast<double>(to_double()); }

        operator float() const { return static_cast<float>(to_double()); }

        arbitrary_precision_number_t& operator=(const arbitrary_precision_number_t& v)
        {
            if (this != &v) {

                mp_prec_t tp = mpfr_get_prec(get_mpfr_handle());
                mp_prec_t vp = mpfr_get_prec(v.get_mpfr_handle());

                if (tp != vp) {
                    clear();
                    mpfr_init2(get_mpfr_handle(), vp);
                }

                mpfr_set(get_mpfr_handle(), v.get_mpfr_handle(), get_default_rounding_mode());
            }

            return *this;
        }

        arbitrary_precision_number_t& operator=(const long double v)
        {
            mpfr_set_ld(get_mpfr_handle(), v, get_default_rounding_mode());
            return *this;
        }

        arbitrary_precision_number_t& operator+=(const arbitrary_precision_number_t& v)
        {
            mpfr_add(get_mpfr_handle(), get_mpfr_handle(), v.get_mpfr_handle(), get_default_rounding_mode());
            return *this;
        }

        arbitrary_precision_number_t& operator+=(const long double u)
        {
            *this += arbitrary_precision_number_t(u);
            return *this;
        }

        arbitrary_precision_number_t& operator++()
        {
            return *this += 1;
        }

        const arbitrary_precision_number_t operator++(int)
        {
            arbitrary_precision_number_t x(*this);
            *this += 1;
            return x;
        }

        arbitrary_precision_number_t& operator--()
        {
            return *this -= 1;
        }

        const arbitrary_precision_number_t operator--(int)
        {
            arbitrary_precision_number_t x(*this);
            *this -= 1;
            return x;
        }

        arbitrary_precision_number_t& operator-=(const arbitrary_precision_number_t& v)
        {
            mpfr_sub(get_mpfr_handle(), get_mpfr_handle(), v.get_mpfr_handle(), get_default_rounding_mode());
            return *this;
        }

        arbitrary_precision_number_t& operator-=(const long double v)
        {
            *this -= arbitrary_precision_number_t(v);
            return *this;
        }

        arbitrary_precision_number_t& operator-=(const int v)
        {
            mpfr_sub_si(get_mpfr_handle(), get_mpfr_handle(), v, get_default_rounding_mode());
            return *this;
        }

        const arbitrary_precision_number_t operator-() const
        {
            arbitrary_precision_number_t u(*this); // copy
            mpfr_neg(u.get_mpfr_handle(), u.get_mpfr_handle(), get_default_rounding_mode());
            return u;
        }

        arbitrary_precision_number_t& operator*=(const arbitrary_precision_number_t& v)
        {
            mpfr_mul(get_mpfr_handle(), get_mpfr_handle(), v.get_mpfr_handle(), get_default_rounding_mode());
            return *this;
        }

        arbitrary_precision_number_t& operator*=(const long double v)
        {
            *this *= arbitrary_precision_number_t(v);
            return *this;
        }

        arbitrary_precision_number_t& operator/=(const arbitrary_precision_number_t& v)
        {
            mpfr_div(get_mpfr_handle(), get_mpfr_handle(), v.get_mpfr_handle(), get_default_rounding_mode());
            return *this;
        }

        arbitrary_precision_number_t& operator/=(const long double v)
        {
            *this /= arbitrary_precision_number_t(v);
            return *this;
        }

        long double to_double() const
        {
            return mpfr_get_ld(get_mpfr_handle(), get_default_rounding_mode());
        }

        std::string to_string() const
        {
            // NOTE: number of decimal digits in the significand is dependent on the current precision and rounding mode
            mpfr_exp_t decimalLocation = 0;
            char* s = mpfr_get_str(NULL, &decimalLocation, 10, 0, get_mpfr_handle(), get_default_rounding_mode());
            std::string out = std::string(s);
            if (out[0] == '-') // first char is minus sign
            {
                // The generated string is a fraction, with an implicit radix point immediately to the left of the
                // first digit. For example, the number −3.1416 would be returned as "−31416" in the string and
                // 1 written at "&decimalLocation".
                decimalLocation += 1; // account for minus sign.
            }
            out.insert(decimalLocation, ".");
            mpfr_free_str(s);
            return out;
        }

        static bool is_nan(const arbitrary_precision_number_t& op)
        {
            return (mpfr_nan_p(op.get_mpfr_handle()) != 0);
        }

        static bool is_zero(const arbitrary_precision_number_t& op)
        {
            return (mpfr_zero_p(op.get_mpfr_handle()) != 0);
        }

    private:
        mpfr_t m_mpfr_val;
    };

    extern std::ostream& operator<<(std::ostream& os, arbitrary_precision_number_t const& m);

    extern arbitrary_precision_number_t operator*(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern arbitrary_precision_number_t operator+(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern arbitrary_precision_number_t operator-(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern arbitrary_precision_number_t operator/(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern arbitrary_precision_number_t operator/(const long double b, const arbitrary_precision_number_t& a);

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

    extern bool operator>(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern bool operator>(const arbitrary_precision_number_t& a, const long double b);

    extern bool operator>=(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern bool operator>=(const arbitrary_precision_number_t& a, const long double b);

    extern bool operator<(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern bool operator<(const arbitrary_precision_number_t& a, const long double b);

    extern bool operator<=(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern bool operator<=(const arbitrary_precision_number_t& a, const long double b);

    extern bool operator==(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern bool operator==(const arbitrary_precision_number_t& a, const long double b);

    extern bool operator!=(const arbitrary_precision_number_t& a, const arbitrary_precision_number_t& b);

    extern bool operator!=(const arbitrary_precision_number_t& a, const long double b);

    using real_number_t = arbitrary_precision_number_t;
#else
    using real_number_t = fixed_precision_number_t;
#endif // #if defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)

} // namespace mcut {
} // namespace math {

#endif // NUMBER_H_
