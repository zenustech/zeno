
///////////////////////////////////////////////////////////////////////////////////
/*
Copyright (c) 2006-2008, Alexander Gessler

All rights reserved.

Redistribution and use of this software in source and binary forms, 
with or without modification, are permitted provided that the following 
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the ASSIMP team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the ASSIMP Development Team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
///////////////////////////////////////////////////////////////////////////////////

#ifndef UM_HALF_INL_INCLUDED
#define UM_HALF_INL_INCLUDED

#ifdef _MSC_VER
	#include <intrin.h>
	#pragma intrinsic(_BitScanReverse)
#endif

// ------------------------------------------------------------------------------------------------
inline HalfFloat::HalfFloat(float other)
{
	IEEESingle f;
	f.Float = other;

	IEEE.Sign = f.IEEE.Sign;

	if ( !f.IEEE.Exp) 
	{ 
		IEEE.Frac = 0;
		IEEE.Exp = 0;
	}
	else if (f.IEEE.Exp==0xff) 
	{ 
		// NaN or INF
		IEEE.Frac = (f.IEEE.Frac!=0) ? 1 : 0;
		IEEE.Exp = 31;
	}
	else 
	{ 
		// regular number
		int new_exp = f.IEEE.Exp-127;

		if (new_exp<-24) 
		{ // this maps to 0
			IEEE.Frac = 0;
			IEEE.Exp = 0;
		}

		else if (new_exp<-14) 
		{
			// this maps to a denorm
			IEEE.Exp = 0;
			unsigned int exp_val = (unsigned int) (-14 - new_exp);  // 2^-exp_val
			switch (exp_val) 
			{
			case 0: 
				IEEE.Frac = 0; 
				break;
			case 1: IEEE.Frac = 512 + (f.IEEE.Frac>>14); break;
			case 2: IEEE.Frac = 256 + (f.IEEE.Frac>>15); break;
			case 3: IEEE.Frac = 128 + (f.IEEE.Frac>>16); break;
			case 4: IEEE.Frac = 64 + (f.IEEE.Frac>>17); break;
			case 5: IEEE.Frac = 32 + (f.IEEE.Frac>>18); break;
			case 6: IEEE.Frac = 16 + (f.IEEE.Frac>>19); break;
			case 7: IEEE.Frac = 8 + (f.IEEE.Frac>>20); break;
			case 8: IEEE.Frac = 4 + (f.IEEE.Frac>>21); break;
			case 9: IEEE.Frac = 2 + (f.IEEE.Frac>>22); break;
			case 10: IEEE.Frac = 1; break;
			}
		}
		else if (new_exp>15) 
		{ // map this value to infinity
			IEEE.Frac = 0;
			IEEE.Exp = 31;
		}
		else 
		{
			IEEE.Exp = new_exp+15;
			IEEE.Frac = (f.IEEE.Frac >> 13);
		}
	}
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat::HalfFloat(const double p_Reference)
{
	const IEEEDouble & l_Reference = reinterpret_cast<const IEEEDouble &>(p_Reference);

	// Copy the sign bit.
	this->IEEE.Sign = l_Reference.IEEE.Sign;

	// Check for special values: Is the exponent zero?
	if(0 == l_Reference.IEEE.Exp) 
	{
		// A zero exponent indicates either a zero or a subnormal number. A subnormal float can not
		//	be represented as a half, so either one will be saved as a zero.
		this->IEEE.Exp = 0;
		this->IEEE.Frac = 0;
	}
	// Is the exponent all one?
	else if(IEEEDouble_MaxExpontent == l_Reference.IEEE.Exp)
	{
		this->IEEE.Exp = MAX_EXPONENT_VALUE;
		// A zero fraction indicates an Infinite value.
		if(0 == l_Reference.IEEE.Frac)
			this->IEEE.Frac = 0;
		// A nonzero fraction indicates NaN. Such a fraction contains further information, e.g. to
		//	distinguish a QNaN from a SNaN. However, we can not just shift-copy the fraction:
		//	if the first five bits were zero we would save an infinite value, so we abandon the
		//	fraction information and set it to a nonzero value.
		else
			this->IEEE.Frac = 1;
	}
	// A usual value?
	else {
		// First, we have to adjust the exponent. It is stored as an unsigned int, to reconstruct
		//	its original value we have to subtract its bias (half of its range).
		const int64_t l_AdjustedExponent = l_Reference.IEEE.Exp - IEEEDouble_ExponentBias;

		// Very small values will be rounded to zero.
		if(-24 > l_AdjustedExponent)
		{
			this->IEEE.Frac = 0;
			this->IEEE.Exp = 0;
		}
		// Some small values can be stored as subnormal values.
		else if(-14 > l_AdjustedExponent) 
		{
			// The exponent of subnormal values is always be zero.
			this->IEEE.Exp = 0;
			// The exponent will now be stored in the fraction.
			const int16_t l_NewExponent = int16_t(-14 - l_AdjustedExponent);  // 2 ^ -l_NewExponent
			this->IEEE.Frac = (1024 >> l_NewExponent) + int16_t(l_Reference.IEEE.Frac >> (42 + l_NewExponent));
		}
		// Very large numbers will be rounded to infinity.
		else if(15 < l_AdjustedExponent) 
		{
			// Exponent all one, fraction zero.
			this->IEEE.Exp = MAX_EXPONENT_VALUE;
			this->IEEE.Frac = 0;
		}
		// All remaining numbers can be converted directly.
		else 
		{
			// We reconstructed the exponent by subtracting the bias. To store it as an unsigned
			//	int, we need to add the bias again.
			this->IEEE.Exp = l_AdjustedExponent + BIAS;
			// When storing the fraction, we abandon its least significant bits by right-shifting.
			//	The fraction of a double is 42 bits wider than that of a half, so we shift 42 bits.
			this->IEEE.Frac = (l_Reference.IEEE.Frac >> 42);
		};
	}; // else usual number
} 
// ------------------------------------------------------------------------------------------------
inline HalfFloat::HalfFloat(uint16_t _m,uint16_t _e,uint16_t _s)
{
    IEEE.Frac = _m;
	IEEE.Exp = _e;
	IEEE.Sign = _s;
}
// ------------------------------------------------------------------------------------------------
HalfFloat::operator float() const
{
	IEEESingle sng;
	sng.IEEE.Sign = IEEE.Sign;

	if (!IEEE.Exp) 
	{  
		if (!IEEE.Frac)
		{
			sng.IEEE.Frac=0;
			sng.IEEE.Exp=0;
		}
		else
		{
			const float half_denorm = (1.0f/16384.0f); 
			float mantissa = ((float)(IEEE.Frac)) / 1024.0f;
			float sgn = (IEEE.Sign)? -1.0f :1.0f;
			sng.Float = sgn*mantissa*half_denorm;
		}
	}
	else if (31 == IEEE.Exp)
	{
		sng.IEEE.Exp = 0xff;
		sng.IEEE.Frac = (IEEE.Frac!=0) ? 1 : 0;
	}
	else 
	{
		sng.IEEE.Exp = IEEE.Exp+112;
		sng.IEEE.Frac = (IEEE.Frac << 13);
	}
	return sng.Float;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat::operator double(void) const
{
	IEEEDouble l_Result;

	// Copy the sign bit.
	l_Result.IEEE.Sign = this->IEEE.Sign;

	// In a zero, both the exponent and the fraction are zero.
	if((0 == this->IEEE.Exp) && (0 == this->IEEE.Frac)) 
	{
		l_Result.IEEE.Exp = 0;
		l_Result.IEEE.Frac = 0;
	}
	// If the exponent is zero and the fraction is nonzero, the number is subnormal.
	else if((0 == this->IEEE.Exp) && (0 != this->IEEE.Frac)) 
	{
		// sign * 2^-14 * fraction
		l_Result.Double = (this->IEEE.Sign ? -1.0 : +1.0) / 16384.0 * (double(this->IEEE.Frac) / 1024.0);
	}
	// Is the exponent all one?
	else if(MAX_EXPONENT_VALUE == this->IEEE.Exp) 
	{
		l_Result.IEEE.Exp = IEEEDouble_MaxExpontent;
		// A zero fraction indicates an infinite value.
		if(0 == this->IEEE.Frac)
			l_Result.IEEE.Frac = 0;
		// A nonzero fraction indicates a NaN. We can re-use the fraction information: a double
		//	fraction is 42 bits wider than a half fraction, so we can just left-shift it. Any
		//	information on QNaNs or SNaNs will be preserved.
		else
			l_Result.IEEE.Frac = uint64_t(this->IEEE.Frac) << 42;
	}
	// A usual value?
	else 
	{
		// The exponent is stored as an unsigned int. To reconstruct its original value, we have to
		//	subtract its bias. To re-store it in a wider bit field, we must add the bias of the new
		//	bit field.
		l_Result.IEEE.Exp = uint64_t(this->IEEE.Exp) - BIAS + IEEEDouble_ExponentBias;
		// A double fraction is 42 bits wider than a half fraction, so we can just left-shift it.
		l_Result.IEEE.Frac = uint64_t(this->IEEE.Frac) << 42;
	}
	return l_Result.Double;
} 
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::IsNaN() const
{
	return IEEE.Frac != 0 && IEEE.Exp == MAX_EXPONENT_VALUE;
}
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::IsInfinity() const
{
	return IEEE.Frac == 0 && IEEE.Exp == MAX_EXPONENT_VALUE;
}
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::IsDenorm() const
{
	return IEEE.Exp == 0;
}
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::GetSign() const
{
	return IEEE.Sign == 0;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator= (HalfFloat other)
{
	bits = other.GetBits();
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator= (float other)
{
	*this = (HalfFloat)other;
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator= (const double p_Reference)
{
	return (*this) = HalfFloat(p_Reference);
}
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::operator== (HalfFloat other) const
{
	// +0 and -0 are considered to be equal
	if (!(bits << 1u) && !(other.bits << 1u))return true;

	return bits == other.bits && !this->IsNaN();
}
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::operator!= (HalfFloat other) const
{
	// +0 and -0 are considered to be equal
	if (!(bits << 1u) && !(other.bits << 1u))return false;

	return bits != other.bits || this->IsNaN();
}
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::operator<  (HalfFloat other) const
{
	// NaN comparisons are always false
	if (this->IsNaN() || other.IsNaN())
		return false;
	
	// this works since the segment oder is s,e,m.
	return (int16_t)this->bits < (int16_t)other.GetBits();
}
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::operator>  (HalfFloat other) const
{
	// NaN comparisons are always false
	if (this->IsNaN() || other.IsNaN())
		return false;
	
	// this works since the segment oder is s,e,m.
	return (int16_t)this->bits > (int16_t)other.GetBits();
}
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::operator<= (HalfFloat other) const
{
	return !(*this > other);
}
// ------------------------------------------------------------------------------------------------
inline bool HalfFloat::operator>= (HalfFloat other) const
{
	return !(*this < other);
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator += (HalfFloat other)
{
	*this = (*this) + other;
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator -= (HalfFloat other)
{
	*this = (*this) - other;
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator *= (HalfFloat other)
{
	*this = (float)(*this) * (float)other;
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator /= (HalfFloat other)
{
	*this = (float)(*this) / (float)other;
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator += (float other)
{
	*this = (*this) + (HalfFloat)other;
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator -= (float other)
{
	*this = (*this) - (HalfFloat)other;
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator *= (float other)
{
	*this = (float)(*this) * other;
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator /= (float other)
{
	*this = (float)(*this) / other;
	return *this;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator++()
{
	// setting the exponent to bias means using 0 as exponent - thus we
	// can set the mantissa to any value we like, we'll always get 1.0
	return this->operator+=(HalfFloat(0,BIAS,0));
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat HalfFloat::operator++(int)
{
	HalfFloat f = *this;
	this->operator+=(HalfFloat(0,BIAS,0));
	return f;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat& HalfFloat::operator--()
{
	return this->operator-=(HalfFloat(0,BIAS,0));
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat HalfFloat::operator--(int)
{
	HalfFloat f = *this;
	this->operator-=(HalfFloat(0,BIAS,0));
	return f;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat HalfFloat::operator-() const
{
	return HalfFloat(IEEE.Frac,IEEE.Exp,~IEEE.Sign);
}
// ------------------------------------------------------------------------------------------------
inline uint16_t HalfFloat::GetBits() const
{
	return bits;
}
// ------------------------------------------------------------------------------------------------
inline uint16_t& HalfFloat::GetBits()
{
	return bits;
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat operator+ (HalfFloat one, HalfFloat two) 
{
#if (!defined HALFFLOAT_NO_CUSTOM_IMPLEMENTATIONS)

	if (one.IEEE.Exp == HalfFloat::MAX_EXPONENT_VALUE)	
	{
		// if one of the components is NaN the result becomes NaN, too.
		if (0 != one.IEEE.Frac || two.IsNaN())
			return HalfFloat(1,HalfFloat::MAX_EXPONENT_VALUE,0);

		// otherwise this must be infinity
		return HalfFloat(0,HalfFloat::MAX_EXPONENT_VALUE,one.IEEE.Sign | two.IEEE.Sign);
	}
	else if (two.IEEE.Exp == HalfFloat::MAX_EXPONENT_VALUE)	
	{
		if (one.IsNaN() || 0 != two.IEEE.Frac)
			return HalfFloat(1,HalfFloat::MAX_EXPONENT_VALUE,0);

		return HalfFloat(0,HalfFloat::MAX_EXPONENT_VALUE,one.IEEE.Sign | two.IEEE.Sign);
	}

	HalfFloat out;
	long m1,m2,temp;

	// compute the difference between the two exponents. shifts with negative
	// numbers are undefined, thus we need two code paths
	register int expDiff = one.IEEE.Exp - two.IEEE.Exp;
	
	if (0 == expDiff)
	{
		// the exponents are equal, thus we must just add the hidden bit
		temp = two.IEEE.Exp;

		if (0 == one.IEEE.Exp)m1 = one.IEEE.Frac;
		else m1 = (int)one.IEEE.Frac | ( 1 << HalfFloat::BITS_MANTISSA ); 

		if (0 == two.IEEE.Exp)m2 = two.IEEE.Frac;
		else m2 = (int)two.IEEE.Frac | ( 1 << HalfFloat::BITS_MANTISSA );
	}
	else 
	{
		if (expDiff < 0)
		{
			expDiff = -expDiff;
			std::swap(one,two);
		}

		m1 = (int)one.IEEE.Frac | ( 1 << HalfFloat::BITS_MANTISSA ); 

		if (0 == two.IEEE.Exp)m2 = two.IEEE.Frac;
		else m2 = (int)two.IEEE.Frac | ( 1 << HalfFloat::BITS_MANTISSA );

		if (expDiff < ((sizeof(long)<<3)-(HalfFloat::BITS_MANTISSA+1)))
		{
			m1 <<= expDiff;
			temp = two.IEEE.Exp;
		}
		else 
		{
			if (0 != two.IEEE.Exp)
			{
				// arithmetic underflow
				if (expDiff > HalfFloat::BITS_MANTISSA)return HalfFloat(0,0,0); 
				else
				{
					m2 >>= expDiff;
				}
			}
			temp = one.IEEE.Exp;
		}
	}
	
	// convert from sign-bit to two's complement representation
	if (one.IEEE.Sign)m1 = -m1;
	if (two.IEEE.Sign)m2 = -m2;
	m1 += m2;
	if (m1 < 0)
	{
		out.IEEE.Sign = 1;
		m1 = -m1; 
	}
	else out.IEEE.Sign = 0;

	// and renormalize the result to fit in a half
	if (0 == m1)return HalfFloat(0,0,0);

#ifdef _MSC_VER
	_BitScanReverse((unsigned long*)&m2,m1);
#else
	m2 = __builtin_clz(m1);
#endif
	expDiff = m2 - HalfFloat::BITS_MANTISSA;
	temp += expDiff;
	if (expDiff >= HalfFloat::MAX_EXPONENT_VALUE)
	{
		// arithmetic overflow. return INF and keep the sign
		return HalfFloat(0,HalfFloat::MAX_EXPONENT_VALUE,out.IEEE.Sign);
	}
	else if (temp <= 0)
	{
		// this maps to a denorm
		m1 <<= (-expDiff-1);
		temp = 0;
	}
	else
	{
		// rebuild the normalized representation, take care of the hidden bit
		if (expDiff < 0)m1 <<= (-expDiff);
		else m1 >>= expDiff; // m1 >= 0
	}
	out.IEEE.Frac = m1;
	out.IEEE.Exp = temp;
	return out;

#else
	return HalfFloat((float)one + (float)two);
#endif
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat operator- (HalfFloat one, HalfFloat two) 
{
	return HalfFloat(one + (-two));
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat operator* (HalfFloat one, HalfFloat two) 
{
	return HalfFloat((float)one * (float)two);
}
// ------------------------------------------------------------------------------------------------
inline HalfFloat operator/ (HalfFloat one, HalfFloat two)
{
	return HalfFloat((float)one / (float)two);
}
// ------------------------------------------------------------------------------------------------
inline float operator+ (HalfFloat one, float two)
{
	return (float)one + two;
}
// ------------------------------------------------------------------------------------------------
inline float operator- (HalfFloat one, float two)
{
	return (float)one - two;
}
// ------------------------------------------------------------------------------------------------
inline float operator* (HalfFloat one, float two)
{
	return (float)one * two;
}
// ------------------------------------------------------------------------------------------------
inline float operator/ (HalfFloat one, float two)
{
	return (float)one / two;
}
// ------------------------------------------------------------------------------------------------
inline float operator+ (float one, HalfFloat two)
{
	return two + one;
}
// ------------------------------------------------------------------------------------------------
inline float operator- (float one, HalfFloat two)
{
	return two - one;
}
// ------------------------------------------------------------------------------------------------
inline float operator* (float one, HalfFloat two)
{
	return two * one;
}
// ------------------------------------------------------------------------------------------------
inline float operator/ (float one, HalfFloat two)
{
	return two / one;
}

#endif //!! UM_HALF_INL_INCLUDED
