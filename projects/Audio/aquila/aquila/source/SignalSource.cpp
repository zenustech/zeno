/**
 * @file SignalSource.cpp
 *
 * A base signal source class.
 *
 * This file is part of the Aquila DSP library.
 * Aquila is free software, licensed under the MIT/X11 License. A copy of
 * the license is provided with the library in the LICENSE file.
 *
 * @package Aquila
 * @version 3.0.0-dev
 * @author Zbigniew Siciarz
 * @date 2007-2014
 * @license http://www.opensource.org/licenses/mit-license.php MIT
 * @since 3.0.0
 */

#include "SignalSource.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>


namespace Aquila
{
    /**
     * Add a constant value to each sample.
     *
     * @param x value to add
     * @return updated source
     */
    SignalSource& SignalSource::operator+=(SampleType x)
    {
        std::transform(
            std::begin(m_data),
            std::end(m_data),
            std::begin(m_data),
            [x] (SampleType y) { return x + y; }
        );
        return *this;
    }

    /**
     * Per-sample addition of other signal source.
     *
     * @param rhs source on the right-hand side of the operator
     * @return sum of two sources
     */
    SignalSource& SignalSource::operator+=(const SignalSource& rhs)
    {
        std::transform(
            std::begin(m_data),
            std::end(m_data),
            std::begin(rhs.m_data),
            std::begin(m_data),
            [] (SampleType x, SampleType y) { return x + y; }
        );
        return *this;
    }

    /**
     * Multiply each sample by a constant value.
     *
     * @param x multiplier
     * @return updated source
     */
    SignalSource& SignalSource::operator*=(SampleType x)
    {
        std::transform(
            std::begin(m_data),
            std::end(m_data),
            std::begin(m_data),
            [x] (SampleType y) { return x * y; }
        );
        return *this;
    }

    /**
     * Per-sample multiplication with other signal source.
     *
     * @param rhs source on the right-hand side of the operator
     * @return product of two sources
     */
    SignalSource& SignalSource::operator*=(const SignalSource& rhs)
    {
        std::transform(
            std::begin(m_data),
            std::end(m_data),
            std::begin(rhs.m_data),
            std::begin(m_data),
            [] (SampleType x, SampleType y) { return x * y; }
        );
        return *this;
    }

    SignalSource operator+(const SignalSource& lhs, SampleType x)
    {
        SignalSource result(lhs);
        return result += x;
    }

    SignalSource operator+(SignalSource&& lhs, SampleType x)
    {
        lhs += x;
        return std::move(lhs);
    }

    SignalSource operator+(SampleType x, const SignalSource& rhs)
    {
        SignalSource result(rhs);
        return result += x;
    }

    SignalSource operator+(SampleType x, SignalSource&& rhs)
    {
        rhs += x;
        return std::move(rhs);
    }

    SignalSource operator+(const SignalSource& lhs, const SignalSource& rhs)
    {
        SignalSource result(lhs);
        return result += rhs;
    }

    SignalSource operator+(SignalSource&& lhs, const SignalSource& rhs)
    {
        lhs += rhs;
        return std::move(lhs);
    }

    SignalSource operator+(const SignalSource& lhs, SignalSource&& rhs)
    {
        rhs += lhs;
        return std::move(rhs);
    }

    SignalSource operator*(const SignalSource& lhs, SampleType x)
    {
        SignalSource result(lhs);
        return result *= x;
    }

    SignalSource operator*(SignalSource&& lhs, SampleType x)
    {
        lhs *= x;
        return std::move(lhs);
    }

    SignalSource operator*(SampleType x, const SignalSource& rhs)
    {
        SignalSource result(rhs);
        return result *= x;
    }

    SignalSource operator*(SampleType x, SignalSource&& rhs)
    {
        rhs *= x;
        return std::move(rhs);
    }

    SignalSource operator*(const SignalSource& lhs, const SignalSource& rhs)
    {
        SignalSource result(lhs);
        return result *= rhs;
    }

    SignalSource operator*(SignalSource&& lhs, const SignalSource& rhs)
    {
        lhs *= rhs;
        return std::move(lhs);
    }

    SignalSource operator*(const SignalSource& lhs, SignalSource&& rhs)
    {
        rhs *= lhs;
        return std::move(rhs);
    }

    /**
     * Calculates mean value of the signal.
     *
     * @param source signal source
     * @return signal mean
     */
    double mean(const SignalSource& source)
    {
        double sum = std::accumulate(std::begin(source), std::end(source), 0.0);
        return sum / source.getSamplesCount();
    }

    /**
     * Calculates energy of the signal.
     *
     * @param source signal source
     * @return signal energy
     */
    double energy(const SignalSource& source)
    {
        return std::accumulate(
            std::begin(source),
            std::end(source),
            0.0,
            [] (double acc, SampleType value) {
                return acc + value * value;
            }
        );
    }

    /**
     * Calculates power of the signal.
     *
     * @param source signal source
     * @return signal power
     */
    double power(const SignalSource& source)
    {
        return energy(source) / source.getSamplesCount();
    }

    /**
     * Calculates Euclidean (L2) norm of the signal.
     *
     * @param source signal source
     * @return norm
     */
    double norm(const SignalSource& source)
    {
        return std::sqrt(energy(source));
    }

    /**
     * Calculates root mean square level of the signal.
     *
     * @param source signal source
     * @return RMS level
     */
    double rms(const SignalSource& source)
    {
        return std::sqrt(power(source));
    }
}
