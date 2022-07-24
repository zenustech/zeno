/**
 * @file Fft.h
 *
 * An interface for FFT calculation classes.
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

#ifndef FFT_H
#define FFT_H

#include "../global.h"
#include <cstddef>

namespace Aquila
{
    /**
     * An interface for FFT calculation classes.
     *
     * The abstract interface for FFT algorithm allows switching between
     * implementations at runtime, or selecting a most effective implementation
     * for the current platform.
     *
     * The FFT objects are not intended to be copied.
     *
     * Some of FFT implementations prepare a "plan" or create a coefficient
     * cache only once, and then run the transform on multiple sets of data.
     * Aquila expresses this approach by defining a meaningful constructor
     * for the base FFT interface. A derived class should calculate the
     * plan once - in the constructor (based on FFT length). Later calls
     * to fft() / ifft() should reuse the already created plan/cache.
     */
    class AQUILA_EXPORT Fft
    {
    public:
        /**
         * Initializes the transform for a given input length.
         *
         * @param length input signal size (usually a power of 2)
         */
        Fft(std::size_t length): N(length)
        {
        }

        /**
         * Destroys the transform object - does nothing.
         *
         * As the derived classes may perform some deinitialization in
         * their destructors, it must be declared as virtual.
         */
        virtual ~Fft()
        {
        }

        /**
         * Applies the forward FFT transform to the signal.
         *
         * @param x input signal
         * @return calculated spectrum
         */
        virtual SpectrumType fft(const SampleType x[]) = 0;

        /**
         * Applies the inverse FFT transform to the spectrum.
         *
         * @param spectrum input spectrum
         * @param x output signal
         */
        virtual void ifft(SpectrumType spectrum, double x[]) = 0;

    protected:
        /**
         * Signal and spectrum length.
         */
        std::size_t N;

    private:
        Fft( const Fft& );
        const Fft& operator=( const Fft& );
    };
}

#endif // FFT_H
