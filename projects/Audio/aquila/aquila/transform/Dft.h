/**
 * @file Dft.h
 *
 * A reference implementation of the Discrete Fourier Transform.
 *
 * Note - this is a direct application of the DFT equation and as such it
 * surely isn't optimal. The implementation here serves only as a reference
 * model to compare with.
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

#ifndef DFT_H
#define DFT_H

#include "Fft.h"

namespace Aquila
{
    /**
     * A straightforward implementation of the Discrete Fourier Transform.
     */
    class AQUILA_EXPORT Dft : public Fft
    {
    public:
        /**
         * Initializes the transform for a given input length.
         *
         * @param length input signal size
         */
        Dft(std::size_t length):
            Fft(length)
        {
        }

        /**
         * Destroys the transform object.
         */
        ~Dft()
        {
        }

        virtual SpectrumType fft(const SampleType x[]);
        virtual void ifft(SpectrumType spectrum, double x[]);

    private:
        /**
         * Complex unit (0.0 + 1.0j).
         */
        static const ComplexType j;
    };
}

#endif // DFT_H
