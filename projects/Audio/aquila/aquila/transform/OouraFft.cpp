/**
 * @file OouraFft.cpp
 *
 * A wrapper for the FFT algorithm found in Ooura mathematical packages.
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

#include "OouraFft.h"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace Aquila
{
    /**
     * Initializes the transform for a given input length.
     *
     * Prepares the work area for Ooura's algorithm.
     *
     * @param length input signal size (usually a power of 2)
     */
    OouraFft::OouraFft(std::size_t length):
        Fft(length),
        // according to the description: "length of ip >= 2+sqrt(n)"
        ip(new int[static_cast<std::size_t>(2 + std::sqrt(static_cast<double>(N)))]),
        w(new double[N / 2])
    {
        ip[0] = 0;
    }

    /**
     * Destroys the transform object and cleans up working area.
     */
    OouraFft::~OouraFft()
    {
        delete [] w;
        delete [] ip;
    }

    /**
     * Applies the transformation to the signal.
     *
     * @param x input signal
     * @return calculated spectrum
     */
    SpectrumType OouraFft::fft(const SampleType x[])
    {
        static_assert(
            sizeof(ComplexType[2]) == sizeof(double[4]),
            "complex<double> has the same memory layout as two consecutive doubles"
        );
        // create a temporary storage array and copy input to even elements
        // of the array (real values), leaving imaginary components at 0
        double* a = new double[2 * N];
        for (std::size_t i = 0; i < N; ++i)
        {
            a[2 * i] = x[i];
            a[2 * i + 1] = 0.0;
        }

        // let's call the C function from Ooura's package
        cdft(2*N, -1, a, ip, w);

        // convert the array back to complex values and return as vector
        ComplexType* tmpPtr = reinterpret_cast<ComplexType*>(a);
        SpectrumType spectrum(tmpPtr, tmpPtr + N);
        delete [] a;

        return spectrum;
    }

    /**
     * Applies the inverse transform to the spectrum.
     *
     * @param spectrum input spectrum
     * @param x output signal
     */
    void OouraFft::ifft(SpectrumType spectrum, double x[])
    {
        static_assert(
            sizeof(ComplexType[2]) == sizeof(double[4]),
                "complex<double> has the same memory layout as two consecutive doubles"
        );
        // interpret the vector as consecutive pairs of doubles (re,im)
        // and copy to input/output array
        double* tmpPtr = reinterpret_cast<double*>(&spectrum[0]);
        double* a = new double[2 * N];
        std::copy(tmpPtr, tmpPtr + 2 * N, a);

        // Ooura's function
        cdft(2*N, 1, a, ip, w);

        // copy the data to the double array and scale it
        for (std::size_t i = 0; i < N; ++i)
        {
            x[i] = a[2 * i] / static_cast<double>(N);
        }
        delete [] a;
    }
}
