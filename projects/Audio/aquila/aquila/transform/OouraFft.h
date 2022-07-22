/**
 * @file OouraFft.h
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

#ifndef OOURAFFT_H
#define OOURAFFT_H

#include "Fft.h"

extern "C" {
    void cdft(int, int, double *, int *, double *);
    void rdft(int, int, double *, int *, double *);
}

namespace Aquila
{
    /**
     * A wrapper for the FFT algorithm found in Ooura mathematical packages.
     */
    class AQUILA_EXPORT OouraFft : public Fft
    {
    public:
        OouraFft(std::size_t length);
        ~OouraFft();

        virtual SpectrumType fft(const SampleType x[]);
        virtual void ifft(SpectrumType spectrum, double x[]);

    private:
        /**
         * Work area for bit reversal.
         */
        int* ip;

        /**
         * Cos/sin table.
         */
        double* w;
    };
}

#endif // OOURAFFT_H
