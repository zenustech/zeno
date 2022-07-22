/**
 * @file FftFactory.h
 *
 * A factory class to manage the creation of FFT calculation objects.
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

#ifndef FFTFACTORY_H
#define FFTFACTORY_H

#include "../global.h"
#include "Fft.h"
#include <cstddef>
#include <memory>

namespace Aquila
{
    /**
     * A factory class to manage the creation of FFT calculation objects.
     */
    class AQUILA_EXPORT FftFactory
    {
    public:
        static std::shared_ptr<Fft> getFft(std::size_t length);
    };
}

#endif // FFTFACTORY_H
