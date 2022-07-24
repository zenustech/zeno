/**
 * @file FftFactory.cpp
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

#include "FftFactory.h"
#include "OouraFft.h"

namespace Aquila
{
    /**
     * Returns "the best possible" FFT object.
     *
     * And now for some clarification about what is "the best possible":
     * This method will perhaps take into consideration some optimisation
     * hints, as optimizing for size or for speed. These hints will affect
     * the choice of FFT implementation - hidden from the caller who gets
     * only a pointer to the base abstract Fft class.
     *
     * As of now, the fastest implementation in Aquila is using Ooura's
     * mathematical packages, so this one is always returned.
     *
     * @param length FFT length (number of samples)
     * @return the FFT object (wrapped in a shared_ptr)
     */
    std::shared_ptr<Fft> FftFactory::getFft(std::size_t length)
    {
        return std::shared_ptr<Fft>(new OouraFft(length));
    }
}
