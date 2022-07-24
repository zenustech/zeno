/**
 * @file WhiteNoiseGenerator.cpp
 *
 * White noise generator.
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

#include "WhiteNoiseGenerator.h"
#include "../../functions.h"

namespace Aquila
{
    /**
     * Creates the generator object.
     *
     * @param sampleFrequency sample frequency of the signal
     */
    WhiteNoiseGenerator::WhiteNoiseGenerator(FrequencyType sampleFrequency):
        Generator(sampleFrequency)
    {
    }

    /**
     * Fills the buffer with generated white noise samples.
     *
     * @param samplesCount how many samples to generate
     */
    void WhiteNoiseGenerator::generate(std::size_t samplesCount)
    {
        m_data.resize(samplesCount);
        for (std::size_t i = 0; i < samplesCount; ++i)
        {
            m_data[i] = m_amplitude * (randomDouble() - 0.5);
        }
    }
}
