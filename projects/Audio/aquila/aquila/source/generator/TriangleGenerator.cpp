/**
 * @file TriangleGenerator.cpp
 *
 * Triangle (and sawtooth) wave generator.
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

#include "TriangleGenerator.h"
#include <cmath>

namespace Aquila
{
    /**
     * Creates the generator object.
     *
     * @param sampleFrequency sample frequency of the signal
     */
    TriangleGenerator::TriangleGenerator(FrequencyType sampleFrequency):
        Generator(sampleFrequency), m_width(1.0)
    {
    }

    /**
     * Fills the buffer with generated triangle wave samples.
     *
     * The default behaviour is to generate a sawtooth wave. To change that
     * to a triangle wave, call setWidth() with some value between 0 and 1.
     *
     * @param samplesCount how many samples to generate
     */
    void TriangleGenerator::generate(std::size_t samplesCount)
    {
        m_data.resize(samplesCount);

        double dt = 1.0 / m_sampleFrequency, period = 1.0 / m_frequency;
        double risingLength = m_width * period;
        double fallingLength = period - risingLength;
        double risingIncrement =
            (risingLength != 0) ? (2.0 * m_amplitude / risingLength) : 0;
        double fallingDecrement =
            (fallingLength != 0) ? (2.0 * m_amplitude / fallingLength) : 0;

        double t = 0;
        for (std::size_t i = 0; i < samplesCount; ++i)
        {
            if (t > period)
            {
                t -= period;
            }
            if (t < risingLength)
            {
                m_data[i] = -m_amplitude + t * risingIncrement;
            }
            else
            {
                m_data[i] = m_amplitude - (t - risingLength) * fallingDecrement;
            }
            t += dt;
        }
    }
}
