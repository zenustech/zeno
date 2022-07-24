/**
 * @file Generator.h
 *
 * An interface for signal generators.
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

#ifndef GENERATOR_H
#define GENERATOR_H

#include "../SignalSource.h"
#include "../../global.h"
#include <cstddef>

namespace Aquila
{
    /**
     * The base interface for signal generators.
     */
    class AQUILA_EXPORT Generator : public SignalSource
    {
    public:
        /**
         * Creates the generator object.
         *
         * @param sampleFrequency sample frequency of the data in array
         */
        Generator(FrequencyType sampleFrequency);

        /**
         * Sets frequency of the generated signal.
         *
         * @param frequency signal frequency
         * @return the current object for fluent interface
         */
        Generator& setFrequency(FrequencyType frequency)
        {
            m_frequency = frequency;

            return *this;
        }

        /**
         * Sets amplitude of the generated signal.
         *
         * @param amplitude signal amplitude
         * @return the current object for fluent interface
         */
        Generator& setAmplitude(SampleType amplitude)
        {
            m_amplitude = amplitude;

            return *this;
        }

        /**
         * Sets phase shift of the generated wave.
         *
         * @param phase phase shift (0 < phase <= 1)
         * @return the current object for fluent interface
         */
        Generator& setPhase(double phase)
        {
            m_phase = phase;

            return *this;
        }

        /**
         * Generates a given number of samples.
         *
         * @param samplesCount how many samples to generate
         */
        virtual void generate(std::size_t samplesCount) = 0;

    protected:
        /**
         * Frequency of the generated signal (not always used).
         */
        FrequencyType m_frequency;

        /**
         * Amplitude of the generated signal (not always used).
         */
        SampleType m_amplitude;

        /**
         * Phase shift as a fraction of whole period (default = 0.0).
         */
        double m_phase;
    };
}

#endif // GENERATOR_H
