/**
 * @file SineSynthesizer.h
 *
 * Simple sine tone synthesis.
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

#ifndef SINESYNTHESIZER_H
#define SINESYNTHESIZER_H

#include "../global.h"
#include "Synthesizer.h"
#include "../source/generator/SineGenerator.h"

namespace Aquila
{
    /**
     * Sine wave synthesizer.
     */
    class AQUILA_EXPORT SineSynthesizer : public Synthesizer
    {
    public:
        /**
         * Creates the synthesizer object.
         *
         * @param sampleFrequency sample frequency of the audio signal
         */
        SineSynthesizer(FrequencyType sampleFrequency):
            Synthesizer(sampleFrequency), m_generator(sampleFrequency)
        {
            m_generator.setAmplitude(8192);
        }

    protected:
        void playFrequency(FrequencyType note, unsigned int duration);

    private:
        /**
         * Underlying sine generator.
         */
        SineGenerator m_generator;
    };
}

#endif // SINESYNTHESIZER_H
