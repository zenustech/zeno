/**
 * @file KarplusStrongSynthesizer.h
 *
 * Plucked string synthesis using Karplus-Strong algorithm.
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

#ifndef KARPLUSSTRONGSYNTHESIZER_H
#define KARPLUSSTRONGSYNTHESIZER_H

#include "../global.h"
#include "Synthesizer.h"
#include "../source/generator/WhiteNoiseGenerator.h"

namespace Aquila
{
    /**
     * Very simple guitar synthesizer using Karplus-Strong algorithm.
     */
    class AQUILA_EXPORT KarplusStrongSynthesizer : public Synthesizer
    {
    public:
        /**
         * Creates the synthesizer object.
         *
         * @param sampleFrequency sample frequency of the audio signal
         */
        KarplusStrongSynthesizer(FrequencyType sampleFrequency):
            Synthesizer(sampleFrequency), m_generator(sampleFrequency),
            m_alpha(0.99)
        {
        }

        /**
         * Sets feedback loop parameter.
         */
        void setAlpha(double alpha)
        {
            m_alpha = alpha;
        }

    protected:
        virtual void playFrequency(FrequencyType frequency, unsigned int duration);

    private:
        /**
         * Generator used to provide initial noise burst.
         */
        WhiteNoiseGenerator m_generator;

        /**
         * Feedback loop parameter.
         */
        double m_alpha;
    };
}

#endif // KARPLUSSTRONGSYNTHESIZER_H
