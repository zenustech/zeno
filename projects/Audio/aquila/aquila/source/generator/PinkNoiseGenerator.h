/**
 * @file PinkNoiseGenerator.h
 *
 * Pink noise generator.
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

#ifndef PINKNOISEGENERATOR_H
#define PINKNOISEGENERATOR_H

#include "Generator.h"

namespace Aquila
{
    /**
     * Pink noise generator using Voss algorithm.
     */
    class AQUILA_EXPORT PinkNoiseGenerator : public Generator
    {
    public:
        PinkNoiseGenerator(FrequencyType sampleFrequency);

        virtual void generate(std::size_t samplesCount);

    private:
        double pinkSample();

        /**
         * Number of white noise samples to use in Voss algorithm.
         */
        enum { whiteSamplesNum = 20 };

        /**
         * An internal buffer for white noise samples.
         */
        double whiteSamples[whiteSamplesNum];

        /**
         * A key marking which of the white noise samples should change.
         */
        int key;

        /**
         * Maximum key value.
         */
        int maxKey;
    };
}

#endif // PINKNOISEGENERATOR_H
