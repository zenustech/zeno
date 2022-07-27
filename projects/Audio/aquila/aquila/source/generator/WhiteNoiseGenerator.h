/**
 * @file WhiteNoiseGenerator.h
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

#ifndef WHITENOISEGENERATOR_H
#define WHITENOISEGENERATOR_H

#include "Generator.h"

namespace Aquila
{
    /**
     * White noise generator.
     */
    class AQUILA_EXPORT WhiteNoiseGenerator : public Generator
    {
    public:
        WhiteNoiseGenerator(FrequencyType sampleFrequency);

        virtual void generate(std::size_t samplesCount);
    };
}

#endif // WHITENOISEGENERATOR_H
