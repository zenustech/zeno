/**
 * @file TriangleGenerator.h
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

#ifndef TRIANGLEGENERATOR_H
#define TRIANGLEGENERATOR_H

#include "Generator.h"

namespace Aquila
{
    /**
     * Triangle (and sawtooth) wave generator.
     */
    class AQUILA_EXPORT TriangleGenerator : public Generator
    {
    public:
        TriangleGenerator(FrequencyType sampleFrequency);

        virtual void generate(std::size_t samplesCount);

        /**
         * Sets slope width of the generated triangle wave.
         *
         * Slope width is a fraction of period in which signal is rising.
         *
         * @param width slope width (0 < width <= 1)
         * @return the current object for fluent interface
         */
        TriangleGenerator& setWidth(double width)
        {
            m_width = width;

            return *this;
        }

    private:
        /**
         * Slope width, default = 1.0 (generates sawtooth wave).
         */
        double m_width;
    };
}

#endif // TRIANGLEGENERATOR_H
