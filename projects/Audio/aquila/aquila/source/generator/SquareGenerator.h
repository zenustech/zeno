/**
 * @file SquareGenerator.h
 *
 * Square wave generator.
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

#ifndef SQUAREGENERATOR_H
#define SQUAREGENERATOR_H

#include "Generator.h"

namespace Aquila
{
    /**
     * Square wave generator.
     */
    class AQUILA_EXPORT SquareGenerator : public Generator
    {
    public:
        SquareGenerator(FrequencyType sampleFrequency);

        virtual void generate(std::size_t samplesCount);

        /**
         * Sets duty cycle of the generated square wave.
         *
         * Duty cycle is a fraction of period in which the signal is positive.
         *
         * @param duty duty cycle (0 < duty <= 1)
         * @return the current object for fluent interface
         */
        SquareGenerator& setDuty(double duty)
        {
            m_duty = duty;

            return *this;
        }

    private:
        /**
         * Duty cycle, default = 0.5.
         */
        double m_duty;
    };
}

#endif // SQUAREGENERATOR_H
