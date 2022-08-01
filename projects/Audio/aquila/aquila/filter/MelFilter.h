/**
 * @file MelFilter.h
 *
 * Triangular filters in Mel frequency scale.
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
 * @since 0.3.3
 */

#ifndef MELFILTER_H
#define MELFILTER_H

#include "../global.h"
#include <cstddef>
#include <cmath>
#include <vector>

namespace Aquila
{
    /**
     * Encapsulation of a single Mel-frequency filter.
     */
    class AQUILA_EXPORT MelFilter
    {
    public:
        explicit MelFilter(FrequencyType sampleFrequency);
        MelFilter(MelFilter&& other);
        MelFilter& operator=(const MelFilter& other);

        void createFilter(std::size_t filterNum, FrequencyType melFilterWidth,
                          std::size_t N);

        double apply(const SpectrumType& dataSpectrum) const;

        /**
         * Converts frequency from linear to Mel scale.
         *
         * @param linearFrequency frequency in linear scale
         * @return frequency in Mel scale
         */
        static FrequencyType linearToMel(FrequencyType linearFrequency)
        {
            return 1127.01048 * std::log(1.0 + linearFrequency / 700.0);
        }

        /**
         * Converts frequency from Mel to linear scale.
         *
         * @param melFrequency frequency in Mel scale
         * @return frequency in linear scale
         */
        static FrequencyType melToLinear(FrequencyType melFrequency)
        {
            return 700.0 * (std::exp(melFrequency / 1127.01048) - 1.0);
        }

        /**
         * Returns sample frequency for which the filter was designed.
         *
         * @return sample frequency
         */
        FrequencyType getSampleFrequency() const
        {
            return m_sampleFrequency;
        }

    private:
        FrequencyType m_sampleFrequency;

        /**
         * Filter spectrum (real-valued).
         */
        std::vector<double> m_spectrum;

        void generateFilterSpectrum(FrequencyType minFreq,
                                    FrequencyType centerFreq,
                                    FrequencyType maxFreq, std::size_t N);
    };
}

#endif // MELFILTER_H
