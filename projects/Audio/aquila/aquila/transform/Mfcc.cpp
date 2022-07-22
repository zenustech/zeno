/**
 * @file Mfcc.cpp
 *
 * Calculation of MFCC signal features.
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

#include "Mfcc.h"
#include "Dct.h"
#include "../source/SignalSource.h"
#include "../filter/MelFilterBank.h"

namespace Aquila
{
    /**
     * Calculates a set of MFCC features from a given source.
     *
     * @param source input signal
     * @param numFeatures how many features to calculate
     * @return vector of MFCC features of length numFeatures
     */
    std::vector<double> Mfcc::calculate(const SignalSource &source,
                                        std::size_t numFeatures)
    {
        auto spectrum = m_fft->fft(source.toArray());

        Aquila::MelFilterBank bank(source.getSampleFrequency(), m_inputSize);
        auto filterOutput = bank.applyAll(spectrum);

        Aquila::Dct dct;
        return dct.dct(filterOutput, numFeatures);
    }
}
