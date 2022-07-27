/**
 * @file Spectrogram.cpp
 *
 * Spectrogram calculation.
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

#include "Spectrogram.h"
#include "FftFactory.h"
#include "../source/Frame.h"
#include "../source/FramesCollection.h"

namespace Aquila
{
    /**
     * Creates the spectrogram from a collection of signal frames.
     *
     * Calculates frame spectra immediately after initialization.
     *
     * @param frames input frames
     */
    Spectrogram::Spectrogram(FramesCollection& frames):
        m_frameCount(frames.count()),
        m_spectrumSize(frames.getSamplesPerFrame()),
        m_fft(FftFactory::getFft(m_spectrumSize)),
        m_data(new SpectrogramDataType(m_frameCount))
    {
        std::size_t i = 0;
        for (auto it = frames.begin(); it != frames.end(); ++it, ++i)
        {
            (*m_data)[i] = m_fft->fft(it->toArray());
        }
    }
}
