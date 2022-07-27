/**
 * @file HammingWindow.cpp
 *
 * Hamming window.
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

#include "HammingWindow.h"
#include <cmath>

namespace Aquila
{
    /**
     * Creates Hamming window of given size.
     *
     * @param size window length
     */
    HammingWindow::HammingWindow(std::size_t size):
        SignalSource()
    {
        m_data.reserve(size);
        for (std::size_t n = 0; n < size; ++n)
        {
            m_data.push_back(
                0.53836 - 0.46164 * std::cos(2.0 * M_PI * n / double(size - 1))
            );
        }
    }
}
