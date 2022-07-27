/**
 * @file BarlettWindow.cpp
 *
 * Barlett (triangular) window.
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

#include "BarlettWindow.h"
#include <cmath>

namespace Aquila
{
    /**
     * Creates Barlett (triangular) window of given size.
     *
     * @param size window length
     */
    BarlettWindow::BarlettWindow(std::size_t size):
        SignalSource()
    {
        m_data.reserve(size);
        for (std::size_t n = 0; n < size; ++n)
        {
            m_data.push_back(
                1.0 - (2.0 * std::fabs(n - (size - 1) / 2.0)) / (double(size - 1))
            );
        }
    }
}
