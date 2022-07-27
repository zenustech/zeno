/**
 * @file FlattopWindow.cpp
 *
 * Flat-top window.
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

#include "FlattopWindow.h"
#include <cmath>

namespace Aquila
{
    /**
     * Creates flat-top window of given size.
     *
     * @param size window length
     */
    FlattopWindow::FlattopWindow(std::size_t size):
        SignalSource()
    {
        m_data.reserve(size);
        for (std::size_t n = 0; n < size; ++n)
        {
            m_data.push_back(
                1.0 - 1.93 * std::cos(2.0 * M_PI * n / double(size - 1)) +
                    1.29 * std::cos(4.0 * M_PI * n / double(size - 1)) -
                    0.388 * std::cos(6.0 * M_PI * n / double(size - 1)) +
                    0.0322 * std::cos(8.0 * M_PI * n / double(size - 1))
            );
        }
    }
}
