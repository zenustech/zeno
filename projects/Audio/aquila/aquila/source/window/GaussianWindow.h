/**
 * @file GaussianWindow.h
 *
 * Gaussian (triangular) window. Based on the formula given at:
 * http://en.wikipedia.org/wiki/Window_function#Gaussian_window
 *
 * This file is part of the Aquila DSP library.
 * Aquila is free software, licensed under the MIT/X11 License. A copy of
 * the license is provided with the library in the LICENSE file.
 *
 * @package Aquila
 * @version 3.0.0-dev
 * @author Chris Vandevelde
 * @date 2007-2014
 * @license http://www.opensource.org/licenses/mit-license.php MIT
 * @since 3.0.0
 */

#ifndef GAUSSIANWINDOW_H
#define GAUSSIANWINDOW_H

#include "../../global.h"
#include "../SignalSource.h"
#include <cstddef>

namespace Aquila
{
    /**
     * Creates Gaussian window of given size, with optional sigma parameter.
     */
    class AQUILA_EXPORT GaussianWindow : public SignalSource
    {
    public:
        GaussianWindow(std::size_t size, double sigma = 0.5);
    };
}

#endif // GAUSSIANWINDOW_H
