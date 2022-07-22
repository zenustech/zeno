/**
 * @file HannWindow.h
 *
 * Hann window.
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

#ifndef HANNWINDOW_H
#define HANNWINDOW_H

#include "../../global.h"
#include "../SignalSource.h"
#include <cstddef>

namespace Aquila
{
    /**
     * Hann window.
     */
    class AQUILA_EXPORT HannWindow : public SignalSource
    {
    public:
        HannWindow(std::size_t size);
    };
}

#endif // HANNWINDOW_H
