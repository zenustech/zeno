/**
 * @file DtwPoint.h
 *
 * A single point of the Dynamic Time Warping algorithm.
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
 * @since 0.5.7
 */

#ifndef DTWPOINT_H
#define DTWPOINT_H

#include "../global.h"
#include <cstddef>

namespace Aquila
{
    /**
     * A struct representing a single point in the DTW array.
     */
    struct AQUILA_EXPORT DtwPoint
    {
        /**
         * Creates the point with default values.
         */
        DtwPoint():
            x(0), y(0), dLocal(0.0), dAccumulated(0.0), previous(0)
        {
        }

        /**
         * Creates the point and associates it with given coordinates.
         *
         * @param x_ x coordinate in DTW array
         * @param y_ y coordinate in DTW array
         * @param distanceLocal value of local distance at (x, y)
         */
        DtwPoint(std::size_t x_, std::size_t y_, double distanceLocal = 0.0):
            x(x_), y(y_), dLocal(distanceLocal),
            // at the edges set accumulated distance to local. otherwise 0
            dAccumulated((0 == x || 0 == y) ? dLocal : 0.0),
            previous(0)
        {
        }

        /**
         * X coordinate of the point in the DTW array.
         */
        std::size_t x;

        /**
         * Y coordinate of the point in the DTW array.
         */
        std::size_t y;

        /**
         * Local distance at this point.
         */
        double dLocal;

        /**
         * Accumulated distance at this point.
         */
        double dAccumulated;

        /**
         * Non-owning pointer to previous point in the lowest-cost path.
         */
        DtwPoint* previous;
    };
}

#endif // DTWPOINT_H
