/**
 * @file Dtw.cpp
 *
 * An implementation of the Dynamic Time Warping algorithm.
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

#include "Dtw.h"

namespace Aquila
{
    /**
     * Computes the distance between two sets of data.
     *
     * @param from first vector of features
     * @param to second vector of features
     * @return double DTW distance
     */
    double Dtw::getDistance(const DtwDataType& from, const DtwDataType& to)
    {
        m_fromSize = from.size();
        m_toSize = to.size();

        // fill the local distances array
        m_points.clear();
        m_points.resize(m_fromSize);
        for (std::size_t i = 0; i < m_fromSize; ++i)
        {
            m_points[i].reserve(m_toSize);
            for (std::size_t j = 0; j < m_toSize; ++j)
            {
                // use emplace_back, once all compilers support it correctly
                m_points[i].push_back(
                    DtwPoint(i, j, m_distanceFunction(from[i], to[j])
                ));
            }
        }

        // the actual pathfinding algorithm
        DtwPoint *top = nullptr, *center = nullptr, *bottom = nullptr, *previous = nullptr;
        for (std::size_t i = 1; i < m_fromSize; ++i)
        {
            for (std::size_t j = 1; j < m_toSize; ++j)
            {
                center = &m_points[i - 1][j - 1];
                if (Neighbors == m_passType)
                {
                    top = &m_points[i - 1][j];
                    bottom = &m_points[i][j - 1];
                }
                else // Diagonals
                {
                    if (i > 1 && j > 1)
                    {
                        top = &m_points[i - 2][j - 1];
                        bottom = &m_points[i - 1][j - 2];
                    }
                    else
                    {
                        top = &m_points[i - 1][j];
                        bottom = &m_points[i][j - 1];
                    }
                }

                if (top->dAccumulated < center->dAccumulated)
                {
                    previous = top;
                }
                else
                {
                    previous = center;
                }

                if (bottom->dAccumulated < previous->dAccumulated)
                {
                    previous = bottom;
                }

                m_points[i][j].dAccumulated = m_points[i][j].dLocal + previous->dAccumulated;
                m_points[i][j].previous = previous;
            }
        }

        return getFinalPoint().dAccumulated;
    }

    /**
     * Returns the lowest-cost path in the DTW array.
     *
     * @return path
     */
    DtwPathType Dtw::getPath() const
    {
        DtwPathType path;
        DtwPoint finalPoint = getFinalPoint();
        DtwPoint* point = &finalPoint;

        path.push_back(*point);
        while(point->previous)
        {
            point = point->previous;
            path.push_back(*point);
        }

        return path;
    }
}
