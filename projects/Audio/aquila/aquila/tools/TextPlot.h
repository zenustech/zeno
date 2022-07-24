/**
 * @file TextPlot.h
 *
 * A simple console-based data plotter.
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

#ifndef TEXTPLOT_H
#define TEXTPLOT_H

#include "../global.h"
#include "../source/SignalSource.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace Aquila
{
    /**
     * A utility class to "draw" data plots in the console applications.
     *
     * TextPlot is used for quick-and-dirty data plotting in console programs.
     * It also serves as a way to take a peek into signal data. The class is
     * very simple and - intentionally - lacks a lot of configuration options.
     * A more serious tool will be neccessary to create pretty images.
     * For example, a frontend to gnuplot is taken under consideration.
     */
    class AQUILA_EXPORT TextPlot
    {
    public:
        TextPlot(const std::string& title = "Data plot",
                 std::ostream& out = std::cout);

        /**
         * Returns the title of the plot.
         *
         * @return plot title
         */
        std::string getTitle() const
        {
            return m_title;
        }

        /**
         * Sets the title of the plot.
         *
         * @param title plot title
         */
        void setTitle(const std::string& title)
        {
            m_title = title;
        }

        /**
         * Returns plot width.
         *
         * @return plot width
         */
        std::size_t getWidth() const
        {
            return m_width;
        }

        /**
         * Returns plot height.
         *
         * @return plot height
         */
        std::size_t getHeight() const
        {
            return m_height;
        }

        /**
         * Sets plot dimensions.
         *
         * @param width plot width, currently ignored
         * @param height plot height
         */
        void setSize(std::size_t width, std::size_t height)
        {
            m_width = width;
            m_height = height;
        }

        /**
         * Plots all data from a given source.
         *
         * @param source data source
         */
        void plot(const SignalSource& source)
        {
            PlotMatrixType plotData(source.length());
            doPlot(plotData, source.begin(), source.end());
        }

        /**
         * An overload for plotting vectors.
         *
         * @param data a numeric vector
         */
        template<typename Numeric>
        void plot(const std::vector<Numeric>& data)
        {
            PlotMatrixType plotData(data.size());
            doPlot(plotData, data.begin(), data.end());
        }

        /**
         * An overload for plotting regular C-style arrays.
         *
         * In theory, we could use the template argument trick to omit the
         * `length` argument, however, that is possible only for static arrays.
         *
         * @param data the array
         * @param length array size
         */
        template<typename Numeric>
        void plot(Numeric data[], std::size_t length)
        {
            PlotMatrixType plotData(length);
            doPlot(plotData, data, data + length);
        }

        void plotSpectrum(SpectrumType spectrum);

    private:
        /**
         * The internal representation of the plot.
         */
        typedef std::vector<std::vector<char>> PlotMatrixType;

        /**
         * Prepares an internal "pixel" matrix and calls drawPlotMatrix().
         *
         * The template parameter is an iterator type, therefore the plotter
         * can be more generic.
         *
         * @param plot reference to the plot matrix
         * @param begin an iterator pointing to the beginning of plotted data
         * @param end an iterator pointing "one past end" of plotted data
         */
        template <typename Iterator>
        void doPlot(PlotMatrixType& plotData, Iterator begin, Iterator end)
        {
            const double max = *std::max_element(begin, end);
            const double min = *std::min_element(begin, end);
            const double range = max - min;

            for (std::size_t xPos = 0; xPos < plotData.size(); ++xPos)
            {
                plotData[xPos].resize(m_height, ' ');
                double normalizedValue = (*begin++ - min) / range;
                std::size_t yPos = m_height - static_cast<std::size_t>(
                    std::ceil(m_height * normalizedValue));

                // bound the value so it stays within vector dimension
                if (yPos >= m_height)
                    yPos = m_height - 1;
                plotData[xPos][yPos] = '*';
            }

            drawPlotMatrix(plotData);
        }

        void drawPlotMatrix(const PlotMatrixType& plotData);

        /**
         * Plot title.
         */
        std::string m_title;

        /**
         * Output stream.
         */
        std::ostream& m_out;

        /**
         * Plot width - currently ignored.
         */
        std::size_t m_width;

        /**
         * Plot height.
         */
        std::size_t m_height;
    };
}

#endif // TEXTPLOT_H
