/**
 * @file TextPlot.cpp
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

#include "TextPlot.h"

namespace Aquila
{
    /**
     * Creates the plot object.
     *
     * @param title plot title (optional, default is "Data plot")
     * @param out where to output the plot data (default is std::cout)
     */
    TextPlot::TextPlot(const std::string &title, std::ostream &out):
        m_title(title), m_out(out), m_width(64), m_height(16)
    {
    }

    /**
     * "Draws" the plot to the output stream.
     *
     * @param plot internal plot data
     */
    void TextPlot::drawPlotMatrix(const PlotMatrixType &plotData)
    {
        const std::size_t length = plotData.size();

        m_out << "\n" << m_title << "\n";
        // output the plot data, flushing only at the end
        for (unsigned int y = 0; y < m_height; ++y)
        {
            for (unsigned int x = 0; x < length; ++x)
            {
                m_out << plotData[x][y];
            }
            m_out << "\n";
        }
        m_out.flush();
    }

    /**
     * A shorthand for plotting only the first half of magnitude spectrum.
     *
     * @param spectrum a vector of complex numbers
     */
    void TextPlot::plotSpectrum(Aquila::SpectrumType spectrum)
    {
        std::size_t halfLength = spectrum.size() / 2;
        std::vector<double> absSpectrum(halfLength);
        for (std::size_t i = 0; i < halfLength; ++i)
        {
            absSpectrum[i] = std::abs(spectrum[i]);
        }
        plot(absSpectrum);
    }
}
