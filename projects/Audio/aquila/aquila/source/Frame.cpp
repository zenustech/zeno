/**
 * @file Frame.cpp
 *
 * Handling signal frames.
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
 * @since 0.2.2
 */

#include "Frame.h"

namespace Aquila
{
    /**
     * Creates the frame object - sets signal source and frame boundaries.
     *
     * Frame should not change original data, so the source is a const
     * reference.
     *
     * @param source const reference to signal source
     * @param indexBegin position of first sample of this frame in the source
     * @param indexEnd position of last sample of this frame in the source
     */
    Frame::Frame(const SignalSource& source, unsigned int indexBegin,
            unsigned int indexEnd):
        SignalSource(source.getSampleFrequency()),
        m_source(&source), m_begin(indexBegin),
        m_end((indexEnd > source.getSamplesCount()) ? source.getSamplesCount() : indexEnd)
    {
    }

    /**
     * Copy constructor.
     *
     * @param other reference to another frame
     */
    Frame::Frame(const Frame &other):
        SignalSource(other.m_sampleFrequency),
        m_source(other.m_source), m_begin(other.m_begin), m_end(other.m_end)
    {
    }

    /**
     * Move constructor.
     *
     * @param other rvalue reference to another frame
     */
    Frame::Frame(Frame&& other):
        SignalSource(other.m_sampleFrequency),
        m_source(other.m_source), m_begin(other.m_begin), m_end(other.m_end)
    {
    }

    /**
     * Assignes another frame to this one using copy-and-swap idiom.
     *
     * @param other reference to another frame
     * @return reference to the current object
     */
    Frame& Frame::operator=(const Frame& other)
    {
        Frame temp(other);
        swap(temp);

        return *this;
    }
}
