/**
 * @file Frame.h
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

#ifndef FRAME_H
#define FRAME_H

#include "../global.h"
#include "SignalSource.h"
#include <algorithm>
#include <cstddef>

namespace Aquila
{
    /**
     * An ecapsulation of a single frame of the signal.
     *
     * The Frame class wraps a signal frame (short fragment of a signal).
     * Frame itself can be considered as a signal source, being a "window"
     * over original signal data. It is a lightweight object which can be
     * copied by value. No data are copied - only the pointer to source
     * and frame boundaries.
     *
     * Frame samples are accessed by STL-compatible iterators, as is the
     * case with all SignalSource-derived classes. Frame sample number N
     * is the same as sample number FRAME_BEGIN+N in the original source.
     *
     * Example (source size = N, frame size = M, frame starts at 8th sample):
     *
     * @verbatim
     * sample number:          0       8                                       N
     * original source:        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     * frame:                          |xxxxxxxxxxxx|
     * sample number in frame:         0             M              @endverbatim
     */
    class AQUILA_EXPORT Frame : public SignalSource
    {
    public:
        Frame(const SignalSource& source, unsigned int indexBegin,
                unsigned int indexEnd);
        Frame(const Frame& other);
        Frame(Frame&& other);
        Frame& operator=(const Frame& other);

        /**
         * Returns the frame length.
         *
         * @return frame length as a number of samples
         */
        virtual std::size_t getSamplesCount() const
        {
            return m_end - m_begin;
        }

        /**
         * Returns number of bits per sample.
         *
         * @return sample size in bits
         */
        virtual unsigned short getBitsPerSample() const
        {
            return m_source->getBitsPerSample();
        }

        /**
         * Gives access to frame samples, indexed from 0 to length()-1.
         *
         * @param position index of the sample in the frame
         * @return sample value
         */
        virtual SampleType sample(std::size_t position) const
        {
            return m_source->sample(m_begin + position);
        }

        /**
         * Returns sample data (read-only!) as a const C-style array.
         *
         * Calculates, using C++ pointer arithmetics, where does the frame
         * start in the original source, after its convertion to an array.
         *
         * @return C-style array containing sample data
         */
        virtual const SampleType* toArray() const
        {
            return m_source->toArray() + static_cast<std::ptrdiff_t>(m_begin);
        }

    private:
        /**
         * A non-owning pointer to constant original source (eg. a WAVE file).
         */
        const SignalSource* m_source;

        /**
         * First and last sample of this frame in the data array/vector.
         */
        unsigned int m_begin, m_end;

        /**
         * Swaps the frame with another one - exception safe.
         *
         * @param other reference to swapped frame
         */
        void swap(Frame& other)
        {
            std::swap(m_begin, other.m_begin);
            std::swap(m_end, other.m_end);
            std::swap(m_source, other.m_source);
        }
    };
}

#endif // FRAME_H
