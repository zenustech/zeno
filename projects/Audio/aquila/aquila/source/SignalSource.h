/**
 * @file SignalSource.h
 *
 * A base signal source class.
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

#ifndef SIGNALSOURCE_H
#define SIGNALSOURCE_H

#include "../global.h"
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace Aquila
{
    /**
     * An abstraction of any signal source.
     *
     * This is the base class for all signal sources cooperating with the
     * library, be it an array, a text file or a WAVE binary file. Most of
     * algorithms defined in the library expect a pointer or a reference
     * to SignalSource. The library ships with a few derived
     * classes for a quick start, including WaveFile, generators etc.
     *
     * Signal sources support the concept of iteration. Use
     * SignalSource::begin() and SignalSource::end() to obtain iterator objects,
     * which allow per-sample data access. The iterators work well with
     * C++ standard library algorithms, so feel free to use them instead of
     * manually looping and calling SignalSource::sample().
     */
    class AQUILA_EXPORT SignalSource
    {
    public:
        /**
         * The default constructor sets sample frequency to 0.
         */
        SignalSource():
            m_data(), m_sampleFrequency(0)
        {
        }

        /**
         * CThis overload initializes sample frequency of the source.
         *
         * @param sampleFrequency sample frequency in Hz
         */
        SignalSource(FrequencyType sampleFrequency):
            m_data(), m_sampleFrequency(sampleFrequency)
        {
        }

        /**
         * Initialize the source from data sampled at a given frequency.
         *
         * @param data pointer to a C-style array of samples (numeric values)
         * @param dataLength length of the array
         * @param sampleFrequency sample frequency in Hz
         */
        template <typename Numeric>
        SignalSource(Numeric* data, std::size_t dataLength,
                     FrequencyType sampleFrequency = 0):
            m_data(data, data + dataLength), m_sampleFrequency(sampleFrequency)
        {
        }

        /**
         * Create the source from a vector of samples.
         *
         * @param data vector of samples
         * @param sampleFrequency sample frequency in Hz
         */
        SignalSource(const std::vector<SampleType>& data,
                     FrequencyType sampleFrequency = 0):
            m_data(data), m_sampleFrequency(sampleFrequency)
        {
        }

        /**
         * Create the source from a (temporary) vector of samples.
         *
         * @param data vector of samples
         * @param sampleFrequency sample frequency in Hz
         */
        SignalSource(std::vector<SampleType>&& data,
                     FrequencyType sampleFrequency = 0):
            m_data(std::move(data)), m_sampleFrequency(sampleFrequency)
        {
        }

        /**
         * The destructor does nothing, but must be defined as virtual.
         */
        virtual ~SignalSource() {}

        /**
         * Returns sample frequency of the signal.
         *
         * @return sample frequency in Hz
         */
        virtual FrequencyType getSampleFrequency() const
        {
            return m_sampleFrequency;
        }

        /**
         * Sets sample frequency of the signal.
         *
         * @param sampleFrequency sample frequency in Hz
         */
        virtual void setSampleFrequency(FrequencyType sampleFrequency)
        {
            m_sampleFrequency = sampleFrequency;
        }

        /**
         * Returns number of bits per signal sample.
         *
         * @return sample size in bits
         */
        virtual unsigned short getBitsPerSample() const
        {
            return 8 * sizeof(SampleType);
        }

        /**
         * Returns number of samples in the source.
         *
         * @return samples count
         */
        virtual std::size_t getSamplesCount() const
        {
            return m_data.size();
        }

        /**
         * Returns sample located at the "position" in the signal.
         *
         * @param position sample index in the signal
         * @return sample value
         */
        virtual SampleType sample(std::size_t position) const
        {
            return m_data[position];
        }

        /**
         * Returns sample data (read-only!) as a const C-style array.
         *
         * Because vector guarantees to be contiguous in memory, we can
         * return the address of the first element in the vector.
         * It is valid only until next operation which modifies the vector.
         *
         * @return C-style array containing sample data
         */
        virtual const SampleType* toArray() const
        {
            return &m_data[0];
        }

        /**
         * Returns number of samples in the source.
         *
         * This method is an alias to getSamplesCount() and it should not be
         * implemented in derived classes.
         *
         * @return samples count
         */
        std::size_t length() const
        {
            return getSamplesCount();
        }

        class iterator;

        /**
         * Returns an iterator pointing to the first sample in the source.
         *
         * @return iterator
         */
        iterator begin() const
        {
            return iterator(this, 0);
        }

        /**
         * Returns an iterator pointing to the "one past last" sample.
         *
         * @return iterator
         */
        iterator end() const
        {
            return iterator(this, getSamplesCount());
        }

        /**
         * Iterator class enabling sequential data access.
         *
         * It is a forward iterator with a range from the first sample in the
         * source to "one past last" sample.
         */
        class AQUILA_EXPORT iterator :
            public std::iterator<std::forward_iterator_tag, int>
        {
        public:
            /**
             * Creates an iterator associated with a given source.
             *
             * @param source pointer to a source on which the iterator will work
             * @param i index of the sample in the source
             */
            explicit iterator(const SignalSource* source = nullptr, unsigned int i = 0):
                m_source(source), idx(i)
            {
            }

            /**
             * Assigns another iterator in a memberwise fashion.
             *
             * @param other right-hand value iterator
             * @return reference to self
             */
            iterator& operator=(const iterator& other)
            {
                m_source = other.m_source;
                idx = other.idx;
                return (*this);
            }

            /**
             * Compares two iterators for equality.
             *
             * Iterators are equal only when they belong to the same source
             * and point to the same sample in the source.
             *
             * @param other right-hand value iterator
             * @return true, if the iterators are equal
             */
            bool operator==(const iterator& other) const
            {
                return m_source == other.m_source && idx == other.idx;
            }

            /**
             * Compares two iterators for inequality.
             *
             * Negates the equality operator.
             *
             * @param other right-hand value iterator
             * @return true only when the iterators are not equal
             */
            bool operator!=(const iterator& other) const
            {
                return !operator==(other);
            }

            /**
             * Moves the iterator one sample to the right (prefix version).
             *
             * @return reference to self
             */
            iterator& operator++()
            {
                ++idx;
                return (*this);
            }

            /**
             * Moves the iterator one sample to the right (postfix version).
             *
             * @return a copy of self before incrementing
             */
            iterator operator++(int)
            {
                iterator tmp(*this);
                ++(*this);
                return tmp;
            }

            /**
             * Dereferences the iterator.
             *
             * @return signal sample value.
             */
            SampleType operator*() const
            {
                return m_source->sample(idx);
            }

            /**
             * Returns the distance from the beginning of the source.
             *
             * @return number of samples between beginning and current position
             */
            std::size_t getPosition() const
            {
                return idx;
            }

        private:
            /**
             * Signal source - as a pointer - the iterators must be copyable.
             */
            const SignalSource* m_source;

            /**
             * Iterator's position in the source.
             */
            std::size_t idx;
        };

        SignalSource& operator+=(SampleType x);
        SignalSource& operator+=(const SignalSource& rhs);
        SignalSource& operator*=(SampleType x);
        SignalSource& operator*=(const SignalSource& rhs);

    protected:
        /**
         * Actual sample data.
         */
        std::vector<SampleType> m_data;

        /**
         * Sample frequency of the data.
         */
        FrequencyType m_sampleFrequency;
    };

    SignalSource operator+(const SignalSource& lhs, SampleType x);
    SignalSource operator+(SignalSource&& lhs, SampleType x);
    SignalSource operator+(SampleType x, const SignalSource& rhs);
    SignalSource operator+(SampleType x, SignalSource&& rhs);
    SignalSource operator+(const SignalSource& lhs, const SignalSource& rhs);
    SignalSource operator+(SignalSource&& lhs, const SignalSource& rhs);
    SignalSource operator+(const SignalSource& lhs, SignalSource&& rhs);

    SignalSource operator*(const SignalSource& lhs, SampleType x);
    SignalSource operator*(SignalSource&& lhs, SampleType x);
    SignalSource operator*(SampleType x, const SignalSource& rhs);
    SignalSource operator*(SampleType x, SignalSource&& rhs);
    SignalSource operator*(const SignalSource& lhs, const SignalSource& rhs);
    SignalSource operator*(SignalSource&& lhs, const SignalSource& rhs);
    SignalSource operator*(const SignalSource& lhs, SignalSource&& rhs);

    /***************************************************************************
     *
     * Free-standing functions closely related to signals.
     *
     **************************************************************************/

    double mean(const SignalSource& source);

    double energy(const SignalSource& source);

    double power(const SignalSource& source);

    double norm(const SignalSource& source);

    double rms(const SignalSource& source);
}

#endif // SIGNALSOURCE_H
