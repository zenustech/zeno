/**
 * @file SoundBufferAdapter.cpp
 *
 * A wrapper around SignalSource to use as a sound buffer in SFML.
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

#include "SoundBufferAdapter.h"
#include "../source/SignalSource.h"
#include <SFML/System.hpp>
#include <algorithm>

namespace Aquila
{
    /**
     * Creates the buffer with no initial data.
     */
    SoundBufferAdapter::SoundBufferAdapter():
        SoundBuffer()
    {
    }

    /**
     * Copy constructor.
     *
     * @param other buffer instance to copy from
     */
    SoundBufferAdapter::SoundBufferAdapter(const SoundBufferAdapter &other):
        SoundBuffer(other)
    {
    }

    /**
     * Creates the buffer with initial data provided by signal source.
     *
     * @param source signal source
     */
    SoundBufferAdapter::SoundBufferAdapter(const SignalSource &source):
        SoundBuffer()
    {
        loadFromSignalSource(source);
    }

    /**
     * Destructor - does nothing by itself.
     *
     * Relies on virtual call to the destructor of the parent class.
     */
    SoundBufferAdapter::~SoundBufferAdapter()
    {
    }

    /**
     * Loads sound data from an instance of SignalSource-subclass.
     *
     * Data read from source are converted to SFML-compatible sample array
     * and loaded into the buffer.
     *
     * Name capitalized for consistency with SFML coding style.
     *
     * @todo get rid of copying data around, let's come up with some better way
     *
     * @param source signal source
     * @return true if successfully loaded
     */
    bool SoundBufferAdapter::loadFromSignalSource(const SignalSource &source)
    {
        sf::Int16* samples = new sf::Int16[source.getSamplesCount()];
        std::copy(source.begin(), source.end(), samples);
        bool result = loadFromSamples(samples,
                                     source.getSamplesCount(),
                                     1,
                                     static_cast<unsigned int>(source.getSampleFrequency()));
        delete [] samples;

        return result;
    }
}
