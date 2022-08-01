/**
 * @file Synthesizer.h
 *
 * Base class for SFML-based audio synthesizers.
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

#ifndef SYNTHESIZER_H
#define SYNTHESIZER_H

#include "../global.h"
#include "../wrappers/SoundBufferAdapter.h"
#include <SFML/System.hpp>
#include <SFML/Audio.hpp>
#include <map>
#include <string>

namespace Aquila
{
    /**
     * Type of the mapping from note names to frequencies.
     */
    typedef std::map<std::string, FrequencyType> NoteMapType;

    NoteMapType initNoteMap();

    /**
     * An abstract class from which sound synthesizers should be derived.
     */
    class AQUILA_EXPORT Synthesizer
    {
    public:
        /**
         * Creates the synthesizer object.
         *
         * @param sampleFrequency sample frequency of the audio signal
         */
        Synthesizer(FrequencyType sampleFrequency):
            m_sampleFrequency(sampleFrequency), m_buffer()
        {
        }

        /**
         * No-op virtual destructor.
         */
        virtual ~Synthesizer() {}

        void playNote(std::string note, unsigned int duration = 500);

    protected:
        /**
         * Plays a tone at given frequency.
         *
         * Must be overriden in child classes.
         *
         * @param frequency base frequency of the generated sound
         * @param duration tone duration in milliseconds
         */
        virtual void playFrequency(FrequencyType frequency, unsigned int duration) = 0;

        /**
         * Sample frequency of the generated signal.
         */
        const FrequencyType m_sampleFrequency;

        /**
         * Audio buffer for playback.
         */
        SoundBufferAdapter m_buffer;

        /**
         * A mapping from note names to frequencies.
         */
        static NoteMapType m_notes;
    };
}

#endif // SYNTHESIZER_H
