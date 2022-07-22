/**
 * @file Synthesizer.cpp
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

#include "Synthesizer.h"
#include <SFML/System/Sleep.hpp>

namespace Aquila
{
    /**
     * Creates the mapping between note names and frequencies.
     *
     * @return initialized note map
     */
    NoteMapType initNoteMap()
    {
        NoteMapType notes;
        notes["C2"] = 65.406;
        notes["C2S"] = 69.296;
        notes["D2"] = 73.416;
        notes["D2S"] = 77.782;
        notes["E2"] = 82.407;
        notes["F2"] = 87.307;
        notes["F2S"] = 92.499;
        notes["G2"] = 97.999;
        notes["G2S"] = 103.83;
        notes["A2"] = 110.0;
        notes["A2S"] = 116.54;
        notes["B2"] = 123.47;
        notes["C3"] = 130.81;
        notes["C3S"] = 138.59;
        notes["D3"] = 146.83 ;
        notes["D3S"] = 155.56;
        notes["E3"] = 164.81;
        notes["F3"] = 174.61;
        notes["F3S"] = 185.0;
        notes["G3"] = 196.0;
        notes["G3S"] = 207.65;
        notes["A3"] = 220.00;
        notes["A3S"] = 233.08;
        notes["B3"] = 246.94;
        notes["C4"] = 261.626;
        notes["C4S"] = 277.18;
        notes["D4"] = 293.665;
        notes["D4S"] = 311.13;
        notes["E4"] = 329.628;
        notes["F4"] = 349.228;
        notes["F4S"] = 369.99;
        notes["G4"] = 391.995;
        notes["G4S"] = 415.305;
        notes["A4"] = 440.0;
        notes["A4S"] = 466.164;
        notes["B4"] = 493.883;
        notes["C5"] = 523.251;
        notes["C5S"] = 554.365;
        notes["D5"] = 587.33;
        notes["D5S"] = 622.254;
        notes["E5"] = 659.255;
        notes["F5"] = 698.456;
        notes["F5S"] = 739.989;
        notes["G5"] = 783.991;
        notes["G5S"] = 830.609;
        notes["A5"] = 880.0;
        notes["A5S"] = 932.33;
        notes["B5"] = 987.77;
        notes["C6"] = 1046.5;
        return notes;
    }

    NoteMapType Synthesizer::m_notes = initNoteMap();

    /**
     * Plays a single note.
     *
     * This method only does the lookup from note name to frequency and
     * delegates the actual playing to pure virtual method playFrequency.
     *
     * Unrecognized note names are silent for the given duration.
     *
     * @param note note name (@see m_notes)
     * @param duration duration in milliseconds
     */
    void Synthesizer::playNote(std::string note, unsigned int duration)
    {
        if (m_notes.count(note) == 0)
        {
            sf::sleep(sf::milliseconds(duration));
        }
        else
        {
            FrequencyType frequency = m_notes[note];
            playFrequency(frequency, duration);
        }
    }
}
