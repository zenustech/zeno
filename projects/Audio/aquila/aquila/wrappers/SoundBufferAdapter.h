/**
 * @file SoundBufferAdapter.h
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

#ifndef SOUNDBUFFERADAPTER_H
#define SOUNDBUFFERADAPTER_H

#include "../global.h"
#include <SFML/Audio.hpp>

namespace Aquila
{
    class SignalSource;

    /**
     * A wrapper around SignalSource to use as a sound buffer in SFML.
     */
    class AQUILA_EXPORT SoundBufferAdapter : public sf::SoundBuffer
    {
    public:
        SoundBufferAdapter();
        SoundBufferAdapter(const SoundBufferAdapter& other);
        SoundBufferAdapter(const SignalSource& source);
        ~SoundBufferAdapter();

        bool loadFromSignalSource(const SignalSource& source);
    };
}

#endif // SOUNDBUFFERADAPTER_H
