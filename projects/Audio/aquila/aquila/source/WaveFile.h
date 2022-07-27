/**
 * @file WaveFile.h
 *
 * WAVE file handling as a signal source.
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
 * @since 0.0.7
 */

#ifndef WAVEFILE_H
#define WAVEFILE_H

#include "../global.h"
#include "SignalSource.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Aquila
{
    /**
     * Which channel to use when reading stereo recordings.
     */
    enum StereoChannel { LEFT, RIGHT };

    /**
     * .wav file header structure.
     */
    struct WaveHeader
    {
        char   RIFF[4];
        std::uint32_t DataLength;
        char   WAVE[4];
        char   fmt_[4];
        std::uint32_t SubBlockLength;
        std::uint16_t formatTag;
        std::uint16_t Channels;
        std::uint32_t SampFreq;
        std::uint32_t BytesPerSec;
        std::uint16_t BytesPerSamp;
        std::uint16_t BitsPerSamp;
        char   data[4];
        std::uint32_t WaveSize;
    };

    /**
     * Wave file data access.
     *
     * Binary files in WAVE format (.wav extension) can serve as data input for
     * Aquila. With this class, you can read the metadata and the actual
     * waveform data from the file. The supported formats are:
     *
     * - 8-bit mono
     * - 8-bit stereo*
     * - 16-bit mono
     * - 16-bit stereo*
     *
     * For stereo data, only only one of the channels is loaded from file.
     * By default this is the left channel, but you can control this from the
     * constructor parameter.
     *
     * There are no requirements for sample frequency of the data.
     */
    class AQUILA_EXPORT WaveFile : public SignalSource
    {
    public:
        /**
         * Audio channel representation.
         */
        typedef decltype(m_data) ChannelType;

        explicit WaveFile(const std::string& filename,
                          StereoChannel channel = LEFT);
        ~WaveFile();

        void load(const std::string& filename, StereoChannel channel);
        static void save(const SignalSource& source, const std::string& file);

        /**
         * Returns the filename.
         *
         * @return full path to currently loaded file
         */
        std::string getFilename() const
        {
            return m_filename;
        }

        /**
         * Returns number of channels.
         *
         * @return 1 for mono, 2 for stereo, other types are not recognized
         */
        unsigned short getChannelsNum() const
        {
            return m_header.Channels;
        }

        /**
         * Checks if this is a mono recording.
         *
         * @return true if there is only one channel
         */
        bool isMono() const
        {
            return 1 == getChannelsNum();
        }

        /**
         * Checks if this is a stereo recording.
         *
         * @return true if there are two channels
         */
        bool isStereo() const
        {
            return 2 == getChannelsNum();
        }

        /**
         * Returns the number of bytes per second.
         *
         * @return product of sample frequency and bytes per sample
         */
        unsigned int getBytesPerSec() const
        {
            return m_header.BytesPerSec;
        }

        /**
         * Returns number of bytes per sample.
         *
         * @return 1 for 8b-mono, 2 for 8b-stereo or 16b-mono, 4 for 16b-stereo
         */
        unsigned int getBytesPerSample() const
        {
            return m_header.BytesPerSamp;
        }

        /**
         * Returns number of bits per sample
         *
         * @return 8 or 16
         */
        virtual unsigned short getBitsPerSample() const
        {
            return m_header.BitsPerSamp;
        }

        /**
         * Returns the recording size (without header).
         *
         * The return value is a raw byte count. To know the real sample count,
         * it must be divided by bytes per sample.
         *
         * @return byte count
         */
        unsigned int getWaveSize() const
        {
            return m_header.WaveSize;
        }

        unsigned int getAudioLength() const;

    private:
        /**
         * Full path of the .wav file.
         */
        std::string m_filename;

        /**
         * Header structure.
         */
        WaveHeader m_header;
    };
}

#endif // WAVEFILE_H
