/**
 * @file WaveFileHandler.cpp
 *
 * A utility class to handle loading and saving of .wav files.
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

#include "WaveFileHandler.h"
#include "WaveFile.h"
#include <cstdint>
#include <cstring>
#include <fstream>

namespace Aquila
{
    /**
     * Create the handler and tell it which file to read later.
     *
     * @param filename .wav file name
     */
    WaveFileHandler::WaveFileHandler(const std::string& filename):
        m_filename(filename)
    {
    }

    /**
     * Reads WAVE header and audio channel data from file.
     *
     * @param header reference to header instance which will be filled
     * @param leftChannel reference to left audio channel
     * @param rightChannel reference to right audio channel
     */
    void WaveFileHandler::readHeaderAndChannels(WaveHeader &header,
        WaveFile::ChannelType& leftChannel, WaveFile::ChannelType& rightChannel)
    {
        // first we read header from the stream
        // then as we know now the data size, we create a temporary
        // buffer and read raw data into that buffer
        std::fstream fs;
        fs.open(m_filename.c_str(), std::ios::in | std::ios::binary);
        fs.read((char*)(&header), sizeof(WaveHeader));
        short* data = new short[header.WaveSize/2];
        fs.read((char*)data, header.WaveSize);
        fs.close();

        // initialize data channels (using right channel only in stereo mode)
        unsigned int channelSize = header.WaveSize/header.BytesPerSamp;
        leftChannel.resize(channelSize);
        if (2 == header.Channels)
            rightChannel.resize(channelSize);

        // most important conversion happens right here
        if (16 == header.BitsPerSamp)
        {
            if (2 == header.Channels)
                decode16bitStereo(leftChannel, rightChannel, data, channelSize);
            else
                decode16bit(leftChannel, data, channelSize);
        }
        else
        {
            if (2 == header.Channels)
                decode8bitStereo(leftChannel, rightChannel, data, channelSize);
            else
                decode8bit(leftChannel, data, channelSize);
        }

        // clear the buffer
        delete [] data;
    }

    /**
     * Saves the given signal source as a .wav file.
     *
     * @param source source of the data to save
     */
    void WaveFileHandler::save(const SignalSource& source)
    {
        WaveHeader header;
        createHeader(source, header);
        std::ofstream fs;
        fs.open(m_filename.c_str(), std::ios::out | std::ios::binary);
        fs.write((const char*)(&header), sizeof(WaveHeader));

        std::size_t waveSize = header.WaveSize;
        short* data = new short[waveSize/2];
        if (16 == header.BitsPerSamp)
        {
            encode16bit(source, data, waveSize/2);
        }
        else
        {
            encode8bit(source, data, waveSize/2);
        }
        fs.write((const char*)data, waveSize);

        delete [] data;
        fs.close();
    }


    /**
     * Populates a .wav file header with values obtained from the source.
     *
     * @param source data source
     * @param header the header which will be filled with correct parameters
     */
    void WaveFileHandler::createHeader(const SignalSource &source, WaveHeader &header)
    {
        std::uint32_t frequency = static_cast<std::uint32_t>(source.getSampleFrequency());
        // saving only mono files at the moment
        std::uint16_t channels = 1;
        std::uint16_t bitsPerSample = source.getBitsPerSample();
        // higher dynamic sources will be converted down to 16 bits per sample
        if (bitsPerSample > 16)
            bitsPerSample = 16;
        std::uint32_t bytesPerSec = frequency * channels * bitsPerSample / 8;
        std::uint32_t waveSize = source.getSamplesCount() * channels * bitsPerSample / 8;

        strncpy(header.RIFF, "RIFF", 4);
        // DataLength is the file size excluding first two header fields -
        // - RIFF and DataLength itself, which together take 8 bytes to store
        header.DataLength = waveSize + sizeof(WaveHeader) - 8;
        strncpy(header.WAVE, "WAVE", 4);
        strncpy(header.fmt_, "fmt ", 4);
        header.SubBlockLength = 16;
        header.formatTag = 1;
        header.Channels = channels;
        header.SampFreq = frequency;
        header.BytesPerSec = bytesPerSec;
        header.BytesPerSamp = header.Channels * bitsPerSample / 8;
        header.BitsPerSamp = bitsPerSample;
        strncpy(header.data, "data", 4);
        header.WaveSize = waveSize;
    }

    /**
     * Decodes 16 bit mono data into a suitable audio channel format.
     *
     * @param channel a reference to audio channel
     * @param data raw data buffer
     * @param channelSize expected number of samples in channel
     */
    void WaveFileHandler::decode16bit(WaveFile::ChannelType& channel, short* data, std::size_t channelSize)
    {
        for (std::size_t i = 0; i < channelSize; ++i)
        {
            channel[i] = data[i];
        }
    }

    /**
     * Decodes 16 bit stereo data into two audio channels.
     *
     * @param leftChannel a reference to left audio channel
     * @param rightChannel a reference to right audio channel
     * @param data raw data buffer
     * @param channelSize expected number of samples (same for both channels)
     */
    void WaveFileHandler::decode16bitStereo(WaveFile::ChannelType& leftChannel,
        WaveFile::ChannelType& rightChannel, short* data, std::size_t channelSize)
    {
        for (std::size_t i = 0; i < channelSize; ++i)
        {
            leftChannel[i] = data[2*i];
            rightChannel[i] = data[2*i+1];
        }
    }

    /**
     * Decodes 8 bit mono data into a suitable audio channel format.
     *
     * @param channel a reference to audio channel
     * @param data raw data buffer
     * @param channelSize expected number of samples in channel
     */
    void WaveFileHandler::decode8bit(WaveFile::ChannelType& channel, short* data, std::size_t channelSize)
    {
        // low byte and high byte of a 16b word
        unsigned char lb, hb;
        for (std::size_t i = 0; i < channelSize; ++i)
        {
            splitBytes(data[i / 2], lb, hb);
            // only one channel collects samples
            channel[i] = lb - 128;
        }
    }

    /**
     * Decodes 8 bit stereo data into two audio channels.
     *
     * @param leftChannel a reference to left audio channel
     * @param rightChannel a reference to right audio channel
     * @param data raw data buffer
     * @param channelSize expected number of samples (same for both channels)
     */
    void WaveFileHandler::decode8bitStereo(WaveFile::ChannelType& leftChannel,
        WaveFile::ChannelType& rightChannel, short* data, std::size_t channelSize)
    {
        // low byte and high byte of a 16b word
        unsigned char lb, hb;
        for (std::size_t i = 0; i < channelSize; ++i)
        {
            splitBytes(data[i / 2], lb, hb);
            // left channel is in low byte, right in high
            // values are unipolar, so we move them by half
            // of the dynamic range
            leftChannel[i] = lb - 128;
            rightChannel[i] = hb - 128;
        }
    }

    /**
     * Encodes the source data as an array of 16-bit values.
     *
     * @param source original signal source
     * @param data the data buffer to be written
     * @param dataSize size of the buffer
     */
    void WaveFileHandler::encode16bit(const SignalSource& source, short* data, std::size_t dataSize)
    {
        for (std::size_t i = 0; i < dataSize; ++i)
        {
            short sample = static_cast<short>(source.sample(i));
            data[i] = sample;
        }
    }

    /**
     * Encodes the source data as an array of 8-bit values stored in shorts.
     *
     * @param source original signal source
     * @param data the data buffer to be written
     * @param dataSize size of the buffer
     */
    void WaveFileHandler::encode8bit(const SignalSource& source, short* data, std::size_t dataSize)
    {
        for (std::size_t i = 0; i < dataSize; ++i)
        {
            unsigned char sample1 = static_cast<unsigned char>(source.sample(2 * i) + 128);
            unsigned char sample2 = static_cast<unsigned char>(source.sample(2 * i + 1) + 128);
            short hb = sample1, lb = sample2;
            data[i] = ((hb << 8) & 0xFF00) | (lb & 0x00FF);
        }
    }

    /**
     * Splits a 16-b number to lower and upper byte.
     *
     * @param twoBytes number to split
     * @param lb lower byte (by reference)
     * @param hb upper byte (by reference)
     */
    void WaveFileHandler::splitBytes(short twoBytes, unsigned char& lb, unsigned char& hb)
    {
        lb = twoBytes & 0x00FF;
        hb = (twoBytes >> 8) & 0x00FF;
    }
}
