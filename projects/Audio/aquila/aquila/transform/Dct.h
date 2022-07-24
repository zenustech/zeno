/**
 * @file Dct.h
 *
 * Discrete Cosine Transform (DCT) calculation.
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

#ifndef DCT_H
#define DCT_H

#include "../global.h"
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

namespace Aquila
{
    /**
     * An implementation of the Discrete Cosine Transform.
     */
    class AQUILA_EXPORT Dct
    {
    public:
        /**
         * Initializes the transform.
         */
        Dct():
            cosineCache()
        {
        }

        /**
         * Destroys the transform object.
         */
        ~Dct()
        {
            clearCosineCache();
        }

        std::vector<double> dct(const std::vector<double>& data, std::size_t outputLength);

    private:
        /**
         * Key type for the cache, using input and output length.
         */
        typedef std::pair<std::size_t, std::size_t> cosineCacheKeyType;

        /**
         * Cache type.
         */
        typedef std::map<cosineCacheKeyType, double**> cosineCacheType;

        /**
         * Cache object, implemented as a map.
         */
        cosineCacheType cosineCache;

        double** getCachedCosines(std::size_t inputLength, std::size_t outputLength);

        void clearCosineCache();
    };
}

#endif // DCT_H
