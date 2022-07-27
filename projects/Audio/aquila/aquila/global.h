/**
 * @file global.h
 *
 * Global library typedefs and constants.
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
 * @since 2.4.1
 */

#ifndef GLOBAL_H
#define GLOBAL_H

#include <complex>
#include <vector>

#if defined (_WIN32) && defined(BUILD_SHARED_LIBS)
#  if defined(Aquila_EXPORTS)
#    define AQUILA_EXPORT  __declspec(dllexport)
#  else
#    define AQUILA_EXPORT  __declspec(dllimport)
#  endif
#else
#    define AQUILA_EXPORT
#endif

/**
 * Main library namespace.
 */
namespace Aquila
{
    /**
     * Library version in an easily comparable format.
     */
    const long VERSION_NUMBER = 0x300000;

    /**
     * Library version as a string.
     */
    const char* const VERSION_STRING = "3.0.0-dev";

    /**
     * Sample value type.
     */
    typedef double SampleType;

    /**
     * Sample frequency type.
     */
    typedef double FrequencyType;

    /**
     * Our standard complex number type, using double precision.
     */
    typedef std::complex<double> ComplexType;

    /**
     * Spectrum type - a vector of complex values.
     */
    typedef std::vector<ComplexType> SpectrumType;
}

#endif // GLOBAL_H
