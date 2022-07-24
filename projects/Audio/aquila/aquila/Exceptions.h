/**
 * @file Exceptions.h
 *
 * Exception class definitions.
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
 * @since 2.0.0
 */

#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include "global.h"
#include <stdexcept>
#include <string>

namespace Aquila
{
    /**
     * Base exception class of the library.
     *
     * Class clients should rather catch exceptions of specific types, such as
     * Aquila::FormatException, however it is allowed to catch Aquila::Exception
     * as the last resort (but catch(...)).
     */
    class AQUILA_EXPORT Exception : public std::runtime_error
    {
    public:
        /**
         * Creates an exception object.
         *
         * @param message exception message
         */
        Exception(const std::string& message):
            runtime_error(message)
        {
        }
    };

    /**
     * Data format-related exception.
     */
    class AQUILA_EXPORT FormatException : public Exception
    {
    public:
        /**
         * Creates a data format exception object.
         *
         * @param message exception message
         */
        FormatException(const std::string& message):
            Exception(message)
        {
        }
    };

    /**
     * Runtime configuration exception.
     */
    class AQUILA_EXPORT ConfigurationException : public Exception
    {
    public:
        /**
         * Creates a configuration exception object.
         *
         * @param message exception message
         */
        ConfigurationException(const std::string& message):
            Exception(message)
        {
        }
    };
}

#endif // EXCEPTIONS_H
