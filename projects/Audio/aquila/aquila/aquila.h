/**
 * @file aquila.h
 *
 * Library "master" header - includes all component headers.
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
 *
 * @mainpage
 *
 * @section what-is-aquila What is Aquila?
 * Aquila is an open source and cross-platform DSP (Digital Signal Processing)
 * library for C++11.
 *
 * Aquila provides a set of classes for common DSP operations, such as FFT, DCT,
 * Mel-frequency filtering, calculating spectrograms etc. It supports reading
 * and writing signals in various formats, such as raw binary files, text files
 * or WAVE audio recordings.
 *
 * @section motivation Motivation
 * The initial goal of this project was to develop computer software capable
 * of recognizing birds' songs. Since then the library was redesigned and
 * extended with more general DSP tools. There are still a few major
 * shortcomings, for example the lack of general purpose filter classes, but
 * hopefully this will change soon.
 */

#ifndef AQUILA_H
#define AQUILA_H

#include "global.h"
#include "Exceptions.h"
#include "functions.h"
#include "source.h"
#include "transform.h"
#include "filter.h"
#include "ml.h"
#include "tools.h"

#endif // AQUILA_H
