/**
 * @file source.h
 *
 * Convenience header that includes all signal source-related headers.
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


#ifndef AQUILA_SOURCE_H
#define AQUILA_SOURCE_H

#include "source/SignalSource.h"
#include "source/Frame.h"
#include "source/FramesCollection.h"
#include "source/PlainTextFile.h"
#include "source/RawPcmFile.h"
#include "source/WaveFile.h"
#include "source/WaveFileHandler.h"
#include "source/generator/Generator.h"
#include "source/generator/SineGenerator.h"
#include "source/generator/SquareGenerator.h"
#include "source/generator/TriangleGenerator.h"
#include "source/generator/PinkNoiseGenerator.h"
#include "source/generator/WhiteNoiseGenerator.h"
#include "source/window/BarlettWindow.h"
#include "source/window/BlackmanWindow.h"
#include "source/window/FlattopWindow.h"
#include "source/window/GaussianWindow.h"
#include "source/window/HammingWindow.h"
#include "source/window/HannWindow.h"
#include "source/window/RectangularWindow.h"

#endif // AQUILA_SOURCE_H
