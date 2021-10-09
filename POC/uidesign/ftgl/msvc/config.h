/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2008 Jeff Myers <JeffM2501@users.sourceforge.net>
 * Copyright (c) 2008 Daniel Remenak <dtremenak@users.sourceforge.net>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// Default font file
#define FONT_FILE "C:\\Windows\\Fonts\\Arial.ttf"

// GLUT
#define HAVE_GL_GLUT_H

// M_PI and friends on VC
#define _USE_MATH_DEFINES

// quell spurious "'this': used in base member initializer list" warnings
#ifdef _MSC_VER
#pragma warning(disable: 4355)
#endif

// quell spurious portable-function deprecation warnings
#define _CRT_SECURE_NO_DEPRECATE 1
#define _POSIX_ 1

// use __FUNCTION__
#define __FUNC__ __FUNCTION__

#define PACKAGE_VERSION "2.3.0"
