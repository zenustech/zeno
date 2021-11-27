Using FTGL and Visual Studio
============================

FTGL on windows can be built a ether a dynamic link library (DLL) with export
lib (lib) or a static library (lib). All files will be built in the "build"
subdirectory that will be created in this directory.

FTGL requires the Freetype2 library (version 2.3.9).

Set up FreeType2
----------------

You will need to download the FreeType sources from http://www.freetype.org/,
or directly from git://git.sv.nongnu.org/freetype/freetype2.git if you have
installed Git.

Open the Visual Studio solution and build FreeType.

Finally, define the FREETYPE environment variable to the full path to your
FreeType sources, eg. C:\Users\john\Desktop\freetype

Build FTGL
----------

The "vc9" directory contains projects for use with Visual Studio 2008. The
"vc8" directory contains projects for Visual Studio 2005. These projects can
build both the dynamic and static libraries.

The "vc71" directory contains projects for use with Visual C++ 2003 and can
only build a dynamic library. It is no longer supported.

Use FTGL
--------

To use FTGL in your own projects you will need to link against either the
static lib, or the DLL export lib. All builds use the multithreaded runtime.
If built with the export lib, your project will need to ship with the
FTGL .dll file in order to be usable.

Your project will also need to include the freetype2 and OpenGL .dll files
in order to be usable.

For instructions on using Freetype go to www.freetype.org
For instructions on using OpenGL go to www.opengl.org
