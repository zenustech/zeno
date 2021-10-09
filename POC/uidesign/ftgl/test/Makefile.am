
if HAVE_CPPUNIT
if HAVE_GLUT
cppunit_tests = CTest CXXTest
endif
endif

noinst_PROGRAMS = $(cppunit_tests)
TESTS = $(cppunit_tests)

CXXTest_SOURCES = \
    $(DEACTIVATED) \
    CXXTest.cpp \
    Fontdefs.h \
    FTBBox-Test.cpp \
    FTBitmapFont-Test.cpp \
    FTBitmapGlyph-Test.cpp \
    FTCharmap-Test.cpp \
    FTCharToGlyphIndexMap-Test.cpp \
    FTContour-Test.cpp \
    FTExtrudeFont-Test.cpp \
    FTExtrudeGlyph-Test.cpp \
    FTFace-Test.cpp \
    FTFont-Test.cpp \
    FTGlyph-Test.cpp \
    FTGlyphContainer-Test.cpp \
    FTlayout-Test.cpp \
    FTLibrary-Test.cpp \
    FTList-Test.cpp \
    FTMesh-Test.cpp \
    FTOutlineFont-Test.cpp \
    FTOutlineGlyph-Test.cpp \
    FTPixmapFont-Test.cpp \
    FTPixmapGlyph-Test.cpp \
    FTPoint-Test.cpp \
    FTPolygonFont-Test.cpp \
    FTPolygonGlyph-Test.cpp \
    FTSize-Test.cpp \
    FTTesselation-Test.cpp \
    FTTextureFont-Test.cpp \
    FTTextureGlyph-Test.cpp \
    FTVectoriser-Test.cpp \
    FTVector-Test.cpp \
    HPGCalc_afm.cpp \
    HPGCalc_pfb.cpp \
    $(NULL)

AM_CPPFLAGS = \
    $(FT2_CPPFLAGS) \
    -I$(top_srcdir)/src \
    -I$(top_srcdir)/src/FTFont \
    -I$(top_srcdir)/src/FTGlyph \
    -I$(top_srcdir)/src/FTLayout \
    $(NULL)

CXXTest_CXXFLAGS = $(FT2_CFLAGS) $(GL_CFLAGS)
CXXTest_LDFLAGS = $(FT2_LIBS) $(GLUT_LIBS) -lcppunit
CXXTest_LDADD = ../src/libftgl.la

CTest_SOURCES = \
    CTest.c \
    $(NULL)
CTest_CPPFLAGS = \
    -I$(top_srcdir)/include \
    -I$(top_srcdir)/src \
    -I$(top_srcdir)/src/FTGlyph \
    -I$(top_srcdir)/src/FTFont \
    -I$(top_srcdir)/src/FTLayout
CTest_CFLAGS = $(FT2_CFLAGS) $(GL_CFLAGS)
CTest_LDFLAGS = $(FT2_LIBS) $(GLUT_LIBS)
CTest_LDADD = ../src/libftgl.la

MAINTAINERCLEANFILES = Makefile.in

NULL =

