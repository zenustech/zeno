


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/cmake)


ADD_DEFINITIONS(-DPACKAGE_VERSION="${VERSION_SERIES}.${VERSION_MAJOR}.${VERSION_MINOR}")

IF(WIN32)
	ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
ENDIF()

SET(libftgl_la_SOURCES
    FTBuffer.cpp
    FTCharmap.cpp
    FTCharmap.h
    FTCharToGlyphIndexMap.h
    FTCleanup.cpp
    FTCleanup.h
    FTContour.cpp
    FTContour.h
    FTFace.cpp
    FTFace.h
    FTGL.cpp
    FTGlyphContainer.cpp
    FTGlyphContainer.h
    FTInternals.h
    FTLibrary.cpp
    FTLibrary.h
    FTList.h
    FTPoint.cpp
    FTSize.cpp
    FTSize.h
    FTVector.h
    FTVectoriser.cpp
    FTVectoriser.h
    FTUnicode.h
)

SET(ftgl_headers
    FTGL/ftgl.h
    FTGL/FTBBox.h
    FTGL/FTBuffer.h
    FTGL/FTPoint.h
    FTGL/FTGlyph.h
    FTGL/FTBitmapGlyph.h
    FTGL/FTBufferGlyph.h
    FTGL/FTExtrdGlyph.h
    FTGL/FTOutlineGlyph.h
    FTGL/FTPixmapGlyph.h
    FTGL/FTPolyGlyph.h
    FTGL/FTTextureGlyph.h
    FTGL/FTFont.h
    FTGL/FTGLBitmapFont.h
    FTGL/FTBufferFont.h
    FTGL/FTGLExtrdFont.h
    FTGL/FTGLOutlineFont.h
    FTGL/FTGLPixmapFont.h
    FTGL/FTGLPolygonFont.h
    FTGL/FTGLTextureFont.h
    FTGL/FTGLTriangleExtractorFont.h
    FTGL/FTLayout.h
    FTGL/FTSimpleLayout.h
)

SET(ftglyph_sources
    FTGlyph/FTGlyph.cpp
    FTGlyph/FTGlyphImpl.h
    FTGlyph/FTGlyphGlue.cpp
    FTGlyph/FTBitmapGlyph.cpp
    FTGlyph/FTBitmapGlyphImpl.h
    FTGlyph/FTBufferGlyph.cpp
    FTGlyph/FTBufferGlyphImpl.h
    FTGlyph/FTExtrudeGlyph.cpp
    FTGlyph/FTExtrudeGlyphImpl.h
    FTGlyph/FTOutlineGlyph.cpp
    FTGlyph/FTOutlineGlyphImpl.h
    FTGlyph/FTPixmapGlyph.cpp
    FTGlyph/FTPixmapGlyphImpl.h
    FTGlyph/FTPolygonGlyph.cpp
    FTGlyph/FTPolygonGlyphImpl.h
    FTGlyph/FTTextureGlyph.cpp
    FTGlyph/FTTextureGlyphImpl.h
    FTGlyph/FTTriangleExtractorGlyph.cpp
    FTGlyph/FTTriangleExtractorGlyphImpl.h
)

SET(ftfont_sources
    FTFont/FTFont.cpp
    FTFont/FTFontImpl.h
    FTFont/FTFontGlue.cpp
    FTFont/FTBitmapFont.cpp
    FTFont/FTBitmapFontImpl.h
    FTFont/FTBufferFont.cpp
    FTFont/FTBufferFontImpl.h
    FTFont/FTExtrudeFont.cpp
    FTFont/FTExtrudeFontImpl.h
    FTFont/FTOutlineFont.cpp
    FTFont/FTOutlineFontImpl.h
    FTFont/FTPixmapFont.cpp
    FTFont/FTPixmapFontImpl.h
    FTFont/FTPolygonFont.cpp
    FTFont/FTPolygonFontImpl.h
    FTFont/FTTextureFont.cpp
    FTFont/FTTextureFontImpl.h
    FTFont/FTTriangleExtractorFont.cpp
    FTFont/FTTriangleExtractorFontImpl.h
)

SET(ftlayout_sources
    FTLayout/FTLayout.cpp
    FTLayout/FTLayoutImpl.h
    FTLayout/FTLayoutGlue.cpp
    FTLayout/FTSimpleLayout.cpp
    FTLayout/FTSimpleLayoutImpl.h
)


ADD_LIBRARY(ftgl ${libftgl_la_SOURCES} ${ftgl_headers} ${ftglyph_sources} ${ftfont_sources} ${ftlayout_sources})
TARGET_LINK_LIBRARIES(ftgl
	${FREETYPE_LIBRARIES}
	${OPENGL_LIBRARIES}
)

IF(NOT BUILD_SHARED_LIBS)
	TARGET_LINK_LIBRARIES(ftgl ${OPENGL_LIBRARIES})
ENDIF()

SET_TARGET_PROPERTIES(
    ftgl
  PROPERTIES
    VERSION ${VERSION_SERIES}.${VERSION_MAJOR}.${VERSION_MINOR}
	SOVERSION ${FTGL_SOVERSION}
    DEBUG_POSTFIX "d"
)

INSTALL(TARGETS   ftgl                         	  
    EXPORT  FTGL-targets
	RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/FTGL/*.h")
INSTALL(FILES ${files} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/FTGL)

