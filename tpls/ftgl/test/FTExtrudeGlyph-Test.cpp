#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include "Fontdefs.h"

#include "FTGL/ftgl.h"
#include "FTInternals.h"

extern void buildGLContext();

class FTExtrudeGlyphTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTExtrudeGlyphTest);
        CPPUNIT_TEST(testConstructor);
        CPPUNIT_TEST(testRender);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTExtrudeGlyphTest() : CppUnit::TestCase("FTExtrudeGlyph Test")
        {
        }

        FTExtrudeGlyphTest(const std::string& name) : CppUnit::TestCase(name) {}

        ~FTExtrudeGlyphTest()
        {
        }

        void testConstructor()
        {
            setUpFreetype();

            buildGLContext();

            FTExtrudeGlyph* extrudedGlyph = new FTExtrudeGlyph(face->glyph,
                                                      0.0f, 0.0f, 0.0f, true);
            CPPUNIT_ASSERT(extrudedGlyph->Error() == 0);

            CPPUNIT_ASSERT(glGetError() == GL_NO_ERROR);

            tearDownFreetype();
        }

        void testRender()
        {
            setUpFreetype();

            buildGLContext();

            FTExtrudeGlyph* extrudedGlyph = new FTExtrudeGlyph(face->glyph,
                                                      0.0f, 0.0f, 0.0f, true);
            CPPUNIT_ASSERT(extrudedGlyph->Error() == 0);
            extrudedGlyph->Render(FTPoint(0, 0, 0), FTGL::RENDER_FRONT |
                                                    FTGL::RENDER_BACK |
                                                    FTGL::RENDER_SIDE);

            CPPUNIT_ASSERT(glGetError() == GL_NO_ERROR);

            tearDownFreetype();
        }

        void setUp()
        {}

        void tearDown()
        {}

    private:
        FT_Library   library;
        FT_Face      face;

        void setUpFreetype()
        {
            FT_Error error = FT_Init_FreeType(&library);
            assert(!error);
            error = FT_New_Face(library, FONT_FILE, 0, &face);
            assert(!error);

            FT_Set_Char_Size(face, 0L, FONT_POINT_SIZE * 64, RESOLUTION, RESOLUTION);

            error = FT_Load_Char(face, CHARACTER_CODE_A, FT_LOAD_DEFAULT);
            assert(!error);
        }

        void tearDownFreetype()
        {
            FT_Done_Face(face);
            FT_Done_FreeType(library);
        }

};

CPPUNIT_TEST_SUITE_REGISTRATION(FTExtrudeGlyphTest);

