#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include "Fontdefs.h"

#include "FTGL/ftgl.h"
#include "FTInternals.h"

extern void buildGLContext();

class FTBitmapFontTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTBitmapFontTest);
        CPPUNIT_TEST(testConstructor);
        CPPUNIT_TEST(testRender);
        CPPUNIT_TEST(testPenPosition);
        CPPUNIT_TEST(testDisplayList);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTBitmapFontTest() : CppUnit::TestCase("FTBitmapFont Test")
        {
        }

        FTBitmapFontTest(const std::string& name) : CppUnit::TestCase(name) {}

        ~FTBitmapFontTest()
        {
        }

        void testConstructor()
        {
            buildGLContext();

            FTBitmapFont* bitmapFont = new FTBitmapFont(FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(bitmapFont->Error(), 0);

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete bitmapFont;
        }

        void testRender()
        {
            buildGLContext();

            FTBitmapFont* bitmapFont = new FTBitmapFont(FONT_FILE);
            bitmapFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(bitmapFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());

            bitmapFont->FaceSize(18);
            bitmapFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(bitmapFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete bitmapFont;
        }


        void testPenPosition()
        {
            buildGLContext();
            float rasterPosition[4];

            glRasterPos2f(0.0f,0.0f);

            glGetFloatv(GL_CURRENT_RASTER_POSITION, rasterPosition);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, rasterPosition[0], 0.01);

            FTBitmapFont* bitmapFont = new FTBitmapFont(FONT_FILE);
            bitmapFont->FaceSize(18);

            bitmapFont->Render(GOOD_ASCII_TEST_STRING);
            bitmapFont->Render(GOOD_ASCII_TEST_STRING);

            glGetFloatv(GL_CURRENT_RASTER_POSITION, rasterPosition);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(122, rasterPosition[0], 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, rasterPosition[1], 0.01);
            delete bitmapFont;
        }


        void testDisplayList()
        {
            buildGLContext();

            FTBitmapFont* bitmapFont = new FTBitmapFont(FONT_FILE);
            bitmapFont->FaceSize(18);

            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                bitmapFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete bitmapFont;
        }

        void setUp()
        {}

        void tearDown()
        {}

    private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(FTBitmapFontTest);

