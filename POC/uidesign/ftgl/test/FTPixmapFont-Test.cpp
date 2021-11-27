#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include "Fontdefs.h"

#include "FTGL/ftgl.h"
#include "FTInternals.h"

extern void buildGLContext();

class FTPixmapFontTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTPixmapFontTest);
        CPPUNIT_TEST(testConstructor);
//        CPPUNIT_TEST(testRender);
        CPPUNIT_TEST(testDisplayList);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTPixmapFontTest() : CppUnit::TestCase("FTPixmapFont Test")
        {
        }

        FTPixmapFontTest(const std::string& name) : CppUnit::TestCase(name) {}

        ~FTPixmapFontTest()
        {
        }

        void testConstructor()
        {
            buildGLContext();

            FTPixmapFont* pixmapFont = new FTPixmapFont(FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(pixmapFont->Error(), 0);

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete pixmapFont;
        }

        void testRender()
        {
            buildGLContext();

            FTPixmapFont* pixmapFont = new FTPixmapFont(FONT_FILE);

            pixmapFont->Render(GOOD_ASCII_TEST_STRING);
            CPPUNIT_ASSERT_EQUAL(pixmapFont->Error(), 0x97);   // Invalid pixels per em
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());

            pixmapFont->FaceSize(18);
            pixmapFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(pixmapFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete pixmapFont;
        }

        void testDisplayList()
        {
            buildGLContext();

            FTPixmapFont* pixmapFont = new FTPixmapFont(FONT_FILE);
            pixmapFont->FaceSize(18);

            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                pixmapFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete pixmapFont;
        }

        void setUp()
        {}

        void tearDown()
        {}

    private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(FTPixmapFontTest);

