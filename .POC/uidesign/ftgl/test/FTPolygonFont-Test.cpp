#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include "Fontdefs.h"

#include "FTGL/ftgl.h"
#include "FTInternals.h"

extern void buildGLContext();

class FTPolygonFontTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTPolygonFontTest);
        CPPUNIT_TEST(testConstructor);
//        CPPUNIT_TEST(testRender);
        CPPUNIT_TEST(testBadDisplayList);
        CPPUNIT_TEST(testGoodDisplayList);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTPolygonFontTest() : CppUnit::TestCase("FTPolygonFont Test")
        {
        }

        FTPolygonFontTest(const std::string& name) : CppUnit::TestCase(name) {}

        ~FTPolygonFontTest()
        {
        }

        void testConstructor()
        {
            buildGLContext();

            FTPolygonFont* polygonFont = new FTPolygonFont(FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(polygonFont->Error(), 0);

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete polygonFont;
        }

        void testRender()
        {
            buildGLContext();

            FTPolygonFont* polygonFont = new FTPolygonFont(FONT_FILE);

            polygonFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(polygonFont->Error(), 0x97);   // Invalid pixels per em
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());

            polygonFont->FaceSize(18);
            polygonFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(polygonFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete polygonFont;
        }

        void testBadDisplayList()
        {
            buildGLContext();

            FTPolygonFont* polygonFont = new FTPolygonFont(FONT_FILE);
            polygonFont->FaceSize(18);

            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                polygonFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL((int)glGetError(), GL_INVALID_OPERATION);
            delete polygonFont;
        }

        void testGoodDisplayList()
        {
            buildGLContext();

            FTPolygonFont* polygonFont = new FTPolygonFont(FONT_FILE);
            polygonFont->FaceSize(18);

            polygonFont->UseDisplayList(false);
            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                polygonFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete polygonFont;
        }

        void setUp()
        {}

        void tearDown()
        {}

    private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(FTPolygonFontTest);

