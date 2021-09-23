#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include "Fontdefs.h"

#include "FTGL/ftgl.h"
#include "FTInternals.h"

extern void buildGLContext();

class FTExtrudeFontTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTExtrudeFontTest);
        CPPUNIT_TEST(testConstructor);
//        CPPUNIT_TEST(testRender);
        CPPUNIT_TEST(testBadDisplayList);
        CPPUNIT_TEST(testGoodDisplayList);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTExtrudeFontTest() : CppUnit::TestCase("FTExtrudeFont Test")
        {
        }

        FTExtrudeFontTest(const std::string& name) : CppUnit::TestCase(name) {}

        ~FTExtrudeFontTest()
        {
        }

        void testConstructor()
        {
            buildGLContext();

            FTExtrudeFont* extrudedFont = new FTExtrudeFont(FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(extrudedFont->Error(), 0);

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete extrudedFont;
        }

        void testRender()
        {
            buildGLContext();

            FTExtrudeFont* extrudedFont = new FTExtrudeFont(FONT_FILE);
            extrudedFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(extrudedFont->Error(), 0x97);   // Invalid pixels per em
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());

            extrudedFont->FaceSize(18);
            extrudedFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(extrudedFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete extrudedFont;
        }

        void testBadDisplayList()
        {
            buildGLContext();

            FTExtrudeFont* extrudedFont = new FTExtrudeFont(FONT_FILE);
            extrudedFont->FaceSize(18);

            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                extrudedFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL((int)glGetError(), GL_INVALID_OPERATION);
            delete extrudedFont;
        }

        void testGoodDisplayList()
        {
            buildGLContext();

            FTExtrudeFont* extrudedFont = new FTExtrudeFont(FONT_FILE);
            extrudedFont->FaceSize(18);

            extrudedFont->UseDisplayList(false);
            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                extrudedFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete extrudedFont;
        }

        void setUp()
        {}

        void tearDown()
        {}

    private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(FTExtrudeFontTest);

