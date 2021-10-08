#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include "Fontdefs.h"

#include "FTGL/ftgl.h"
#include "FTInternals.h"

extern void buildGLContext();

class FTOutlineFontTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTOutlineFontTest);
        CPPUNIT_TEST(testConstructor);
//        CPPUNIT_TEST(testRender);
        CPPUNIT_TEST(testBadDisplayList);
        CPPUNIT_TEST(testGoodDisplayList);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTOutlineFontTest() : CppUnit::TestCase("FTOutlineFont Test")
        {
        }

        FTOutlineFontTest(const std::string& name) : CppUnit::TestCase(name) {}

        ~FTOutlineFontTest()
        {
        }

        void testConstructor()
        {
            buildGLContext();

            FTOutlineFont* outlineFont = new FTOutlineFont(FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(outlineFont->Error(), 0);

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete outlineFont;
        }

        void testRender()
        {
            buildGLContext();

            FTOutlineFont* outlineFont = new FTOutlineFont(FONT_FILE);
            outlineFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(outlineFont->Error(), 0x97);   // Invalid pixels per em
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());

            outlineFont->FaceSize(18);
            outlineFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(outlineFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete outlineFont;
        }

        void testBadDisplayList()
        {
            buildGLContext();

            FTOutlineFont* outlineFont = new FTOutlineFont(FONT_FILE);
            outlineFont->FaceSize(18);

            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                outlineFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL((int)glGetError(), GL_INVALID_OPERATION);
            delete outlineFont;
        }

        void testGoodDisplayList()
        {
            buildGLContext();

            FTOutlineFont* outlineFont = new FTOutlineFont(FONT_FILE);
            outlineFont->FaceSize(18);

            outlineFont->UseDisplayList(false);
            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                outlineFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete outlineFont;
        }

        void setUp()
        {}

        void tearDown()
        {}

    private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(FTOutlineFontTest);

