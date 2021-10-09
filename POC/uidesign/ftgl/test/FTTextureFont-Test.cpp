#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include "Fontdefs.h"

#include "FTGL/ftgl.h"
#include "FTInternals.h"

extern void buildGLContext();

class FTTextureFontTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTTextureFontTest);
        CPPUNIT_TEST(testConstructor);
        CPPUNIT_TEST(testResizeBug);
        CPPUNIT_TEST(testRender);
        CPPUNIT_TEST(testDisplayList);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTTextureFontTest() : CppUnit::TestCase("FTTextureFontTest Test")
        {
        }

        FTTextureFontTest(const std::string& name) : CppUnit::TestCase(name) {}

        ~FTTextureFontTest()
        {
        }

        void testConstructor()
        {
            buildGLContext();

            FTTextureFont* textureFont = new FTTextureFont(FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(textureFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete textureFont;
        }

        void testResizeBug()
        {
            buildGLContext();

            FTTextureFont* textureFont = new FTTextureFont(FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(textureFont->Error(), 0);

            textureFont->FaceSize(18);
            textureFont->Render("first");

            textureFont->FaceSize(38);
            textureFont->Render("second");

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete textureFont;
        }

        void testRender()
        {
            buildGLContext();

            FTTextureFont* textureFont = new FTTextureFont(FONT_FILE);

            textureFont->Render(GOOD_ASCII_TEST_STRING);
            CPPUNIT_ASSERT_EQUAL(textureFont->Error(), 0x97);   // Invalid pixels per em
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());

            textureFont->FaceSize(18);
            textureFont->Render(GOOD_ASCII_TEST_STRING);

            CPPUNIT_ASSERT_EQUAL(textureFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete textureFont;
        }

        void testDisplayList()
        {
            buildGLContext();

            FTTextureFont* textureFont = new FTTextureFont(FONT_FILE);
            textureFont->FaceSize(18);

            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                textureFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete textureFont;
        }

        void setUp()
        {}

        void tearDown()
        {}

    private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(FTTextureFontTest);

