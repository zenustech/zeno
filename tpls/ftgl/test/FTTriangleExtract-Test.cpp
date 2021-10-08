#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include "Fontdefs.h"

#include "FTGL/ftgl.h"
#include "FTInternals.h"

extern void buildGLContext();

class FTTriangleExtractTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTTriangleExtractTest);
        CPPUNIT_TEST(testConstructor);
        CPPUNIT_TEST(testRender);
        CPPUNIT_TEST(testPenPosition);
        CPPUNIT_TEST(testDisplayList);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTTriangleExtractTest() : CppUnit::TestCase("FTBitmapFont Test")
        {
        }

        FTTriangleExtractTest(const std::string& name) : CppUnit::TestCase(name) {}

        ~FTTriangleExtractTest()
        {
        }

        void testConstructor()
        {
            buildGLContext();

            std::vector<float> vertices;
            FTGLTriangleExtractorFont* triangleFont = new FTGLTriangleExtractorFont(FONT_FILE, vertices);
            CPPUNIT_ASSERT_EQUAL(triangleFont->Error(), 0);

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete triangleFont;
        }

        void testRender()
        {
            buildGLContext();

			std::vector<float> vertices;
            FTGLTriangleExtractorFont* triangleFont = new FTGLTriangleExtractorFont(FONT_FILE, vertices);
            triangleFont->FaceSize(5);

            triangleFont->Render("test");
            CPPUNIT_ASSERT_EQUAL(triangleFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
			CPPUNIT_ASSERT(vertices.size() >= 3555);
			vertices.clear();

            triangleFont->Render(GOOD_ASCII_TEST_STRING);
            CPPUNIT_ASSERT_EQUAL(triangleFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
			CPPUNIT_ASSERT(vertices.size() >= 10000);
			vertices.clear();

            triangleFont->FaceSize(18);
            triangleFont->Render(GOOD_ASCII_TEST_STRING);
            CPPUNIT_ASSERT_EQUAL(triangleFont->Error(), 0);
            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
			CPPUNIT_ASSERT(vertices.size() >= 10000);

            delete triangleFont;
        }


        void testPenPosition()
        {
            buildGLContext();
            float rasterPosition[4];

            glRasterPos2f(0.0f,0.0f);

            glGetFloatv(GL_CURRENT_RASTER_POSITION, rasterPosition);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, rasterPosition[0], 0.01);

            std::vector<float> vertices;
            FTGLTriangleExtractorFont* triangleFont = new FTGLTriangleExtractorFont(FONT_FILE, vertices);
            triangleFont->FaceSize(18);

            triangleFont->Render(GOOD_ASCII_TEST_STRING);
			CPPUNIT_ASSERT(vertices.size() >= 10000);
            triangleFont->Render(GOOD_ASCII_TEST_STRING);
			CPPUNIT_ASSERT(vertices.size() >= 20000);

//            glGetFloatv(GL_CURRENT_RASTER_POSITION, rasterPosition);
//            CPPUNIT_ASSERT_DOUBLES_EQUAL(122, rasterPosition[0], 0.01);
//            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, rasterPosition[1], 0.01);
            delete triangleFont;
        }


        void testDisplayList()
        {
            buildGLContext();

            std::vector<float> vertices;
            FTGLTriangleExtractorFont* triangleFont = new FTGLTriangleExtractorFont(FONT_FILE, vertices);
            triangleFont->FaceSize(18);

            int glList = glGenLists(1);
            glNewList(glList, GL_COMPILE);

                triangleFont->Render(GOOD_ASCII_TEST_STRING);

            glEndList();

            CPPUNIT_ASSERT_EQUAL(GL_NO_ERROR, (int)glGetError());
            delete triangleFont;
        }

        void setUp()
        {}

        void tearDown()
        {}

    private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(FTTriangleExtractTest);

