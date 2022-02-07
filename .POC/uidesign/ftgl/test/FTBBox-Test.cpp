#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include "Fontdefs.h"

#include "FTGL/ftgl.h"


class FTBBoxTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTBBoxTest);
        CPPUNIT_TEST(testDefaultConstructor);
        CPPUNIT_TEST(testGlyphConstructor);
        CPPUNIT_TEST(testBitmapConstructor);
        CPPUNIT_TEST(testMoveBBox);
        CPPUNIT_TEST(testPlusEquals);
        CPPUNIT_TEST(testSetDepth);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTBBoxTest() : CppUnit::TestCase("FTBBox Test")
        {}

        FTBBoxTest(const std::string& name) : CppUnit::TestCase(name) {}

        void testDefaultConstructor()
        {
            FTBBox boundingBox;

            CPPUNIT_ASSERT(boundingBox.Lower().X() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Lower().Y() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Lower().Z() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Upper().X() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Upper().Y() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Upper().Z() == 0.0f);
        }


        void testGlyphConstructor()
        {
            setUpFreetype(GOOD_FONT_FILE);

//            FTBBox boundingBox2((FT_GlyphSlot)(0));

//            CPPUNIT_ASSERT(boundingBox2.Lower().X() == 0.0f);
//            CPPUNIT_ASSERT(boundingBox2.Lower().Y() == 0.0f);
//            CPPUNIT_ASSERT(boundingBox2.Lower().Z() == 0.0f);
//            CPPUNIT_ASSERT(boundingBox2.Upper().X() == 0.0f);
//            CPPUNIT_ASSERT(boundingBox2.Upper().Y() == 0.0f);
//            CPPUNIT_ASSERT(boundingBox2.Upper().Z() == 0.0f);

            FTBBox boundingBox(face->glyph);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(2, boundingBox.Lower().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(-15, boundingBox.Lower().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0, boundingBox.Lower().Z(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(35, boundingBox.Upper().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(38, boundingBox.Upper().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0, boundingBox.Upper().Z(), 0.01);


            tearDownFreetype();
        }

        void testBitmapConstructor()
        {
            setUpFreetype(GOOD_FONT_FILE);

            FT_Load_Char(face, CHARACTER_CODE_G, FT_LOAD_MONOCHROME);

            CPPUNIT_ASSERT(ft_glyph_format_bitmap != face->glyph->format);

            FTBBox boundingBox3(face->glyph);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(2, boundingBox3.Lower().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(-15, boundingBox3.Lower().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0, boundingBox3.Lower().Z(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(35, boundingBox3.Upper().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(38, boundingBox3.Upper().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0, boundingBox3.Upper().Z(), 0.01);

        }

        void testMoveBBox()
        {
            FTBBox  boundingBox;
            FTPoint firstMove(3.5f, 1.0f, -2.5f);
            FTPoint secondMove(-3.5f, -1.0f, 2.5f);

            boundingBox += firstMove;

            CPPUNIT_ASSERT(boundingBox.Lower().X() ==  3.5f);
            CPPUNIT_ASSERT(boundingBox.Lower().Y() ==  1.0f);
            CPPUNIT_ASSERT(boundingBox.Lower().Z() == -2.5f);
            CPPUNIT_ASSERT(boundingBox.Upper().X() ==  3.5f);
            CPPUNIT_ASSERT(boundingBox.Upper().Y() ==  1.0f);
            CPPUNIT_ASSERT(boundingBox.Upper().Z() == -2.5f);

            boundingBox += secondMove;

            CPPUNIT_ASSERT(boundingBox.Lower().X() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Lower().Y() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Lower().Z() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Upper().X() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Upper().Y() == 0.0f);
            CPPUNIT_ASSERT(boundingBox.Upper().Z() == 0.0f);
        }

        void testPlusEquals()
        {
            setUpFreetype(GOOD_FONT_FILE);

            FTBBox boundingBox1;
            FTBBox boundingBox2(face->glyph);

            boundingBox1 |= boundingBox2;

            CPPUNIT_ASSERT_DOUBLES_EQUAL(2, boundingBox2.Lower().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(-15, boundingBox2.Lower().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0, boundingBox2.Lower().Z(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(35, boundingBox2.Upper().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(38, boundingBox2.Upper().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0, boundingBox2.Upper().Z(), 0.01);

            float advance  = 40;

            boundingBox2 += FTPoint(advance, 0, 0);
            boundingBox1 |= boundingBox2;

            CPPUNIT_ASSERT_DOUBLES_EQUAL(42, boundingBox2.Lower().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(-15, boundingBox2.Lower().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0, boundingBox2.Lower().Z(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(75, boundingBox2.Upper().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(38, boundingBox2.Upper().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0, boundingBox2.Upper().Z(), 0.01);

            tearDownFreetype();
        }

        void testSetDepth()
        {
            setUpFreetype(GOOD_FONT_FILE);

            FTBBox boundingBox(face->glyph);

            boundingBox.SetDepth(37.754);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(2, boundingBox.Lower().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(-15, boundingBox.Lower().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0, boundingBox.Lower().Z(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(35, boundingBox.Upper().X(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(38, boundingBox.Upper().Y(), 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(37.754, boundingBox.Upper().Z(), 0.01);

            tearDownFreetype();
        }

        void setUp()
        {}


        void tearDown()
        {}

    private:
        FT_Library   library;
        FT_Face      face;

        void setUpFreetype(const char *fontName)
        {
            FT_Error error = FT_Init_FreeType(&library);
            CPPUNIT_ASSERT(!error);
            error = FT_New_Face(library, fontName, 0, &face);
            CPPUNIT_ASSERT(!error);

            FT_Set_Char_Size(face, 0L, FONT_POINT_SIZE * 64, RESOLUTION, RESOLUTION);

            error = FT_Load_Char(face, CHARACTER_CODE_G, FT_LOAD_RENDER);
            CPPUNIT_ASSERT(!error);
        }

        void tearDownFreetype()
        {
            FT_Done_Face(face);
            FT_Done_FreeType(library);
        }

};

CPPUNIT_TEST_SUITE_REGISTRATION(FTBBoxTest);

