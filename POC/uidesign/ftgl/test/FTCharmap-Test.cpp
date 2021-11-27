#include <iostream>

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <cppunit/TestSuite.h>
#include <assert.h>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H


#include "Fontdefs.h"
#include "FTFace.h"
#include "FTCharmap.h"


class FTCharmapTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTCharmapTest);
        CPPUNIT_TEST(testConstructor);
        CPPUNIT_TEST(testSetEncoding);
        CPPUNIT_TEST(testGetGlyphListIndex);
//        CPPUNIT_TEST(testGetFontIndex);
//        CPPUNIT_TEST(testInsertCharacterIndex);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTCharmapTest() : CppUnit::TestCase("FTCharmap Test")
        {
            setUpFreetype();
        }

        FTCharmapTest(const std::string& name) : CppUnit::TestCase(name) {}

        ~FTCharmapTest()
        {
            tearDownFreetype();
        }


        void testConstructor()
        {
            CPPUNIT_ASSERT_EQUAL(0, charmap->Error());
            CPPUNIT_ASSERT_EQUAL(ft_encoding_unicode, charmap->Encoding());
        }


        void testSetEncoding()
        {
            CPPUNIT_ASSERT(charmap->CharMap(ft_encoding_unicode));

            CPPUNIT_ASSERT_EQUAL(0, charmap->Error());
            CPPUNIT_ASSERT_EQUAL(ft_encoding_unicode, charmap->Encoding());

            CPPUNIT_ASSERT(!charmap->CharMap(ft_encoding_johab));

            CPPUNIT_ASSERT_EQUAL(0x06, charmap->Error()); // invalid argument
            CPPUNIT_ASSERT_EQUAL(ft_encoding_unicode, charmap->Encoding());
        }


        void testGetGlyphListIndex()
        {
            charmap->CharMap(ft_encoding_johab);

            CPPUNIT_ASSERT_EQUAL(0x06, charmap->Error()); // invalid argument
            CPPUNIT_ASSERT_EQUAL(0U, charmap->GlyphListIndex(CHARACTER_CODE_A));
            CPPUNIT_ASSERT_EQUAL(0U, charmap->GlyphListIndex(BIG_CHARACTER_CODE));
            CPPUNIT_ASSERT_EQUAL(0U, charmap->GlyphListIndex(NULL_CHARACTER_CODE));

            charmap->CharMap(ft_encoding_unicode);

            CPPUNIT_ASSERT_EQUAL(0, charmap->Error());
            CPPUNIT_ASSERT_EQUAL(0U, charmap->GlyphListIndex(CHARACTER_CODE_A));
            CPPUNIT_ASSERT_EQUAL(0U, charmap->GlyphListIndex(BIG_CHARACTER_CODE));
            CPPUNIT_ASSERT_EQUAL(0U, charmap->GlyphListIndex(NULL_CHARACTER_CODE));

            // Check that the error flag is reset.
            charmap->CharMap(ft_encoding_johab);
            CPPUNIT_ASSERT_EQUAL(0x06, charmap->Error()); // invalid argument
            charmap->CharMap(ft_encoding_unicode);
            CPPUNIT_ASSERT_EQUAL(0, charmap->Error());
        }


        void testGetFontIndex()
        {
            charmap->CharMap(ft_encoding_johab);

            CPPUNIT_ASSERT_EQUAL(0x06, charmap->Error()); // invalid argument
            CPPUNIT_ASSERT_EQUAL(FONT_INDEX_OF_A, charmap->FontIndex(CHARACTER_CODE_A));
            CPPUNIT_ASSERT_EQUAL(BIG_FONT_INDEX, charmap->FontIndex(BIG_CHARACTER_CODE));
            CPPUNIT_ASSERT_EQUAL(NULL_FONT_INDEX, charmap->FontIndex(NULL_CHARACTER_CODE));
            charmap->CharMap(ft_encoding_unicode);

            CPPUNIT_ASSERT_EQUAL(0, charmap->Error());

            CPPUNIT_ASSERT_EQUAL(FONT_INDEX_OF_A, charmap->FontIndex(CHARACTER_CODE_A));
            CPPUNIT_ASSERT_EQUAL(BIG_FONT_INDEX, charmap->FontIndex(BIG_CHARACTER_CODE));
            CPPUNIT_ASSERT_EQUAL(NULL_FONT_INDEX, charmap->FontIndex(NULL_CHARACTER_CODE));

        }


        void testInsertCharacterIndex()
        {
            CPPUNIT_ASSERT_EQUAL(0U, charmap->GlyphListIndex(CHARACTER_CODE_A));
            CPPUNIT_ASSERT_EQUAL(FONT_INDEX_OF_A, charmap->FontIndex(CHARACTER_CODE_A));

            charmap->InsertIndex(69, CHARACTER_CODE_A);
            CPPUNIT_ASSERT_EQUAL(FONT_INDEX_OF_A, charmap->FontIndex(CHARACTER_CODE_A));
            CPPUNIT_ASSERT_EQUAL(69U, charmap->GlyphListIndex(CHARACTER_CODE_A));

            charmap->InsertIndex(999, CHARACTER_CODE_G);
            CPPUNIT_ASSERT_EQUAL(999U, charmap->GlyphListIndex(CHARACTER_CODE_G));
        }

        void setUp()
        {
            charmap = new FTCharmap(face);
        }


        void tearDown()
        {
            delete charmap;
        }

    private:
        FTFace*      face;
        FTCharmap* charmap;

        void setUpFreetype()
        {
            face = new FTFace(GOOD_FONT_FILE);
            CPPUNIT_ASSERT(!face->Error());
        }

        void tearDownFreetype()
        {
            delete face;
        }
};

CPPUNIT_TEST_SUITE_REGISTRATION(FTCharmapTest);

