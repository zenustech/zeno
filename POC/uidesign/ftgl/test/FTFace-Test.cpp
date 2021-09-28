#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/TestCaller.h"
#include "cppunit/TestCase.h"
#include "cppunit/TestSuite.h"

#include "Fontdefs.h"
#include "FTFace.h"


class FTFaceTest : public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(FTFaceTest);
        CPPUNIT_TEST(testOpenFace);
        CPPUNIT_TEST(testOpenFaceFromMemory);
        CPPUNIT_TEST(testAttachFile);
        CPPUNIT_TEST(testAttachMemoryData);
        CPPUNIT_TEST(testGlyphCount);
        CPPUNIT_TEST(testSetFontSize);
        CPPUNIT_TEST(testGetCharmapList);
        CPPUNIT_TEST(testKerning);
    CPPUNIT_TEST_SUITE_END();

    public:
        FTFaceTest() : CppUnit::TestCase("FTFace test") {};
        FTFaceTest(const std::string& name) : CppUnit::TestCase(name) {};


        void testOpenFace()
        {
            FTFace face1(BAD_FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(face1.Error(), 0x06);

            FTFace face2(GOOD_FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(face2.Error(), 0);
        }


        void testOpenFaceFromMemory()
        {
            FTFace face1((unsigned char*)100, 0);
            CPPUNIT_ASSERT_EQUAL(face1.Error(), 0x02);

            FTFace face2(HPGCalc_pfb.dataBytes, HPGCalc_pfb.numBytes);
            CPPUNIT_ASSERT_EQUAL(face2.Error(), 0);
        }


        void testAttachFile()
        {
            CPPUNIT_ASSERT(!testFace->Attach(TYPE1_AFM_FILE));
            CPPUNIT_ASSERT_EQUAL(testFace->Error(), 0x07); // unimplemented feature

            FTFace test(TYPE1_FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(test.Error(), 0);

            CPPUNIT_ASSERT(test.Attach(TYPE1_AFM_FILE));
            CPPUNIT_ASSERT_EQUAL(test.Error(), 0);
        }


        void testAttachMemoryData()
        {
            CPPUNIT_ASSERT(!testFace->Attach((unsigned char*)100, 0));
            CPPUNIT_ASSERT_EQUAL(testFace->Error(), 0x07); // unimplemented feature

            FTFace test(TYPE1_FONT_FILE);
            CPPUNIT_ASSERT_EQUAL(test.Error(), 0);

            CPPUNIT_ASSERT(test.Attach(HPGCalc_afm.dataBytes, HPGCalc_afm.numBytes));
            CPPUNIT_ASSERT_EQUAL(test.Error(), 0);
        }


        void testGlyphCount()
        {
            CPPUNIT_ASSERT_EQUAL(testFace->GlyphCount(), 14099U);
        }


        void testSetFontSize()
        {
            FTSize size = testFace->Size(FONT_POINT_SIZE, RESOLUTION);
            CPPUNIT_ASSERT_EQUAL(testFace->Error(), 0);
        }


        void testGetCharmapList()
        {
            CPPUNIT_ASSERT_EQUAL(testFace->CharMapCount(), 2U);

            FT_Encoding* charmapList = testFace->CharMapList();

            CPPUNIT_ASSERT_EQUAL(charmapList[0], ft_encoding_unicode);
            CPPUNIT_ASSERT_EQUAL(charmapList[1], ft_encoding_adobe_standard);
        }


        void testKerning()
        {
            FTFace test(ARIAL_FONT_FILE);
            FTPoint kerningVector = test.KernAdvance('A', 'A');
            CPPUNIT_ASSERT_EQUAL(kerningVector.X(), 0.);
            CPPUNIT_ASSERT_EQUAL(kerningVector.Y(), 0.);
            CPPUNIT_ASSERT_EQUAL(kerningVector.Z(), 0.);

            kerningVector = test.KernAdvance(0x6FB3, 0x9580);
            CPPUNIT_ASSERT_EQUAL(kerningVector.X(), 0.);
            CPPUNIT_ASSERT_EQUAL(kerningVector.Y(), 0.);
            CPPUNIT_ASSERT_EQUAL(kerningVector.Z(), 0.);
        }


        void setUp()
        {
            testFace = new FTFace(GOOD_FONT_FILE);
        }


        void tearDown()
        {
            delete testFace;
        }

    private:
        FTFace* testFace;

};

CPPUNIT_TEST_SUITE_REGISTRATION(FTFaceTest);

