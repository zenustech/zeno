/*
 * FTGL - OpenGL font library
 *
 * Copyright (c) 2009 Sam Hocevar <sam@hocevar.net>
 *               2009 Mathew Eis (kingrobot)
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef     __FTCleanup__
#define     __FTCleanup__

#include <set>

#include "FTFace.h"

/**
 * A dummy object type for items to be stored in the cleanup list
 */
typedef void* FTCleanupObject;

/**
 * FTCleanup is used as a "callback" by FTLibrary to
 * make sure things are cleaned up in the right order
 */
class FTCleanup
{
    protected:

        /**
         * singleton instance
         */
        static FTCleanup* _instance;

        /**
         * Constructors
         */
         FTCleanup();

        /**
         * Destructor
         */
         ~FTCleanup();

    public:

        /**
         * Generate the instance if necessary and return it
         *
         * @return The FTCleanup instance
         */
        static FTCleanup* Instance()
        {
            if (FTCleanup::_instance == 0)
                FTCleanup::_instance = new FTCleanup;
            return FTCleanup::_instance;
        }

        /**
        * Destroy the FTCleanup instance
        */
        static void DestroyAll()
        {
            delete FTCleanup::_instance;
        }

        /**
         * Add an FT_Face to the cleanup list
         *
         * @param obj The reference to the FT_Face to be deleted on cleanup
         */
        void RegisterObject(FT_Face **obj);

        /**
         * Remove an FT_Face from the cleanup list
         *
         * @param obj The reference to the FT_Face to be removed from the list
         */
        void UnregisterObject(FT_Face **obj);

    private:

        std::set<FT_Face **> cleanupFT_FaceItems;
};

#endif  //  __FTCleanup__

