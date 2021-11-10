/*
    Copyright 2005-2009 Intel Corporation.  All Rights Reserved.

    This file is part of Threading Building Blocks.

    Threading Building Blocks is free software; you can redistribute it
    and/or modify it under the terms of the GNU General Public License
    version 2 as published by the Free Software Foundation.

    Threading Building Blocks is distributed in the hope that it will be
    useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Threading Building Blocks; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

    As a special exception, you may use this file as part of a free software
    library without restriction.  Specifically, if other files instantiate
    templates or use macros or inline functions from this file, or you compile
    this file and link it with other files to produce an executable, this
    file does not by itself cause the resulting executable to be covered by
    the GNU General Public License.  This exception does not however
    invalidate any other reasons why the executable file might be covered by
    the GNU General Public License.
*/

#include "hunk.h"
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"

int Limit;
int Count;

struct Exception {};

class MinimalType: NoCopy {
    //! Used to detect use of bitwise copy instead of copy constructor
    MinimalType* my_self_ptr;
public:
    bool is_valid() const {return my_self_ptr==this;}
    MinimalType() {
        my_self_ptr = this;
        if( Count>=Limit ) 
            throw Exception();
        ++Count;
    }
    ~MinimalType() {
        my_self_ptr = NULL;
        --Count;
    }
};

typedef rml::internal::hunk<MinimalType> MyHunkType;

void SimpleTest() {
    // Trials differ in whether method clear is used or not before destruction
    for( int trial=0; trial<2; ++trial ) {
        for( int n=0; n<25; ++n ) {
            {
                MyHunkType h;
                Limit = n;
                ASSERT( !h.is_allocated(), NULL );
                h.allocate( n );
                ASSERT( Count==n, NULL );
                ASSERT( h.is_allocated(), NULL );
                ASSERT( h.size()==size_t(n), NULL );
                size_t k=0;
                for( MyHunkType::iterator i=h.begin(); i!=h.end(); ++i, ++k ) {
                    ASSERT( i->is_valid(), NULL );
                    MinimalType& element = h[k];
                    ASSERT( &element==h.begin()+k, NULL );
                }
                if( trial&1 ) {
                    h.clear();
                    ASSERT( Count==0, NULL );
                    ASSERT( !h.is_allocated(), NULL );
                }
            }
            ASSERT( Count==0, NULL );
        }
    }
}

void ExceptionTest() {
    for( Limit=0; Limit<25; ++Limit ) 
        for( int n=Limit+1; n<=Limit+25; ++n ) {
            MyHunkType h;
            ASSERT( Count==0, NULL );
            try {
                h.allocate( n );
            } catch( Exception ) {
                ASSERT( Count==0, "did not undo constructions properly?" );
            }
        }
}

int main() {
    SimpleTest();
    ExceptionTest();
    printf("done\n");
    return 0;
}
