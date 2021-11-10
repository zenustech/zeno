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

#include "ring.h"
#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"

/** Must be relatively prime to 17. */
const size_t N = 20;

struct MyJob: ::rml::job, private NoCopy {
    size_t tag;
};

size_t Count;

struct MyClient: NoAssign {
    //! Is virtual because what the test is testing is virtual in the real production case.
    virtual void tally( ::rml::job& job ) {
        ++Count;
        ASSERT( static_cast<MyJob&>(job).tag>0, NULL ); 
    }
    virtual ~MyClient() {}
};

struct MyConnection: private NoCopy {
    MyClient client() const {return MyClient();}
};

MyConnection Connections[N];

MyJob Job[N];
rml::internal::job_automaton JobAutomaton[N];

typedef rml::internal::ring<MyConnection> MyRingType;

void CheckSizeAndRandomizeCursor( MyRingType& r, size_t size ) {
    Count = 0;
    r.for_each_job( &MyClient::tally ) ;
    ASSERT( Count==size, NULL );
    if( size ) {
        // March cursor around to semi-random point
        size_t delta = size_t(rand()) % size;
        for( size_t m=0; m<delta; ++m )
            r.advance_cursor();
    }
}

#include <cstdlib>

int main() {
    srand(2);
    // Trials differ in randomization of cursor
    for( int trial=0; trial<20; ++trial ) {
        MyRingType r;
        ASSERT( r.cursor()==NULL, NULL );
        // Add some jobs
        for( size_t i=0; i<N; ++i ) {
            MyJob& job = Job[i*17%N];
            MyJob* current_job = static_cast<MyJob*>(r.cursor() ? &r.cursor()->job() : NULL);
            r.insert( Connections[i*61%N], job, JobAutomaton[i*17%N] );
            ASSERT( r.size()==i+1, NULL );
            if( current_job )
                ASSERT( &r.cursor()->job()==current_job, "insertion did not preserve cursor position" );
            else 
                ASSERT( &r.cursor()->job()==&job, "cursor not positioned after inserting into empty ring" );
            job.tag = i+1;
            CheckSizeAndRandomizeCursor(r,i+1);
            // Walk around ring once
            size_t sum = 0;
            for( size_t k=0; k<=i; ++k ) {
                // Use const object here for sake of testing that cursor() is const.
                const MyRingType& cr = r;
                sum += static_cast<MyJob&>(cr.cursor()->job()).tag;
                r.advance_cursor();
            }
            ASSERT( sum==(i+1)*(i+2)/2, NULL );
        }
        // Remove the jobs
        for( size_t k=0; k<N; ++k ) {
            // Get current and next cursor position.
            MyJob* current_job = static_cast<MyJob*>(&r.cursor()->job());
            r.advance_cursor();
            MyJob* next_job = static_cast<MyJob*>(&r.cursor()->job());
            // Reposition cursor
            for( size_t m=0; m<N-k-1; ++m )
                r.advance_cursor();
            ASSERT( current_job==&r.cursor()->job(), NULL );

            size_t i = (k*23+trial)%N;
            rml::internal::job_automaton* ja = r.erase( Connections[i*61%N] );
            rml::job* j = Job+(ja-JobAutomaton);
            ASSERT( r.size()==N-k-1, NULL );
            if( current_job!=j ) {
                ASSERT( &r.cursor()->job()==current_job, "erasure did not preserve cursor position" );
            } else if( k+1<N ) {
                ASSERT( next_job==&r.cursor()->job(), NULL );
            } else {
                ASSERT( !r.cursor(), NULL );
            }
            ASSERT( ja==JobAutomaton+i*17%N, "erase returned wrong job_automaton" );
            static_cast<MyJob*>(j)->tag = 0;
            CheckSizeAndRandomizeCursor(r,N-k-1);
        }
        ASSERT( r.cursor()==NULL, NULL );
    }
    // JobAutomaton objects must be plugged before being destroyed.
    for( size_t k=0; k<N; ++k ) {
        rml::job* j;
        JobAutomaton[k].try_plug(j);
    }
    printf("done\n");
    return 0;
}
