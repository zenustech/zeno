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

#ifndef __RML_ring_H
#define __RML_ring_H

#include <vector>
#include <utility>
#include "tbb/cache_aligned_allocator.h"
#include "rml_base.h"
#include "job_automaton.h"

namespace rml {

namespace internal {

//! Circular sequence of pair(Connection&,job&) with a cursor.
template<typename Connection> 
class ring {
public:
    typedef Connection connection_type;
    typedef ::rml::job job_type;
    class value_type {
    public:
        Connection& connection() const {return *my_connection;}
        job_type& job() const {return *my_job;}
        job_automaton& automaton() const {return *my_automaton;}
        value_type( Connection& c, job_type& j, job_automaton& ja ) : my_connection(&c), my_job(&j), my_automaton(&ja) {}
    private:
        Connection* my_connection;
        job_type* my_job;
        job_automaton* my_automaton;
    };
private:
    typedef std::vector<value_type,tbb::cache_aligned_allocator<value_type> > array_type;
    /** Logically, a linked list would make more sense.  But it is not as cache friendly. */
    array_type my_array;
    typename array_type::const_iterator my_cursor;

public:
    ring() {
        my_cursor = my_array.begin();
    }

    //! Return number of items in the ring
    size_t size() const {return my_array.size();}

    //! Insert an new item into the sequence
    /** Where the item is inserted is implementation-dependent. */
    void insert( connection_type& client, job_type& job, job_automaton& ja ) {
        // Save cursor position
        typename array_type::difference_type i = my_cursor-my_array.begin();
        my_array.push_back(value_type(client,job,ja));
        // Restore cursor
        my_cursor = my_array.begin()+i;
    }

    //! Erase item in the sequence that corresponds to given connection_type.
    /** If item was present, returns pointer to job_automaton of that item.
        Otherwise returns NULL. */
    job_automaton* erase( const connection_type& c ) {
        // Search for matching item.
        typename array_type::iterator i = my_array.begin();
        for(;;) {
            if( i==my_array.end() )
                return NULL;
            if( &i->connection()==&c )
                break;
            ++i;
        }
        // Delete found item without corrupting cursor.
        typename array_type::difference_type j = my_cursor-my_array.begin();
        if( i<my_cursor ) {
            --j;
        } else if( my_cursor+1==my_array.end() ) {
            __TBB_ASSERT(i==my_cursor,NULL);
            j=0;
        }
        job_automaton* result = &i->automaton();
        __TBB_ASSERT( result, NULL );
        my_array.erase(i); 
        __TBB_ASSERT( !my_array.empty() || j==0, NULL );
        my_cursor=my_array.begin()+j;
        return result;
    }

    //! Apply client method to all jobs in the ring.
    /** Requres signature "Client& Connection::client()". 
        Templatized over Client so that method can be tested without having to define a full Client type. */
    template<typename Client>
    void for_each_job( void (Client::*method)(job&) ) {
        for( typename array_type::iterator i=my_array.begin(); i!=my_array.end(); ++i ) {
            value_type& v = *i;
            if( v.automaton().try_acquire() ) {
                (i->connection().client().*method)(v.job());
                v.automaton().release();
            }
        } 
    }

    //! NULL if ring is empty, otherwise pointer to value at current cursor position.
    const value_type* cursor() const {return my_array.empty() ? NULL : &*my_cursor;}

    //! Advance cursor to next item.
    void advance_cursor() {
        if( ++my_cursor==my_array.end() )
            my_cursor = my_array.begin();
    }

};

} // namespace internal
} // namespace rml

#endif /* __RML_ring_H */
