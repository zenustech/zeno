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

#ifndef __RML_hunk_H
#define __RML_hunk_H

#include "tbb/cache_aligned_allocator.h"

namespace rml {

namespace internal {

//! Bare-bones container for sequence 
/* The sequence avoids false sharing of its elements with objects that are not in the sequence.
   The container works even for element types that are not assignable or copyable (unlike std::vctor<T>). */
template<typename T>
class hunk {
    //! Pointer to beginning of storage
    T* my_array;
    //! Pointer to end of storage
    T* my_end;
    typedef tbb::cache_aligned_allocator<T> allocator_type;
public:
    //! Iterator for iterating over the sequence
    typedef T* iterator;

    //! Construct hunk with no sequence
    hunk() : my_array(NULL), my_end(NULL) {}

    //! Allocate a sequence
    void allocate( size_t n );

    //! Deallocate a sequence
    /** Assertion failure if sequence was not allocated. */
    void clear();

    //! True if sequence as been allocated
    bool is_allocated() const {return my_array!=NULL;}

    //! Beginning of sequence
    iterator begin() {return my_array;}

    //! Beginning of sequence
    iterator end() {return my_end;}

    //! Return kth element
    T& operator[]( size_t k ) {
        __TBB_ASSERT( k<size(), NULL );
        return my_array[k];
    }

    //! Number of element in the sequence
    size_t size() const {return my_end-my_array;}

    ~hunk() {
        if( is_allocated() )
            clear();
    }
};

template<typename T>
void hunk<T>::allocate( size_t n ) { 
    my_array = allocator_type().allocate(n);
    my_end = my_array;
    try {
        T* limit = my_array+n;
        for( ; my_end!=limit; ++my_end ) 
            new(my_end) T;
    } catch(...) {
        clear();
        throw;
    }
}

template<typename T>
void hunk<T>::clear() {
    __TBB_ASSERT( is_allocated(), NULL );
    size_t n = size();
    while( my_end!=my_array ) 
        (--my_end)->T::~T();
    allocator_type().deallocate(my_array,n);
    my_array = NULL;
    my_end = NULL;
}

} // namespace internal
} // namespace rml

#endif /* __RML_hunk_H */
