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

#include "tbb/concurrent_vector.h"
#include "tbb/cache_aligned_allocator.h"
#include "tbb/tbb_exception.h"
#include "tbb_misc.h"
#include "itt_notify.h"
#include <cstring>

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (disable: 4267)
#endif

using namespace std;

namespace tbb {

namespace internal {
class concurrent_vector_base_v3::helper {
public:
    //! memory page size
    static const size_type page_size = 4096;

    inline static bool incompact_predicate(size_type size) { // assert size != 0, see source/test/test_vector_layout.cpp
        return size < page_size || ((size-1)%page_size < page_size/2 && size < page_size * 128); // for more details
    }

    inline static size_type find_segment_end(const concurrent_vector_base_v3 &v) {
        segment_t *s = v.my_segment;
        segment_index_t u = s==v.my_storage? pointers_per_short_table : pointers_per_long_table;
        segment_index_t k = 0;
        while( k < u && s[k].array > __TBB_BAD_ALLOC )
            ++k;
        return k;
    }

    //! assign first segment size. k - is index of last segment to be allocated, not a count of segments
    static void assign_first_segment_if_neccessary(concurrent_vector_base_v3 &v, segment_index_t k) {
        if( !v.my_first_block ) {
            /* There was a suggestion to set first segment according to incompact_predicate:
            while( k && !helper::incompact_predicate(segment_size( k ) * element_size) )
                --k; // while previous vector size is compact, decrement
            // reasons to not do it:
            // * constructor(n) is not ready to accept fragmented segments
            // * backward compatibility due to that constructor
            // * current version gives additional guarantee and faster init.
            // * two calls to reserve() will give the same effect.
            */
            v.my_first_block.compare_and_swap(k+1, 0); // store number of segments
        }
    }

    inline static void *allocate_segment(concurrent_vector_base_v3 &v, size_type n) {
        void *ptr = v.vector_allocator_ptr(v, n);
        if(!ptr) throw bad_alloc(); // check for bad allocation, throw exception
        return ptr;
    }

    //! Publish segment so other threads can see it.
    inline static void publish_segment( segment_t& s, void* rhs ) {
    // see also itt_store_pointer_with_release_v3()
        ITT_NOTIFY( sync_releasing, &s.array );
        __TBB_store_with_release( s.array, rhs );
    }

    static size_type enable_segment(concurrent_vector_base_v3 &v, size_type k, size_type element_size) {
        __TBB_ASSERT( v.my_segment[k].array <= __TBB_BAD_ALLOC, "concurrent operation during growth?" );
        if( !k ) {
            assign_first_segment_if_neccessary(v, default_initial_segments-1);
            try {
                publish_segment(v.my_segment[0], allocate_segment(v, segment_size(v.my_first_block) ) );
            } catch(...) { // intercept exception here, assign __TBB_BAD_ALLOC value, re-throw exception
                publish_segment(v.my_segment[0], __TBB_BAD_ALLOC); throw;
            }
            return 2;
        }
        size_type m = segment_size(k);
        if( !v.my_first_block ) // push_back only
            spin_wait_while_eq( v.my_first_block, segment_index_t(0) );
        if( k < v.my_first_block ) {
            segment_t* s = v.my_segment;
            // s[0].array is changed only once ( 0 -> !0 ) and points to uninitialized memory
            void *array0 = __TBB_load_with_acquire(s[0].array);
            if( !array0 ) {
                // sync_prepare called only if there is a wait
                ITT_NOTIFY(sync_prepare, &s[0].array );
                spin_wait_while_eq( s[0].array, (void*)0 );
                array0 = __TBB_load_with_acquire(s[0].array);
            }
            ITT_NOTIFY(sync_acquired, &s[0].array);
            if( array0 <= __TBB_BAD_ALLOC ) { // check for __TBB_BAD_ALLOC of initial segment
                publish_segment(s[k], __TBB_BAD_ALLOC); // and assign __TBB_BAD_ALLOC here
                throw bad_last_alloc(); // throw custom exception
            }
            publish_segment( s[k],
                    static_cast<void*>( static_cast<char*>(array0) + segment_base(k)*element_size )
            );
        } else {
            try {
                publish_segment(v.my_segment[k], allocate_segment(v, m));
            } catch(...) { // intercept exception here, assign __TBB_BAD_ALLOC value, re-throw exception
                publish_segment(v.my_segment[k], __TBB_BAD_ALLOC); throw;
            }
        }
        return m;
    }

    inline static void extend_table_if_necessary(concurrent_vector_base_v3 &v, size_type k) {
        if(k >= pointers_per_short_table && v.my_segment == v.my_storage)
            extend_segment_table(v);
    }

    static void extend_segment_table(concurrent_vector_base_v3 &v) {
        segment_t* s = (segment_t*)NFS_Allocate( pointers_per_long_table, sizeof(segment_t), NULL );
        // if( !s ) throw bad_alloc() -- implemented in NFS_Allocate
        memset( s, 0, pointers_per_long_table*sizeof(segment_t) );
        // If other threads are trying to set pointers in the short segment, wait for them to finish their
        // assigments before we copy the short segment to the long segment. Note: grow_to_at_least depends on it
        for(segment_index_t i = 0; i < pointers_per_short_table; i++)
            if(!v.my_storage[i].array)
                spin_wait_while_eq(v.my_storage[i].array, (void*)0);

        for( segment_index_t i = 0; i < pointers_per_short_table; i++)
            s[i] = v.my_storage[i];
        if( v.my_segment.compare_and_swap( s, v.my_storage ) != v.my_storage )
            NFS_Free( s );
    }
};

concurrent_vector_base_v3::~concurrent_vector_base_v3() {
    segment_t* s = my_segment;
    if( s != my_storage ) {
        // Clear short segment.
        for( segment_index_t i = 0; i < pointers_per_short_table; i++)
            my_storage[i].array = NULL;
#if 0 && TBB_USE_DEBUG // TODO: current copy constructor couses it to fail
        for( segment_index_t i = 0; i < pointers_per_long_table; i++)
            __TBB_ASSERT( my_segment[i].array <= __TBB_BAD_ALLOC, "should have been freed by clear");
#endif
        my_segment = my_storage;
        NFS_Free( s );
    }
}

concurrent_vector_base_v3::size_type concurrent_vector_base_v3::internal_capacity() const {
    return segment_base( helper::find_segment_end(*this) );
}

void concurrent_vector_base_v3::internal_throw_exception(size_type t) const {
    switch(t) {
        case 0: throw out_of_range("Index out of requested size range");
        case 1: throw range_error ("Index out of allocated segment slots");
        case 2: throw range_error ("Index is not allocated");
    }
}

void concurrent_vector_base_v3::internal_reserve( size_type n, size_type element_size, size_type max_size ) {
    if( n>max_size ) {
        throw length_error("argument to ConcurrentVector::reserve exceeds ConcurrentVector::max_size()");
    }
    __TBB_ASSERT( n, NULL );
    helper::assign_first_segment_if_neccessary(*this, segment_index_of(n-1));
    segment_index_t k = helper::find_segment_end(*this);
    try {
        for( ; segment_base(k)<n; ++k ) {
            helper::extend_table_if_necessary(*this, k);
            if(my_segment[k].array <= __TBB_BAD_ALLOC)
                helper::enable_segment(*this, k, element_size);
        }
    } catch(...) {
        my_segment[k].array = NULL; throw; // repair and rethrow
    }
}

void concurrent_vector_base_v3::internal_copy( const concurrent_vector_base_v3& src, size_type element_size, internal_array_op2 copy ) {
    size_type n = src.my_early_size;
    my_early_size = n;
    my_segment = my_storage;
    if( n ) {
        helper::assign_first_segment_if_neccessary(*this, segment_index_of(n));
        size_type b;
        for( segment_index_t k=0; (b=segment_base(k))<n; ++k ) {
            if( (src.my_segment == (segment_t*)src.my_storage && k >= pointers_per_short_table)
                || src.my_segment[k].array <= __TBB_BAD_ALLOC ) {
                my_early_size = b; break;
            }
            helper::extend_table_if_necessary(*this, k);
            size_type m = helper::enable_segment(*this, k, element_size);
            if( m > n-b ) m = n-b; 
            copy( my_segment[k].array, src.my_segment[k].array, m );
        }
    }
}

void concurrent_vector_base_v3::internal_assign( const concurrent_vector_base_v3& src, size_type element_size, internal_array_op1 destroy, internal_array_op2 assign, internal_array_op2 copy ) {
    size_type n = src.my_early_size;
    while( my_early_size>n ) { 
        segment_index_t k = segment_index_of( my_early_size-1 );
        size_type b=segment_base(k);
        size_type new_end = b>=n ? b : n;
        __TBB_ASSERT( my_early_size>new_end, NULL );
        if( my_segment[k].array <= __TBB_BAD_ALLOC) // check vector was broken before
            throw bad_last_alloc(); // throw custom exception
        // destructors are supposed to not throw any exceptions
        destroy( (char*)my_segment[k].array+element_size*(new_end-b), my_early_size-new_end );
        my_early_size = new_end;
    }
    size_type dst_initialized_size = my_early_size;
    my_early_size = n;
    helper::assign_first_segment_if_neccessary(*this, segment_index_of(n));
    size_type b;
    for( segment_index_t k=0; (b=segment_base(k))<n; ++k ) {
        helper::extend_table_if_necessary(*this, k);
        if(!my_segment[k].array)
            helper::enable_segment(*this, k, element_size);
        if( (src.my_segment == (segment_t*)src.my_storage && k >= pointers_per_short_table)
            || src.my_segment[k].array <= __TBB_BAD_ALLOC ) { // if source is damaged
                my_early_size = b; break;
        }
        size_type m = k? segment_size(k) : 2;
        if( m > n-b ) m = n-b;
        size_type a = 0;
        if( dst_initialized_size>b ) {
            a = dst_initialized_size-b;
            if( a>m ) a = m;
            assign( my_segment[k].array, src.my_segment[k].array, a );
            m -= a;
            a *= element_size;
        }
        if( m>0 )
            copy( (char*)my_segment[k].array+a, (char*)src.my_segment[k].array+a, m );
    }
    __TBB_ASSERT( src.my_early_size==n, "detected use of ConcurrentVector::operator= with right side that was concurrently modified" );
}

void* concurrent_vector_base_v3::internal_push_back( size_type element_size, size_type& index ) {
    __TBB_ASSERT( sizeof(my_early_size)==sizeof(uintptr), NULL );
    size_type tmp = __TBB_FetchAndIncrementWacquire(&my_early_size);
    index = tmp;
    segment_index_t k_old = segment_index_of( tmp );
    size_type base = segment_base(k_old);
    helper::extend_table_if_necessary(*this, k_old);
    segment_t& s = my_segment[k_old];
    if( !__TBB_load_with_acquire(s.array) ) { // do not check for __TBB_BAD_ALLOC because it's hard to recover after __TBB_BAD_ALLOC correctly
        if( base==tmp ) {
            helper::enable_segment(*this, k_old, element_size);
        } else {
            ITT_NOTIFY(sync_prepare, &s.array);
            spin_wait_while_eq( s.array, (void*)0 );
            ITT_NOTIFY(sync_acquired, &s.array);
        }
    } else {
        ITT_NOTIFY(sync_acquired, &s.array);
    }
    if( s.array <= __TBB_BAD_ALLOC ) // check for __TBB_BAD_ALLOC
        throw bad_last_alloc(); // throw custom exception
    size_type j_begin = tmp-base;
    return (void*)((char*)s.array+element_size*j_begin);
}

void concurrent_vector_base_v3::internal_grow_to_at_least( size_type new_size, size_type element_size, internal_array_op2 init, const void *src ) {
    internal_grow_to_at_least_with_result( new_size, element_size, init, src );
}

concurrent_vector_base_v3::size_type concurrent_vector_base_v3::internal_grow_to_at_least_with_result( size_type new_size, size_type element_size, internal_array_op2 init, const void *src ) {
    size_type e = my_early_size;
    while( e<new_size ) {
        size_type f = my_early_size.compare_and_swap(new_size,e);
        if( f==e ) {
            internal_grow( e, new_size, element_size, init, src );
            break;
        }
        e = f;
    }
    // Check/wait for segments allocation completes
    segment_index_t i, k_old = segment_index_of( new_size-1 );
    if( k_old >= pointers_per_short_table && my_segment == my_storage ) {
        spin_wait_while_eq( my_segment, my_storage );
    }
    for( i = 0; i <= k_old; ++i ) {
        segment_t &s = my_segment[i];
        if(!s.array) {
            ITT_NOTIFY(sync_prepare, &s.array);
            atomic_backoff backoff;
            do backoff.pause();
            while( !__TBB_load_with_acquire(my_segment[i].array) ); // my_segment may change concurrently
            ITT_NOTIFY(sync_acquired, &s.array);
        }
        if( my_segment[i].array <= __TBB_BAD_ALLOC )
            throw bad_last_alloc();
    }
    __TBB_ASSERT( internal_capacity() >= new_size, NULL);
    return e;
}

concurrent_vector_base_v3::size_type concurrent_vector_base_v3::internal_grow_by( size_type delta, size_type element_size, internal_array_op2 init, const void *src ) {
    size_type result = my_early_size.fetch_and_add(delta);
    internal_grow( result, result+delta, element_size, init, src );
    return result;
}

void concurrent_vector_base_v3::internal_grow( const size_type start, size_type finish, size_type element_size, internal_array_op2 init, const void *src ) {
    __TBB_ASSERT( start<finish, "start must be less than finish" );
    size_type tmp = start;
    helper::assign_first_segment_if_neccessary(*this, segment_index_of(finish));
    do {
        segment_index_t k_old = segment_index_of( tmp );
        size_type base = segment_base(k_old);
        helper::extend_table_if_necessary(*this, k_old);
        segment_t& s = my_segment[k_old];
        if( !__TBB_load_with_acquire(s.array) ) { // do not check for __TBB_BAD_ALLOC because it's hard to recover after __TBB_BAD_ALLOC correctly
            if( base==tmp ) {
                helper::enable_segment(*this, k_old, element_size);
            } else {
                ITT_NOTIFY(sync_prepare, &s.array);
                spin_wait_while_eq( s.array, (void*)0 );
                ITT_NOTIFY(sync_acquired, &s.array);
            }
        } else {
            ITT_NOTIFY(sync_acquired, &s.array);
        }
        if( s.array <= __TBB_BAD_ALLOC ) // check for __TBB_BAD_ALLOC
            throw bad_last_alloc(); // throw custom exception
        size_type n = k_old?segment_size(k_old):2;
        size_type j_begin = tmp-base;
        size_type j_end = n > finish-base ? finish-base : n;
        init( (void*)((char*)s.array+element_size*j_begin), src, j_end-j_begin );
        tmp = base+j_end;
    } while( tmp<finish );
}

void concurrent_vector_base_v3::internal_resize( size_type n, size_type element_size, size_type max_size, const void *src,
                                                internal_array_op1 destroy, internal_array_op2 init ) {
    size_type j = my_early_size;
    my_early_size = n;
    if( n > j ) { // construct items
        internal_reserve(n, element_size, max_size);
        segment_index_t k = segment_index_of( j );
        size_type i = my_first_block; // it should be read after call to reserve
        if( k < i ) k = 0; // process solid segment at a time
        segment_index_t b = segment_base( k );
        n -= b; j -= b; // rebase as offsets from segment k
        size_type sz = k ? b : segment_size( i ); // sz==b for k>0
        while( sz < n ) { // work for more than one segment
            void *array = my_segment[k].array;
            if( array <= __TBB_BAD_ALLOC )
                throw bad_last_alloc(); // throw custom exception
            init( (void*)((char*)array+element_size*j), src, sz-j );
            n -= sz; j = 0; // offsets from next segment
            if( !k ) k = i;
            else { ++k; sz <<= 1; }
        }
        void *array = my_segment[k].array;
        if( array <= __TBB_BAD_ALLOC )
            throw bad_last_alloc(); // throw custom exception
        init( (void*)((char*)array+element_size*j), src, n-j );
    } else {
        segment_index_t k = segment_index_of( n );
        size_type i = my_first_block;
        if( k < i ) k = 0; // process solid segment at a time
        segment_index_t b = segment_base( k );
        n -= b; j -= b; // rebase as offsets from segment k
        size_type sz = k ? b : segment_size( i ); // sz==b for k>0
        while( sz < j ) { // work for more than one segment
            void *array = my_segment[k].array;
            if( array > __TBB_BAD_ALLOC )
                destroy( (void*)((char*)array+element_size*n), sz-n);
            j -= sz; n = 0;
            if( !k ) k = i;
            else { ++k; sz <<= 1; }
        }
        void *array = my_segment[k].array;
        if( array > __TBB_BAD_ALLOC )
            destroy( (void*)((char*)array+element_size*n), j-n);
    }
}

concurrent_vector_base_v3::segment_index_t concurrent_vector_base_v3::internal_clear( internal_array_op1 destroy ) {
    __TBB_ASSERT( my_segment, NULL );
    const size_type k_end = helper::find_segment_end(*this);
    size_type finish = my_early_size;
    // Set "my_early_size" early, so that subscripting errors can be caught.
    my_early_size = 0;
    while( finish > 0 ) {
        segment_index_t k_old = segment_index_of(finish-1);
        size_type base = segment_base(k_old);
        size_type j_end = finish-base;
        finish = base;
        if( k_old <= k_end ) {
            segment_t& s = my_segment[k_old];
            __TBB_ASSERT( j_end, NULL );
            if( s.array > __TBB_BAD_ALLOC)
                destroy( s.array, j_end ); // destructors are supposed to not throw any exceptions
        }
    }
    return k_end;
}

void *concurrent_vector_base_v3::internal_compact( size_type element_size, void *table, internal_array_op1 destroy, internal_array_op2 copy )
{
    const size_type my_size = my_early_size;
    const segment_index_t k_end = helper::find_segment_end(*this); // allocated segments
    const segment_index_t k_stop = my_size? segment_index_of(my_size-1) + 1 : 0; // number of segments to store existing items: 0=>0; 1,2=>1; 3,4=>2; [5-8]=>3;..
    const segment_index_t first_block = my_first_block; // number of merged segments, getting values from atomics

    segment_index_t k = first_block;
    if(k_stop < first_block)
        k = k_stop;
    else
        while (k < k_stop && helper::incompact_predicate(segment_size( k ) * element_size) ) k++;
    if(k_stop == k_end && k == first_block)
        return NULL;

    segment_t *const segment_table = my_segment;
    internal_segments_table &old = *static_cast<internal_segments_table*>( table );
    memset(&old, 0, sizeof(old));

    if ( k != first_block && k ) // first segment optimization
    {
        // exception can occur here
        void *seg = old.table[0] = helper::allocate_segment( *this, segment_size(k) );
        old.first_block = k; // fill info for freeing new segment if exception occurs
        // copy items to the new segment
        size_type my_segment_size = segment_size( first_block );
        for (segment_index_t i = 0, j = 0; i < k && j < my_size; j = my_segment_size) {
            __TBB_ASSERT( segment_table[i].array > __TBB_BAD_ALLOC, NULL);
            void *s = static_cast<void*>(
                static_cast<char*>(seg) + segment_base(i)*element_size );
            if(j + my_segment_size >= my_size) my_segment_size = my_size - j;
            // exception can occur here
            copy( s, segment_table[i].array, my_segment_size );
            my_segment_size = i? segment_size( ++i ) : segment_size( i = first_block );
        }
        // commit the changes
        memcpy(old.table, segment_table, k * sizeof(segment_t));
        for (segment_index_t i = 0; i < k; i++) {
            segment_table[i].array = static_cast<void*>(
                static_cast<char*>(seg) + segment_base(i)*element_size );
        }
        old.first_block = first_block; my_first_block = k; // now, first_block != my_first_block
        // destroy original copies
        my_segment_size = segment_size( first_block ); // old.first_block actually
        for (segment_index_t i = 0, j = 0; i < k && j < my_size; j = my_segment_size) {
            if(j + my_segment_size >= my_size) my_segment_size = my_size - j;
            // destructors are supposed to not throw any exceptions
            destroy( old.table[i], my_segment_size );
            my_segment_size = i? segment_size( ++i ) : segment_size( i = first_block );
        }
    }
    // free unnecessary segments allocated by reserve() call
    if ( k_stop < k_end ) {
        old.first_block = first_block;
        memcpy(old.table+k_stop, segment_table+k_stop, (k_end-k_stop) * sizeof(segment_t));
        memset(segment_table+k_stop, 0, (k_end-k_stop) * sizeof(segment_t));
        if( !k ) my_first_block = 0;
    }
    return table;
}

void concurrent_vector_base_v3::internal_swap(concurrent_vector_base_v3& v)
{
    size_type my_sz = my_early_size, v_sz = v.my_early_size;
    if(!my_sz && !v_sz) return;
    size_type tmp = my_first_block; my_first_block = v.my_first_block; v.my_first_block = tmp;
    bool my_short = (my_segment == my_storage), v_short  = (v.my_segment == v.my_storage);
    if ( my_short && v_short ) { // swap both tables
        char tbl[pointers_per_short_table * sizeof(segment_t)];
        memcpy(tbl, my_storage, pointers_per_short_table * sizeof(segment_t));
        memcpy(my_storage, v.my_storage, pointers_per_short_table * sizeof(segment_t));
        memcpy(v.my_storage, tbl, pointers_per_short_table * sizeof(segment_t));
    }
    else if ( my_short ) { // my -> v
        memcpy(v.my_storage, my_storage, pointers_per_short_table * sizeof(segment_t));
        my_segment = v.my_segment; v.my_segment = v.my_storage;
    }
    else if ( v_short ) { // v -> my
        memcpy(my_storage, v.my_storage, pointers_per_short_table * sizeof(segment_t));
        v.my_segment = my_segment; my_segment = my_storage;
    } else {
        segment_t *ptr = my_segment; my_segment = v.my_segment; v.my_segment = ptr;
    }
    my_early_size = v_sz; v.my_early_size = my_sz;
}

} // namespace internal

} // tbb
