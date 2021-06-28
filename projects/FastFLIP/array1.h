#ifndef ARRAY1_H
#define ARRAY1_H

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

// In this file:
//   Array1<T>: a dynamic 1D array for plain-old-data (not objects)
//   WrapArray1<T>: a 1D array wrapper around an existing array (perhaps objects, perhaps data)
// For the most part std::vector operations are supported, though for the Wrap version
// note that memory is never allocated/deleted and constructor/destructors are never called
// from within the class, thus only shallow copies can be made and some operations such as
// resize() and push_back() are limited.
// Note: for the most part assertions are done with assert(), not exceptions...

namespace LosTopos {

// gross template hacking to determine if a type is integral or not
struct Array1True {};
struct Array1False {};
template<typename T> struct Array1IsIntegral{ typedef Array1False type; }; // default: no (specializations to yes follow)
template<> struct Array1IsIntegral<bool>{ typedef Array1True type; };
template<> struct Array1IsIntegral<char>{ typedef Array1True type; };
template<> struct Array1IsIntegral<signed char>{ typedef Array1True type; };
template<> struct Array1IsIntegral<unsigned char>{ typedef Array1True type; };
template<> struct Array1IsIntegral<short>{ typedef Array1True type; };
template<> struct Array1IsIntegral<unsigned short>{ typedef Array1True type; };
template<> struct Array1IsIntegral<int>{ typedef Array1True type; };
template<> struct Array1IsIntegral<unsigned int>{ typedef Array1True type; };
template<> struct Array1IsIntegral<long>{ typedef Array1True type; };
template<> struct Array1IsIntegral<unsigned long>{ typedef Array1True type; };
//template<> struct Array1IsIntegral<long long>{ typedef Array1True type; };
//template<> struct Array1IsIntegral<unsigned long long>{ typedef Array1True type; };

//============================================================================
template<typename T>
struct Array1
{
    // STL-friendly typedefs
    
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef unsigned long size_type;
    typedef long difference_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
    
    // the actual representation
    
    unsigned long n;
    unsigned long max_n;
    T* data;
    
    // STL vector's interface, with additions, but only valid when used with plain-old-data
    
    Array1(void)
    : n(0), max_n(0), data(0)
    {}
    
    // note: default initial values are zero
    Array1(unsigned long n_)
    : n(0), max_n(0), data(0)
    {
        if(n_>ULONG_MAX/sizeof(T)) throw std::bad_alloc();
        data=(T*)std::calloc(n_, sizeof(T));
        if(!data) throw std::bad_alloc();
        n=n_;
        max_n=n_;
    }
    
    Array1(unsigned long n_, const T& value)
    : n(0), max_n(0), data(0)
    {
        if(n_>ULONG_MAX/sizeof(T)) throw std::bad_alloc();
        data=(T*)std::calloc(n_, sizeof(T));
        if(!data) throw std::bad_alloc();
        n=n_;
        max_n=n_;
        for(unsigned long i=0; i<n; ++i) data[i]=value;
    }
    
    Array1(unsigned long n_, const T& value, unsigned long max_n_)
    : n(0), max_n(0), data(0)
    {
        assert(n_<=max_n_);
        if(max_n_>ULONG_MAX/sizeof(T)) throw std::bad_alloc();
        data=(T*)std::calloc(max_n_, sizeof(T));
        if(!data) throw std::bad_alloc();
        n=n_;
        max_n=max_n_;
        for(unsigned long i=0; i<n; ++i) data[i]=value;
    }
    
    Array1(unsigned long n_, const T* data_)
    : n(0), max_n(0), data(0)
    {
        if(n_>ULONG_MAX/sizeof(T)) throw std::bad_alloc();
        data=(T*)std::calloc(n_, sizeof(T));
        if(!data) throw std::bad_alloc();
        n=n_;
        max_n=n_;
        assert(data_);
        std::memcpy(data, data_, n*sizeof(T));
    }
    
    Array1(unsigned long n_, const T* data_, unsigned long max_n_)
    : n(0), max_n(0), data(0)
    {
        assert(n_<=max_n_);
        if(max_n_>ULONG_MAX/sizeof(T)) throw std::bad_alloc();
        data=(T*)std::calloc(max_n_, sizeof(T));
        if(!data) throw std::bad_alloc();
        max_n=max_n_;
        n=n_;
        assert(data_);
        std::memcpy(data, data_, n*sizeof(T));
    }
    
    Array1(const Array1<T> &x)
    : n(0), max_n(0), data(0)
    {
        data=(T*)std::malloc(x.n*sizeof(T));
        if(!data) throw std::bad_alloc();
        n=x.n;
        max_n=x.n;
        std::memcpy(data, x.data, n*sizeof(T));
    }
    
    ~Array1(void)
    {
        std::free(data);
#ifndef NDEBUG
        data=0;
        n=max_n=0;
#endif
    }
    
    const T& operator[](unsigned long i) const
    { return data[i]; }
    
    T& operator[](unsigned long i)
    { return data[i]; }
    
    // these are range-checked (in debug mode) versions of operator[], like at()
    const T& operator()(unsigned long i) const
    {
        assert(i<n);
        return data[i];
    }
    
    T& operator()(unsigned long i)
    {
        assert(i<n);
        return data[i];
    }
    
    Array1<T>& operator=(const Array1<T>& x)
    {
        if(max_n<x.n){
            T* new_data=(T*)std::malloc(x.n*sizeof(T));
            if(!new_data) throw std::bad_alloc();
            std::free(data);
            data=new_data;
            max_n=x.n;
        }
        n=x.n;
        std::memcpy(data, x.data, n*sizeof(T));
        return *this;
    }
    
    bool operator==(const Array1<T>& x) const
    {
        if(n!=x.n) return false;
        for(unsigned long i=0; i<n; ++i) if(!(data[i]==x.data[i])) return false;
        return true;
    }
    
    bool operator!=(const Array1<T>& x) const
    {
        if(n!=x.n) return true;
        for(unsigned long i=0; i<n; ++i) if(data[i]!=x.data[i]) return true;
        return false;
    }
    
    bool operator<(const Array1<T>& x) const
    {
        for(unsigned long i=0; i<n && i<x.n; ++i){
            if(data[i]<x[i]) return true;
            else if(x[i]<data[i]) return false;
        }
        return n<x.n;
    }
    
    bool operator>(const Array1<T>& x) const
    {
        for(unsigned long i=0; i<n && i<x.n; ++i){
            if(data[i]>x[i]) return true;
            else if(x[i]>data[i]) return false;
        }
        return n>x.n;
    }
    
    bool operator<=(const Array1<T>& x) const
    {
        for(unsigned long i=0; i<n && i<x.n; ++i){
            if(data[i]<x[i]) return true;
            else if(x[i]<data[i]) return false;
        }
        return n<=x.n;
    }
    
    bool operator>=(const Array1<T>& x) const
    {
        for(unsigned long i=0; i<n && i<x.n; ++i){
            if(data[i]>x[i]) return true;
            else if(x[i]>data[i]) return false;
        }
        return n>=x.n;
    }
    
    void add_unique(const T& value)
    {
        for(unsigned long i=0; i<n; ++i) if(data[i]==value) return;
        if(n==max_n) grow();
        data[n++]=value;
    }
    
    void assign(const T& value)
    { for(unsigned long i=0; i<n; ++i) data[i]=value; }
    
    void assign(unsigned long num, const T& value)
    { fill(num, value); } 
    
    // note: copydata may not alias this array's data, and this should not be
    // used when T is a full object (which defines its own copying operation)
    void assign(unsigned long num, const T* copydata)
    {
        assert(num==0 || copydata);
        if(num>max_n){
            if(num>ULONG_MAX/sizeof(T)) throw std::bad_alloc();
            std::free(data);
            data=(T*)std::malloc(num*sizeof(T));
            if(!data) throw std::bad_alloc();
            max_n=num;
        }
        n=num;
        std::memcpy(data, copydata, n*sizeof(T));
    }
    
    template<typename InputIterator>
    void assign(InputIterator first, InputIterator last)
    { assign_(first, last, typename Array1IsIntegral<InputIterator>::type()); }
    
    template<typename InputIterator>
    void assign_(InputIterator first, InputIterator last, Array1True )
    { fill(first, last); }
    
    template<typename InputIterator>
    void assign_(InputIterator first, InputIterator last, Array1False )
    {
        unsigned long i=0;
        InputIterator p=first;
        for(; p!=last; ++p, ++i){
            if(i==max_n) grow();
            data[i]=*p;
        }
        n=i;
    }
    
    const T& at(unsigned long i) const
    {
        assert(i<n);
        return data[i];
    }
    
    T& at(unsigned long i)
    {
        assert(i<n);
        return data[i];
    }
    
    const T& back(void) const
    { 
        assert(data && n>0);
        return data[n-1];
    }
    
    T& back(void)
    {
        assert(data && n>0);
        return data[n-1];
    }
    
    const T* begin(void) const
    { return data; }
    
    T* begin(void)
    { return data; }
    
    unsigned long capacity(void) const
    { return max_n; }
    
    void clear(void)
    {
        std::free(data);
        data=0;
        max_n=0;
        n=0;
    }
    
    bool empty(void) const
    { return n==0; }
    
    const T* end(void) const
    { return data+n; }
    
    T* end(void)
    { return data+n; }
    
    void erase(unsigned long index)
    {
        assert(index<n);
        for(unsigned long i=index; i<n-1; ++i)
            data[i]=data[i-1];
        pop_back();
    }
    
    void fill(unsigned long num, const T& value)
    {
        if(num>max_n){
            if(num>ULONG_MAX/sizeof(T)) throw std::bad_alloc();
            std::free(data);
            data=(T*)std::malloc(num*sizeof(T));
            if(!data) throw std::bad_alloc();
            max_n=num;
        }
        n=num;
        for(unsigned long i=0; i<n; ++i) data[i]=value;
    }
    
    const T& front(void) const
    {
        assert(n>0);
        return *data;
    }
    
    T& front(void)
    {
        assert(n>0);
        return *data;
    }
    
    void grow(void)
    {
        unsigned long new_size=(max_n*sizeof(T)<ULONG_MAX/2 ? 2*max_n+1 : ULONG_MAX/sizeof(T));
        T *new_data=(T*)std::realloc(data, new_size*sizeof(T));
        if(!new_data) throw std::bad_alloc();
        data=new_data;
        max_n=new_size;
    }
    
    void insert(unsigned long index, const T& entry)
    {
        assert(index<=n);
        push_back(back());
        for(unsigned long i=n-1; i>index; --i)
            data[i]=data[i-1];
        data[index]=entry;
    }
    
    unsigned long max_size(void) const
    { return ULONG_MAX/sizeof(T); }
    
    void pop_back(void)
    {
        assert(n>0);
        --n;
    }
    
    void push_back(const T& value)
    {
        if(n==max_n) grow();
        data[n++]=value;
    }
    
    reverse_iterator rbegin(void)
    { return reverse_iterator(end()); }
    
    const_reverse_iterator rbegin(void) const
    { return const_reverse_iterator(end()); }
    
    reverse_iterator rend(void)
    { return reverse_iterator(begin()); }
    
    const_reverse_iterator rend(void) const
    { return const_reverse_iterator(begin()); }
    
    void reserve(unsigned long r)
    {
        if(r>ULONG_MAX/sizeof(T)) throw std::bad_alloc();
        T *new_data=(T*)std::realloc(data, r*sizeof(T));
        if(!new_data) throw std::bad_alloc();
        data=new_data;
        max_n=r;
    }
    
    void resize(unsigned long n_)
    {
        if(n_>max_n) reserve(n_);
        n=n_;
    }
    
    void resize(unsigned long n_, const T& value)
    {
        if(n_>max_n) reserve(n_);
        if(n<n_) for(unsigned long i=n; i<n_; ++i) data[i]=value;
        n=n_;
    }
    
    void set_zero(void)
    { std::memset(data, 0, n*sizeof(T)); }
    
    unsigned long size(void) const
    { return n; }
    
    void swap(Array1<T>& x)
    {
        std::swap(n, x.n);
        std::swap(max_n, x.max_n);
        std::swap(data, x.data);
    }
    
    // resize the array to avoid wasted space, without changing contents
    // (Note: realloc, at least on some platforms, will not do the trick)
    void trim(void)
    {
        if(n==max_n) return;
        T *new_data=(T*)std::malloc(n*sizeof(T));
        if(!new_data) return;
        std::memcpy(new_data, data, n*sizeof(T));
        std::free(data);
        data=new_data;
        max_n=n;
    }
};

// some common arrays

typedef Array1<double>             Array1d;
typedef Array1<float>              Array1f;
//typedef Array1<long long>          Array1ll;
//typedef Array1<unsigned long long> Array1ull;
typedef Array1<int>                Array1i;
typedef Array1<unsigned int>       Array1ui;
typedef Array1<short>              Array1s;
typedef Array1<unsigned short>     Array1us;
typedef Array1<char>               Array1c;
typedef Array1<unsigned char>      Array1uc;

//============================================================================
template<typename T>
struct WrapArray1
{
    // STL-friendly typedefs
    
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef unsigned long size_type;
    typedef long difference_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
    
    // the actual representation
    
    unsigned long n;
    unsigned long max_n;
    T* data;
    
    // most of STL vector's interface, with a few changes
    
    WrapArray1(void)
    : n(0), max_n(0), data(0)
    {}
    
    WrapArray1(unsigned long n_, T* data_)
    : n(n_), max_n(n_), data(data_)
    { assert(data || max_n==0); }
    
    WrapArray1(unsigned long n_, T* data_, unsigned long max_n_)
    : n(n_), max_n(max_n_), data(data_)
    {
        assert(n<=max_n);
        assert(data || max_n==0);
    }
    
    // Allow for simple shallow copies of existing arrays
    // Note that if the underlying arrays change where their data is, the WrapArray may be screwed up
    
    WrapArray1(Array1<T>& a)
    : n(a.n), max_n(a.max_n), data(a.data)
    {}
    
    WrapArray1(std::vector<T>& a)
    : n(a.size()), max_n(a.capacity()), data(&a[0])
    {}
    
    void init(unsigned long n_, T* data_, unsigned long max_n_)
    {
        assert(n_<=max_n_);
        assert(data_ || max_n_==0);
        n=n_;
        max_n=max_n_;
        data=data_;
    }
    
    const T& operator[](unsigned long i) const
    { return data[i]; }
    
    T& operator[](unsigned long i)
    { return data[i]; }
    
    // these are range-checked (in debug mode) versions of operator[], like at()
    const T& operator()(unsigned long i) const
    {
        assert(i<n);
        return data[i];
    }
    
    T& operator()(unsigned long i)
    {
        assert(i<n);
        return data[i];
    }
    
    bool operator==(const WrapArray1<T>& x) const
    {
        if(n!=x.n) return false;
        for(unsigned long i=0; i<n; ++i) if(!(data[i]==x.data[i])) return false;
        return true;
    }
    
    bool operator!=(const WrapArray1<T>& x) const
    {
        if(n!=x.n) return true;
        for(unsigned long i=0; i<n; ++i) if(data[i]!=x.data[i]) return true;
        return false;
    }
    
    bool operator<(const WrapArray1<T>& x) const
    {
        for(unsigned long i=0; i<n && i<x.n; ++i){
            if(data[i]<x[i]) return true;
            else if(x[i]<data[i]) return false;
        }
        return n<x.n;
    }
    
    bool operator>(const WrapArray1<T>& x) const
    {
        for(unsigned long i=0; i<n && i<x.n; ++i){
            if(data[i]>x[i]) return true;
            else if(x[i]>data[i]) return false;
        }
        return n>x.n;
    }
    
    bool operator<=(const WrapArray1<T>& x) const
    {
        for(unsigned long i=0; i<n && i<x.n; ++i){
            if(data[i]<x[i]) return true;
            else if(x[i]<data[i]) return false;
        }
        return n<=x.n;
    }
    
    bool operator>=(const WrapArray1<T>& x) const
    {
        for(unsigned long i=0; i<n && i<x.n; ++i){
            if(data[i]>x[i]) return true;
            else if(x[i]>data[i]) return false;
        }
        return n>=x.n;
    }
    
    void add_unique(const T& value)
    {
        for(unsigned long i=0; i<n; ++i) if(data[i]==value) return;
        assert(n<max_n);
        data[n++]=value;
    }
    
    void assign(const T& value)
    { for(unsigned long i=0; i<n; ++i) data[i]=value; }
    
    void assign(unsigned long num, const T& value)
    { fill(num, value); } 
    
    // note: copydata may not alias this array's data, and this should not be
    // used when T is a full object (which defines its own copying operation)
    void assign(unsigned long num, const T* copydata)
    {
        assert(num==0 || copydata);
        assert(num<=max_n);
        n=num;
        std::memcpy(data, copydata, n*sizeof(T));
    }
    
    template<typename InputIterator>
    void assign(InputIterator first, InputIterator last)
    { assign_(first, last, typename Array1IsIntegral<InputIterator>::type()); }
    
    template<typename InputIterator>
    void assign_(InputIterator first, InputIterator last, Array1True )
    { fill(first, last); }
    
    template<typename InputIterator>
    void assign_(InputIterator first, InputIterator last, Array1False )
    {
        unsigned long i=0;
        InputIterator p=first;
        for(; p!=last; ++p, ++i){
            assert(i<max_n);
            data[i]=*p;
        }
        n=i;
    }
    
    const T& at(unsigned long i) const
    {
        assert(i<n);
        return data[i];
    }
    
    T& at(unsigned long i)
    {
        assert(i<n);
        return data[i];
    }
    
    const T& back(void) const
    { 
        assert(data && n>0);
        return data[n-1];
    }
    
    T& back(void)
    {
        assert(data && n>0);
        return data[n-1];
    }
    
    const T* begin(void) const
    { return data; }
    
    T* begin(void)
    { return data; }
    
    unsigned long capacity(void) const
    { return max_n; }
    
    void clear(void)
    { n=0; }
    
    bool empty(void) const
    { return n==0; }
    
    const T* end(void) const
    { return data+n; }
    
    T* end(void)
    { return data+n; }
    
    void erase(unsigned long index)
    {
        assert(index<n);
        for(unsigned long i=index; i<n-1; ++i)
            data[i]=data[i-1];
        pop_back();
    }
    
    void fill(unsigned long num, const T& value)
    {
        assert(num<=max_n);
        n=num;
        for(unsigned long i=0; i<n; ++i) data[i]=value;
    }
    
    const T& front(void) const
    {
        assert(n>0);
        return *data;
    }
    
    T& front(void)
    {
        assert(n>0);
        return *data;
    }
    
    void insert(unsigned long index, const T& entry)
    {
        assert(index<=n);
        push_back(back());
        for(unsigned long i=n-1; i>index; --i)
            data[i]=data[i-1];
        data[index]=entry;
    }
    
    unsigned long max_size(void) const
    { return max_n; }
    
    void pop_back(void)
    {
        assert(n>0);
        --n;
    }
    
    void push_back(const T& value)
    {
        assert(n<max_n);
        data[n++]=value;
    }
    
    reverse_iterator rbegin(void)
    { return reverse_iterator(end()); }
    
    const_reverse_iterator rbegin(void) const
    { return const_reverse_iterator(end()); }
    
    reverse_iterator rend(void)
    { return reverse_iterator(begin()); }
    
    const_reverse_iterator rend(void) const
    { return const_reverse_iterator(begin()); }
    
    void reserve(unsigned long r)
    { assert(r<=max_n); }
    
    void resize(unsigned long n_)
    {
        assert(n_<=max_n);
        n=n_;
    }
    
    void resize(unsigned long n_, const T& value)
    {
        assert(n_<=max_n);
        if(n<n_) for(unsigned long i=n; i<n_; ++i) data[i]=value;
        n=n_;
    }
    
    // note: shouldn't be used when T is a full object (setting to zero may not make sense)
    void set_zero(void)
    { std::memset(data, 0, n*sizeof(T)); }
    
    unsigned long size(void) const
    { return n; }
    
    void swap(WrapArray1<T>& x)
    {
        std::swap(n, x.n);
        std::swap(max_n, x.max_n);
        std::swap(data, x.data);
    }
};

// some common arrays

typedef WrapArray1<double>             WrapArray1d;
typedef WrapArray1<float>              WrapArray1f;
//typedef WrapArray1<long long>          WrapArray1ll;
//typedef WrapArray1<unsigned long long> WrapArray1ull;
typedef WrapArray1<int>                WrapArray1i;
typedef WrapArray1<unsigned int>       WrapArray1ui;
typedef WrapArray1<short>              WrapArray1s;
typedef WrapArray1<unsigned short>     WrapArray1us;
typedef WrapArray1<char>               WrapArray1c;
typedef WrapArray1<unsigned char>      WrapArray1uc;

}

#endif
