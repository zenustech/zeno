#ifndef ARRAY3_H
#define ARRAY3_H

#include "array1.h"
#include <algorithm>
#include <cassert>
#include <vector>

template<class T, class ArrayT=std::vector<T> >
struct Array3
{
   // STL-friendly typedefs

   typedef typename ArrayT::iterator iterator;
   typedef typename ArrayT::const_iterator const_iterator;
   typedef typename ArrayT::size_type size_type;
   typedef long difference_type;
   typedef T& reference;
   typedef const T& const_reference;
   typedef T value_type;
   typedef T* pointer;
   typedef const T* const_pointer;
   typedef typename ArrayT::reverse_iterator reverse_iterator;
   typedef typename ArrayT::const_reverse_iterator const_reverse_iterator;

   // the actual representation

   int ni, nj, nk;
   ArrayT a;

   // the interface

   Array3(void)
      : ni(0), nj(0), nk(0)
   {}

   Array3(int ni_, int nj_, int nk_)
      : ni(ni_), nj(nj_), nk(nk_), a(ni_*nj_*nk_)
   { assert(ni_>=0 && nj_>=0 && nk_>=0); }

   Array3(int ni_, int nj_, int nk_, ArrayT& a_)
      : ni(ni_), nj(nj_), nk(nk_), a(a_)
   { assert(ni_>=0 && nj_>=0 && nk_>=0); }

   Array3(int ni_, int nj_, int nk_, const T& value)
      : ni(ni_), nj(nj_), nk(nk_), a(ni_*nj_*nk_, value)
   { assert(ni_>=0 && nj_>=0 && nk_>=0); }

   Array3(int ni_, int nj_, int nk_, const T& value, size_type max_n_)
      : ni(ni_), nj(nj_), nk(nk_), a(ni_*nj_*nk_, value, max_n_)
   { assert(ni_>=0 && nj_>=0 && nk_>=0); }

   Array3(int ni_, int nj_, int nk_, T* data_)
      : ni(ni_), nj(nj_), nk(nk_), a(ni_*nj_*nk_, data_)
   { assert(ni_>=0 && nj_>=0 && nk_>=0); }

   Array3(int ni_, int nj_, int nk_, T* data_, size_type max_n_)
      : ni(ni_), nj(nj_), nk(nk_), a(ni_*nj_*nk_, data_, max_n_)
   { assert(ni_>=0 && nj_>=0 && nk_>=0); }

   ~Array3(void)
   {
#ifndef NDEBUG
      ni=nj=0;
#endif
   }

   const T& operator()(int i, int j, int k) const
   {
      assert(i>=0 && i<ni && j>=0 && j<nj && k>=0 && k<nk);
      return a[i+ni*(j+nj*k)];
   }

   T& operator()(int i, int j, int k)
   {
      assert(i>=0 && i<ni && j>=0 && j<nj && k>=0 && k<nk);
      return a[i+ni*(j+nj*k)];
   }

   bool operator==(const Array3<T>& x) const
   { return ni==x.ni && nj==x.nj && nk==x.nk && a==x.a; }

   bool operator!=(const Array3<T>& x) const
   { return ni!=x.ni || nj!=x.nj || nk!=x.nk || a!=x.a; }

   bool operator<(const Array3<T>& x) const
   {
      if(ni<x.ni) return true; else if(ni>x.ni) return false;
      if(nj<x.nj) return true; else if(nj>x.nj) return false;
      if(nk<x.nk) return true; else if(nk>x.nk) return false;
      return a<x.a;
   }

   bool operator>(const Array3<T>& x) const
   {
      if(ni>x.ni) return true; else if(ni<x.ni) return false;
      if(nj>x.nj) return true; else if(nj<x.nj) return false;
      if(nk>x.nk) return true; else if(nk<x.nk) return false;
      return a>x.a;
   }

   bool operator<=(const Array3<T>& x) const
   {
      if(ni<x.ni) return true; else if(ni>x.ni) return false;
      if(nj<x.nj) return true; else if(nj>x.nj) return false;
      if(nk<x.nk) return true; else if(nk>x.nk) return false;
      return a<=x.a;
   }

   bool operator>=(const Array3<T>& x) const
   {
      if(ni>x.ni) return true; else if(ni<x.ni) return false;
      if(nj>x.nj) return true; else if(nj<x.nj) return false;
      if(nk>x.nk) return true; else if(nk<x.nk) return false;
      return a>=x.a;
   }

   void assign(const T& value)
   { a.assign(value); }

   void assign(int ni_, int nj_, int nk_, const T& value)
   {
      a.assign(ni_*nj_*nk_, value);
      ni=ni_;
      nj=nj_;
      nk=nk_;
   }
    
   void assign(int ni_, int nj_, int nk_, const T* copydata)
   {
      a.assign(ni_*nj_*nk_, copydata);
      ni=ni_;
      nj=nj_;
      nk=nk_;
   }
    
   const T& at(int i, int j, int k) const
   {
      assert(i>=0 && i<ni && j>=0 && j<nj && k>=0 && k<nk);
      return a[i+ni*(j+nj*k)];
   }

   T& at(int i, int j, int k)
   {
      assert(i>=0 && i<ni && j>=0 && j<nj && k>=0 && k<nk);
      return a[i+ni*(j+nj*k)];
   }

   const T& back(void) const
   { 
      assert(a.size());
      return a.back();
   }

   T& back(void)
   {
      assert(a.size());
      return a.back();
   }

   const_iterator begin(void) const
   { return a.begin(); }

   iterator begin(void)
   { return a.begin(); }

   size_type capacity(void) const
   { return a.capacity(); }

   void clear(void)
   {
      a.clear();
      ni=nj=nk=0;
   }

   bool empty(void) const
   { return a.empty(); }

   const_iterator end(void) const
   { return a.end(); }

   iterator end(void)
   { return a.end(); }

   void fill(int ni_, int nj_, int nk_, const T& value)
   {
      a.fill(ni_*nj_*nk_, value);
      ni=ni_;
      nj=nj_;
      nk=nk_;
   }
    
   const T& front(void) const
   {
      assert(a.size());
      return a.front();
   }

   T& front(void)
   {
      assert(a.size());
      return a.front();
   }

   size_type max_size(void) const
   { return a.max_size(); }

   reverse_iterator rbegin(void)
   { return reverse_iterator(end()); }

   const_reverse_iterator rbegin(void) const
   { return const_reverse_iterator(end()); }

   reverse_iterator rend(void)
   { return reverse_iterator(begin()); }

   const_reverse_iterator rend(void) const
   { return const_reverse_iterator(begin()); }

   void reserve(int reserve_ni, int reserve_nj, int reserve_nk)
   { a.reserve(reserve_ni*reserve_nj*reserve_nk); }

   void resize(int ni_, int nj_, int nk_)
   {
      assert(ni_>=0 && nj_>=0 && nk_>=0);
      a.resize(ni_*nj_*nk_);
      ni=ni_;
      nj=nj_;
      nk=nk_;
   }

   void resize(int ni_, int nj_, int nk_, const T& value)
   {
      assert(ni_>=0 && nj_>=0 && nk_>=0);
      a.resize(ni_*nj_*nk_, value);
      ni=ni_;
      nj=nj_;
      nk=nk_;
   }

   void set_zero(void)
   { a.set_zero(); }

   size_type size(void) const
   { return a.size(); }

   void swap(Array3<T>& x)
   {
      std::swap(ni, x.ni);
      std::swap(nj, x.nj);
      std::swap(nk, x.nk);
      a.swap(x.a);
   }

   void trim(void)
   { a.trim(); }
};

// some common arrays

typedef Array3<double, Array1<double> > Array3d;
typedef Array3<float, Array1<float> > Array3f;
typedef Array3<long long, Array1<long long> > Array3ll;
typedef Array3<unsigned long long, Array1<unsigned long long> > Array3ull;
typedef Array3<int, Array1<int> > Array3i;
typedef Array3<unsigned int, Array1<unsigned int> > Array3ui;
typedef Array3<short, Array1<short> > Array3s;
typedef Array3<unsigned short, Array1<unsigned short> > Array3us;
typedef Array3<char, Array1<char> > Array3c;
typedef Array3<unsigned char, Array1<unsigned char> > Array3uc;

#endif
