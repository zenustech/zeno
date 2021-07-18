#ifndef HASHGRID_H
#define HASHGRID_H

#include "hashtable.h"
#include "vec.h"

//========================================================= first do 2D ============================

template<class DataType>
struct HashGrid2
{
   double dx, overdx; // side-length of a grid cell and its reciprocal
   HashTable<Vec2i,DataType> grid;

   explicit HashGrid2(double dx_=1, int expected_size=512)
      : dx(dx_), overdx(1/dx_), grid(expected_size)
   {}

   // only do this with an empty grid
   void set_grid_size(double dx_)
   { assert(size()==0); dx=dx_; overdx=1/dx; }

   void add_point(const Vec2d &x, const DataType &datum)
   { grid.add(round(x*overdx), datum); }

   void delete_point(const Vec2d &x, const DataType &datum)
   { grid.delete_entry(round(x*overdx), datum); }

   void add_box(const Vec2d &xmin, const Vec2d &xmax, const DataType &datum)
   {
      Vec2i imin=round(xmin*overdx), imax=round(xmax*overdx);
      for(int j=imin[1]; j<=imax[1]; ++j) for(int i=imin[0]; i<=imax[0]; ++i)
         grid.add(Vec2i(i,j), datum);
   }

   void delete_box(const Vec2d &xmin, const Vec2d &xmax, const DataType &datum)
   {
      Vec2i imin=round(xmin*overdx), imax=round(xmax*overdx);
      for(int j=imin[1]; j<=imax[1]; ++j) for(int i=imin[0]; i<=imax[0]; ++i)
         grid.delete_entry(Vec2i(i,j), datum);
   }

   unsigned int size(void) const
   { return grid.size(); }

   void clear(void)
   { grid.clear(); }

   void reserve(unsigned int expected_size)
   { grid.reserve(expected_size); }

   bool find_first_point(const Vec2d &x, DataType &datum) const
   { return grid.get_entry(round(x*overdx), datum); }

   bool find_point(const Vec2d &x, std::vector<DataType> &data_list) const
   {
      data_list.resize(0);
      grid.append_all_entries(round(x*overdx), data_list);
      return data_list.size()>0;
   }

   bool find_box(const Vec2d &xmin, const Vec2d &xmax, std::vector<DataType> &data_list) const
   {
      data_list.resize(0);
      Vec2i imin=round(xmin*overdx), imax=round(xmax*overdx);
      for(int j=imin[1]; j<=imax[1]; ++j) for(int i=imin[0]; i<=imax[0]; ++i)
         grid.append_all_entries(Vec2i(i,j), data_list);
      return data_list.size()>0;
   }
};

//==================================== and now in 3D =================================================

template<class DataType>
struct HashGrid3
{
   double dx, overdx; // side-length of a grid cell and its reciprocal
   HashTable<Vec3i,DataType> grid;

   explicit HashGrid3(double dx_=1, int expected_size=512)
      : dx(dx_), overdx(1/dx_), grid(expected_size)
   {}

   // only do this with an empty grid
   void set_grid_size(double dx_)
   { assert(size()==0); dx=dx_; overdx=1/dx; }

   void add_point(const Vec3d &x, const DataType &datum)
   { grid.add(round(x*overdx), datum); }

   void delete_point(const Vec3d &x, const DataType &datum)
   { grid.delete_entry(round(x*overdx), datum); }

   void add_box(const Vec3d &xmin, const Vec3d &xmax, const DataType &datum)
   {
      Vec3i imin=round(xmin*overdx), imax=round(xmax*overdx);
      for(int k=imin[2]; k<=imax[2]; ++k) for(int j=imin[1]; j<=imax[1]; ++j) for(int i=imin[0]; i<=imax[0]; ++i)
         grid.add(Vec3i(i,j,k), datum);
   }

   void delete_box(const Vec3d &xmin, const Vec3d &xmax, const DataType &datum)
   {
      Vec3i imin=round(xmin*overdx), imax=round(xmax*overdx);
      for(int k=imin[2]; k<=imax[2]; ++k) for(int j=imin[1]; j<=imax[1]; ++j) for(int i=imin[0]; i<=imax[0]; ++i)
         grid.delete_entry(Vec3i(i,j,k), datum);
   }
   
   unsigned int size(void) const
   { return grid.size(); }

   void clear(void)
   { grid.clear(); }

   void reserve(unsigned int expected_size)
   { grid.reserve(expected_size); }

   bool find_first_point(const Vec3d &x, DataType &index) const
   { return grid.get_entry(round(x*overdx), index); }

   bool find_point(const Vec3d &x, std::vector<DataType> &data_list) const
   {
      data_list.resize(0);
      grid.append_all_entries(round(x*overdx), data_list);
      return data_list.size()>0;
   }

   bool find_box(const Vec3d &xmin, const Vec3d &xmax, std::vector<DataType> &data_list) const
   {
      data_list.resize(0);
      Vec3i imin=round(xmin*overdx), imax=round(xmax*overdx);
      for(int k=imin[2]; k<=imax[2]; ++k) for(int j=imin[1]; j<=imax[1]; ++j) for(int i=imin[0]; i<=imax[0]; ++i)
         grid.append_all_entries(Vec3i(i, j, k), data_list);
      return data_list.size()>0;
   }
};

#endif
