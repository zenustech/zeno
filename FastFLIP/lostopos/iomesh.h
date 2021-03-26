// ---------------------------------------------------------
//
//  iomesh.h
//  Christopher Batty, Fang Da 2014
//
//  Non-member functions for reading and writing various mesh file formats.
//
// ---------------------------------------------------------

#ifndef IOMESH_H
#define IOMESH_H

// ---------------------------------------------------------
// Nested includes
// ---------------------------------------------------------

#include <vector>
#include <vec.h>
#include <fstream>
#include <set>

// ---------------------------------------------------------
//  Forwards and typedefs
// ---------------------------------------------------------
namespace LosTopos{
class NonDestructiveTriMesh;
}

namespace Gluvi
{
   struct Target3D;
}

// ---------------------------------------------------------
//  Interface declarations
// ---------------------------------------------------------

// ---------------------------------------------------------
//
// Read/write mesh in our own binary format
//
// ---------------------------------------------------------

bool write_binary_file( const LosTopos::NonDestructiveTriMesh &mesh,  const std::vector<LosTopos::Vec3d> &x, const std::vector<double> &masses, double curr_t, const char *filename_format, ...);
bool write_binary_file_with_velocities( const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<LosTopos::Vec3d> &x, const std::vector<double> &masses, const std::vector<LosTopos::Vec3d> &v, double curr_t, const char *filename_format, ...);
bool write_binary_file_with_newpositions( const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<LosTopos::Vec3d> &x, const std::vector<double> &masses, const std::vector<LosTopos::Vec3d> &new_positions, double curr_t, const char *filename_format, ...);

bool read_binary_file( LosTopos::NonDestructiveTriMesh &mesh, std::vector<LosTopos::Vec3d> &x, std::vector<double> &masses, double& curr_t, const char *filename_format, ...);
bool read_binary_file_with_velocities( LosTopos::NonDestructiveTriMesh &mesh, std::vector<LosTopos::Vec3d> &x, std::vector<double> &masses, std::vector<LosTopos::Vec3d> &v, double& curr_t, const char *filename_format, ...);
bool read_binary_file_with_newpositions( LosTopos::NonDestructiveTriMesh &mesh, std::vector<LosTopos::Vec3d> &x, std::vector<double> &masses, std::vector<LosTopos::Vec3d> &new_positions, double& curr_t, const char *filename_format, ...);

// ---------------------------------------------------------
//
// Read/write mesh in Wavefront OBJ format (ASCII)
//
// ---------------------------------------------------------

bool write_objfile(const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<LosTopos::Vec3d> &x, const char *filename_format, ...);
bool read_objfile(LosTopos::NonDestructiveTriMesh &mesh, std::vector<LosTopos::Vec3d> &x, const char *filename_format, ...);
bool read_objfile(std::vector<LosTopos::Vec3st> &tris, std::vector<LosTopos::Vec3d> &x, const char *filename_format, ...);
bool write_objfile_per_region(const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<LosTopos::Vec3d> &x, int label, const std::set<int> & excluding_regions, const char *filename_format, ...);
bool write_objfile_per_region_pair(const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<LosTopos::Vec3d> &x, LosTopos::Vec2i label, const char *filename_format, ...);
bool write_objfile_excluding_regions(const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<LosTopos::Vec3d> &x, const std::set<int> & labels, const char *filename_format, ...);
// ---------------------------------------------------------
//
// Write mesh in Renderman RIB format (geometry only)
//
// ---------------------------------------------------------

bool write_ribfile(const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<float> &x, const char *filename_format, ...);
bool write_ribfile(const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<float> &x, std::ostream &output);
bool write_ribfile(const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<float> &x, FILE *output);

// ---------------------------------------------------------
///
/// Write an RIB file for the shadow map for the given light
///
// ---------------------------------------------------------

bool output_shadow_rib( Gluvi::Target3D& light, const std::vector<LosTopos::Vec3d>& positions,  const LosTopos::NonDestructiveTriMesh& mesh, const char *filename_format, ...);

// ---------------------------------------------------------
///
/// Write a render-ready RIB file.
///
// ---------------------------------------------------------

bool output_rib( const std::vector<LosTopos::Vec3d>& positions, const LosTopos::NonDestructiveTriMesh& mesh, const char *filename_format, ...);

// ---------------------------------------------------------
//
// Write mesh in PBRT format (geometry only)
//
// ---------------------------------------------------------

bool write_pbrtfile(const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<LosTopos::Vec3f> &x, const char *filename_format, ...);
bool write_pbrtfile(const LosTopos::NonDestructiveTriMesh &mesh, const std::vector<float> &x, std::ostream &output);


// ---------------------------------------------------------
//
// Write an STL vector to an ASCII file.  Not really mesh-related, but useful.
//
// ---------------------------------------------------------

template<class T> void dump_vector_to_file( const char* filename, const std::vector<T, std::allocator<T> >& vec )
{
   std::ofstream outfile( filename, std::ios::out|std::ios::trunc );
   for ( unsigned int i = 0; i < vec.size(); ++i )
   {
      outfile << vec[i] << std::endl;
   }         
   outfile.close();
}


#endif
