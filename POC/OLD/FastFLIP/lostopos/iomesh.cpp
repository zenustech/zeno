#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "iomesh.h"

#include <cstdarg>
#include <cstdlib>
#include <cmath>
#include <fstream>

#include <nondestructivetrimesh.h>
#include <bfstream.h>
#include <map>

#define LINESIZE 1024 // maximum line size when reading .OBJ files

using namespace LosTopos;

// ---------------------------------------------------------
///
/// Write mesh in binary format
///
// ---------------------------------------------------------

bool write_binary_file( const NonDestructiveTriMesh &mesh, 
                       const std::vector<Vec3d> &x,
                       const std::vector<double> &masses,
                       double curr_t, 
                       const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);   
   bofstream outfile( filename_format, ap );
   va_end(ap);
   
   outfile.write_endianity();
   
   outfile << curr_t;
   
   outfile << (unsigned int)x.size();
   for ( unsigned int i = 0; i < x.size(); ++i )
   {
      outfile << x[i][0];
      outfile << x[i][1];
      outfile << x[i][2];
   }
   
   assert( x.size() == masses.size() );
   
   for ( unsigned int i = 0; i < masses.size(); ++i )
   {
      outfile << masses[i];
   }
   
   outfile << (unsigned int) mesh.num_triangles();
   
   for ( unsigned int t = 0; t < mesh.num_triangles(); ++t )
   {
      Vec3st tri = mesh.get_triangle(t);
      outfile <<  (unsigned int)tri[0];
      outfile <<  (unsigned int)tri[1];
      outfile <<  (unsigned int)tri[2];      
   }

   for ( unsigned int t = 0; t < mesh.num_triangles(); ++t )
   {
      const Vec2i& label_pair = mesh.get_triangle_label(t);
      outfile << label_pair[0];
      outfile << label_pair[1];
   }
   
   outfile.close();
   
   return outfile.good();
}

// ---------------------------------------------------------
///
/// Write mesh in binary format, with per-vertex velocities
///
// ---------------------------------------------------------

bool write_binary_file_with_velocities( const NonDestructiveTriMesh &mesh, 
                                       const std::vector<Vec3d> &x,
                                       const std::vector<double> &masses,                                       
                                       const std::vector<Vec3d> &v,
                                       double curr_t, 
                                       const char *filename_format, ...)
{
   
   va_list ap;
   va_start(ap, filename_format);   
   bofstream outfile( filename_format, ap );
   va_end(ap);
   
   outfile.write_endianity();
   
   outfile << curr_t;
   
   outfile <<  (unsigned int)x.size();
   
   for ( unsigned int i = 0; i < x.size(); ++i )
   {
      outfile << x[i][0];
      outfile << x[i][1];
      outfile << x[i][2];
   }
   
   assert( x.size() == masses.size() );
   
   for ( unsigned int i = 0; i < masses.size(); ++i )
   {
      outfile << masses[i];
   }
   
   for ( unsigned int i = 0; i < v.size(); ++i )
   {
      outfile << v[i][0];
      outfile << v[i][1];
      outfile << v[i][2];
   }
   
   outfile <<  (unsigned int)mesh.num_triangles();
   
   for ( unsigned int t = 0; t < mesh.num_triangles(); ++t )
   {
      Vec3st tri = mesh.get_triangle(t);
      outfile <<  (unsigned int)tri[0];
      outfile <<  (unsigned int)tri[1];
      outfile <<  (unsigned int)tri[2];      
   }
   
   outfile.close();
   
   return outfile.good();
}


// ---------------------------------------------------------
///
///
///
// ---------------------------------------------------------

bool write_binary_file_with_newpositions( const NonDestructiveTriMesh &mesh, 
                                          const std::vector<Vec3d> &x, 
                                          const std::vector<double> &masses, 
                                          const std::vector<Vec3d> &new_positions, 
                                          double curr_t, 
                                          const char *filename_format, ...)
{
   
   va_list ap;
   va_start(ap, filename_format);   
   bofstream outfile( filename_format, ap );
   va_end(ap);
   
   outfile.write_endianity();
   
   outfile << curr_t;
   
   outfile <<  (unsigned int)x.size();
   
   for ( unsigned int i = 0; i < x.size(); ++i )
   {
      outfile << x[i][0];
      outfile << x[i][1];
      outfile << x[i][2];
   }
   
   assert( x.size() == masses.size() );
   
   for ( unsigned int i = 0; i < masses.size(); ++i )
   {
      outfile << masses[i];
   }
   
   for ( unsigned int i = 0; i < new_positions.size(); ++i )
   {
      outfile << new_positions[i][0];
      outfile << new_positions[i][1];
      outfile << new_positions[i][2];
   }
   
   outfile <<  (unsigned int)mesh.num_triangles();
   
   for ( unsigned int t = 0; t < mesh.num_triangles(); ++t )
   {
      Vec3st tri = mesh.get_triangle(t);
      outfile <<  (unsigned int)tri[0];
      outfile <<  (unsigned int)tri[1];
      outfile <<  (unsigned int)tri[2];      
   }
   
   outfile.close();
   
   return outfile.good();   
}

// ---------------------------------------------------------
///
/// Read mesh in binary format
///
// ---------------------------------------------------------

bool read_binary_file( NonDestructiveTriMesh &mesh, 
                      std::vector<Vec3d> &x, 
                      std::vector<double> &masses,
                      double& curr_t, 
                      const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);   
   bifstream infile( filename_format, ap );
   va_end(ap);
   
   assert( infile.good() );
   
   infile.read_endianity();
   
   infile >> curr_t;
   
   unsigned int nverts;
   infile >> nverts;
   x.resize( nverts );
   std::cout << "Num vertices: " << nverts << std::endl;
   for ( unsigned int i = 0; i < nverts; ++i )
   {
      infile >> x[i][0];
      infile >> x[i][1];
      infile >> x[i][2];      
   }
   
   masses.resize( nverts );
   for ( unsigned int i = 0; i < masses.size(); ++i )
   {
      infile >> masses[i];
   }
   
   unsigned int ntris;
   infile >> ntris;
   mesh.m_tris.resize( ntris );
   for ( unsigned int t = 0; t < ntris; ++t )
   {
      unsigned int a,b,c;
      infile >> a >> b >> c;
      mesh.m_tris[t][0] = a;
      mesh.m_tris[t][1] = b;
      mesh.m_tris[t][2] = c;
   }
   
   mesh.m_triangle_labels.resize( ntris );
   for ( unsigned int t = 0; t < ntris; ++t )
   {
      Vec2i label;
      infile >> label[0];
      infile >> label[1];
      mesh.m_triangle_labels[t] = label;
   }
   
   infile.close();
   
   return infile.good();
}


// ---------------------------------------------------------
///
/// Read mesh in binary format, with per-vertex velocities
///
// ---------------------------------------------------------

bool read_binary_file_with_velocities( NonDestructiveTriMesh &mesh, 
                                      std::vector<Vec3d> &x, 
                                      std::vector<double> &masses,
                                      std::vector<Vec3d> &v, 
                                      double& curr_t, 
                                      const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);   
   bifstream infile( filename_format, ap );
   va_end(ap);
   
   infile.read_endianity();
   
   infile >> curr_t;
   
   unsigned int nverts;
   infile >> nverts;
   
   x.resize( nverts );
   for ( unsigned int i = 0; i < nverts; ++i )
   {
      infile >> x[i][0];
      infile >> x[i][1];
      infile >> x[i][2];      
   }
   
   masses.resize( nverts );
   for ( unsigned int i = 0; i < masses.size(); ++i )
   {
      infile >> masses[i];
   }
   
   v.resize( nverts );
   for ( unsigned int i = 0; i < nverts; ++i )
   {
      infile >> v[i][0];
      infile >> v[i][1];
      infile >> v[i][2];      
   }
   
   unsigned int ntris;
   infile >> ntris;
   mesh.m_tris.resize( ntris );
   for ( unsigned int t = 0; t < ntris; ++t )
   {
      unsigned int a,b,c;
      infile >> a >> b >> c;

      mesh.m_tris[t][0] = a;
      mesh.m_tris[t][1] = b;
      mesh.m_tris[t][2] = c;
   }
   
   infile.close();
   
   return infile.good();
}


// ---------------------------------------------------------
///
/// 
///
// ---------------------------------------------------------

bool read_binary_file_with_newpositions( NonDestructiveTriMesh &mesh, 
                                      std::vector<Vec3d> &x, 
                                      std::vector<double> &masses,
                                      std::vector<Vec3d> &new_positions, 
                                      double& curr_t, 
                                      const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);   
   bifstream infile( filename_format, ap );
   va_end(ap);
   
   infile.read_endianity();
   
   infile >> curr_t;
   
   unsigned int nverts;
   infile >> nverts;
   
   x.resize( nverts );
   for ( unsigned int i = 0; i < nverts; ++i )
   {
      infile >> x[i][0];
      infile >> x[i][1];
      infile >> x[i][2];      
   }
   
   masses.resize( nverts );
   for ( unsigned int i = 0; i < masses.size(); ++i )
   {
      infile >> masses[i];
   }
   
   new_positions.resize( nverts );
   for ( unsigned int i = 0; i < nverts; ++i )
   {
      infile >> new_positions[i][0];
      infile >> new_positions[i][1];
      infile >> new_positions[i][2];      
   }
   
   unsigned int ntris;
   infile >> ntris;
   mesh.m_tris.resize( ntris );
   for ( unsigned int t = 0; t < ntris; ++t )
   {
      unsigned int a,b,c;
      infile >> a >> b >> c;
      mesh.m_tris[t][0] = a;
      mesh.m_tris[t][1] = b;
      mesh.m_tris[t][2] = c;
   }
   
   infile.close();
   
   return infile.good();
}

void generate_normals(const std::vector<Vec3d> & vertices, const std::vector<Vec3st> & triangles, std::vector<Vec3d> & normals, std::vector<Vec3st> & normal_indices)
{
    std::vector<std::vector<size_t> > vf_map(vertices.size());
    for (size_t i = 0; i < triangles.size(); i++)
    {
        vf_map[triangles[i][0]].push_back(i);
        vf_map[triangles[i][1]].push_back(i);
        vf_map[triangles[i][2]].push_back(i);
    }
    
    std::map<std::pair<size_t, size_t>, std::vector<size_t> > ef_map;
    for (size_t i = 0; i < triangles.size(); i++)
    {
        const Vec3st & t = triangles[i];
        std::pair<size_t, size_t> e;
        e = std::pair<size_t, size_t>(t[0], t[1]); if (e.first > e.second) std::swap(e.first, e.second);
        ef_map[e].push_back(i);
        e = std::pair<size_t, size_t>(t[1], t[2]); if (e.first > e.second) std::swap(e.first, e.second);
        ef_map[e].push_back(i);
        e = std::pair<size_t, size_t>(t[2], t[0]); if (e.first > e.second) std::swap(e.first, e.second);
        ef_map[e].push_back(i);
    }
    
    // floodfill to identify manifold patches
    std::vector<int> manifold_patch_labels(triangles.size(), -1);
    int nextid = 0;
    for (size_t i = 0; i < triangles.size(); i++)
    {
        if (manifold_patch_labels[i] < 0)
        {
            int label = nextid++;
            std::deque<size_t> open(1, i);
            while (open.size() > 0)
            {
                size_t top = open.back();
                const Vec3st & t = triangles[top];
                open.pop_back();
                
                manifold_patch_labels[top] = label;
                
                for (int j = 0; j < 3; j++)
                {
                    std::pair<size_t, size_t> e;
                    e = std::pair<size_t, size_t>(t[(j + 0) % 3], t[(j + 1) % 3]); if (e.first > e.second) std::swap(e.first, e.second);
                    const std::vector<size_t> & et = ef_map[e];
                    if (et.size() == 2)
                    {
                        size_t other_triangle = (et[0] == i ? et[1] : et[0]);
                        if (manifold_patch_labels[other_triangle] < 0)
                            open.push_back(other_triangle);
                    }
                }
            }
        }
    }
    
    // create the normals
    normals.clear();
    normal_indices.resize(triangles.size(), Vec3st(0, 0, 0));
    std::vector<int> index_to_label(nextid);
    std::vector<size_t> label_to_index(nextid);
    for (size_t i = 0; i < vertices.size(); i++)
    {
        std::set<int> manifold_patches;
        for (size_t j = 0; j < vf_map[i].size(); j++)
            manifold_patches.insert(manifold_patch_labels[vf_map[i][j]]);

        int k = 0;
        for (std::set<int>::iterator j = manifold_patches.begin(); j != manifold_patches.end(); j++, k++)
        {
            index_to_label[k] = *j;
            label_to_index[*j] = normals.size();
            normals.push_back(Vec3d(0, 0, 0));
        }
        
        for (size_t j = 0; j < vf_map[i].size(); j++)
        {
            const Vec3st & t = triangles[vf_map[i][j]];
            
            Vec3d facenormal = cross(vertices[t[1]] - vertices[t[0]], vertices[t[2]] - vertices[t[0]]);
            facenormal /= mag(facenormal);
            
            size_t index = label_to_index[manifold_patch_labels[vf_map[i][j]]];
            normals[index] += facenormal;
            
            Vec3st & ni = normal_indices[vf_map[i][j]];
            if      (t[0] == i) ni[0] = index;
            else if (t[1] == i) ni[1] = index;
            else if (t[2] == i) ni[2] = index;
            else assert("error");
        }
    }
    
    for (size_t i = 0; i < normals.size(); i++)
    {
        normals[i] /= mag(normals[i]);
    }
    
}

// ---------------------------------------------------------
///
/// Write mesh in Wavefront OBJ format
///
// ---------------------------------------------------------

bool write_objfile(const NonDestructiveTriMesh &mesh, const std::vector<Vec3d> &x, const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);
#ifdef _MSC_VER
   int len=_vscprintf(filename_format, ap) +1;// _vscprintf doesn't count terminating '\0'
   char *filename=new char[len];
   vsprintf(filename, filename_format, ap);
#else
   char *filename;
   vasprintf(&filename, filename_format, ap);
#endif
   std::cout << "Writing " << filename << std::endl;
   
   std::ofstream output(filename, std::ofstream::binary);
#ifdef _MSC_VER
   delete [] filename;
#else
   std::free(filename);
#endif
   va_end(ap);

   if(!output.good()) return false;
    
    std::vector<Vec3d> normals;
    std::vector<Vec3st> normal_indices;
    generate_normals(x, mesh.m_tris, normals, normal_indices);

   output<<"# generated by editmesh"<<std::endl;
   for(unsigned int i=0; i<x.size(); ++i)
      output<<"v "<<x[i]<<std::endl;
   for(unsigned int i=0; i<normals.size(); ++i)
      output<<"vn "<<normals[i]<<std::endl;
   for(unsigned int t=0; t<mesh.m_tris.size(); ++t)
      output<<"f "<<mesh.m_tris[t][0]+1<<"//"<<normal_indices[t][0]+1<<" "<<mesh.m_tris[t][1]+1<<"//"<<normal_indices[t][1]+1<<" "<<mesh.m_tris[t][2]+1<<"//"<<normal_indices[t][2]+1<<std::endl; // correct for 1-based indexing in OBJ files
   return output.good();
}


// ---------------------------------------------------------
///
/// Write mesh in Wavefront OBJ format
///
// ---------------------------------------------------------

bool write_objfile_per_region(const NonDestructiveTriMesh &mesh, const std::vector<Vec3d> &x, int label, const std::set<int> & excluding_regions, const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);
#ifdef _MSC_VER
   int len=_vscprintf(filename_format, ap) +1;// _vscprintf doesn't count terminating '\0'
   char *filename=new char[len];
   vsprintf(filename, filename_format, ap);
#else
   char *filename;
   vasprintf(&filename, filename_format, ap);
#endif
   std::cout << "Writing " << filename << std::endl;

   std::ofstream output(filename);
#ifdef _MSC_VER
   delete [] filename;
#else
   std::free(filename);
#endif
   va_end(ap);

   if(!output.good()) return false;

   std::vector<LosTopos::Vec3d> new_points;
   std::vector<LosTopos::Vec3st> new_tris;
   std::vector<bool> vert_needed(x.size(), false);

   for(size_t i = 0; i < mesh.m_tris.size(); ++i) {
      Vec2i cur_label = mesh.get_triangle_label(i);
      if(cur_label[0] == label || cur_label[1] == label) {
         Vec3st tri = mesh.m_tris[i];
         for(int j = 0; j < 3; ++j) {
            vert_needed[tri[j]] = true;
         }   
      }
   }

   
   std::vector<int> new_indices(x.size(), -1);
   
   for(size_t i = 0; i < vert_needed.size(); ++i) {
      if(vert_needed[i]) {
         new_points.push_back(x[i]);
         new_indices[i] = (int)new_points.size()-1;
      }
      else 
         new_indices[i] = -1;
      
   }

   for(size_t i = 0; i < mesh.m_tris.size(); ++i) {
      Vec2i cur_label = mesh.get_triangle_label(i);
      if((cur_label[0] == label || cur_label[1] == label) && (excluding_regions.find(cur_label[0]) == excluding_regions.end() && excluding_regions.find(cur_label[1]) == excluding_regions.end())) {
         Vec3st old_tri = mesh.m_tris[i];
         Vec3st new_tri(new_indices[old_tri[0]], new_indices[old_tri[1]], new_indices[old_tri[2]]);
         //swap the orientation depending on which label is on the "front", for good measure.
         if(cur_label[1] == label)
            std::swap(new_tri[1], new_tri[2]);
         new_tris.push_back(new_tri);
      }
   }
   
    std::vector<Vec3d> normals;
    std::vector<Vec3st> normal_indices;
    generate_normals(new_points, new_tris, normals, normal_indices);
    
   output<<"# generated by VoronoiFluid3D - meeting all your wacky fluid sim needs since 2010"<<std::endl;
   for(unsigned int i=0; i<new_points.size(); ++i)
      output<<"v "<<new_points[i]<<std::endl;
   for(unsigned int i=0; i<normals.size(); ++i)
      output<<"vn "<<normals[i]<<std::endl;
   for(unsigned int t=0; t<new_tris.size(); ++t)
      output<<"f "<<new_tris[t][0]+1<<"//"<<normal_indices[t][0]+1<<" "<<new_tris[t][1]+1<<"//"<<normal_indices[t][1]+1<<" "<<new_tris[t][2]+1<<"//"<<normal_indices[t][2]+1<<std::endl; // correct for 1-based indexing in OBJ files
   output.close();

   return output.good();
}

bool write_objfile_per_region_pair(const NonDestructiveTriMesh &mesh, const std::vector<Vec3d> &x, Vec2i label, const char *filename_format, ...)
{
    va_list ap;
    va_start(ap, filename_format);
#ifdef _MSC_VER
    int len=_vscprintf(filename_format, ap) +1;// _vscprintf doesn't count terminating '\0'
    char *filename=new char[len];
    vsprintf(filename, filename_format, ap);
#else
    char *filename;
    vasprintf(&filename, filename_format, ap);
#endif
    std::cout << "Writing " << filename << std::endl;
    
    std::ofstream output(filename);
#ifdef _MSC_VER
    delete [] filename;
#else
    std::free(filename);
#endif
    va_end(ap);
    
    if(!output.good()) return false;
    
    std::vector<LosTopos::Vec3d> new_points;
    std::vector<LosTopos::Vec3st> new_tris;
    std::vector<bool> vert_needed(x.size(), false);
    
    for(size_t i = 0; i < mesh.m_tris.size(); ++i) {
        Vec2i cur_label = mesh.get_triangle_label(i);
        if(cur_label == label || cur_label == Vec2i(label[1], label[0])) {
            Vec3st tri = mesh.m_tris[i];
            for(int j = 0; j < 3; ++j) {
                vert_needed[tri[j]] = true;
            }   
        }
    }
    
    
    std::vector<int> new_indices(x.size(), -1);
    
    for(size_t i = 0; i < vert_needed.size(); ++i) {
        if(vert_needed[i]) {
            new_points.push_back(x[i]);
            new_indices[i] = (int)new_points.size()-1;
        }
        else 
            new_indices[i] = -1;
        
    }
    
    for(size_t i = 0; i < mesh.m_tris.size(); ++i) {
        Vec2i cur_label = mesh.get_triangle_label(i);
        if(cur_label == label || cur_label == Vec2i(label[1], label[0])) {
            Vec3st old_tri = mesh.m_tris[i];
            Vec3st new_tri(new_indices[old_tri[0]], new_indices[old_tri[1]], new_indices[old_tri[2]]);
            //swap the orientation depending on which label is on the "front", for good measure.
            if(cur_label[1] == label[0])
                std::swap(new_tri[1], new_tri[2]);
            new_tris.push_back(new_tri);
        }
    }
    
    /* std::vector<Vec3d> normals;
    std::vector<Vec3st> normal_indices;
    generate_normals(new_points, new_tris, normals, normal_indices);*/
    
    output<<"# generated by VoronoiFluid3D - meeting all your wacky fluid sim needs since 2010"<<std::endl;
    for(unsigned int i=0; i<new_points.size(); ++i)
        output<<"v "<<new_points[i]<<std::endl;
    /*for(unsigned int i=0; i<normals.size(); ++i)
        output<<"vn "<<normals[i]<<std::endl;*/
    //for(unsigned int t=0; t<new_tris.size(); ++t)
    //    output<<"f "<<new_tris[t][0]+1<<"//"<<normal_indices[t][0]+1<<" "<<new_tris[t][1]+1<<"//"<<normal_indices[t][1]+1<<" "<<new_tris[t][2]+1<<"//"<<normal_indices[t][2]+1<<std::endl; // correct for 1-based indexing in OBJ files
    for(unsigned int t=0; t<new_tris.size(); ++t)
       output<<"f "<<new_tris[t][0]+1<< " "<<new_tris[t][1]+1<<" "<<new_tris[t][2]+1<<std::endl; // correct for 1-based indexing in OBJ files
    output.close();
    
    return output.good();
}

bool write_objfile_excluding_regions(const NonDestructiveTriMesh &mesh, const std::vector<Vec3d> &x, const std::set<int> & labels, const char *filename_format, ...)
{
    va_list ap;
    va_start(ap, filename_format);
#ifdef _MSC_VER
    int len=_vscprintf(filename_format, ap) +1;// _vscprintf doesn't count terminating '\0'
    char *filename=new char[len];
    vsprintf(filename, filename_format, ap);
#else
    char *filename;
    vasprintf(&filename, filename_format, ap);
#endif
    std::cout << "Writing " << filename << std::endl;
    
    std::ofstream output(filename);
#ifdef _MSC_VER
    delete [] filename;
#else
    std::free(filename);
#endif
    va_end(ap);
    
    if(!output.good()) return false;
    
    std::vector<LosTopos::Vec3d> new_points;
    std::vector<LosTopos::Vec3st> new_tris;
    std::vector<bool> vert_needed(x.size(), false);
    
    for(size_t i = 0; i < mesh.m_tris.size(); ++i) {
        Vec2i cur_label = mesh.get_triangle_label(i);
        if(labels.find(cur_label[0]) == labels.end() && labels.find(cur_label[1]) == labels.end()) {
            Vec3st tri = mesh.m_tris[i];
            for(int j = 0; j < 3; ++j) {
                vert_needed[tri[j]] = true;
            }   
        }
    }
    
    
    std::vector<int> new_indices(x.size(), -1);
    
    for(size_t i = 0; i < vert_needed.size(); ++i) {
        if(vert_needed[i]) {
            new_points.push_back(x[i]);
            new_indices[i] = (int)new_points.size()-1;
        }
        else 
            new_indices[i] = -1;
        
    }
    
    for(size_t i = 0; i < mesh.m_tris.size(); ++i) {
        Vec2i cur_label = mesh.get_triangle_label(i);
        if(labels.find(cur_label[0]) == labels.end() && labels.find(cur_label[1]) == labels.end()) {
            Vec3st old_tri = mesh.m_tris[i];
            Vec3st new_tri(new_indices[old_tri[0]], new_indices[old_tri[1]], new_indices[old_tri[2]]);
            new_tris.push_back(new_tri);
        }
    }
    
    std::vector<Vec3d> normals;
    std::vector<Vec3st> normal_indices;
    generate_normals(new_points, new_tris, normals, normal_indices);
    
    output<<"# generated by VoronoiFluid3D - meeting all your wacky fluid sim needs since 2010"<<std::endl;
    for(unsigned int i=0; i<new_points.size(); ++i)
        output<<"v "<<new_points[i]<<std::endl;
    for(unsigned int i=0; i<normals.size(); ++i)
        output<<"vn "<<normals[i]<<std::endl;
    for(unsigned int t=0; t<new_tris.size(); ++t)
        output<<"f "<<new_tris[t][0]+1<<"//"<<normal_indices[t][0]+1<<" "<<new_tris[t][1]+1<<"//"<<normal_indices[t][1]+1<<" "<<new_tris[t][2]+1<<"//"<<normal_indices[t][2]+1<<std::endl; // correct for 1-based indexing in OBJ files
    output.close();
    
    return output.good();
}


// ---------------------------------------------------------
///
/// Helper for reading OBJ file
///
// ---------------------------------------------------------

static bool read_int(const char *s, int &value, bool &leading_slash, int &position)
{
   leading_slash=false;
   for(position=0; s[position]!=0; ++position){
      switch(s[position]){
         case '/':
            leading_slash=true;
            break;
	 case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
            goto found_int;
      }
   }
   return false;
   
   found_int:
   value=0;
   for(;; ++position){
      switch(s[position]){
	 case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
            value=10*value+s[position]-'0';
            break;
         default:
            return true;
      }
   }

#ifndef _MSC_VER //this yields an annoying warning on VC++, so strip it.
   return true; // should never get here, but keeps compiler happy
#endif
}


// ---------------------------------------------------------
///
/// Helper for reading OBJ file
///
// ---------------------------------------------------------

static void read_face_list(const char *s, std::vector<int> &vertex_list)
{
   vertex_list.clear();
   int v, skip;
   bool leading_slash;
   for(int i=0;;){
      if(read_int(s+i, v, leading_slash, skip)){
         if(!leading_slash)
            vertex_list.push_back(v-1); // correct for 1-based index
         i+=skip;
      }else
         break;
   }
}

// ---------------------------------------------------------
///
/// Read mesh in Wavefront OBJ format
///
// ---------------------------------------------------------

bool read_objfile(NonDestructiveTriMesh &mesh, std::vector<Vec3d> &x, const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);

#ifdef _MSC_VER
   int len=_vscprintf(filename_format, ap) +1;// _vscprintf doesn't count // terminating '\0'
   char *filename=new char[len];
   vsprintf(filename, filename_format, ap);
#else
   char *filename;
   vasprintf(&filename, filename_format, ap);
#endif

   std::ifstream input(filename, std::ifstream::binary);

#ifdef _MSC_VER
   delete [] filename;
#else
   std::free(filename);
#endif

   va_end(ap);

   if(!input.good()) return false;

   x.clear();
   mesh.clear();

   char line[LINESIZE];
   std::vector<int> vertex_list;
   while(input.good()){
      input.getline(line, LINESIZE);
      switch(line[0]){
         case 'v': // vertex data
            if(line[1]==' '){
               Vec3d new_vertex;
               std::sscanf(line+2, "%lf %lf %lf", &new_vertex[0], &new_vertex[1], &new_vertex[2]);
               x.push_back(new_vertex);
            }
            break;
         case 'f': // face data
            if(line[1]==' '){
               read_face_list(line+2, vertex_list);
               for(int j=0; j<(int)vertex_list.size()-2; ++j)
                  mesh.m_tris.push_back(Vec3st(vertex_list[0], vertex_list[j+1], vertex_list[j+2]));
            }
            break;
      }
   }
   return true;
}

bool read_objfile(std::vector<Vec3st> &tris, std::vector<Vec3d> &x, const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);

#ifdef _MSC_VER
   int len=_vscprintf(filename_format, ap) +1;// _vscprintf doesn't count // terminating '\0'
   char *filename=new char[len];
   vsprintf(filename, filename_format, ap);
#else
   char *filename;
   vasprintf(&filename, filename_format, ap);
#endif

   std::ifstream input(filename, std::ifstream::binary);

#ifdef _MSC_VER
   delete [] filename;
#else
   std::free(filename);
#endif

   va_end(ap);

   if(!input.good()) return false;

   x.clear();
   tris.clear();

   char line[LINESIZE];
   std::vector<int> vertex_list;
   while(input.good()){
      input.getline(line, LINESIZE);
      
      // remove leading whitespaces
      while (line[0] == ' ')
      {
         char * p = line;
         do {
            *p = *(p + 1);
            p++;
         } while (*(p + 1));
      }
      
      switch(line[0]){
      case 'v': // vertex data
         if(line[1]==' '){
            Vec3d new_vertex;
            std::sscanf(line+2, "%lf %lf %lf", &new_vertex[0], &new_vertex[1], &new_vertex[2]);
            x.push_back(new_vertex);
         }
         break;
      case 'f': // face data
         if(line[1]==' '){
            read_face_list(line+2, vertex_list);
            for(int j=0; j<(int)vertex_list.size()-2; ++j)
               tris.push_back(Vec3st(vertex_list[0], vertex_list[j+1], vertex_list[j+2]));
         }
         break;
      }
   }
   return true;
}
// ---------------------------------------------------------
///
/// Write mesh in Renderman RIB format.
///
// ---------------------------------------------------------

bool write_ribfile(const NonDestructiveTriMesh &mesh, const std::vector<float> &x, const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);
   
#ifdef _MSC_VER
   int len=_vscprintf(filename_format, ap) +1;// _vscprintf doesn't count // terminating '\0'
   char *filename=new char[len];
   vsprintf(filename, filename_format, ap);
#else
   char *filename;
   vasprintf(&filename, filename_format, ap);
#endif
   
   std::ofstream output(filename, std::ofstream::binary);
#ifdef _MSC_VER
   delete [] filename;
#else
   std::free(filename);
#endif
   va_end(ap);

   if(!output.good()) return false;
    return write_ribfile(mesh, x, output);
}

// ---------------------------------------------------------
///
/// Write mesh in Renderman RIB format.
///
// ---------------------------------------------------------

bool write_ribfile(const NonDestructiveTriMesh &mesh, const std::vector<float> &x, std::ostream &output)
{
   output<<"# generated by editmesh"<<std::endl;
   output<<"PointsPolygons"<<std::endl;
   output<<" [ ";
   for(unsigned int i=0; i<mesh.m_tris.size(); ++i){
      output<<"3 ";
      if(i%38==37 && i!=mesh.m_tris.size()-1) output<<std::endl;
   }
   output<<"]"<<std::endl;
   output<<" [ ";
   for(unsigned int i=0; i<mesh.m_tris.size(); ++i){
      output<<mesh.m_tris[i]<<"  ";
      if(i%6==5 && i!=mesh.m_tris.size()-1) output<<std::endl;
   }
   output<<"]"<<std::endl;
   output<<" \"P\" [";
   for(unsigned int i=0; i<x.size(); ++i){
      output<<x[i]<<"  ";
      if(i%4==3 && i!=x.size()-1) output<<std::endl;
   }
   output<<"]"<<std::endl;
   
   return output.good();
}

// ---------------------------------------------------------
///
/// Write mesh in Renderman RIB format.
///
// ---------------------------------------------------------

bool write_ribfile(const NonDestructiveTriMesh &mesh, const std::vector<float> &x, FILE *output)
{
   fprintf( output, "# generated by editmesh\n" );
   fprintf( output, "PointsPolygons\n" );
   fprintf( output, " [ " );
   for(unsigned int i=0; i<mesh.m_tris.size(); ++i){
      fprintf( output, "3 " );
      if(i%38==37 && i!=mesh.m_tris.size()-1) fprintf( output, "\n" );
   }
   fprintf( output, "]\n" );
   fprintf( output, " [ " );
   for(unsigned int i=0; i<mesh.m_tris.size(); ++i){
      fprintf( output, " %ld %ld %ld ", (long)mesh.m_tris[i][0], (long)mesh.m_tris[i][1], (long)mesh.m_tris[i][2] );
      if(i%6==5 && i!=mesh.m_tris.size()-1) fprintf( output, "\n" ); 
   }
   fprintf( output, "]\n" );
   fprintf( output, " \"P\" [" );
   for(unsigned int i=0; i<x.size(); ++i){
      fprintf( output, " %f ", x[i] );
      if(i%4==3 && i!=x.size()-1) fprintf( output, "\n" ); 
   }
   fprintf( output, "]\n" );
   
   return true; 
}


// ---------------------------------------------------------
//
// Write mesh in PBRT format
//
// ---------------------------------------------------------

bool write_pbrtfile(const NonDestructiveTriMesh &mesh, const std::vector<float> &x, const char *filename_format, ...)
{
   va_list ap;
   va_start(ap, filename_format);
#ifdef _MSC_VER
   int len=_vscprintf(filename_format, ap) +1;// _vscprintf doesn't count // terminating '\0'
   char *filename=new char[len];
   vsprintf(filename, filename_format, ap);
#else
   char *filename;
   vasprintf(&filename, filename_format, ap);
#endif
   std::ofstream output(filename, std::ofstream::binary);
   
#ifdef _MSC_VER
   delete [] filename;
#else
   std::free(filename);
#endif
   
   va_end(ap);

   if(!output.good()) return false;
    return write_ribfile(mesh, x, output);
}

// ---------------------------------------------------------
//
// Write mesh in PBRT format
//
// ---------------------------------------------------------

bool write_pbrtfile(const NonDestructiveTriMesh &mesh, const std::vector<float> &x, std::ostream &output)
{
   output<<"# generated by editmesh"<<std::endl;

   //output<<"\"integer nlevels\" [3]"<<std::endl;
   output<<"\"point P\" ["<<std::endl;
   for(unsigned int i=0; i<x.size(); ++i){
      output<<x[i]<<"  ";
      if(i%4==3 && i!=x.size()-1) output<<std::endl;
   }
   output<<"]"<<std::endl;
   output<<" \"integer indices\" ["<<std::endl;
   for(unsigned int i=0; i<mesh.m_tris.size(); ++i){
      output<<mesh.m_tris[i]<<"  ";
      if(i%6==5 && i!=mesh.m_tris.size()-1) output<<std::endl;
   }
   output<<"]"<<std::endl;

   return output.good();
}





