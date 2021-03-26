// ---------------------------------------------------------
//
//  surftrack.h
//  Tyson Brochu 2008
//  Christopher Batty, Fang Da 2014
//
//  The SurfTrack class: a dynamic mesh with topological changes and mesh maintenance operations.
//
// ---------------------------------------------------------

#ifndef LOSTOPOS_SURFTRACK_H
#define LOSTOPOS_SURFTRACK_H

#include <dynamicsurface.h>
#include <edgecollapser.h>
#include <edgeflipper.h>
#include <edgesplitter.h>
#include <meshmerger.h>
#include <meshpincher.h>
#include <meshsmoother.h>
#include <meshcutter.h>
#include <t1transition.h>
#include <meshsnapper.h>

// ---------------------------------------------------------
//  Forwards and typedefs
// ---------------------------------------------------------

namespace LosTopos {

class SubdivisionScheme;
typedef std::vector<size_t> TriangleSet;

// ---------------------------------------------------------
//  Class definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// Structure for setting up a SurfTrack object with some initial parameters.  This is passed to the SurfTrack constructor.
///
// ---------------------------------------------------------

struct SurfTrackInitializationParameters
{
    
    ///  Constructor. Sets default values for parameters which are not likely to be specified.
    ///
    SurfTrackInitializationParameters();
    
    /// Elements closer than this are considered "near" (or proximate)
    ///
    double m_proximity_epsilon;
    
    /// Coefficient of friction to apply during collisions
    ///
    double m_friction_coefficient;
    
    /// Smallest triangle area to allow
    ///
    double m_min_triangle_area;
    
    /// Whether to enable T1 transition operations
    ///
    bool m_t1_transition_enabled;
    
    /// The velocity field callback
    ///
    T1Transition::VelocityFieldCallback * m_velocity_field_callback;
    
    /// Collision epsilon to use during mesh improvement operations (i.e. if any mesh elements are closer than this, the operation is 
    /// aborted).  NOTE: This should be greater than collision_epsilon, to prevent improvement operations from moving elements into 
    /// a collision configuration.
    ///
    double m_improve_collision_epsilon;
    
    /// The min edge length for a given vertex, as a ratio to its target edge length (e.g. for the old LosTopos this is typically 0.5)
    ///
    double m_min_to_target_ratio;
    
    /// The max edge length for a given vertex, as a ratio to its target edge length (e.g. for the old LosTopos this is typically 1.5)
    ///
    double m_max_to_target_ratio;

    /// The coefficient for the curvature's effect on target edge length: target <= coef / max(abs(kappa))
    ///
    double m_target_edge_length_coef_curvature;
    
    /// The coefficient for the velocity's effect on target edge length: target <= coef / ||laplacian v||
    ///
    double m_target_edge_length_coef_velocity;
    
    /// The upper bound of the ratio between the target edge lengths on adjacent vertices, used in the grading field smoothing step to eliminate abrupt spatial variation of the target edge length
    ///
    double m_max_adjacent_target_edge_length_ratio;

    /// Whether to set the min and max edge lengths as fractions of the initial average edge length
    ///
    bool m_use_fraction;
    
    // If use_fraction is true, the following three values are taken to be fractions of the average edge length of the new surface.
    // If use_fraction is false, these are absolute.
    
    /// In the isotropic adaptive scenario every vertex has a target edge length (from which the min and max are computed) determined by its (scalar) grading field. This parameter specifies the lower bound.
    ///
    double m_min_target_edge_length;
    
    /// In the isotropic adaptive scenario every vertex has a target edge length (from which the min and max are computed) determined by its (scalar) grading field. This parameter specifies the upper bound.
    ///
    double m_max_target_edge_length;
    
    /// set true to use the adaptive remeshing (based on curvature and velocity) for solid interior too; set false to always use a very coarse target edge length for solid.
    ///
    bool m_refine_solid;
    
    /// set true to use the adaptive remeshing (based on curvature and velocity) for triple junction, which typically results in fine subdivisions depending on the contact angle; set false to always use m_max_target_edge_length regardless of curvature/velocity.
    ///
    bool m_refine_triple_junction;
    
    /// Maximum change in volume allowed for one operation
    ///
    double m_max_volume_change;
    
    /// Smallest interior angle at a triangle vertex allowed
    ///
    double m_min_triangle_angle;
    
    /// Largest interior angle at a triangle vertex allowed
    ///
    double m_max_triangle_angle;   
    

    /// Largest interior angle at a triangle vertex allowed before large-angle split pass kicks in
    ///
    double m_large_triangle_angle_to_split;
    
    ///////////////////////////////////////////////////////////////////////

    /// Whether to scale by curvature when computing edge lengths, in order to refine high-curvature regions
    ///
    bool m_use_curvature_when_splitting;

    /// Whether to scale by curvature when computing edge lengths, in order to coarsen low-curvature regions
    ///
    bool m_use_curvature_when_collapsing;
    
    /// The minimum curvature scaling allowed
    ///
    double m_min_curvature_multiplier;
    
    /// The maximum curvature scaling allowed
    ///
    double m_max_curvature_multiplier;
    
    /// boolean, whether to allow vertices to move during improvement
    int m_allow_vertex_movement_during_collapse;

    /// boolean, whether to allow vertices to move during improvement
    int m_perform_smoothing;
    
    /// Elements within this distance will trigger a merge attempt   
    ///
    double m_merge_proximity_epsilon;
    
    /// For Droplets: A different (typically smaller than or same as m_merge_proximity_epsilon) merging epsilon for liquid sheet puncture: it is harder for a liquid sheet to puncture than it is for an air sheet
    ///
    double m_merge_proximity_epsilon_for_liquid_sheet_puncture;

    /// Type of subdivision to use when collapsing or splitting (butterfly, quadric error minimization, etc.)
    ///
    SubdivisionScheme *m_subdivision_scheme;   
    
    /// Whether to enforce collision-free surfaces (including during mesh maintenance operations)
    ///
    bool m_collision_safety;
    
    /// Whether to allow changes in topology
    ///
    bool m_allow_topology_changes;
    
    /// Whether to allow non-manifold (edges incident on more than two triangles)
    ///
    bool m_allow_non_manifold;
    
    /// Whether to allow mesh improvement
    ///
    bool m_perform_improvement;

    /// Whether to perform remeshing on mesh boundary edges (in the case of open surfaces, e.g. sheets)
    ///
    bool m_remesh_boundaries;
  
    /// Pull apart distance, in terms of absolute length
    double m_pull_apart_distance;

    /// Whether to be verbose in outputting data
    ///
    bool m_verbose;
    
};

// ---------------------------------------------------------
///
/// Used to build a list of edges sorted in order of increasing length.
/// 
// ---------------------------------------------------------

struct SortableEdge
{    
   /// Constructor
   ///
   SortableEdge( size_t i, double edge_len ) : 
m_edge_index(i), 
   m_length(edge_len) 
{}

/// Comparison operator for sorting
///
bool operator<( const SortableEdge& other ) const
{
   return (this->m_length < other.m_length);
}

/// The index of the edge
///
size_t m_edge_index;

/// The stored length
///
double m_length;

};

// used to build a heap of vertices where the vertex with shortest target edge length can be retrieved (for the gradint field smoothing)
struct SortableVertex
{
    SortableVertex(size_t _v, double _l) : v(_v), l(_l) { }
    
    bool operator < (const SortableVertex & other) const { return this->l > other.l; }  // for use on an STL heap, which returns the largest value instead of the smallest.
    
    size_t v;
    double l;
};

// ---------------------------------------------------------
///
/// Used to build a list of edges sorted in order of increasing length.
/// 
// ---------------------------------------------------------

struct SortablePair
{    
    /// Constructor
    ///
    SortablePair( size_t i, size_t j, double sep_dist ) : 
    m_vertex0(i), 
    m_vertex1(j), 
    m_length(sep_dist) 
    {}
    
    /// Comparison operator for sorting
    ///
    bool operator<( const SortablePair& other ) const
    {
        return (this->m_length < other.m_length);
    }
    
    /// The index of the verts
    ///
    size_t m_vertex0, m_vertex1;
    
    /// The stored length
    ///
    double m_length;

};

// ---------------------------------------------------------
///
/// Used to build a sorted list of proximity data
/// 
// ---------------------------------------------------------

struct SortableProximity
{    
   /// Constructor
   ///
   SortableProximity( size_t face, size_t vertex, double sep_dist, bool isFaceVert ) : 
      m_index0(face),
      m_index1(vertex), 
      m_face_vert_proximity(isFaceVert),
      m_length(sep_dist)
   {
   }

   /// Comparison operator for sorting
   ///
   bool operator<( const SortableProximity& other ) const
   {
      return (this->m_length < other.m_length);
   }

   /// The indices of the relevant geometry
   ///
   size_t m_index0, m_index1;

   /// Whether it's an edge-edge or a face-vertex proximity

   bool m_face_vert_proximity;

   /// The stored length
   ///
   double m_length;

};

// ---------------------------------------------------------
///
/// Keeps track of a vertex removal or addition.  If it's an addition, it also points to the edge that was split to create it.
///
// ---------------------------------------------------------

struct VertexUpdateEvent
{
    /// Constructor
    ///
    VertexUpdateEvent(bool is_remove = false, 
                      size_t vertex_index = (size_t)~0, 
                      const Vec2st& split_edge = Vec2st((size_t)~0) ) :
    m_is_remove( is_remove ),
    m_vertex_index( vertex_index ),
    m_split_edge( split_edge )
    {}
    
    /// Tag for identifying a vertex removal
    ///
    static const bool VERTEX_REMOVE = true;
    
    /// Tag for identifying a vertex addition
    ///
    static const bool VERTEX_ADD = false;
    
    /// Whether this event is a vertex removal
    ///
    bool m_is_remove;
    
    /// The index of the vertex being added or removed
    ///
    size_t m_vertex_index;   
    
    /// If this is a vertex addition due to edge splitting, the edge that was split
    ///
    Vec2st m_split_edge;
    
};


// ---------------------------------------------------------
///
/// Keeps track of a triangle removal or addition. If addition, contains the three vertices that form the new triangle.
///
// ---------------------------------------------------------

struct TriangleUpdateEvent
{
    /// Constructor
    ///
    TriangleUpdateEvent(bool is_remove = false, 
                        size_t triangle_index = (size_t)~0, 
                        const Vec3st& triangle = Vec3st((size_t)~0) ) :
    m_is_remove( is_remove ),
    m_triangle_index( triangle_index ),
    m_tri( triangle )
    {}
    
    /// Tag for identifying a triangle removal
    ///
    static const bool TRIANGLE_REMOVE = true;
    
    /// Tag for identifying a triangle addition
    ///
    static const bool TRIANGLE_ADD = false;
    
    /// Whether this event is a triangle removal
    ///
    bool m_is_remove;
    
    /// The index of the triangle being added or removed
    ///
    size_t m_triangle_index;  
    
    /// If this is a triangle addition, the triangle added
    ///
    Vec3st m_tri;
    
};


// ---------------------------------------------------------
///
/// Keeps track of a triangle removal or addition. If addition, contains the three vertices that form the new triangle.
///
// ---------------------------------------------------------

struct MeshUpdateEvent
{
  enum EventType {
    EDGE_SPLIT,
    EDGE_FLIP,
    EDGE_COLLAPSE,
    EDGE_CUT,     //for fracturing/cutting
    FLAP_DELETE,  //remove non-manifold flap
    PINCH,        //separate singular vertex to allow topology change
    MERGE,        //zipper two edges together
    SNAP,         //combine two disjoint vertices into one
    FACE_SPLIT,   //subdivide a triangle into 3 triangles
    EDGE_POP,     //T1 transition (unused)
    VERTEX_POP,   //T1 transition
    
    ////////////////////////////////////////////////////////////

  };

  /// Constructors
  ///
  MeshUpdateEvent(EventType eType):m_type(eType), m_deleted_tris(0), m_created_tris(0), m_created_tri_data(0)
  {}

  /// What type of mesh event this is
  ///
  EventType m_type;

  /// The start and end vertices of the edge
  size_t m_v0, m_v1;
  
  // Another identifying vertex, needed for internal cuts
  size_t m_v2; 
    
  // A fourth vertex needed only for zippering
  size_t m_v3;

  /// The index of the triangles involved. 
  ///
  std::vector<size_t> m_deleted_tris;
  std::vector<size_t> m_created_tris;

  /// The data of the new triangles (vertex indices)
  ///
  std::vector<Vec3st> m_created_tri_data;
  
  /// The label data of the new triangles
  ///
  std::vector<Vec2i> m_created_tri_labels;
  
  /// Dirty triangles whose labels have been changed
  ///
  std::vector<std::pair<size_t, Vec2i> > m_dirty_tris;

  /// The indices of the vertices involved. 
  ///
  std::vector<size_t> m_deleted_verts;
  std::vector<size_t> m_created_verts;
  
  /// The positions of the created vertices
  ///
  std::vector<Vec3d> m_created_vert_data;

  /// The location of the final vertex (for a split, collapse, or possibly smooth)
  ///
  Vec3d m_vert_position;


};

// ---------------------------------------------------------
///
/// A DynamicSurface with topological and mesh maintenance operations.
///
// ---------------------------------------------------------

class SurfTrack : public DynamicSurface
{
public:
    
    /// Create a SurfTrack object from a set of vertices and triangles using the specified parameters
    ///
    SurfTrack(const std::vector<Vec3d>& vs, 
              const std::vector<Vec3st>& ts,
              const std::vector<Vec2i>& labels,
              const std::vector<Vec3d>& masses,
              const SurfTrackInitializationParameters& initial_parameters );
    
    /// Destructor
    ///
    ~SurfTrack();
    
private:
    
    /// Disallow copying and assignment by declaring private
    ///
    SurfTrack( const SurfTrack& );
    
    /// Disallow copying and assignment by declaring private
    ///
    SurfTrack& operator=( const SurfTrack& );
    
    
public:
    

    //
    // Mesh bookkeeping
    //
    
    /// Add a triangle to the surface.  Update the underlying TriMesh and acceleration grid. 
    ///
    size_t add_triangle(const Vec3st& t, const Vec2i& label);
    
    /// Remove a triangle from the surface.  Update the underlying TriMesh and acceleration grid. 
    ///
    void remove_triangle(size_t t); 

    /// Efficiently renumber a triangle (replace its verts) for defragging purposes.
    ///
    void renumber_triangle(size_t tri, const Vec3st& verts);
    
    /// Add a vertex to the surface.  Update the acceleration grid. 
    ///
    size_t add_vertex( const Vec3d& new_vertex_position, const Vec3d& new_vertex_mass );
    
    /// Remove a vertex from the surface.  Update the acceleration grid. 
    ///
    void remove_vertex(size_t v);
    
    /// Remove deleted vertices and triangles from the mesh data structures
    ///
    void defrag_mesh();
    void defrag_mesh_from_scratch(std::vector<size_t> & vertices_to_be_mapped);
    void defrag_mesh_from_scratch_manual(std::vector<size_t> & vertices_to_be_mapped);
    void defrag_mesh_from_scratch_copy(std::vector<size_t> & vertices_to_be_mapped);

    /// Check for labels with -1 as their value, or the same label on both sides.
    /// 
    void assert_no_bad_labels();

    //
    // Main operations
    //
    
    /// Run mesh maintenance operations
    ///
    void improve_mesh( );
    

    /// Run split'n'snap merging and t1-transition processing
    ///
    void topology_changes( );
    
    /// Run mesh cutting operations on a given set of edges
    ///
    void cut_mesh( const std::vector< std::pair<size_t, size_t> >& edges);


    //
    // Mesh cleanup
    //
    
    /// Check for and delete flaps and zero-area triangles among the given triangle indices, then separate singular vertices.
    ///
    void trim_degeneracies( std::vector<size_t>& triangle_indices );
    
    /// Check for and delete flaps and zero-area triangles among *all* triangles, then separate singular vertices.
    ///
    inline void trim_degeneracies();
    
    /// Fire an assert if any degenerate triangles or tets (flaps) are found.
    /// 
    void assert_no_degenerate_triangles();
    
    /// Detect any bad angle, i.e. out of range [m_min_triangle_angle, m_max_triangle_angle)
    bool triangle_with_bad_angle(size_t triangle);
    bool any_triangles_with_bad_angles();

    /// Compute the target edge length for one vertex
    ///
    double compute_vertex_target_edge_length(size_t vertex);
    
    /// Compute vertex edge length with external programs
    double compute_vertex_target_edge_length(size_t vertex, std::function<double(LosTopos::Vec3d)> vtx_el_function_of_idx, std::function<bool(LosTopos::Vec3d)> if_overwrite);

    /// Compute all target edge lengths, ensuring spatial smoothness
    ///
    void compute_all_vertex_target_edge_lengths();
    void compute_all_vertex_target_edge_lengths(std::function<double(LosTopos::Vec3d)> vtx_el_function_of_idx, std::function<bool(LosTopos::Vec3d)> if_overwrite);

    double vertex_target_edge_length(size_t v) const { assert(m_target_edge_lengths[v] > 0); return m_target_edge_lengths[v]; }
    double edge_target_edge_length(size_t e) const { return std::min(vertex_target_edge_length(m_mesh.m_edges[e][0]), vertex_target_edge_length(m_mesh.m_edges[e][1])); }
    
    double vertex_min_edge_length(size_t v) const { return vertex_target_edge_length(v) * m_min_to_target_ratio; }
    double vertex_max_edge_length(size_t v) const { return vertex_target_edge_length(v) * m_max_to_target_ratio; }
    double edge_min_edge_length  (size_t e) const { return edge_target_edge_length(e)   * m_min_to_target_ratio; }
    double edge_max_edge_length  (size_t e) const { return edge_target_edge_length(e)   * m_max_to_target_ratio; }

    /// Utility to compute the discrete (integral) laplacian of a given quantity at a vertex (using the cotan formula). Returns the laplacian value and the sum of weights (which is useful in the case of solving a Laplace's equation)
    ///
    template<class T>
    std::pair<T, double> laplacian(size_t vertex, const std::vector<T> & data) const;
    
    //
    // Member variables
    //
    
    /// Edge collapse operation object
    ///
    EdgeCollapser m_collapser;
    
    /// Edge split operation object
    ///
    EdgeSplitter m_splitter;
    
    /// Edge flip operation object
    ///
    EdgeFlipper m_flipper;
    
    /// Null-space surface smoothing
    /// 
    MeshSmoother m_smoother;
    
    /// Surface merging object
    ///
    MeshMerger m_merger;
    
    /// Surface splitting operation object
    ///
    MeshPincher m_pincher;
    
    /// Surface cutting (tearing) operation object
    ///
    MeshCutter m_cutter;
    
    /// Mesh snapping
    ///
    MeshSnapper m_snapper;

    /// T1 transition operation object
    ///
    T1Transition m_t1transition;
    
    /// An option to indicate whether T1 transition operations are enabled
    ///
    bool m_t1_transition_enabled;

    ////////////////////////////////////////////////////////////

    /// Collision epsilon to use during mesh improvement operations
    ///
    double m_improve_collision_epsilon;
    
    /// Maximum volume change allowed when flipping or collapsing an edge
    ///
    double m_max_volume_change;
    
    /// Minimum target edge length of the whole mesh.
    ///
    double m_min_target_edge_length;
    
    /// Maximum target edge length of the whole mesh.
    ///
    double m_max_target_edge_length;
    
    /// set true to use the adaptive remeshing (based on curvature and velocity) for solid interior too; set false to always use a very coarse target edge length for solid.
    ///
    bool m_refine_solid;
    
    /// set true to use the adaptive remeshing (based on curvature and velocity) for triple junction, which typically results in fine subdivisions depending on the contact angle; set false to always use m_max_target_edge_length regardless of curvature/velocity.
    ///
    bool m_refine_triple_junction;
    
    /// Local ratio between min/max edge length and the target edge length at a vertex. Edges incident to this vertex out of this range will be collapsed/split.
    ///
    double m_min_to_target_ratio;
    double m_max_to_target_ratio;
    
    /// The target edge length for each vertex
    ///
    std::vector<double> m_target_edge_lengths;
    
    /// The coefficient for the curvature's effect on target edge length: target <= coef / max(abs(kappa))      (coef is dimensionless)
    ///
    double m_target_edge_length_coef_curvature;
    
    /// The coefficient for the velocity's effect on target edge length: target <= coef / ||laplacian v||       (coef has unit s^-1 and the user is responsible for folding the time step into it)
    ///
    double m_target_edge_length_coef_velocity;
    
    /// The upper bound of the ratio between the target edge lengths on adjacent vertices, used in the grading field smoothing step to eliminate abrupt spatial variation of the target edge length
    ///
    double m_max_adjacent_target_edge_length_ratio;
    
    /// Elements within this distance will trigger a merge attempt
    ///
    double m_merge_proximity_epsilon;

    /// For Droplets: A different (typically smaller) merging epsilon for liquid sheet puncture: it is harder for a liquid sheet to puncture than it is for an air sheet
    ///
    double m_merge_proximity_epsilon_for_liquid_sheet_puncture;

    /// Try to prevent triangles with area less than this
    ///
    double m_min_triangle_area;
    
    /// Don't create triangles with angles less than this.  If angles less than this do exist, try to remove them.
    ///
    double m_min_triangle_angle;
    
    /// Don't create triangles with angles greater than this.  If angles greater than this do exist, try to remove them.
    ///
    double m_max_triangle_angle;
    
    /// Don't create triangles with angles less than this.  Use cosine so we can just compare to dot product. Should match angles above.
    ///
    double m_min_angle_cosine;

    /// Don't create triangles with angles greater than this. Use cosine so we can just compare to dot product. Should match angles above.
    ///
    double m_max_angle_cosine;


    /// Some weaker bounds to use in aggressive mode.
    ///
    double m_hard_min_edge_len, m_hard_max_edge_len;

    /// Split triangles with angles greater than this.
    ///
    double m_large_triangle_angle_to_split;
    
    /// Interpolation scheme, determines edge midpoint location
    ///
    SubdivisionScheme *m_subdivision_scheme;
    
    /// If we allocate our own SubdivisionScheme object, we must delete it in this object's deconstructor.
    ///
    bool should_delete_subdivision_scheme_object;
    
    /// Triangles which are involved in connectivity changes which may introduce degeneracies
    ///
    std::vector<size_t> m_dirty_triangles;
    
    /// Whether to allow merging and separation
    ///
    bool m_allow_topology_changes;
    
    /// Whether to allow non-manifold (edges incident on more than two triangles)
    ///
    bool m_allow_non_manifold;
    
    /// Whether to perform adaptivity operations
    ///
    bool m_perform_improvement;

    /// Whether to perform remeshing on mesh boundary edges (in the case of open surfaces, e.g. sheets)
    ///
    bool m_remesh_boundaries;

    /// Flag that dictates whether to aggressively pursue good angles for only the worst offenders.
    /// 
    bool m_aggressive_mode;

    /// boolean, whether to allow vertices to move during collapses (i.e. use points other than the endpoints)
    int m_allow_vertex_movement_during_collapse;

    /// boolean, whether to do null space smoothing on vertex positions
    int m_perform_smoothing;
    
    
    //Return whether the given edge is a feature as determined by dihedral angles.
    bool edge_is_feature(size_t edge) const;
    bool edge_is_feature(size_t edge, const std::vector<Vec3d>& cached_normals) const;
    
    //Return whether the vertex is on a feature, as determined by dihedral angles
    int vertex_feature_edge_count(size_t vertex) const;
    int vertex_feature_edge_count(size_t vertex, const std::vector<Vec3d>& cached_normals) const;
    

    // Return whether the incident feature curves on a vertex form a smooth ridge (implying # of feature curves = 2)
    bool vertex_feature_is_smooth_ridge(size_t vertex) const;

  
    /// Mesh update event callback
    ///
    class MeshEventCallback
    {
    public:
        virtual void pre_collapse(const SurfTrack & st, size_t e, void ** data) { }
        virtual void post_collapse(const SurfTrack & st, size_t e, size_t merged_vertex, void * data) { }

        virtual void pre_split(const SurfTrack & st, size_t e, void ** data) { }
        virtual void post_split(const SurfTrack & st, size_t e, size_t new_vertex, void * data) { }

        virtual void pre_flip(const SurfTrack & st, size_t e, void ** data) { }
        virtual void post_flip(const SurfTrack & st, size_t e, void * data) { }

        virtual void pre_t1(const SurfTrack & st, size_t v, void ** data) { }
        virtual void post_t1(const SurfTrack & st, size_t v, size_t a, size_t b, void * data) { }

        virtual void pre_facesplit(const SurfTrack & st, size_t f, void ** data) { }
        virtual void post_facesplit(const SurfTrack & st, size_t f, size_t new_vertex, void * data) { }

        virtual void pre_snap(const SurfTrack & st, size_t v0, size_t v1, void ** data) { }
        virtual void post_snap(const SurfTrack & st, size_t v_kept, size_t v_deleted, void * data) { }

        virtual void pre_smoothing(const SurfTrack & st, void ** data) { }
        virtual void post_smoothing(const SurfTrack & st, void * data) { }

        virtual std::ostream & log() { return std::cout; }
    };
    
    MeshEventCallback * m_mesheventcallback;
    
    class SolidVerticesCallback
    {
    public:
        virtual bool generate_collapsed_position(SurfTrack & st, size_t v0, size_t v1, Vec3d & pos) = 0;
        
        virtual bool generate_split_position(SurfTrack & st, size_t v0, size_t v1, Vec3d & pos) = 0;
        
        virtual bool generate_snapped_position(SurfTrack & st, size_t v0, size_t v1, Vec3d & pos) = 0;
        
        virtual Vec3c generate_collapsed_solid_label(SurfTrack & st, size_t v0, size_t v1, const Vec3c & label0, const Vec3c & label1) = 0;
        
        virtual Vec3c generate_split_solid_label(SurfTrack & st, size_t v0, size_t v1, const Vec3c & label0, const Vec3c & label1) = 0;
        
        virtual Vec3c generate_snapped_solid_label(SurfTrack & st, size_t v0, size_t v1, const Vec3c & label0, const Vec3c & label1) = 0;
        
        virtual bool generate_edge_popped_positions(SurfTrack & st, size_t oldv, const Vec2i & cut, Vec3d & pos_upper, Vec3d & pos_lower) = 0;
        
        virtual bool generate_vertex_popped_positions(SurfTrack & st, size_t oldv, int A, int B, Vec3d & pos_a, Vec3d & pos_b) = 0;
      
        virtual bool solid_edge_is_feature(const SurfTrack & st, size_t edge) = 0;
    };
    
    SolidVerticesCallback * m_solid_vertices_callback;
        
    /// History of vertex removal or addition events
    ///
    std::vector<VertexUpdateEvent> m_vertex_change_history;
    
    /// History of triangle removal or addition events
    ///    
    std::vector<TriangleUpdateEvent> m_triangle_change_history;
    
    /// Map of triangle indices, mapping pre-defrag triangle indices to post-defrag indices (deprecated; see defrag_mesh())
    ///
    std::vector<Vec2st> m_defragged_triangle_map;
    
    /// Map of vertex indices, mapping pre-defrag vertex indices to post-defrag indices (deprecated; see defrag_mesh())
    ///
    std::vector<Vec2st> m_defragged_vertex_map;

    /// History of higher level mesh update events (split, flip, collapse, smooth)
    ///    
    std::vector<MeshUpdateEvent> m_mesh_change_history;

    
};

// ---------------------------------------------------------
//  Inline functions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// Search the entire mesh for non-manifold elements and remove them
/// NOTE: SHOULD USE THE VERSION THAT ACCEPTS A SET OF TRIANGLE INDICES INSTEAD.
///
// ---------------------------------------------------------

inline void SurfTrack::trim_degeneracies()
{
    
    std::vector<size_t> triangle_indices;
    triangle_indices.resize( m_mesh.num_triangles() );
    for ( size_t i = 0; i < triangle_indices.size(); ++i )
    {
        triangle_indices[i] = i;
    }
    
    trim_degeneracies( triangle_indices );
}

}

#endif

