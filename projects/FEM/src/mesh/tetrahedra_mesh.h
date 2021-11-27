#pragma once

#include <matrix_helper.hpp>
#include <differentiable_SVD.h>
#include <Eigen/Core>
#include <iostream>
#include <map>
#include <memory>
#include <set>

/**
 * @class <TetrahedraMesh>
 * @brief A tetrahedron class defining the geometric properties and mass distribution of a tetrahedron mesh.
 * 
 * For initializing a tetrahedron mesh for simulation, you should at least call following three methods LoadTetrahedraFromFile,SetConstrainedNodes,SetDensityParam
 */ 
class TetrahedraMesh {
public:
    /**
     * @brief Loading the tetrahedron mesh information from file.
     * @param mesh_ptr The mesh pointer.
     * @param elm_file The path of the .ele file, defining the topology of the mesh.
     * @param node_file The path of .node file, defining the positions of each vertices of the mesh.
     * @param bou_file The path of .bou file, defining the boundary vertex indices of mesh.
     * @param normalized If true, the method will uniformly scale the whole mesh, so that the height of the mesh if 0.5m.
     * @param center If true, the method will translate the whole mesh so that the center of the bounding box is placed the coordinate's origin.
     */
    static int LoadTetrahedraFromFile(std::shared_ptr<TetrahedraMesh>& mesh_ptr,const char* elm_file,const char* node_file,const char* bou_file);
    /**
     * @brief Setting the indices of boundary vertices.
     * @param cons_node_indices indices of constrained vertices
     * @param nm_cons_nodes number of constrained vertices
     */
    void SetConstrainedNodes(const size_t* cons_node_indices, size_t nm_cons_nodes);

    inline size_t GetNumSurfaceDoFs() const {return surface_dofs.size();}
    inline const std::vector<size_t>& GetSurfaceDoFs() const {return surface_dofs;}
    inline size_t GetNumSurfaceVertices() const {return surface_dofs.size() / 3;}

    void SetConstrainedDoFs(const size_t* cons_dofs_indices,size_t nm_cons_nodes);
    /**
     * @brief Set the density distribution of the whole mesh
     * @param rhos_handles the density handles's buffer, for homogeneous matererial, there is only one handle and the length of the handle buffer is 1.
     * @param nm_handles The number of density handles, for homogeneous material, the nm_handles should be 1.
     * @param weight The interpolation matrix for interploting the handles' density value to density of each tet, it is a value buffer of a row-major Ne x Nh matrix
     * with Ne the number of elements and Nh the number of handles. For homogeneous material, the weight matrix should be a Ne x 1 vector, with all the coefficients to be 1.
     */
    void SetDensityParam(const FEM_Scaler* rhos_handles, size_t nm_handles = 1, const FEM_Scaler* weight = nullptr);
    /**
     * @brief setting the density parameter for homogeneous material.
     */
    inline void SetDensityParam(FEM_Scaler rho) {
        std::fill(elm_rho.begin(), elm_rho.end(), rho);
        update_mass();
    }
    inline void SetElmAnisotropic(size_t elm_id,const Mat3x3d& elm_orient,const Vec3d& weight){
        elm_anisotropic_orients[elm_id] = elm_orient;
        elm_anisotropic_weight[elm_id] = weight;
    }
    inline Mat3x3d GetElmAnisotropicOrient(size_t elm_id) const {
        return elm_anisotropic_orients[elm_id];
    }
    inline Vec3d GetElmAnisotropicWeight(size_t elm_id) const {
        return elm_anisotropic_weight[elm_id];
    }
    inline Mat3x3d GetElmActivation(size_t elm_id) const {
        return elm_activation[elm_id];
    }
    inline void SetElmActivation(size_t elm_id,const Mat3x3d& Act){
        elm_activation[elm_id] = Act;
    }
    /**
     * @brief Loading the indices of the marked points of the mesh
     * @param filename The path of .vis file, defining the indices of the marked points.
     * @param vsindices The buffer to store the indices of the marked points.
     */
    static int LoadVisIndices(const char* filename,std::vector<size_t>& vsindices);
    /**
     * @brief When we load the mesh from the file, we might normalized the mesh by scaling it, this function return the scale factor when you import the mesh from file.
     * @return the scale factor of the mesh
     * @see LoadTetrahedraFromFile
     */
    inline FEM_Scaler GetMeshScale() {return mesh_scale;}
    /**
     * @brief There is a connectivity matrix of the mesh vertices, which has the same sparse matrix structure as the tangent stiffness matrix, by givening the value buffer of the 
     * tangent stiffness matrix or the Jacobi matrix of the integrator, we can use this map function to map any such value buffer to the corresponding sparse matrix of tangent
     * stiffness matrix or Jacobi matrix. This method is a const method, you can do any read-only operation with the mapped sparse matrix, but no writing operation is allowed.
     * @param matrix_buffer The matrix value buffer.
     * @return The const sparse matrix constructed from the matrix buffer.
     * @see <MapConnMatrixRef,MapUcConnMatrix>
     */
    inline Eigen::Map<const SpMat> MapConnMatrix(const SPMAT_SCALER* matrix_buffer) const{
        size_t n = GetNumDoF();
        return Eigen::Map<const SpMat>(n,n,conn_matrix.nonZeros(),
            conn_matrix.outerIndexPtr(),
            conn_matrix.innerIndexPtr(),matrix_buffer);
    }
    /**
     * @brief There is a connectivity matrix of the mesh vertices, which has the same sparse matrix structure as the tangent stiffness matrix, by givening the value buffer of the
     * tangent stiffness matrix or the Jacobi matrix of the integrator, we can use this map function to map any such value buffer to the corresponding sparse matrix of tangent
     * stiffness matrix or Jacobi matrix. You can do both writing and reading operation on this mapped sparse matrix.
     * @param matrix_buffer The matrix value buffer.
     * @return The sparse matrix constructed from the matrix buffer.
     * @see <MapConnMatrix,MapUcConnMatrix>
     */
    inline Eigen::Map<SpMat> MapConnMatrixRef(SPMAT_SCALER* matrix_buffer){
        size_t n = GetNumDoF();
        return Eigen::Map<SpMat>(n,n,conn_matrix.nonZeros(),
            conn_matrix.outerIndexPtr(),
            conn_matrix.innerIndexPtr(),matrix_buffer);
    }
    /**
     * @brief This function map the value buffer to a sparse matrix with the same structure as the tangent stiffness matrix and integrator's Jacobi matrix with the constrained 
     * dofs removed.
     * @param matrix_buffer the matrix value buffer
     * @return The sparse matrix constructed from the matrix buffer.
     * @see <MapConnMatrix,MapConnMatrixRef>
     */
    inline Eigen::Map<const SpMat> MapUcConnMatrix(const SPMAT_SCALER* matrix_buffer) const{
        size_t nuc = GetNumUCDoF();
        return Eigen::Map<const SpMat>(nuc,nuc,ucconn_matrix.nonZeros(),
            ucconn_matrix.outerIndexPtr(),
            ucconn_matrix.innerIndexPtr(),matrix_buffer);
    }
    /**
     * @brief Query the vertices poisition of the whole mesh
     * @return The vertices' positions buffer of the mesh, and the buffer is read-only
     */
    inline const FEM_Scaler* GetVertices(void) const {return vertices.data();}
    /**
     * @brief Query the poisition of a specific vertex
     * @param v_id the index of queried vertex
     * @return position of a specific vertex, and the returned position is read-only
     */
    inline Eigen::Map<const Vec3d> GetVertex(size_t v_id) const {return Eigen::Map<const Vec3d>(vertices.data() + v_id*3);}

    /**
     * @brief Quecy the topology of the whole mesh.
     * @return the topology of the whole mesh, and the returned buffer is read-only.
     */
    inline const size_t* GetElements(void) const {return elements.data();}

    /**
     * @brief Quecy the topology of a specific element, i.e the vertices' indices of specified tet.
     * @param elm_id The queried element's index
     * @return the queried element
     */
    inline const size_t* GetElement(size_t elm_id) const {return elements.data() + elm_id*4;}

    /**
     * @return The number of the verices.
     */
    inline size_t GetNumVertices() const { return vertices.size()/3;}
    /**
     * @return The number of the elements.
     */
    inline size_t GetNumElements() const {return elements.size()/4;}
    // inline size_t GetNumEdges() const {return connectivity.size();}
    /**
     * @return The number of DOFs of the mesh.
     */
    inline size_t GetNumDoF() const { return vertices.size(); }
    /**
     * @return The boundary vertices' indices.
     */
    inline const size_t* GetConVertIds() const {return bou_vert_ids.data();}
    // inline const std::set<Triplet,triplet_cmp>& GetConnectivity() const {return connectivity;}
    /**
     * @return The connectivity matrix.
     */
    inline const SpMat& GetConnMatrix() const {return conn_matrix;}
    /**
     * @return The bounding box of the mesh.
     */
    inline const Vec6d& GetBoundingBox() const {return bounding_box;}
    /**
     * @return The connectivity of the mesh with constrained dofs removed.
     */
    inline const std::set<Triplet,triplet_cmp>& GetUcConnectivity(){return uc_connectivity;}
    /**
     * @return The connectivity matrix of the mesh with constrained dofs removed.
     */
    inline const SpMat& GetUcConnMatrix() const {return ucconn_matrix;}
    /**
     * @return Get the constrained dofs, which is read-only
     */
    inline const size_t* GetCDoF() const { return cons_dofs.data();}
    /**
     * @return Get the unconstrained dofs.
     */
    inline const size_t* GetUCDoF() const { return ucons_dofs.data();}
    /**
     * @return Get the number of constrained dofs.
     */
    inline size_t GetNumCDoF() const { return cons_dofs.size();}
    /**
     * @return Get the number of unconstrained dofs.
     */
    inline size_t GetNumUCDoF() const { return ucons_dofs.size();}
    /**
     * @brief Removing the constrained dofs from the sparse matrix.
     * @param from The sparse matrix with constrained dofs to be removed
     * @param to the sparse matrix to store the result
     */
    void RemoveConstrainedDoF(const SpMat &from, SpMat &to) const;
    /**
     * @brief Removing the constrained dofs from the sparse matrix.
     * @param from The sparse matrix with constrained dofs to be removed
     * @param to the sparse matrix to store the result
     */
    void RemoveConstrainedDoF(const SpMat &from, Eigen::Map<SpMat> &to) const;
    /**
     * @brief Removing the constrained dofs from the sparse matrix.
     * @param from The sparse matrix value buffer with constrained dofs to be removed
     * @param to the sparse matrix value buffer to store the result
     */
    void RemoveConstrainedDoF(const FEM_Scaler* sp_from,FEM_Scaler* sp_to) const;
    /**
     * @brief Assemble a 12x1 vector, e.g the force vector of a tet element, to a global vector, e.g the force vector of the whole mesh.
     * @param elm_vec The element vector
     * @param elm_id The elment index
     * @param global_vec the value buffer of the global vector
     */
    void AssembleElmVectorAdd(const Vec12d& elm_vec,size_t elm_id,FEM_Scaler* global_vec) const;
    /**
     * @brief Assemble a 12x12 matrix, e.g the tangent stiffness matrix of a tet element, to a global sparse matrix, e.g the tangent stiffness matrix of the whole mesh.
     * @param elm_hessian The element matrix
     * @param elm_id The elment index
     * @param global_hessian the value buffer of the global sparse matrix
     */
    void AssembleElmMatrixAdd(const Mat12x12d& elm_hessian,size_t elm_id,SpMat& global_hessian) const;
    /**
     * @brief Assemble a 12x12 matrix, e.g the tangent stiffness matrix of a tet element, to a global sparse matrix, e.g the tangent stiffness matrix of the whole mesh.
     * @param elm_hessian The element matrix
     * @param elm_id The elment index
     * @param global_hessian the value buffer of the global sparse matrix
     */
    void AssembleElmMatrixAdd(const Mat12x12d& elm_hessian,size_t elm_id,Eigen::Map<SpMat>& global_hessian) const;
    /**
     * @brief Assemble a 12x12 matrix, e.g the tangent stiffness matrix of a tet element, to a global sparse matrix, e.g the tangent stiffness matrix of the whole mesh.
     * @param elm_hessian The element matrix
     * @param elm_id The elment index
     * @param global_hessian the value buffer of the global sparse matrix
     */
    void AssembleElmMatrixAdd(const Mat12x12d& elm_hessian,size_t elm_id,FEM_Scaler *sp_data) const;
    /**
     * @brief Assemble a 12 x np matrix, e.g the jacobi matrix of force w.r.t the matererial parameters of a tet element, to a global sparse matrix,
     * e.g the Jaocbi matrix of the global force with repect to material params.
     * @param elm_J The element Jacobi matrix w.r.t certain parameters
     * @param elm_id The elment index
     * @param J the value buffer of the global Jacobi matrix w.r.t certain parameters
     */
    void AssembleElmJacobiAdd(const Mat12xXd& elm_J,size_t elm_id,MatXd& J) const;
    /**
     * @brief Retrieve a dofs corresponding to a specific element, e.g the shape of tet from a deformed mesh, from a global vector, e.g the shape of the whole deformed mesh.
     * @param elm_vec The 1x12 vector to store the dofs corresponding to a specific element.
     * @param elm_id The index of the element.
     * @param global_vec The buffer of the global vector.
     */
    void RetrieveElmVector(Vec12d& elm_vec,size_t elm_id,const FEM_Scaler* global_vec) const;  
    /**
     * @return Get the mass of the whole mesh.
     */
    inline const VecXd& GetDoFMass() const {return dof_mass;}
    /**
     * @param vert_id the queried vertex index
     * @return the mass of a specific vertex.
     */
    inline FEM_Scaler GetVertMass(size_t vert_id) const { return vert_mass[vert_id];}
    /**
     * @param dof_id the queried dof's index
     * @return the mass of a specific dof.
     */
    inline FEM_Scaler GetDoFMass(size_t dof_id) const { return vert_mass[dof_id/3]; }
    /**
     * @param elm_id the queried elment's index
     * @return the mass of a specific elment.
     */
    inline FEM_Scaler GetElmMass(size_t elm_id) const { return elm_mass[elm_id]; }
    /**
     * @param elm_id the querid element index.
     * @return the volume of the queried element
     */
    inline FEM_Scaler GetElmVolume(size_t elm_id) const {return elm_volume[elm_id];}
    /**
     * @param elm_id the querid element index.
     * @return the density of the queried element
     */
    inline FEM_Scaler GetElmDensity(size_t elm_id) const {return elm_rho[elm_id];}
    /**
     * @brief given the shaped of specific deformed tet, compute the deformation gradient of this tet.
     * @param elm_id the queried elm_id
     * @param elm_u the deformed shape the tet
     * @param F the deformation gradient
     */
    void ComputeDeformationGradient(size_t elm_id,const Vec12d& elm_u,Mat3x3d& F) const;
    /**
     * @brief Get the dFdx(i.e the gradient of F w.r.t x) of a specific element which can be precomputed. F is the deformation gradient and x the deformed shape.
     * @param elm_id the queried elm_id
     * @return the dFdx of quried tet
     */
    inline const Mat9x12d& GetElmdFdx(size_t elm_id) const {return elm_dFdx[elm_id];}
    /**
     * @brief Get the wBm^T matrix of a specific element which can be precomputed, which is used to map the first piola-stress to element's nodal force.
     * w is the volume of the tet, Bm = [X3-X0,X2-X0,X1-X0]^{-1} and Xi is the element's vertex position of rest shape. Bm can also be computed in the following way:
     * Bm = Minv.block(0,0,3,3) with Minv = M^{-1}, M = [X0,X1,X2,X3;0,0,0,1];
     * @param elm_id the queried elm_id
     * @return wBm^T matrix of the queried element
     * @see GetElmMInv
     */
    inline const Mat3x3d& GetElmWBmT(size_t elm_id) const {return elm_WBmT[elm_id];}
    /**
     * @brief Get the Minv matrix of a specific element which can be precomputed. Minv = M^{-1}, M = [X0,X1,X2,X3;0,0,0,1].
     * @param elm_id the queried element id.
     * @return the Minv matrix of queried element
     * @see GetElmWBmT
     */
    inline const Mat4x4d& GetElmMInv(size_t elm_id) const {return elm_MInv[elm_id];}

    void UpdateConstrainedDoFs(); 

    inline void Translate(const Vec3d& trans) {
        for(size_t i = 0;i < GetNumVertices();++i){
            vertices[i*3 + 0] += trans[0];
            vertices[i*3 + 1] += trans[1];
            vertices[i*3 + 2] += trans[2];
        }
        update_bounding_box();
    }

    inline void Deform(const Mat3x3d& F) {
        for(size_t i = 0;i < GetNumVertices();++i){
            Vec3d vert = GetVertex(i);
            vert = F * vert;
            vertices[i*3 + 0] = vert[0];
            vertices[i*3 + 1] = vert[1];
            vertices[i*3 + 2] = vert[2];
        }

        update_bounding_box();
        DoPreComputation();
    }

    inline void Scale(FEM_Scaler scale) {
        Mat3x3d F = Mat3x3d::Identity() * scale;
        Deform(F);
        DoPreComputation();
    }

private:
    FEM_Scaler mesh_scale;                          /** the mesh scale due to the normalization */
    std::vector<FEM_Scaler> vertices;               /** the rest shape of the mesh */
    size_t num_vertices;                            
    std::vector<size_t> elements;
    size_t num_elements;

    std::vector<size_t> cons_dofs;
    std::vector<size_t> num_cons_dofs;
    std::vector<size_t> ucons_dofs;
    size_t num_ucons_dofs;
    std::vector<size_t> bou_vert_ids;               /** the indices of boundary vertices */

    std::vector<size_t> dof2ucdof_indices;
    std::set<Triplet,triplet_cmp> uc_connectivity;
    std::set<Triplet,triplet_cmp> connectivity;

    Vec6d bounding_box;

    std::vector<FEM_Scaler> elm_rho;
    std::vector<FEM_Scaler> vert_mass;
    std::vector<FEM_Scaler> elm_mass;
    std::vector<FEM_Scaler> elm_volume;
    VecXd dof_mass;

    std::vector<Mat9x12d> elm_dFdx;
    std::vector<Mat3x3d> elm_WBmT;
    std::vector<Mat4x4d> elm_MInv;
    std::vector<Mat3x3d> elm_DmInv;

    SpMat conn_matrix;
    SpMat ucconn_matrix;
    typedef std::pair<size_t,size_t> SpIdx;     /** the structure storing the sparse structure indices, <row_idx,col_idx> */
    // for the assemble
    std::map<SpIdx,size_t> sp_indices_map;      /** the <row_idx,col_idx> to the index of the coefficient on the matrix value buffer */
    std::vector<Mat12x12i> elm_sp_indices;
    std::vector<Mat24x24i> elm_J_indices;
    // for the removal
    std::vector<size_t> sp_uc_dofs;

    std::vector<Mat3x3d> elm_anisotropic_orients;
    std::vector<Vec3d> elm_anisotropic_weight;

    std::vector<Mat3x3d> elm_activation;

    std::vector<size_t> surface_dofs;   

private:
    TetrahedraMesh(const FEM_Scaler* vertices, size_t nm_vertices, const size_t* elements, size_t nm_elements);
    static int LoadElmsFromFile(const char* filename, std::vector<size_t>& elms);
    static int LoadNodesFromFile(const char* filename, VecXd& vertices);
    static int LoadBouIndices(const char* filename, std::vector<size_t>& bindices);

    void build_ucdof_mapping();
    int update_mass();
    void make_dFdx();

    void extract_surface_dofs();

    void DoPreComputation();
    void update_bounding_box();
};
