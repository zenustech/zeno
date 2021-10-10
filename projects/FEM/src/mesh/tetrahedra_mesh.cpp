#include "tetrahedra_mesh.h"
#include <fstream>
#include <algorithm>

int TetrahedraMesh::LoadTetrahedraFromFile(std::shared_ptr<TetrahedraMesh>& mesh_ptr,
            const char* elm_file,
            const char* node_file,
            const char* bou_file){
    std::vector<size_t> elements;
    VecXd vertices;
    std::vector<size_t> bou_indices;

    TetrahedraMesh::LoadElmsFromFile(elm_file,elements);
    TetrahedraMesh::LoadNodesFromFile(node_file,vertices);
    bou_indices.clear();
    if(bou_file)
        TetrahedraMesh::LoadBouIndices(bou_file,bou_indices);

    size_t nm_vertices = vertices.size()/3;
    size_t nm_elms = elements.size()/4;

    mesh_ptr = std::shared_ptr<TetrahedraMesh>(new TetrahedraMesh(vertices.data(),nm_vertices,elements.data(),nm_elms));
    // mesh_ptr->SetConstrainedNodes(bou_indices.data(),bou_indices.size());  
    mesh_ptr->SetConstrainedDoFs(bou_indices.data(),bou_indices.size());

    // the material is by default to be isotropic
    mesh_ptr->elm_anisotropic_weight.resize(nm_elms,Vec3d::Constant(1.0));
    mesh_ptr->elm_anisotropic_orients.resize(nm_elms,Mat3x3d::Identity());
    mesh_ptr->elm_activation.resize(nm_elms,Mat3x3d::Identity());

    return 0;  
}

int TetrahedraMesh::LoadElmsFromFile(const char* filename,std::vector<size_t>& elements){
    size_t nm_elms,elm_size,v_start_idx,elm_idx;
    std::ifstream fin;
    size_t d2;
    try {
        fin.open(filename);
        if (!fin.is_open()) {
            std::cerr << "ERROR::TET::FAILED::" << std::string(filename) << std::endl;
            return -1;
        }
        fin >> nm_elms >> elm_size >> v_start_idx;
        elements.resize(nm_elms * elm_size);

        for(size_t elm_id = 0;elm_id < nm_elms;++elm_id) {
            fin >> elm_idx;
            for (size_t i = 0; i < elm_size; ++i) {
                fin >> elements[elm_id * elm_size + i];
                elements[elm_id * elm_size + i] -= v_start_idx;
            }
        }
        fin.close();
    }catch(std::exception &e){
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int TetrahedraMesh::LoadNodesFromFile(const char* filename,VecXd& vertices){
    size_t num_vertices,space_dimension,d1,d2;
    std::ifstream fin;
    try {
        fin.open(filename);
        if (!fin.is_open()) {
            std::cerr << "ERROR::TET::FAILED::" << std::string(filename) << std::endl;
            return -1;
        }
        fin >> num_vertices >> space_dimension >> d1 >> d2;
        vertices.resize(num_vertices * space_dimension);

        for(size_t vert_id = 0;vert_id < num_vertices;++vert_id) {
            fin >> d1;
            for (size_t i = 0; i < space_dimension; ++i)
                fin >> vertices[vert_id * space_dimension + i];
        }
        fin.close();
    }catch(std::exception &e){
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int TetrahedraMesh::LoadBouIndices(const char* filename,std::vector<size_t>& bindices){
    size_t num_fixed_dofs,start_idx;
    std::ifstream fin;
    try{
        fin.open(filename);
        if(!fin.is_open()){
            std::cerr << "ERROR::BOU::FAILED::" << std::string(filename) << std::endl;
            return -1;
        }

        fin >> num_fixed_dofs >> start_idx;
        bindices.resize(num_fixed_dofs);
        for(size_t c_id = 0;c_id < num_fixed_dofs;++c_id){
            fin >> bindices[c_id];
            bindices[c_id] -= start_idx;
        }
        fin.close();
    }catch(std::exception &e){
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int TetrahedraMesh::LoadVisIndices(const char* filename,std::vector<size_t>& vsindices){
    size_t num_vis_points,start_idx;
    std::ifstream fin;
    try{
        fin.open(filename);
        if(!fin.is_open()){
            std::cerr << "ERROR::BOU::FAILED::" << std::string(filename) << std::endl;
            return -1;
        }

        fin >> num_vis_points >> start_idx;
        vsindices.resize(num_vis_points);
        for(size_t cv_id = 0;cv_id < num_vis_points;++cv_id){
            fin >> vsindices[cv_id];
            vsindices[cv_id] -= start_idx;
        }

        fin.close();
    }catch(std::exception &e){
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}

void TetrahedraMesh::update_bounding_box() {
    for (size_t i = 0; i < 3; ++i) {
        bounding_box[i] = 1e8;
        bounding_box[i + 3] = -1e8;
    }
    for (size_t i = 0; i < GetNumVertices(); ++i) {
        const auto& vert = GetVertex(i);
        for (size_t j = 0; j < 3; ++j) {
            if (vert[j] > bounding_box[j + 3])
                bounding_box[j + 3] = vert[j];
            if (vert[j] < bounding_box[j])
                bounding_box[j] = vert[j];
        }
    }
}

TetrahedraMesh::TetrahedraMesh(const SPMAT_SCALER* _vertices, size_t _nm_vertices, const size_t* _elements, size_t _nm_elements){
    vertices.resize(_nm_vertices * 3);
    elements.resize(_nm_elements * 4);
    memcpy(vertices.data(),_vertices,sizeof(FEM_Scaler) * _nm_vertices * 3);
    memcpy(elements.data(), _elements, sizeof(size_t) * _nm_elements * 4);

    size_t reserve_size = GetNumDoF() * 20;
    connectivity.clear();
    size_t nm_insertions = 0;
    for (size_t elm_id = 0; elm_id < GetNumElements(); ++elm_id) {
        const auto& elm = GetElement(elm_id);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                for (size_t k = 0; k < 3; ++k)
                    for (size_t l = 0; l < 3; ++l) {
                        size_t row = elm[i] * 3 + k;
                        size_t col = elm[j] * 3 + l;
                        if(row > col)
                            continue;
                        if(row == col){
                            connectivity.insert(Triplet(row, col, 1.0));
                            ++nm_insertions;
                        }else{
                            connectivity.insert(Triplet(row, col, 1.0));
                            connectivity.insert(Triplet(col, row, 1.0));
                            nm_insertions += 2;
                        }
                    }
    }
    // connectivity.shrink_to_fit();
    update_bounding_box();

    elm_rho.resize(GetNumElements());
    elm_mass.resize(GetNumElements());
    elm_volume.resize(GetNumElements());
    vert_mass.resize(GetNumVertices());
    dof_mass = VecXd::Zero(GetNumDoF());

    elm_WBmT.clear();
    elm_WBmT.resize(GetNumElements());
    elm_MInv.clear();
    elm_MInv.resize(GetNumElements());
    elm_DmInv.clear();
    elm_DmInv.resize(GetNumElements());

    DoPreComputation();

    conn_matrix = SpMat(GetNumDoF(),GetNumDoF());
    conn_matrix.setZero();
    conn_matrix.setFromTriplets(connectivity.begin(),connectivity.end());
    conn_matrix.makeCompressed();
    if(conn_matrix.nonZeros() != connectivity.size()){
        std::cout << "conn_matrix_size:" << conn_matrix.nonZeros() << std::endl;
        std::cout << "connectivity_size:" << connectivity.size() << std::endl;
        std::cout << "nm_insertions:" << nm_insertions << std::endl;
        throw std::runtime_error("conn_not_matched");
    }

    sp_indices_map.clear();
    size_t array_idx = 0;
    for(size_t k = 0;k < size_t(conn_matrix.outerSize());++k){
        for(SpMat::InnerIterator it(conn_matrix,k);it;++it){
            std::pair<SpIdx,size_t> pair(SpIdx(it.row(),it.col()),array_idx);
            sp_indices_map.insert(pair);
            ++array_idx;
        }
    }

    elm_sp_indices.resize(GetNumElements());
    for(size_t elm_id = 0;elm_id < GetNumElements();++elm_id){
        auto tet = GetElement(elm_id);
        for(size_t i = 0;i < 4;++i) {
            for(size_t j = 0;j < 4;++j)
                for (size_t r = 0; r < 3; ++r)
                    for (size_t c = 0; c < 3; ++c){
                        size_t row_idx = tet[i]*3 + r;
                        size_t col_idx = tet[j]*3 + c;
                        auto it = sp_indices_map.find(SpIdx(row_idx,col_idx));
                        if(it == sp_indices_map.end()){
                            throw std::runtime_error("could not find entry");
                        }
                        elm_sp_indices[elm_id](i*3 + r,j*3 + c) = it->second;
                    }
        }
    }

    extract_surface_dofs();
}

 void TetrahedraMesh::DoPreComputation(){
    Mat4x4d M;
    for (size_t elm_id = 0; elm_id < GetNumElements(); ++elm_id) {
        auto elm = GetElement(elm_id);
        for (size_t i = 0; i < 4; ++i)
            M.block(0, i, 3, 1).noalias() = GetVertex(elm[i]);
        M.bottomRows(1).setConstant(1.0);
        elm_MInv[elm_id] = M.inverse();

        Mat3x3d Dm;
        for(size_t i = 1; i < 4;++i)
            Dm.col(i-1) = GetVertex(elm[i]) - GetVertex(elm[0]);     

        elm_DmInv[elm_id] = Dm.inverse();

        // if(M.determinant() < 0){
        //     std::cerr << "inverted tets detected " << elm_id << std::endl;
        // }
        elm_volume[elm_id] = fabs(M.determinant());
        for (size_t j = 1; j < 4; ++j)
            elm_volume[elm_id] /= j;
        elm_WBmT[elm_id].noalias() = elm_volume[elm_id] * elm_MInv[elm_id].topLeftCorner(3, 3).transpose();
    }
    elm_dFdx.resize(GetNumElements());
    make_dFdx();
 }

void TetrahedraMesh::UpdateConstrainedDoFs(){
    ucons_dofs.resize(GetNumDoF() - cons_dofs.size());
    for(size_t cdof_idx = 0,dof = 0,ucdof_count = 0;dof < GetNumDoF();++dof){
        if(cdof_idx >= GetNumCDoF() || dof != cons_dofs[cdof_idx]){
            ucons_dofs[ucdof_count] = dof;
            ++ucdof_count;
        }
        else
            ++cdof_idx;
    }
    assert(GetNumDoF() == (GetNumCDoF() + GetNumUCDoF()));
    build_ucdof_mapping();

    uc_connectivity.clear();
    size_t nm_insertions = 0;
    for(size_t elm_id = 0;elm_id < GetNumElements();++elm_id) {
        auto elm = GetElement(elm_id);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 4; ++j)
                for (size_t k = 0; k < 3; ++k)
                    for (size_t l = 0; l < 3; ++l) {
                        size_t row = dof2ucdof_indices[elm[i] * 3 + k];
                        size_t col = dof2ucdof_indices[elm[j] * 3 + l];
                        if(row == -1 || col == -1 || row > col)
                            continue;
                        if(row == col){
                            uc_connectivity.insert(Triplet(row,col,1.0));
                            nm_insertions++;
                        }else{
                            uc_connectivity.insert(Triplet(row,col,1.0));
                            uc_connectivity.insert(Triplet(col,row,1.0));
                            nm_insertions += 2;
                        }
                    }
    }


    ucconn_matrix = SpMat(GetNumUCDoF(),GetNumUCDoF());
    ucconn_matrix.setFromTriplets(uc_connectivity.begin(),uc_connectivity.end());
    ucconn_matrix.makeCompressed();

    if(ucconn_matrix.nonZeros() != uc_connectivity.size()){
        std::cout << "ucconn_matrix_size:" << ucconn_matrix.nonZeros() << std::endl;
        std::cout << "ucconnectivity_size:" << uc_connectivity.size() << std::endl;
        std::cout << "nm_insertions:" << nm_insertions << std::endl;
        throw std::runtime_error("ucconn_not_matched");
    }

    sp_uc_dofs.resize(ucconn_matrix.nonZeros());
    size_t uc_idx = 0;
    size_t idx = 0;
    for(size_t k = 0;k < size_t(conn_matrix.outerSize());++k)
        for(SpMat::InnerIterator it(conn_matrix,k);it;++it){
            size_t row = it.row();
            size_t col = it.col();
            if(dof2ucdof_indices[row] == -1 || dof2ucdof_indices[col] == -1){
                idx++;
                continue;
            }
            sp_uc_dofs[uc_idx] = idx;
            ++uc_idx;
            ++idx;
        }

    // std::cout << "idx:" << idx << "\t" << "uc_idx:" << uc_idx << "\t" << "sp_uc:" << sp_uc_dofs.size() << "\tsp:" << connectivity.size() << std::endl;
    if(uc_idx != sp_uc_dofs.size()){
        std::cerr << "sp_uc_dofs initialize fail" << std::endl;
        std::cout << "uc_idx:" << uc_idx << "\t" << "sp_uc_dofs:" << sp_uc_dofs.size() << std::endl;
        std::cout << "idx:" << idx << std::endl;
        throw std::runtime_error("sp_uc_dofs initialize fail");
    }
}

void TetrahedraMesh::SetConstrainedNodes(const size_t* cons_node_indices,size_t nm_cons_nodes){
    cons_dofs.resize(nm_cons_nodes*3);
    for(size_t i = 0;i < nm_cons_nodes;++i){
        for(size_t j = 0;j < 3;++j)
            cons_dofs[i*3 + j] = cons_node_indices[i]*3 + j;
    }
    std::sort(cons_dofs.begin(),cons_dofs.end());
    UpdateConstrainedDoFs();
}

void TetrahedraMesh::SetConstrainedDoFs(const size_t* cons_dofs_indices,size_t nm_cons_dofs){
    cons_dofs.resize(nm_cons_dofs);
    memcpy(cons_dofs.data(),cons_dofs_indices,sizeof(size_t) * nm_cons_dofs);
    std::sort(cons_dofs.begin(),cons_dofs.end());
    UpdateConstrainedDoFs();
}

void TetrahedraMesh::AssembleElmVectorAdd(const Vec12d& elm_vec,size_t elm_id,SPMAT_SCALER* global_vec) const{
   const auto& elm = GetElement(elm_id);
   auto gv_map = Eigen::Map<VecXd>(global_vec,GetNumDoF(),1);
   for(size_t i = 0;i < 4;++i)
            gv_map.segment(elm[i]*3,3) += elm_vec.segment(i*3,3);
}
void TetrahedraMesh::AssembleElmMatrixAdd(const Mat12x12d& elm_hessian,size_t elm_id,SpMat& global_hessian) const{
    const auto& elm = GetElement(elm_id);
    for(size_t i = 0;i < 4;++i) {
        for(size_t j = 0;j < 4;++j)
            for (size_t r = 0; r < 3; ++r)
                for (size_t c = 0; c < 3; ++c)
                    global_hessian.coeffRef(elm[i] * 3 + r, elm[j] * 3 + c) += elm_hessian(i * 3 + r, j * 3 + c);
    } 
}

void TetrahedraMesh::AssembleElmMatrixAdd(const Mat12x12d& elm_hessian,size_t elm_id,SPMAT_SCALER *sp_data) const{
    const auto& elm = GetElement(elm_id);
    const auto& sp_idx = elm_sp_indices[elm_id]; 
     for(size_t i = 0;i < 4;++i) {
        for(size_t j = 0;j < 4;++j)
            for (size_t r = 0; r < 3; ++r)
                for (size_t c = 0; c < 3; ++c){
                    size_t idx = elm_sp_indices[elm_id](i*3 + r,j*3 + c);
                    sp_data[idx] += elm_hessian(i*3 + r,j*3 + c);
                }
    }  
}

void TetrahedraMesh::AssembleElmJacobiAdd(const 
Mat12xXd& elm_J,size_t elm_id,MatXd& J) const{
    const auto& elm = GetElement(elm_id);
    assert(elm_id < GetNumElements());
    assert(elm_J.cols() == J.cols());
    assert(J.rows() == GetNumDoF());
    for(size_t i = 0;i < 4;++i){
        size_t v_id = elm[i];
        J.block(3*v_id,0,3,elm_J.cols()) = J.block(3 * v_id, 0, 3, elm_J.cols()) + elm_J.block(3*i,0,3,elm_J.cols());
    }     
}

void TetrahedraMesh::RetrieveElmVector(Vec12d& elm_vec,size_t elm_id,const SPMAT_SCALER* global_vec) const{
    const auto& elm = GetElement(elm_id);
    auto gv_map = Eigen::Map<const VecXd>(global_vec,GetNumDoF(),1);
    for(size_t i = 0;i < 4;++i)
        elm_vec.segment(i*3,3) = gv_map.segment(elm[i]*3,3);
}

void TetrahedraMesh::build_ucdof_mapping(){
    dof2ucdof_indices.resize(GetNumDoF());
    std::fill(dof2ucdof_indices.begin(),dof2ucdof_indices.end(),-1);
    for(size_t i = 0;i < GetNumUCDoF();++i){
        size_t ucdof = GetUCDoF()[i];
        dof2ucdof_indices[ucdof] = i;
    }
}

void TetrahedraMesh::RemoveConstrainedDoF(const SpMat& _src,SpMat& _des) const {
    assert(_src.rows() == GetNumDoF() && _src.cols() == GetNumDoF());
    assert(_des.rows() == GetNumUCDoF() && _des.cols() == GetNumUCDoF());
    _des.setZero();
    for(size_t i = 0;i < GetNumDoF();++i)
        for(SpMat::InnerIterator it(_src,i);it;++it){
            size_t row = dof2ucdof_indices[it.row()];
            size_t col = dof2ucdof_indices[it.col()];
            if(row == -1 || col == -1)
                continue;
            _des.coeffRef(row,col) = it.value();
        }
}

void TetrahedraMesh::RemoveConstrainedDoF(const SpMat &_src, Eigen::Map<SpMat> &_des) const {
    for(size_t i = 0;i < GetNumDoF();++i)
        for(SpMat::InnerIterator it(_src,i);it;++it){
            size_t row = dof2ucdof_indices[it.row()];
            size_t col = dof2ucdof_indices[it.col()];
            if(row == -1 || col == -1)
                continue;
            _des.coeffRef(row,col) = it.value();
        }    
}

void TetrahedraMesh::RemoveConstrainedDoF(const SPMAT_SCALER* sp_from,SPMAT_SCALER* sp_to) const {
    MatHelper::RetrieveDoFs(sp_from,sp_to,sp_uc_dofs.size(),sp_uc_dofs.data());
}

void TetrahedraMesh::SetDensityParam(const FEM_Scaler* rhos_handles,size_t nm_handles,const FEM_Scaler* weight){
    if (nm_handles == 1 || !weight) {
        for (size_t elm_id = 0; elm_id < GetNumElements(); ++elm_id)
                elm_rho[elm_id] = rhos_handles[0];
    }else {
        auto w_map = Eigen::Map<const MatXd>(weight, GetNumElements(), nm_handles);
        std::fill(elm_rho.begin(), elm_rho.end(), 0);
        memset(elm_rho.data(), 0, sizeof(FEM_Scaler) * GetNumElements());
        for (size_t elm_id = 0; elm_id < GetNumElements(); ++elm_id)
            for (size_t h_id = 0; h_id < nm_handles; ++h_id)
                elm_rho[elm_id] += w_map(elm_id, h_id) * rhos_handles[h_id];
    }
    update_mass();
}

void TetrahedraMesh::ComputeDeformationGradient(size_t elm_id,const Vec12d &elm_u,Mat3x3d& F) const {
    Mat4x4d G;
    for(size_t i = 0;i < 4;++i)
        G.block(0,i,3,1) = elm_u.segment(i*3,3);
    G.bottomRows(1).setConstant(1.0);
    G = G * elm_MInv[elm_id];
    F = G.topLeftCorner(3,3);
}

//    Compute the elment and nodal mass
int TetrahedraMesh::update_mass(){
    std::fill(vert_mass.begin(),vert_mass.end(),0.);
    memset(vert_mass.data(),0,sizeof(FEM_Scaler) * GetNumVertices());
    for(size_t elm_id = 0;elm_id < GetNumElements();++elm_id){
        const auto& elm = this->GetElement(elm_id);
        elm_mass[elm_id] = elm_volume[elm_id]*elm_rho[elm_id];
        for(size_t i = 0;i < 4;++i)
            vert_mass[elm[i]] += elm_mass[elm_id]/4.0;
    }
    for(size_t i = 0;i < GetNumVertices();++i)
        dof_mass.segment(i*3,3).setConstant(vert_mass[i]);
    return 0;
}


void sortVec3i(Vec3i& s) {
        if(s[0] > s[1]){
            size_t tmp = s[0];
            s[0] = s[1];
            s[1] = tmp;
        }
        if(s[1] > s[2]){
            size_t tmp = s[1];
            s[1] = s[2];
            s[2] = tmp;            
        }   
        if(s[0] > s[1]){
            size_t tmp = s[0];
            s[0] = s[1];
            s[1] = tmp;
        }

    assert(s[0] < s[1] && s[1] < s[2]);
}

bool compare_vec3i(const Vec3i &_s1,const Vec3i &_s2) {
    Vec3i s1 = _s1,s2 = _s2;
    sortVec3i(s1);
    sortVec3i(s2);

    if(s1[0] < s2[0])
        return true;
    if(s1[0] == s2[0] && s1[1] < s2[1])
        return true;
    if(s1[0] == s2[0] && s1[1] == s2[1] && s1[2] < s2[2])
        return true;
    return false;
}   

void TetrahedraMesh::extract_surface_dofs() {
    std::vector<Vec3i> triangles;
    triangles.clear();

    for(size_t i = 0;i < GetNumElements();++i){
        auto tet = GetElement(i);
        triangles.emplace_back(tet[0],tet[1],tet[2]);
        triangles.emplace_back(tet[1],tet[3],tet[2]);
        triangles.emplace_back(tet[0],tet[2],tet[3]);
        triangles.emplace_back(tet[0],tet[3],tet[1]);
    }

    std::sort(triangles.begin(),triangles.end(),compare_vec3i);

    for(size_t i = 0;i < triangles.size();++i){
        Vec3i T = triangles[i];
        sortVec3i(T);
    }

    std::vector<Vec3i> surf_triangles;
    surf_triangles.clear();

    for(size_t i = 0;i < triangles.size();++i){
        if(i == triangles.size() - 1)
            surf_triangles.push_back(triangles[i]);
        else{
            if(!compare_vec3i(triangles[i],triangles[i+1]) && !compare_vec3i(triangles[i+1],triangles[i])){
                ++i;continue;
            }else{
                surf_triangles.push_back(triangles[i]);
            }
        }
    }

    for(size_t i = 0;i < surf_triangles.size();++i){
        Vec3i S = surf_triangles[i];
        sortVec3i(S);
    }
    surface_dofs.resize(surf_triangles.size() * 3 * 3);
    for(size_t i = 0;i < surf_triangles.size();++i){
        size_t offset = i*9;
        for(size_t j = 0;j < 3;++j){
            surface_dofs[offset + j*3 + 0] = surf_triangles[i][j] * 3 + 0;
            surface_dofs[offset + j*3 + 1] = surf_triangles[i][j] * 3 + 1;
            surface_dofs[offset + j*3 + 2] = surf_triangles[i][j] * 3 + 2;
        }
    }

    // std::cout << "nm_surf_dofs : " << GetNumSurfaceDoFs() << std::endl;
    // std::cout << "surf_dofs : " << std::endl;
    // for(size_t i = 0;i < GetNumSurfaceVertices();++i)
    //     std::cout << GetSurfaceDoFs()[3 * i] << "\t" << GetSurfaceDoFs()[3*i + 1] << "\t" << GetSurfaceDoFs()[i*3 + 2] << std::endl;
}

void TetrahedraMesh::make_dFdx(){
    for(size_t elm_id = 0;elm_id < GetNumElements();++elm_id) {
        Mat9x12d& PFPx = elm_dFdx[elm_id];
        const Mat3x3d& DmInv = elm_DmInv[elm_id];
        PFPx.setZero();
        // for(size_t i = 0;i < 4;++i) {
        //     for (size_t j = 0; j < 3; ++j)
        //         elm_dFdx[elm_id].block(j * 3, i * 3 + j, 3, 1) =  elm_MInv[elm_id].block(i, 0, 1, 3).transpose();
        // }

        FEM_Scaler m = DmInv(0,0);
        FEM_Scaler n = DmInv(0,1);
        FEM_Scaler o = DmInv(0,2);
        FEM_Scaler p = DmInv(1,0);
        FEM_Scaler q = DmInv(1,1);
        FEM_Scaler r = DmInv(1,2);
        FEM_Scaler s = DmInv(2,0);
        FEM_Scaler t = DmInv(2,1);
        FEM_Scaler u = DmInv(2,2);

        FEM_Scaler t1 = - m - p - s;
        FEM_Scaler t2 = - n - q - t;
        FEM_Scaler t3 = - o - r - u;

        PFPx << t1, 0, 0, m, 0, 0, p, 0, 0, s, 0, 0, 
                 0,t1, 0, 0, m, 0, 0, p, 0, 0, s, 0,
                 0, 0,t1, 0, 0, m, 0, 0, p, 0, 0, s,
                t2, 0, 0, n, 0, 0, q, 0, 0, t, 0, 0,
                 0,t2, 0, 0, n, 0, 0, q, 0, 0, t, 0,
                 0, 0,t2, 0, 0, n, 0, 0, q, 0, 0, t,
                t3, 0, 0, o, 0, 0, r, 0, 0, u, 0, 0,
                 0,t3, 0, 0, o, 0, 0, r, 0, 0, u, 0,
                 0, 0,t3, 0, 0, o, 0, 0, r, 0, 0, u;
    }    
}
