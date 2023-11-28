// This file is part of Directional, a library for directional field processing.
// Copyright (C) 2022 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DIRECTIONAL_CARTESIAN_FIELD_H
#define DIRECTIONAL_CARTESIAN_FIELD_H

#include <set>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "./IntrinsicFaceTangentBundle.h"

/***
 The class implements general cartesian fields in intrinsic dimension 2, which are attached to a tangent bundle. These fields can be of any degree N, where the unifying principle is that they are represented by Cartesian coordinates (intrinsically and possibly extrinsically). The class supports either direct raw fields (just a list of vectors in each tangent space in order), or power and polyvector fields, representing fields as root of polynomials irrespective of order.

 This class assumes extrinsic representation in 3D space.
 ***/

namespace zeno::directional{

    enum class fieldTypeEnum{RAW_FIELD, POWER_FIELD, POLYVECTOR_FIELD};

    //The data structure for seamless integration
    struct IntegrationData {
        int N;                                              // # uncompressed parametric functions
        int n;                                              // # independent parameteric functions
        Eigen::MatrixXi linRed;                             // Linear Reduction tying the n dofs to the full N
        Eigen::MatrixXi periodMat;                          // Function spanning integers
        Eigen::SparseMatrix<float> vertexTrans2CutMat;     // Map between the whole mesh (vertex + translational jump) representation to the vertex-based representation on the cut mesh
        Eigen::SparseMatrix<float> constraintMat;          // Linear constraints (resulting from non-singular nodes)
        Eigen::SparseMatrix<float> linRedMat;              // Global uncompression of n->N
        Eigen::SparseMatrix<float> intSpanMat;             // Spanning the translational jump lattice
        Eigen::SparseMatrix<float> singIntSpanMat;         // Layer for the singularities
        Eigen::VectorXi constrainedVertices;                // Constrained vertices (fixed points in the parameterization)
        Eigen::VectorXi integerVars;                        // Variables that are to be rounded.
        Eigen::MatrixXi face2cut;                           // |F|x3 map of which edges of faces are seams
        Eigen::VectorXf nVertexFunction;                    // Final compressed result (used for meshing)

        Eigen::VectorXi fixedIndices;                       // Translation fixing indices
        Eigen::VectorXf fixedValues;                        // Translation fixed values
        Eigen::VectorXi singularIndices;                    // Singular-vertex indices

        //integer versions, for exact seamless parameterizations (good for error-free meshing)
        Eigen::SparseMatrix<int> vertexTrans2CutMatInteger;
        Eigen::SparseMatrix<int> constraintMatInteger;
        Eigen::SparseMatrix<int> linRedMatInteger;
        Eigen::SparseMatrix<int> intSpanMatInteger;
        Eigen::SparseMatrix<int> singIntSpanMatInteger;

        float lengthRatio;                                 // Global scaling of functions
        //Flags
        bool integralSeamless;                              // Whether to do full translational seamless.
        bool roundSeams;                                    // Whether to round seams or round singularities
        bool verbose;                                       // Output the integration log.
        bool localInjectivity;                              //Enforce local injectivity; might result in failure!

        IntegrationData(int _N):lengthRatio(0.02), integralSeamless(false), roundSeams(true), verbose(false), localInjectivity(false) {
            N = _N;
            n = ( N % 2 == 0 ? N / 2 : N);
            if (N%2==0) {
                linRed.resize(N, N / 2);
                linRed << Eigen::MatrixXi::Identity(N/2,N/2),-Eigen::MatrixXi::Identity(N/2,N/2);
            }
            else
                linRed = Eigen::MatrixXi::Identity(N,n);
            periodMat = Eigen::MatrixXi::Identity(n,n);
        }
        ~IntegrationData(){}
    };


    class CartesianField{
    public:
        const IntrinsicFaceTangentBundle* tb;           //Referencing the tangent bundle on which the field is defined

        int N;                                          //Degree of field (how many vectors are in each point);
        fieldTypeEnum fieldType;                        //The representation of the field (for instance, either a raw field or a power/polyvector field)

        Eigen::MatrixXf intField;                       //Intrinsic representation (depending on the local basis of the face). Size #T x 2N
        Eigen::MatrixXf extField;                       //Ambient coordinates. Size #T x 3N

        Eigen::VectorXi matching;                       //Matching(i)=j when vector k in EF(i,0) matches to vector (k+j)%N in EF(i,1)
        Eigen::VectorXf effort;                         //Effort of the entire matching (sum of deviations from parallel transport)
        std::vector<int> sing_local_cycles{};

        CartesianField(){}
        CartesianField(const IntrinsicFaceTangentBundle& _tb):tb(&_tb){}
        ~CartesianField(){}

        // Initializing the field with the proper tangent spaces
        void init(const IntrinsicFaceTangentBundle& _tb, const fieldTypeEnum _fieldType, const int _N) {
            tb = &_tb;
            fieldType = _fieldType;
            N=_N;
            intField.resize(tb->num_f, 2 * N);
            extField.resize(tb->num_f, 3 * N);
        };

        void set_intrinsic_field(const Eigen::MatrixXf& _intField){
            assert (!(fieldType == fieldTypeEnum::POWER_FIELD) || (_intField.cols() == 2));
            assert ((_intField.cols() == 2 * N) || !(fieldType == fieldTypeEnum::POLYVECTOR_FIELD || fieldType == fieldTypeEnum::RAW_FIELD));
            intField = _intField;
            extField = tb->project_to_extrinsic(Eigen::VectorXi(), intField);
        }

        void set_extrinsic_field(const Eigen::MatrixXf& _extField){
            assert(_extField.cols() == 3 * N);
            extField = _extField;
            intField = tb->project_to_intrinsic(Eigen::VectorXi::LinSpaced(extField.rows(), 0, extField.rows() - 1), extField);
        }

        // Takes a field in raw form and computes both the principal effort and the consequent principal matching on every edge.
        // Important: if the Raw field in not CCW ordered, the result is meaningless.
        // The input and output are both a RAW_FIELD type cartesian field, in which the matching, effort, and singularities are set.
        void principal_matching(bool update_matching = true);

        // Given a mesh and the singularities of a polyvector field, cut the mesh
        // to disk topology in such a way that the singularities lie at the boundary of
        // the disk, as described in the paper "Mixed Integer Quadrangulation" by
        // Bommes et al. 2009.
        // Inputs:
        //   singularities    #S by 1 list of the indices of the singular vertices
        // Outputs:
        //   cuts             #F by 3 list of boolean flags, indicating the edges that need to be cut
        //
        void cut_mesh_with_singularities(Eigen::MatrixXi& cuts);

        // Reorders the vectors in a tangent space (preserving CCW direction) so that the prescribed matching across most TB edges is an identity, except for seams.
        // Important: if the Raw field in not CCW ordered, the result is unpredictable.
        // Input:
        //  rawField:   a RAW_FIELD uncombed cartesian field object
        //  cuts: #Fx3 prescribing the TB edges (corresponding to mesh faces) that must be a seam.
        // Output:
        //  combed_field: the combed field object, also RAW_FIELD
        void combing(CartesianField& combed_field,
                     const Eigen::MatrixXi& cuts);
                     
        // Setting up the seamless integration algorithm.
        // Output:
        //  intData:      updated integration data.
        //  meshCut:      a mesh which is face-corresponding with meshWhole, but is cut so that it has disc-topology.
        //  combedField:  The raw field combed so that all singularities are on the seams of the cut mesh
        void setup_integration(IntegrationData& intData,
                               Eigen::MatrixXf& cut_verts,
                               Eigen::MatrixXi& cut_faces,
                               CartesianField& combedField);

        // Integrates an N-directional fields into an N-function by solving the seamless Poisson equation. Respects *valid* linear reductions where the field is reducible to an n-field for n<=M, and consequently the function is reducible to an n-function.
        // Input:
        //  intData:            Integration data, which must be obtained from directional::setup_integration(). This is altered by the function.
        //  meshCut:            Cut mesh (obtained from setup_integration())
        // Output:
        //  NFunction:          #cV x N parameterization functions per cut vertex (full version with all symmetries unpacked)
        //  NCornerFunctions   (3*N) x #F parameterization functions per corner of whole mesh
        bool integrate(IntegrationData& intData,
                       const zeno::pmp::SurfaceMesh* meshCut,
                       Eigen::MatrixXf& NFunction,
                       Eigen::MatrixXf& NCornerFunctions);
    
    private:
        // Computes cycle-based indices from adjaced-space efforts of a directional field.
        // accepts a cartesian field object and operates on it as input and output.
        void effort_to_indices();

        // Use Dijkstra's algorithm to find a shortest path from source to any target in the set.
        void dijkstra(const int &source,
                      const std::set<int> &targets,
                      std::vector<int> &path);

        // assuming corner functions are arranged packets of N per vertex.
        void branched_gradient(const zeno::pmp::SurfaceMesh* meshCut,
                               const int N,
                               Eigen::SparseMatrix<float>& G);

        bool iterative_rounding(const zeno::pmp::SurfaceMesh* meshCut,
                                const Eigen::SparseMatrix<float>& A,
                                const Eigen::MatrixXf& rawField,
                                const Eigen::VectorXi& fixedIndices,
                                const Eigen::VectorXf& fixedValues,
                                const Eigen::VectorXi& singularIndices,
                                const Eigen::VectorXi& integerIndices,
                                const float lengthRatio,
                                const Eigen::VectorXf& b,
                                const Eigen::SparseMatrix<float> C,
                                const Eigen::SparseMatrix<float> G,
                                const int N,
                                const int n,
                                const Eigen::SparseMatrix<float>& x2CornerMat,
                                const bool fullySeamless,
                                const bool roundSeams,
                                const bool localInjectivity,
                                Eigen::VectorXf& fullx);
    };
}

#endif 
