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
 The class implements general cartesian fields in intrinsic dimension 2, which are attached to a tangent bundle. These fields can be of any degree N, where the unifying principle is that
 they are represented by Cartesian coordinates (intrinsically and possibly extrinsically). The class supports either direct raw fields (just a list of vectors in each
 tangent space in order), or power and polyvector fields, representing fields as root of polynomials irrespective of order.

 This class assumes extrinsic representation in 3D space.
 ***/

namespace zeno::directional{

    enum class fieldTypeEnum{RAW_FIELD, POWER_FIELD, POLYVECTOR_FIELD};

    class CartesianField{
    public:

        const IntrinsicFaceTangentBundle* tb;            //Referencing the tangent bundle on which the field is defined

        int N;                              //Degree of field (how many vectors are in each point);
        fieldTypeEnum fieldType;                      //The representation of the field (for instance, either a raw field or a power/polyvector field)

        Eigen::MatrixXf intField;           //Intrinsic representation (depending on the local basis of the face). Size #T x 2N
        Eigen::MatrixXf extField;           //Ambient coordinates. Size #T x 3N

        Eigen::VectorXi matching;           //Matching(i)=j when vector k in mesh->EF(i,0) matches to vector (k+j)%N in mesh->EF(i,1)
        Eigen::VectorXf effort;             //Effort of the entire matching (sum of deviations from parallel transport)
        std::vector<int> sing_local_cycles;

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
            assert (!(fieldType==fieldTypeEnum::POWER_FIELD) || (_intField.cols()==2));
            assert ((_intField.cols() == 2 * N) || !(fieldType == fieldTypeEnum::POLYVECTOR_FIELD || fieldType == fieldTypeEnum::RAW_FIELD));
            intField = _intField;

            extField = tb->project_to_extrinsic(Eigen::VectorXi(), intField);
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
        void cut_mesh_with_singularities(const Eigen::VectorXi& singularities,
                                         Eigen::MatrixXi& cuts);

        // Reorders the vectors in a tangent space (preserving CCW direction) so that the prescribed matching across most TB edges is an identity, except for seams.
        // Important: if the Raw field in not CCW ordered, the result is unpredictable.
        // Input:
        //  rawField:   a RAW_FIELD uncombed cartesian field object
        //  cuts: #Fx3 prescribing the TB edges (corresponding to mesh faces) that must be a seam.
        // Output:
        //  combed_field: the combed field object, also RAW_FIELD
        void combing(CartesianField& combed_field,
                     const Eigen::MatrixXi& cuts);
    
    private:
        // Computes cycle-based indices from adjaced-space efforts of a directional field.
        // accepts a cartesian field object and operates on it as input and output.
        void effort_to_indices();

        // Use Dijkstra's algorithm to find a shortest path from source to any target in the set.
        void dijkstra(const int &source,
                      const std::set<int> &targets,
                      std::vector<int> &path);
        
    };
}

#endif 
