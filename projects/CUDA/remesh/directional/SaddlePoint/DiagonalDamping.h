// This file is part of SaddlePoint, a simple library for Eigen-based sparse nonlinear optimization
//
// Copyright (C) 2021 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SADDLEPOINT_DIAGONAL_DAMPING_H
#define SADDLEPOINT_DIAGONAL_DAMPING_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include <cstdio>
#include <set>


namespace SaddlePoint{
template<class SolverTraits>
class DiagonalDamping{
public:
  float currLambda;
  void init(const Eigen::SparseMatrix<float>& J,
            const Eigen::VectorXf& initSolution,
            const bool verbose,
            Eigen::SparseMatrix<float>& dampJ){
    
    
    //collecting the diagonal values
    Eigen::VectorXf dampVector=Eigen::VectorXf::Zero(initSolution.size());
    std::vector<Eigen::Triplet<float>> dampJTris;
    for (int k=0; k<J.outerSize(); ++k){
      for (Eigen::SparseMatrix<float>::InnerIterator it(J,k); it; ++it){
        dampVector(it.col())+=currLambda*it.value()*it.value();
        dampJTris.push_back(Eigen::Triplet<float>(it.row(), it.col(), it.value()));
      }
    }
    for (int i=0;i<dampVector.size();i++)
      dampJTris.push_back(Eigen::Triplet<float>(J.rows()+i,i,sqrt(dampVector(i))));
    
    dampJ.conservativeResize(J.rows()+dampVector.size(),J.cols());
    dampJ.setFromTriplets(dampJTris.begin(), dampJTris.end());
    
    if (verbose)
      std::cout<<"Initial Lambda: "<<currLambda<<std::endl;
  }
  
  bool update(SolverTraits& ST,
              const Eigen::SparseMatrix<float>& J,
              const Eigen::VectorXf& currSolution,
              const Eigen::VectorXf& direction,
              const bool verbose,
              Eigen::SparseMatrix<float>& dampJ){
    
    Eigen::VectorXf EVec;
    Eigen::SparseMatrix<float> stubJ;
    ST.objective_jacobian(currSolution,EVec, stubJ, false);
    float prevEnergy2=EVec.squaredNorm();
    ST.objective_jacobian(currSolution+direction,EVec, stubJ, false);
    float newEnergy2=EVec.squaredNorm();
    if ((prevEnergy2>newEnergy2)&&(newEnergy2!=std::numeric_limits<float>::infinity()))  //progress; making it more gradient descent
      currLambda/=10.0;
    else
      currLambda*=10.0;
    
    if (verbose)
      std::cout<<"Current Lambda: "<<currLambda<<std::endl;
    //collecting the diagonal values
    Eigen::VectorXf dampVector=Eigen::VectorXf::Zero(currSolution.size());
    std::vector<Eigen::Triplet<float>> dampJTris;
    for (int k=0; k<J.outerSize(); ++k){
      for (Eigen::SparseMatrix<float>::InnerIterator it(J,k); it; ++it){
        dampVector(it.col())+=currLambda*it.value()*it.value();
        dampJTris.push_back(Eigen::Triplet<float>(it.row(), it.col(), it.value()));
      }
    }
    for (int i=0;i<dampVector.size();i++)
      dampJTris.push_back(Eigen::Triplet<float>(J.rows()+i,i,sqrt(dampVector(i))));
    
    dampJ.conservativeResize(J.rows()+dampVector.size(),J.cols());
    dampJ.setFromTriplets(dampJTris.begin(), dampJTris.end());
    
    return (prevEnergy2>newEnergy2);  //this preconditioner always approves new direction
    
  }
  
  DiagonalDamping(float _currLambda=0.01):currLambda(_currLambda){}
  ~DiagonalDamping(){}
};
}

#endif /* DiagonalDamping_h */
