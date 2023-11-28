// This file is part of SaddlePoint, a simple library for Eigen-based sparse nonlinear optimization
//
// Copyright (C) 2021 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SADDLEPOINT_EIGEN_SOLVER_WRAPPER_H
#define SADDLEPOINT_EIGEN_SOLVER_WRAPPER_H
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include <cstdio>

namespace SaddlePoint
{
  //a templated wrapper to all sparse solvers by Eigen. Not doing much and not entirely efficient since the matrix has to be initialized twice, but not too bad.
  //
  
  //TODO: perhaps better to invalidate the analysis stage and do it all in the factorization.
  
  template<class EigenSparseSolver>
  class EigenSolverWrapper{
  public:
    EigenSparseSolver solver;
   
    Eigen::SparseMatrix<float> A;
    //if Symmetric = true that means that (_rows, _cols) only contain the bottom left as input, and the matrix will be symmetrized.
    bool analyze(const Eigen::SparseMatrix<int>& JPattern){
      /*rows=_rows;
      cols=_cols;
      A.resize(rows.maxCoeff()+1, cols.maxCoeff()+1);
      std::vector<Eigen::Triplet<float> > triplets;
      for (int i=0;i<rows.size();i++){
        triplets.push_back(Eigen::Triplet<float> (rows(i), cols(i), 1.0));  //it's just a pattern
        if ((Symmetric)&&(rows(i)!=cols(i)))
          triplets.push_back(Eigen::Triplet<float> (cols(i), rows(i), 1.0));
      }
      A.setZero();
      A.setFromTriplets(triplets.begin(), triplets.end());*/
      //converting to float
      std::vector<Eigen::Triplet<float>> JPatDoubleTris;
      Eigen::SparseMatrix<float> JPatDouble(JPattern.rows(), JPattern.cols());
      for (int k=0; k<JPattern.outerSize(); ++k)
        for (Eigen::SparseMatrix<int>::InnerIterator it(JPattern,k); it; ++it)
          JPatDoubleTris.push_back(Eigen::Triplet<float>(it.col(), it.row(), 1.0));
      
      JPatDouble.setFromTriplets(JPatDoubleTris.begin(), JPatDoubleTris.end());
      solver.analyzePattern(JPatDouble.transpose()*JPatDouble);
      return true; //(solver.info()==Eigen::Success);
    }
    
    bool factorize(const Eigen::SparseMatrix<float> _A){
      A=_A;
      //solver.analyzePattern(A);
      //solver.factorize(A);
      solver.compute(A);
      return (solver.info()==Eigen::Success);
    }
    
    bool solve(const Eigen::MatrixXf& rhs,
               Eigen::MatrixXf& x){
      
      //cout<<"Rhs: "<<rhs<<endl;
      x.conservativeResize(A.cols(), rhs.cols());
      for (int i=0;i<rhs.cols();i++)
        x.col(i) = solver.solve(rhs.col(i));
      return true;
    }
    
    bool solve(const Eigen::VectorXf& rhs,
               Eigen::VectorXf& x){
      
      //cout<<"Rhs: "<<rhs<<endl;
      x = solver.solve(rhs);
      return true;
    }
  };
  
  //a simple SPD linear solution solver
  template<class EigenSparseSolver>
  Eigen::MatrixXf EigenSingleSolveWrapper(const Eigen::SparseMatrix<float> A,Eigen::MatrixXf b, bool Symmetric)
  {
    using namespace Eigen;
    VectorXi I;
    VectorXi J;
    VectorXf S;
    int Counter=0;
    
    for (int k=0; k<A.outerSize(); ++k)
      for (SparseMatrix<float>::InnerIterator it(A,k); it; ++it)
      {
        if ((it.row()>it.col())&&(Symmetric))
          continue;
        
        Counter++;
      }
    
    int NumNonZero=Counter;
    I.resize(NumNonZero);
    J.resize(NumNonZero);
    S.resize(NumNonZero);
    Counter=0;
    for (int k=0; k<A.outerSize(); ++k)
      for (SparseMatrix<float>::InnerIterator it(A,k); it; ++it)
      {
        if ((it.row()>it.col())&&(Symmetric))
          continue;
        
        S(Counter)=it.value();
        I(Counter)=it.row();   // row index
        J(Counter++)=it.col();   // col index (here it is equal to k
        
      }
    
    EigenSolverWrapper<EigenSparseSolver> ls;
    ls.analyze(I,J, Symmetric);
    ls.factorize(S, Symmetric);
    MatrixXf x;
    ls.solve(b,x);
    return x;
  }
  
}


#endif
