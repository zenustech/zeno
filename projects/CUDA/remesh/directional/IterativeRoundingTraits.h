// This file is part of Directional, a library for directional field processing.
// Copyright (C) 2021 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef ITERATIVE_ROUNDING_TRAITS_H
#define ITERATIVE_ROUNDING_TRAITS_H

#include "./SIInitialSolutionTraits.h"
#include "./SaddlePoint/sparse_block.h"


template <class LinearSolver>
class IterativeRoundingTraits{
public:
  
  int xSize;
  int ESize;
  
  Eigen::SparseMatrix<float> A,C,G,G2, UFull, x2CornerMat, UExt;
  Eigen::MatrixXf rawField, rawField2, FN, V,B1,B2, origFieldVolumes,SImagField;
  Eigen::MatrixXi F;
  Eigen::VectorXf b,xPoisson, fixedValues, x0, x0Small, xCurrSmall, xPrevSmall,rawField2Vec,rawFieldVec, xCurr;
  Eigen::VectorXi fixedIndices, integerIndices, singularIndices;
  Eigen::MatrixXi IImagField, JImagField;
  int N,n;
  float lengthRatio, paramLength, fraction;
  float wConst, wBarrier, wClose, s, wPoisson;
  Eigen::VectorXi leftIndices;
  bool roundedSingularities, roundSeams, localInjectivity;
  
  float origValue,roundValue;
  int currRoundIndex;
  
  bool success;
  
  //elements of the jacobian which are fixed
  Eigen::SparseMatrix<float> gObj,gClose,gConst,G2UFullParamLength, gObjCloseConst;
  
  void initial_solution(Eigen::VectorXf& _x0){
    _x0 = x0Small;
  }
  
  bool pre_optimization(const Eigen::VectorXf& prevx){return true;}
  void pre_iteration(const Eigen::VectorXf& prevx){}
  bool post_iteration(const Eigen::VectorXf& x){return false;}

  IterativeRoundingTraits() :xSize(0), ESize(0){}
  ~IterativeRoundingTraits() {}
  
  
  bool initFixedIndices() {
      using namespace Eigen;
      using namespace std;

      xPrevSmall = xCurrSmall;
      xCurr = UFull * xCurrSmall;


      VectorXf roundDiffs(leftIndices.size());
      float minRoundDiff = 3276700.0;
      int minRoundIndex = -1;
      for (int i = 0; i < leftIndices.size(); i++) {
          //cout<<"fraction*xCurr(leftIndices(i)): "<<fraction*xCurr(leftIndices(i))<<endl;
          //cout<<"std::round(fraction*xCurr(leftIndices(i))): "<<std::round(fraction*xCurr(leftIndices(i)))<<endl;
          roundDiffs(i) = std::fabs(fraction * xCurr(leftIndices(i)) - std::round(fraction * xCurr(leftIndices(i))));
          if (roundDiffs(i) < minRoundDiff) {
              minRoundIndex = i;
              minRoundDiff = roundDiffs(i);
              // cout<<"minRoundDiff: "<<minRoundDiff<<endl;
          }
      }

      currRoundIndex = leftIndices(minRoundIndex);
      origValue = xCurr(leftIndices(minRoundIndex));
      roundValue = std::round(fraction * xCurr(leftIndices(minRoundIndex))) / fraction;
      //cout<<"origValue,roundValue: "<<origValue<<","<<roundValue<<endl;
      fixedIndices.conservativeResize(fixedIndices.size() + 1);
      fixedIndices(fixedIndices.size() - 1) = leftIndices[minRoundIndex];  //is this under-performing?
      fixedValues.conservativeResize(fixedValues.size() + 1);
      fixedValues(fixedValues.size() - 1) = roundValue;


      VectorXi newLeftIndices(leftIndices.size() - 1);
      newLeftIndices.head(minRoundIndex) = leftIndices.head(minRoundIndex);
      newLeftIndices.tail(newLeftIndices.size() - minRoundIndex) = leftIndices.tail(newLeftIndices.size() - minRoundIndex);
      leftIndices = newLeftIndices;

      if ((leftIndices.size() == 0) && (!roundSeams) && (!roundedSingularities)) {  //completed rounding singularities;starting to round rest of seams
          leftIndices = integerIndices;
          roundedSingularities = true;
      }

      //Updating fixed elements of the jacobian
      G2UFullParamLength = G2 * UFull * paramLength;

      //Poisson error
      gObj = G2UFullParamLength * wPoisson;

      //Closeness
      int gClose_d = xCurrSmall.size();
      gClose = Eigen::SparseMatrix<float>(gClose_d, gClose_d);
      gClose.reserve(gClose_d);
      for(int i = 0;i<gClose_d;i++) {
        gClose.insert(i,i) = 1.0;
      }
      gClose.finalize();

      gClose = gClose * wClose;  //TODO: bad performance

      //fixedIndices constness
      gConst.resize(fixedIndices.size(), xCurr.size());
      vector<Triplet<float>> gConstTriplets;
      for (int i = 0; i < fixedIndices.size(); i++)
          gConstTriplets.push_back(Triplet<float>(i, fixedIndices(i), 1.0));

      gConst.setFromTriplets(gConstTriplets.begin(), gConstTriplets.end());

      gConst = gConst * UFull * wConst;

      MatrixXi blockIndices(3, 1);
      blockIndices << 0, 1, 2;
      vector<SparseMatrix<float>*> JMats;
      JMats.push_back(&gObj);
      JMats.push_back(&gClose);
      JMats.push_back(&gConst);

      SaddlePoint::sparse_block(blockIndices, JMats, gObjCloseConst);


      if (ESize == 0) {
        SparseMatrix<float> J;
        VectorXf EVec;
        objective_jacobian(Eigen::VectorXf::Random(UFull.cols()), EVec, J, false);
        ESize = EVec.size();
      }
    
    return (minRoundDiff>10e-7); //only proceeding if there is a need to round
  }
  
  
  
  void objective_jacobian(const Eigen::VectorXf& xCurrSmall,  Eigen::VectorXf& EVec, Eigen::SparseMatrix<float>& J, bool computeJacobian){
    using namespace std;
    using namespace Eigen;
    
    
    VectorXf xCurr = UFull*xCurrSmall;
    VectorXf fObj = G2UFullParamLength*xCurrSmall - rawField2Vec;
    VectorXf fClose = (xCurrSmall-xPrevSmall);
    
    VectorXf fConst(fixedIndices.size());
    for (int i=0;i<fixedIndices.size();i++)
      fConst(i) = xCurr(fixedIndices(i))-fixedValues(i);
    
    
    VectorXf currField = G2UFullParamLength*xCurrSmall;
    VectorXf fBarrier = VectorXf::Zero(N*FN.rows());
    VectorXf barSpline = VectorXf::Zero(N*FN.rows());
    VectorXf splineDerivative;
    if (computeJacobian)
      splineDerivative=VectorXf::Zero(N*FN.rows(),1);
    
    for (int i=0;i<FN.rows();i++){
      for (int j=0;j<N;j++){
        RowVector2f currVec=currField.segment(2*N*i+2*j,2);
        RowVector2f nextVec=currField.segment(2*N*i+2*((j+1)%N),2);
        float imagProduct = (currVec(0)*nextVec(1) - currVec(1)*nextVec(0))/origFieldVolumes(i,j);
        float barResult = (imagProduct/s)*(imagProduct/s)*(imagProduct/s) - 3.0*(imagProduct/s)*(imagProduct/s) + 3.0*(imagProduct/s);
        float barResult2 = 1.0/barResult - 1.0;
        if (imagProduct<=0) barResult2 = std::numeric_limits<float>::infinity();
        if (imagProduct>=s) barResult2 = 0.0;
        fBarrier(N*i+j)=barResult2;
        
        if (computeJacobian){
          barSpline(N*i+j)=barResult;
          
          float splineDerivativeLocal=3.0*(imagProduct*imagProduct/(s*s*s)) -6.0*(imagProduct/(s*s)) + 3.0/s;
          if (imagProduct<=0) splineDerivativeLocal=std::numeric_limits<float>::infinity();
          if (imagProduct>=s) splineDerivativeLocal=0.0;
          splineDerivative(N*i+j)=splineDerivativeLocal;
          
          SImagField.row(N*i+j)<<nextVec(1)/origFieldVolumes(i,j), -nextVec(0)/origFieldVolumes(i,j), -currVec(1)/origFieldVolumes(i,j),currVec(0)/origFieldVolumes(i,j);
        }
      }
    }
    
    if (localInjectivity){
      EVec.conservativeResize(fObj.size()+fClose.size()+fConst.size()+fBarrier.size());
      EVec<<fObj*wPoisson,fClose*wClose,fConst*wConst,fBarrier*wBarrier;
    }else{
      EVec.conservativeResize(fObj.size()+fClose.size()+fConst.size());
      EVec<<fObj*wPoisson,fClose*wClose,fConst*wConst;
    }
    
    if (!computeJacobian)
      return;
    
    SparseMatrix<float> gImagField(N*FN.rows(), currField.size());
    vector<Triplet<float>> gImagTriplets;
    for (int i=0;i<IImagField.rows();i++)
      for (int j=0;j<IImagField.cols();j++)
        gImagTriplets.push_back(Triplet<float>(IImagField(i,j), JImagField(i,j), SImagField(i,j)));
    
    gImagField.setFromTriplets(gImagTriplets.begin(), gImagTriplets.end());
    
    VectorXf barDerVec=-splineDerivative.array()/((barSpline.array()*barSpline.array()).array());
    for (int i=0;i<fBarrier.size();i++)
      if (std::abs(fBarrier(i))<10e-9)
        barDerVec(i)=0.0;
      else if (fBarrier(i)==std::numeric_limits<float>::infinity())
        barDerVec(i)=std::numeric_limits<float>::infinity();
    
    SparseMatrix<float> gBarrierFunc(barDerVec.size(), barDerVec.size());
    vector<Triplet<float>> gBarrierFuncTris;
    for (int i=0;i<barDerVec.size();i++)
      gBarrierFuncTris.push_back(Triplet<float>(i,i,barDerVec(i)));
    gBarrierFunc.setFromTriplets(gBarrierFuncTris.begin(), gBarrierFuncTris.end());
    
    SparseMatrix<float> gBarrier = gBarrierFunc*gImagField*G2UFullParamLength*wBarrier;
    
    if (localInjectivity){
      MatrixXi blockIndices(2,1);
      blockIndices<<0,1;
      vector<SparseMatrix<float>*> JMats;
      JMats.push_back(&gObjCloseConst);
      JMats.push_back(&gBarrier);
      SaddlePoint::sparse_block(blockIndices, JMats,J);
    } else J=gObjCloseConst;
  }
  
  
  
  bool post_checking(const Eigen::VectorXf& x){
    //std::cout<<"x:"<<x<<std::endl;
    
    xCurr = UFull*x;
    Eigen::VectorXf roundDiffs(fixedIndices.size());
    //int minRoundDiff=3276700.0;
    //int minRoundIndex=-1;
    for (int i=0;i<fixedIndices.size();i++){
      // std::cout<<"xCurr(fixedIndices[i]): "<<xCurr(fixedIndices[i])<<std::endl;
      roundDiffs(i) = std::abs(fraction*xCurr(fixedIndices[i])-std::round(fraction*xCurr(fixedIndices[i])));
      /*if (roundDiffs(i)<minRoundDiff){
       minRoundIndex=i;
       minRoundDiff=roundDiffs(i);
       }*/
    }
    
    xCurrSmall=x;
    x0Small=xCurrSmall;
    xPrevSmall=x0Small;
    
    //std::cout<<"roundDiffs.maxCoeff(): "<<roundDiffs.maxCoeff()<<std::endl;
    return (roundDiffs.maxCoeff()<=10e-7);
    //} else {
    //  return (leftIndices.size()==0);
    //}
  }
  
  
  void init(const SIInitialSolutionTraits<LinearSolver>& sist, const Eigen::VectorXf& initCurrXandFieldSmall, bool _roundSeams){
    using namespace std;
    using namespace Eigen;
    
    
    A=sist.A; C=sist.C; G=sist.G; G2=sist.G2; UFull=sist.UFull; x2CornerMat=sist.x2CornerMat; UExt=sist.UExt;
    rawField=sist.rawField; rawField2=sist.rawField2; FN=sist.FN, V=sist.V; B1=sist.B1; B2=sist.B2; origFieldVolumes=sist.origFieldVolumes; SImagField=sist.SImagField;
    F=sist.F;
    b=sist.b; xPoisson=sist.xPoisson; fixedValues=sist.fixedValues;  rawField2Vec=sist.rawField2Vec; rawFieldVec=sist.rawFieldVec;
    fixedIndices=sist.fixedIndices; integerIndices=sist.integerIndices; singularIndices=sist.singularIndices;
    IImagField=sist.IImagField; JImagField=sist.JImagField;
    N=sist.N,n=sist.n;
    lengthRatio=sist.lengthRatio; paramLength=sist.paramLength;
    wConst=sist.wConst, wBarrier=sist.wBarrier, wClose=sist.wClose, s=sist.s;
    roundSeams=_roundSeams;
    roundedSingularities=false;
    localInjectivity=sist.localInjectivity;
    
    wPoisson=1;
    wClose = 0.01;
    wConst=10e4;
    
    //Updating initial quantities
    
    
    VectorXf currXandField=UExt*initCurrXandFieldSmall;
    
    xSize=UFull.cols();
    x0=currXandField.head(UFull.rows());
    x0Small=initCurrXandFieldSmall.head(UFull.cols());
    rawField2Vec=currXandField.tail(2*N*FN.rows());
    
    rawField2.conservativeResize(FN.rows(),2*N);
    float avgGradNorm=0.0;
    for (int i=0;i<FN.rows();i++)
      rawField2.row(i)=rawField2Vec.segment(2*N*i,2*N);
    
    for (int i=0;i<FN.rows();i++)
      for (int j=0;j<N;j++)
        avgGradNorm+=rawField2.block(i,2*j,1,2).norm();
    
    avgGradNorm/=(float)(N*FN.rows());
    //cout<<"avgGradNorm: "<<avgGradNorm<<endl;
    
    rawField.array()/=avgGradNorm;
    rawField2.array()/=avgGradNorm;
    x0.array()/=avgGradNorm;
    
    origFieldVolumes.resize(FN.rows(),N);
    for (int i=0;i<FN.rows();i++){
      for (int j=0;j<N;j++){
        RowVector2f currVec=rawField2.block(i,2*j,1,2);
        RowVector2f nextVec=rawField2.block(i,2*((j+1)%N),1,2);
        
        origFieldVolumes(i,j)=currVec.norm()*nextVec.norm();//currVec(0)*nextVec(1)-currVec(1)*nextVec(0);
      }
    }
    
    
    //cout<<"min origFieldVolumes: "<<origFieldVolumes.colwise().minCoeff()<<endl;
    
    fixedIndices=VectorXi::Zero(0);
    fixedValues=VectorXf::Zero(0.0);
    
    if (roundSeams)
      leftIndices=integerIndices;
    else
      leftIndices=singularIndices;
  
    
    xCurrSmall=x0Small;
    xPrevSmall=xCurrSmall;
    fraction=1.0;
    
  
  }
};




#endif /* Iterative Rounding Traits */
