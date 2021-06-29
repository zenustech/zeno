#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

// Simple placeholder code for BLAS calls - replace with calls to a real BLAS library

#include <vector>
#include "tbb/tbb.h"

namespace BLAS{

// dot products ==============================================================
template <typename T>
inline double dot(const std::vector<T> &x, const std::vector<T> &y)
{ 
   //return cblas_ddot((int)x.size(), &x[0], 1, &y[0], 1); 

  /* double sum = 0;
   for(unsigned int i = 0; i < x.size(); ++i)
      sum += x[i]*y[i];*/


   //try parallel sum
   T total = tbb::parallel_reduce(
       tbb::blocked_range<int>(0, x.size()),
       0.0,
       [&](tbb::blocked_range<int> r, double running_total)
       {
           for (int i = r.begin(); i < r.end(); ++i)
           {
               running_total += x[i]*y[i];
           }

           return running_total;
       }, std::plus<T>());

   return total;
}

// inf-norm (maximum absolute value: index of max returned) ==================
template <typename T>
inline int index_abs_max(const std::vector<T> &x)
{ 
   //return cblas_idamax((int)x.size(), &x[0], 1); 
   /*int maxind = 0;
   T maxvalue = 0;
   for(unsigned int i = 0; i < x.size(); ++i) {
      if(fabs(x[i]) > maxvalue) {
         maxvalue = fabs(x[i]);
         maxind = i;
      }
   }
   return maxind;*/

   //parallel reduction style
   auto result =tbb::parallel_reduce(tbb::blocked_range<unsigned int>(0, (unsigned int) x.size()), std::make_pair(int(0),T(0)),
       [&](const tbb::blocked_range<unsigned int>& R, std::pair<int,T> curr_idx_maxval) {
           
           for (unsigned int i = R.begin(); i != R.end(); ++i) {
               if (fabs(x[i]) > curr_idx_maxval.second) {
                   curr_idx_maxval.first = i;
                   curr_idx_maxval.second = fabs(x[i]);
               }
           }
           return curr_idx_maxval;
       },
       [&](const std::pair<int,T>& a, const std::pair<int,T>& b) {
           if (a.second > b.second) {
               return a;
           }
           else {
               return b;
           }
       });
   return result.first;
}

// inf-norm (maximum absolute value) =========================================
// technically not part of BLAS, but useful
template <typename T>
inline T abs_max(const std::vector<T> &x)
{ return std::fabs(x[index_abs_max(x)]); }

// saxpy (y=alpha*x+y) =======================================================
template <typename T>
inline void add_scaled(T alpha, const std::vector<T> &x, std::vector<T> &y)
{ 
   //cblas_daxpy((int)x.size(), alpha, &x[0], 1, &y[0], 1); 
   //for(unsigned int i = 0; i < x.size(); ++i)
   //   y[i] += alpha*x[i];
	assert(y.size() == x.size());
	int num = (int)x.size();
	tbb::parallel_for((int)0, (int)num, (int)1, [&] (int i){
		y[i] += alpha*x[i];
	});
}

}
#endif
