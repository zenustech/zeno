#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

// Simple placeholder code for BLAS calls - replace with calls to a real BLAS library

#include <vector>
#include "tbb/tbb.h"

namespace BLAS{

// dot products ==============================================================
    inline double mean(const std::vector<double> &x)
    {
        double sum = 0;
        for (size_t i = 0;i < x.size();i++)
        {
            sum += x[i];
        }
        return sum / (double)x.size();
    }
    inline void subtractConst(std::vector<double> &x, const double a)
    {
        size_t num = x.size();
        tbb::parallel_for((size_t)0, (size_t)num, (size_t)1, [&](size_t i) {
            x[i] -= a;
        });
    }

    inline double dot(const std::vector<double> &x, const std::vector<double> &y)
    {
        //return cblas_ddot((int)x.size(), &x[0], 1, &y[0], 1);

   auto values = std::vector<double>(x.size());
    tbb::parallel_for( tbb::blocked_range<int>(0,values.size()),
                       [&](tbb::blocked_range<int> r)
                       {
                           for (int i=r.begin(); i<r.end(); ++i)
                           {
                               values[i] = x[i]*y[i];
                           }
                       });
    auto total = tbb::parallel_reduce(
            tbb::blocked_range<int>(0,values.size()),
            0.0,
            [&](tbb::blocked_range<int> r, double running_total)
            {
                for (int i=r.begin(); i<r.end(); ++i)
                {
                    running_total += values[i];
                }

                return running_total;
            }, std::plus<double>() );
    return (double)total;
//        double sum = 0;
//        for(unsigned int i = 0; i < x.size(); ++i)
//            sum += x[i]*y[i];
//        return sum;
    }

// inf-norm (maximum absolute value: index of max returned) ==================

    inline int index_abs_max(const std::vector<double> &x)
    {
        //return cblas_idamax((int)x.size(), &x[0], 1);
        int maxind = 0;
        double maxvalue = 0;
        for(unsigned int i = 0; i < x.size(); ++i) {
            if(fabs(x[i]) > maxvalue) {
                maxvalue = fabs(x[i]);
                maxind = i;
            }
        }
        return maxind;
    }

// inf-norm (maximum absolute value) =========================================
// technically not part of BLAS, but useful

    inline double abs_max(const std::vector<double> &x)
    {
    auto values = std::vector<double>(x.size());
    tbb::parallel_for( tbb::blocked_range<int>(0,values.size()),
                       [&](tbb::blocked_range<int> r)
                       {
                           for (int i=r.begin(); i<r.end(); ++i)
                           {
                               values[i] = std::fabs(x[i]);
                           }
                       });
    auto total = tbb::parallel_reduce(
            tbb::blocked_range<int>(0,values.size()),
            0.0,
            [&](tbb::blocked_range<int> r, double running_total)
            {
                for (int i=r.begin(); i<r.end(); ++i)
                {
                    running_total = std::max(running_total, values[i]);
                }

                return running_total;
            }, [&](double x, double y)->double{
                return std::max(x,y);
            }
            );
    return (double)total;
        //return std::fabs(x[index_abs_max(x)]);
    }

// saxpy (y=alpha*x+y) =======================================================

    inline void add_scaled(double alpha, const std::vector<double> &x, std::vector<double> &y)
    {
        //cblas_daxpy((int)x.size(), alpha, &x[0], 1, &y[0], 1);
        //for(unsigned int i = 0; i < x.size(); ++i)
        //   y[i] += alpha*x[i];
        assert(y.size() == x.size());
        int num = x.size();
        tbb::parallel_for((int)0, (int)num, (int)1, [&] (int i){

            y[i] += alpha*x[i];
        });
    }

}
#endif