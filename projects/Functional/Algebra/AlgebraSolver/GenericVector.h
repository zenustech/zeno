/**
 * @file GenericVector.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2022-01-08
 * 
 * @copyright Copyright (c) 2022  Ma Pengfei
 * 
 */
#ifndef _GENERIC_VECTOR_H_
#define _GENERIC_VECTOR_H_

// STANDARD
#include <vector>

// CUDA
#include <vector_types.h>
#include <helper_math.h>

// THIRD PARTY
#include <loguru/loguru.hpp>

// NOTE : reason for designing this class
// After studied xinxin's code and dolfin's code, I designed GenericVector 
// which is an interface class. I want GpuVector and StdVector have the same
// interface so that BiCONSTAB can be applied on both GPU and CPU without
// modification. Besides, GenericVector can interact whith std::vector.

// Two classes  : Array1D<T> and GenericVector<T, TV>;
// One function : Array1D<T> flatten(const GenericVector<T, TV>& x);

// -----------------------------------------------------------------------------------
// GenericVector<T, TV> is an interface class, but I don't know how to complete it now.
// blogs for future study : 
// C++：如何正确的使用接口类      https://blog.csdn.net/netyeaxi/article/details/80887646
// C++：如何正确的定义一个接口类  https://blog.csdn.net/netyeaxi/article/details/80724557
template<typename T, typename TV> class GenericVector;

// wrap an raw pointer and size of an array.
template<class T> class Array1D;

template<typename T, typename TV>
Array1D<T> flatten(const GenericVector<T, TV>& x){
    Array1D<T> res;
    res.data = (T *) x.data();
    res.num = x.value_size() * x.size();
    return res;

}

template<class T>
class Array1D {
public:
    T *data;
    size_t num;
};

// TV Including : float, float3, float4, double, double3, double4
// T  Including : float, double
template<typename T, typename TV>
class GenericVector
{
protected:
    int3 _dim;
public:

    GenericVector():_dim({0,1,1}){}

    size_t size() const { return _dim.x*_dim.y*_dim.z; }

    void resize(size_t new_size) {

        CHECK_F(new_size >= 1, "Wrong size.");

        if (new_size != this->size()){
            _dim = make_int3(new_size, 1, 1);
            init();
        }
        _dim= make_int3(new_size, 1, 1);
    }

    void resize(int3 dim) {

        CHECK_F(dim.x>=1 && dim.y >= 1 && dim.z >=1, "Wrong dim.");

        size_t new_size = dim.x*dim.y*dim.z;

        if (new_size != this->size()) {
            _dim = dim;
            init();
        } else {
            _dim = dim;
        }
    };
    
    int3 dim() const {return _dim;}

    constexpr size_t value_size() const { return sizeof(TV) / sizeof(T); }

    virtual bool use_gpu() const = 0;
    
    // Allocate memory
    virtual void init() = 0;

    virtual TV* data() = 0;

    virtual const TV* data() const = 0;

    // NOTE : Try not to use these operator, because it is expensive for data stored in GPU memory.
    // virtual double& operator[](std::size_t __n) = 0;
    // virtual const double& operator[](std::size_t __n) const = 0;
    
    // /// Set all entries to zero.
    // virtual void zero() = 0;

    // /// Return copy of vector
    // virtual std::shared_ptr<GenericVector> copy() const = 0;

    // /// Get block of values
    virtual void get(std::vector<TV>& data) const = 0;
    
    // /// Add block of values
    // virtual void add(const double* block, std::size_t m) = 0;

    // /// Set block of values
    virtual void set(const std::vector<TV>& data) = 0;

    // /// Add multiple of given vector (AXPY operation : y = y + a*x)
    virtual void axpy(T a, const GenericVector& x) = 0;

    virtual void axpy(T a, const GenericVector& x, const GenericVector& y) = 0;

    // /// Replace all entries in the vector by their absolute values
    // virtual void abs() = 0;

    // /// Return inner product with given vector
    virtual double inner(const GenericVector& x) const = 0;

    // /// Return norm of vector ("l2" by default?)
    // virtual double norm(std::string norm_type) const = 0;

    /// Return minimum value of vector
    virtual double min() const = 0;

    /// Return maximum value of vector
    virtual double max() const = 0;

    /// Return sum of vector
    virtual double sum() const = 0;

    // /// Multiply vector by given number
    // virtual const GenericVector& operator*= (double a) = 0;

    // /// Multiply vector by another vector pointwise
    // virtual const GenericVector& operator*= (const GenericVector& x) = 0;

    // /// Divide vector by given number
    // virtual const GenericVector& operator/= (double a) = 0;

    // /// Add given vector
    // virtual const GenericVector& operator+= (const GenericVector& x) = 0;

    // /// Add number to all components of a vector
    // virtual const GenericVector& operator+= (double a) = 0;

    // /// Subtract given vector
    // virtual const GenericVector& operator-= (const GenericVector& x) = 0;

    /// Subtract number from all components of a vector
    virtual const GenericVector& operator-= (T a) = 0;

    /// Assignment operator
    // TODO : what's the difference between GenericVector& and const GenericVector& as return type?
    // StdVector<double, double3>& StdVector<double, double3>::operator=(const StdVector<double, double3>&) do not override this function.
    // virtual GenericVector& operator= (const GenericVector& x) = default;

    /// Assignment operator
    virtual const GenericVector& operator= (T a) = 0; 

};

#endif