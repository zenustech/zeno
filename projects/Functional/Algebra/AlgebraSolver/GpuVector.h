/**
 * @file GpuVector.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2022-01-08
 * 
 * @copyright Copyright (c) 2022  Ma Pengfei
 * 
 */

#ifndef _GPU_VECTOR_H_
#define _GPU_VECTOR_H_

#include "GenericVector.h"
#include "StdVector.h"
#include <gpu_lib.h>


// xinxin use an unnecessary temp array to compute inner product. I think it
// is better to use algrithms of thrust. Here is the reference :
// Thrust快速入门教程（三） —— Algorithms  https://blog.csdn.net/zerolover/article/details/44458985

template<typename T, typename TV>
class GpuVector : public GenericVector<T, TV>
{

private:
    TV* _data;
    using GenericVector<T, TV>::_dim;

    virtual void init() override {

        // free old memory
        if(_data != nullptr) gpu::freeGPUBuffer(_data);
        
        // allocate GPU memory
        gpu::gpu_malloc((void **)&_data, sizeof(TV) * this->size());
    }

    
public:
    virtual bool use_gpu() const { return true;}

    virtual TV* data() override { return _data;}

    virtual const TV* data() const override { return _data;}

    /// interaction with std::vector
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief copy data from GPU memory to HOST memory
     * @param data 
     */
    virtual void get(std::vector<TV>& data) const override {
        CHECK_F(data.size() == this->size(), "Wrong size.");
        gpu::gpu_to_cpu((char*)data.data(), (char*)this->data(), sizeof(TV)*this->size());
    }
    
    /**
     * @brief copy data from HOST memory to GPU memory
     * @param data 
     */
    virtual void set(const std::vector<TV>& data) override {
        CHECK_F(data.size() == this->size(), "Wrong size.");
        gpu::cpu_to_gpu((char*)this->data(), (char*)data.data(), sizeof(TV)*this->size());

    }

    /// linear operations
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    // max(abs(x));
    T abs_max(){
        auto array = flatten(*this);
        return gpu::gpu_abs_max(array.data, array.num);
    }

    virtual double sum() const override {
        auto array = flatten(*this);
        return gpu::gpu_sum(array.data, array.num);
    }

    /// Return minimum value of vector
    virtual double min() const override {
        auto array = flatten(*this);
        return gpu::gpu_min(array.data, array.num);
    }

    /// Return maximum value of vector
    virtual double max() const override {
        auto array = flatten(*this);
        return gpu::gpu_max(array.data, array.num);
    }



    // dot(a,b); 
    // a.inner(b);
    virtual double inner(const GenericVector<T, TV>& x) const override {
        
        CHECK_F(this->use_gpu() == x.use_gpu(), "not on GPU");
        
        auto _x = flatten(x);
        auto _m = flatten(*this);
        
        CHECK_F(_x.num == _m.num, "Wrong size.");
        
        return gpu::gpu_inner(_x.data, _m.data, _x.num);
    }

    // x = y + a*x
    // x.axpy(a,y);
    virtual void axpy(T a, const GenericVector<T, TV>& y)override{

        CHECK_F(this->use_gpu() == y.use_gpu(), "not on GPU.");

        auto _x = flatten(*this);
        auto _y = flatten(y);
        
        CHECK_F(_x.num == _y.num, "Wrong size.");

        gpu::gpu_axpy(_x.data, _x.data, _y.data, a, _x.num);
    }

    // z = a*x + y
    // z.axpy(a,x,y);
    virtual void axpy(T a, const GenericVector<T, TV>& x, const GenericVector<T, TV>& y)override{

        CHECK_F(this->use_gpu() == x.use_gpu(), "not on GPU.");
        CHECK_F(this->use_gpu() == y.use_gpu(), "not on GPU.");

        auto _x = flatten(x);
        auto _y = flatten(y);
        auto _z = flatten(*this);

        CHECK_F(_x.num == _y.num, "Wrong size.");
        CHECK_F(_x.num == _z.num, "Wrong size.");
    
        gpu::gpu_axpy(_z.data, _x.data, _y.data, a, _x.num);
    }

    // // x = a*x = (a-1)*x + x; 
    // // x.axpy(a-1,x);
    // // x *= a;
    // virtual const GenericVector<T, TV>& operator*= (T a) override {
    //     this->axpy(a-1,*this);
    //     return *this;
    // }

    // x = a;
    virtual const GenericVector<T, TV>& operator= (T a) override {
        auto _x = flatten(*this);
        gpu::gpu_fill(_x.data, a, _x.num);
        return *this;
    }

    virtual const GenericVector<T, TV>& operator-= (T a) override {
        CHECK_F(false, "not implemented");
    }

    // construction functions.
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    GpuVector() :  GenericVector<T,TV>(), _data(nullptr){ }

    explicit GpuVector(int3 dim) : GpuVector() {
        CHECK_F(dim.x>=1 && dim.y >= 1 && dim.z >=1, "Wrong dim.");
        _dim = dim;
        init();
    };
    
    explicit GpuVector(const StdVector<T, TV>& cpu_vector): GpuVector() {
        _dim = cpu_vector.dim();
        init();        
        gpu::cpu_to_gpu((char*)_data, (char*)cpu_vector.data(), sizeof(TV)*this->size());
    }
    // Five functions.
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    virtual GpuVector& operator=(const GpuVector& x)
    {
        CHECK_F(this->use_gpu() == x.use_gpu(), "not on GPU.");
        CHECK_F(x.size() == this->size(), "Wrong size.");
        CHECK_F(x.dim().x == this->dim().x && x.dim().y == this->dim().y && x.dim().z == this->dim().z, "Wrong dim.");
        LOG_F(WARNING, "gpu_copy on assignment method.");
        gpu::gpu_copy((char*)this->data(), (char*)x.data(), this->size()*sizeof(TV));
        return *this;
    }

    virtual GpuVector& operator=(GpuVector&& gpu_vector){
        if(this->_data != nullptr) gpu::freeGPUBuffer(this->_data);
        this->_data = gpu_vector.data();
        this->_dim  = gpu_vector.dim();
        gpu_vector.set_nullptr();
        return *this;
    }
    
    explicit GpuVector(const GpuVector& gpu_vector): GpuVector() {
        _dim = gpu_vector.dim();
        init();        
        LOG_F(WARNING, "gpu_copy on construction method.");
        gpu::gpu_copy((char*)_data, (char*)gpu_vector.data(), sizeof(TV)*this->size());
    }
    
    explicit GpuVector(GpuVector&& gpu_vector) : GpuVector() {
        // there is no need to care about freeing memory, because it is a constructor.
        this->_data = gpu_vector.data();
        this->_dim  = gpu_vector.dim();
        gpu_vector.set_nullptr();
    };
    ~GpuVector(){
        if(_data != nullptr) gpu::freeGPUBuffer(_data);
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

private:
    
    void set_nullptr(){ _data = nullptr;}



};

#endif
