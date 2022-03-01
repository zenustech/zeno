/**
 * @file StdVector.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-30
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */
#ifndef _STD_VECTOR_H_
#define _STD_VECTOR_H_

#include "GenericVector.h"
#include <limits>

template<typename T, typename TV>
class StdVector : public GenericVector<T, TV>
{

private:

    using GenericVector<T, TV>::_dim;

    
    virtual void init() override { _data.resize(this->size()); }

public:

    std::vector<TV> _data;

    virtual bool use_gpu() const override { return false;}

    virtual TV* data() override { return _data.data();}

    virtual const TV* data() const override { return _data.data();}

    virtual void get(std::vector<TV>& data) const override { data = _data; }

    virtual void set(const std::vector<TV>& data) override {
        _dim = make_int3(static_cast<int>(data.size()),1,1);
        _data = data;
    }

    virtual double sum() const override {
        auto array = flatten(*this);
        T sum = T(0);
        for (size_t i = 0; i < array.num; i++)
        {
            sum += array.data[i];
        }
        return sum;
    }

    /// Return minimum value of vector
    virtual double min() const override {
        auto array = flatten(*this);
        auto result = std::numeric_limits<T>::max();
        for (size_t i = 0; i < array.num; i++)
        {
            if (array.data[i] < result) result = array.data[i];
        }
        return result;
    }

    /// Return maximum value of vector
    virtual double max() const override {
        auto array = flatten(*this);
        auto result = std::numeric_limits<T>::min();
        for (size_t i = 0; i < array.num; i++)
        {
            if (array.data[i] > result) result = array.data[i];
        }
        return result;
    }

    // dot(a,b); 
    // a.inner(b);
    virtual double inner(const GenericVector<T, TV>& x) const override {
        auto _x = flatten(x);
        auto _m = flatten(*this);        
        CHECK_F(_x.num == _m.num, "Wrong size.");
        
        T sum = 0.0;
        for (size_t i = 0; i < _x.num; i++) sum += _x.data[i]*_m.data[i];
        return sum;
    }

    // x = y + a*x
    // x.axpy(a,y);

    // x = a*x = (a-1)*x + x; 
    // x.axpy(a-1,x);

    virtual void axpy(T a, const GenericVector<T, TV>& y) override {

        auto _x = flatten(*this);
        auto _y = flatten(y);
        
        CHECK_F(_x.num == _y.num, "Wrong size.");

        for (size_t i = 0; i < _x.num; i++) _x.data[i] = _y.data[i] + a*_x.data[i];
    }

    // z = a*x + y
    // z.axpy(a,x,y);
    virtual void axpy(T a, const GenericVector<T, TV>& x, const GenericVector<T, TV>& y) override {

        auto _x = flatten(x);
        auto _y = flatten(y);
        auto _z = flatten(*this);

        CHECK_F(_x.num == _y.num, "Wrong size.");
        CHECK_F(_x.num == _z.num, "Wrong size.");
        
        for (size_t i = 0; i < _x.num; i++) _z.data[i] = a*_x.data[i] + _y.data[i]; 
    }

    virtual const GenericVector<T, TV>& operator= (T a) override {
        auto _x = flatten(*this);
        for (size_t i = 0; i < _x.num; i++) _x.data[i] = a;
        return *this;
    }

    virtual const GenericVector<T, TV>& operator-= (T a) override {
        auto _x = flatten(*this);
        for (size_t i = 0; i < _x.num; i++) _x.data[i] -= a;
        return *this;
    }
    // virtual StdVector& operator=(const StdVector& x)
    // {
    //     _data = x._data;
    //     return *this;
    // }

    
    StdVector() : GenericVector<T,TV>(), _data(0){

    }

    explicit StdVector(int3 dim) : StdVector() {
        CHECK_F(dim.x>=1 && dim.y >= 1 && dim.z >=1, "Wrong dim.");
        _dim = dim;
        init();
    }


    ~StdVector(){}

};

#endif
