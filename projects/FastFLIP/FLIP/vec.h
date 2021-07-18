#ifndef VEC_H
#define VEC_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <ostream>
#include "util.h"

// Defines a thin wrapper around fixed size C-style arrays, using template parameters,
// which is useful for dealing with vectors of different dimensions.
// For example, float[3] is equivalent to Vec<3,float>.
// Entries in the vector are accessed with the overloaded [] operator, so
// for example if x is a Vec<3,float>, then the middle entry is x[1].
// For convenience, there are a number of typedefs for abbreviation:
//   Vec<3,float> -> Vec3f
//   Vec<2,int>   -> Vec2i
// and so on.
// Arithmetic operators are appropriately overloaded, and functions are defined
// for additional operations (such as dot-products, norms, cross-products, etc.)
namespace FLUID
{

    template<unsigned int N, class T>
    struct Vec
    {
        T v[N];

        Vec<N,T>(void)
        {}

        Vec<N,T>(T value_for_all)
        { for(unsigned int i=0; i<N; ++i) v[i]=value_for_all; }

        template<class S>
        Vec<N,T>(const S *source)
        { for(unsigned int i=0; i<N; ++i) v[i]=(T)source[i]; }

        template <class S>
        explicit Vec<N,T>(const Vec<N,S>& source)
        { for(unsigned int i=0; i<N; ++i) v[i]=(T)source[i]; }

        Vec<N,T>(T v0, T v1)
        {
            assert(N==2);
            v[0]=v0; v[1]=v1;
        }

        Vec<N,T>(T v0, T v1, T v2)
        {
            assert(N==3);
            v[0]=v0; v[1]=v1; v[2]=v2;
        }

        Vec<N,T>(T v0, T v1, T v2, T v3)
        {
            assert(N==4);
            v[0]=v0; v[1]=v1; v[2]=v2; v[3]=v3;
        }

        Vec<N,T>(T v0, T v1, T v2, T v3, T v4)
        {
            assert(N==5);
            v[0]=v0; v[1]=v1; v[2]=v2; v[3]=v3; v[4]=v4;
        }

        Vec<N,T>(T v0, T v1, T v2, T v3, T v4, T v5)
        {
            assert(N==6);
            v[0]=v0; v[1]=v1; v[2]=v2; v[3]=v3; v[4]=v4; v[5]=v5;
        }

        T &operator[](int index)
        {
            assert(0<=index && (unsigned int)index<N);
            return v[index];
        }

        const T &operator[](int index) const
        {
            assert(0<=index && (unsigned int)index<N);
            return v[index];
        }

        operator bool(void) const
        {
            for(unsigned int i=0; i<N; ++i) if(v[i]) return true;
            return false;
        }

        Vec<N,T> operator+=(const Vec<N,T> &w)
        {
            for(unsigned int i=0; i<N; ++i) v[i]+=w[i];
            return *this;
        }

        Vec<N,T> operator+(const Vec<N,T> &w) const
        {
            Vec<N,T> sum(*this);
            sum+=w;
            return sum;
        }

        Vec<N,T> operator-=(const Vec<N,T> &w)
        {
            for(unsigned int i=0; i<N; ++i) v[i]-=w[i];
            return *this;
        }

        Vec<N,T> operator-(void) const // unary minus
        {
            Vec<N,T> negative;
            for(unsigned int i=0; i<N; ++i) negative.v[i]=-v[i];
            return negative;
        }

        Vec<N,T> operator-(const Vec<N,T> &w) const // (binary) subtraction
        {
            Vec<N,T> diff(*this);
            diff-=w;
            return diff;
        }

        Vec<N,T> operator*=(T a)
        {
            for(unsigned int i=0; i<N; ++i) v[i]*=a;
            return *this;
        }

        Vec<N,T> operator*(T a) const
        {
            Vec<N,T> w(*this);
            w*=a;
            return w;
        }

        Vec<N,T> operator*(const Vec<N,T> &w) const
        {
            Vec<N,T> componentwise_product;
            for(unsigned int i=0; i<N; ++i) componentwise_product[i]=v[i]*w.v[i];
            return componentwise_product;
        }

        Vec<N,T> operator/=(T a)
        {
            for(unsigned int i=0; i<N; ++i) v[i]/=a;
            return *this;
        }

        Vec<N,T> operator/(T a) const
        {
            Vec<N,T> w(*this);
            w/=a;
            return w;
        }
    };

    typedef Vec<2,double>         Vec2d;
    typedef Vec<2,float>          Vec2f;
    typedef Vec<2,int>            Vec2i;
    typedef Vec<2,unsigned int>   Vec2ui;
    typedef Vec<2,short>          Vec2s;
    typedef Vec<2,unsigned short> Vec2us;
    typedef Vec<2,char>           Vec2c;
    typedef Vec<2,unsigned char>  Vec2uc;

    typedef Vec<3,double>         Vec3d;
    typedef Vec<3,float>          Vec3f;
    typedef Vec<3,int>            Vec3i;
    typedef Vec<3,unsigned int>   Vec3ui;
    typedef Vec<3,short>          Vec3s;
    typedef Vec<3,unsigned short> Vec3us;
    typedef Vec<3,char>           Vec3c;
    typedef Vec<3,unsigned char>  Vec3uc;

    typedef Vec<4,double>         Vec4d;
    typedef Vec<4,float>          Vec4f;
    typedef Vec<4,int>            Vec4i;
    typedef Vec<4,unsigned int>   Vec4ui;
    typedef Vec<4,short>          Vec4s;
    typedef Vec<4,unsigned short> Vec4us;
    typedef Vec<4,char>           Vec4c;
    typedef Vec<4,unsigned char>  Vec4uc;

    typedef Vec<6,double>         Vec6d;
    typedef Vec<6,float>          Vec6f;
    typedef Vec<6,unsigned int>   Vec6ui;
    typedef Vec<6,int>            Vec6i;
    typedef Vec<6,short>          Vec6s;
    typedef Vec<6,unsigned short> Vec6us;
    typedef Vec<6,char>           Vec6c;
    typedef Vec<6,unsigned char>  Vec6uc;


    template<unsigned int N, class T>
    T mag2(const Vec<N,T> &a)
    {
        T l=sqr(a.v[0]);
        for(unsigned int i=1; i<N; ++i) l+=sqr(a.v[i]);
        return l;
    }

    template<unsigned int N, class T>
    T mag(const Vec<N,T> &a)
    { return sqrt(mag2(a))+1e-6; }

    template<unsigned int N, class T>
    inline T dist2(const Vec<N,T> &a, const Vec<N,T> &b)
    {
        T d=sqr(a.v[0]-b.v[0]);
        for(unsigned int i=1; i<N; ++i) d+=sqr(a.v[i]-b.v[i]);
        return d;
    }

    template<unsigned int N, class T>
    inline T dist(const Vec<N,T> &a, const Vec<N,T> &b)
    { return std::sqrt(dist2(a,b)); }

    template<unsigned int N, class T>
    inline void normalize(Vec<N,T> &a)
    { a/=mag(a); }

    template<unsigned int N, class T>
    inline Vec<N,T> normalized(const Vec<N,T> &a)
    { return a/mag(a); }

    template<unsigned int N, class T>
    inline T infnorm(const Vec<N,T> &a)
    {
        T d=std::fabs(a.v[0]);
        for(unsigned int i=1; i<N; ++i) d=max(std::fabs(a.v[i]),d);
        return d;
    }

    template<unsigned int N, class T>
    void zero(Vec<N,T> &a)
    {
        for(unsigned int i=0; i<N; ++i)
            a.v[i] = 0;
    }

    template<unsigned int N, class T>
    std::ostream &operator<<(std::ostream &out, const Vec<N,T> &v)
    {
        out<<v.v[0];
        for(unsigned int i=1; i<N; ++i)
            out<<' '<<v.v[i];
        return out;
    }

    template<unsigned int N, class T>
    std::istream &operator>>(std::istream &in, Vec<N,T> &v)
    {
        in>>v.v[0];
        for(unsigned int i=1; i<N; ++i)
            in>>v.v[i];
        return in;
    }

    template<unsigned int N, class T>
    inline bool operator==(const Vec<N,T> &a, const Vec<N,T> &b)
    {
        bool t = (a.v[0] == b.v[0]);
        unsigned int i=1;
        while(i<N && t) {
            t = t && (a.v[i]==b.v[i]);
            ++i;
        }
        return t;
    }

    template<unsigned int N, class T>
    inline bool operator!=(const Vec<N,T> &a, const Vec<N,T> &b)
    {
        bool t = (a.v[0] != b.v[0]);
        unsigned int i=1;
        while(i<N && !t) {
            t = t || (a.v[i]!=b.v[i]);
            ++i;
        }
        return t;
    }

    template<unsigned int N, class T>
    inline Vec<N,T> operator*(T a, const Vec<N,T> &v)
    {
        Vec<N,T> w(v);
        w*=a;
        return w;
    }

    template<unsigned int N, class T>
    inline T min(const Vec<N,T> &a)
    {
        T m=a.v[0];
        for(unsigned int i=1; i<N; ++i) if(a.v[i]<m) m=a.v[i];
        return m;
    }

    template<unsigned int N, class T>
    inline Vec<N,T> min_union(const Vec<N,T> &a, const Vec<N,T> &b)
    {
        Vec<N,T> m;
        for(unsigned int i=0; i<N; ++i) (a.v[i] < b.v[i]) ? m.v[i]=a.v[i] : m.v[i]=b.v[i];
        return m;
    }

    template<unsigned int N, class T>
    inline Vec<N,T> max_union(const Vec<N,T> &a, const Vec<N,T> &b)
    {
        Vec<N,T> m;
        for(unsigned int i=0; i<N; ++i) (a.v[i] > b.v[i]) ? m.v[i]=a.v[i] : m.v[i]=b.v[i];
        return m;
    }

    template<unsigned int N, class T>
    inline T max(const Vec<N,T> &a)
    {
        T m=a.v[0];
        for(unsigned int i=1; i<N; ++i) if(a.v[i]>m) m=a.v[i];
        return m;
    }

    template<unsigned int N, class T>
    inline T dot(const Vec<N,T> &a, const Vec<N,T> &b)
    {
        T d=a.v[0]*b.v[0];
        for(unsigned int i=1; i<N; ++i) d+=a.v[i]*b.v[i];
        return d;
    }

    template<class T>
    inline Vec<2,T> rotate(const Vec<2,T>& a, float angle)
    {
        T c = cos(angle);
        T s = sin(angle);
        return Vec<2,T>(c*a[0] - s*a[1],s*a[0] + c*a[1]); // counter-clockwise rotation
    }

    template<class T>
    inline Vec<2,T> perp(const Vec<2,T> &a)
    { return Vec<2,T>(-a.v[1], a.v[0]); } // counter-clockwise rotation by 90 degrees

    template<class T>
    inline T cross(const Vec<2,T> &a, const Vec<2,T> &b)
    { return a.v[0]*b.v[1]-a.v[1]*b.v[0]; }

    template<class T>
    inline Vec<3,T> cross(const Vec<3,T> &a, const Vec<3,T> &b)
    { return Vec<3,T>(a.v[1]*b.v[2]-a.v[2]*b.v[1], a.v[2]*b.v[0]-a.v[0]*b.v[2], a.v[0]*b.v[1]-a.v[1]*b.v[0]); }

    template<class T>
    inline T triple(const Vec<3,T> &a, const Vec<3,T> &b, const Vec<3,T> &c)
    { return a.v[0]*(b.v[1]*c.v[2]-b.v[2]*c.v[1])
             +a.v[1]*(b.v[2]*c.v[0]-b.v[0]*c.v[2])
             +a.v[2]*(b.v[0]*c.v[1]-b.v[1]*c.v[0]); }

    template<unsigned int N, class T>
    inline unsigned int hash(const Vec<N,T> &a)
    {
        unsigned int h=a.v[0];
        for(unsigned int i=1; i<N; ++i)
            h=hash(h ^ a.v[i]);
        return h;
    }

    template<unsigned int N, class T>
    inline void assign(const Vec<N,T> &a, T &a0, T &a1)
    {
        assert(N==2);
        a0=a.v[0]; a1=a.v[1];
    }

    template<unsigned int N, class T>
    inline void assign(const Vec<N,T> &a, T &a0, T &a1, T &a2)
    {
        assert(N==3);
        a0=a.v[0]; a1=a.v[1]; a2=a.v[2];
    }

    template<unsigned int N, class T>
    inline void assign(const Vec<N,T> &a, T &a0, T &a1, T &a2, T &a3)
    {
        assert(N==4);
        a0=a.v[0]; a1=a.v[1]; a2=a.v[2]; a3=a.v[3];
    }

    template<unsigned int N, class T>
    inline void assign(const Vec<N,T> &a, T &a0, T &a1, T &a2, T &a3, T &a4, T &a5)
    {
        assert(N==6);
        a0=a.v[0]; a1=a.v[1]; a2=a.v[2]; a3=a.v[3]; a4=a.v[4]; a5=a.v[5];
    }

    template<unsigned int N, class T>
    inline Vec<N,int> round(const Vec<N,T> &a)
    {
        Vec<N,int> rounded;
        for(unsigned int i=0; i<N; ++i)
            rounded.v[i]=lround(a.v[i]);
        return rounded;
    }

    template<unsigned int N, class T>
    inline Vec<N,int> floor(const Vec<N,T> &a)
    {
        Vec<N,int> rounded;
        for(unsigned int i=0; i<N; ++i)
            rounded.v[i]=(int)std::floor(a.v[i]);
        return rounded;
    }

    template<unsigned int N, class T>
    inline Vec<N,int> ceil(const Vec<N,T> &a)
    {
        Vec<N,int> rounded;
        for(unsigned int i=0; i<N; ++i)
            rounded.v[i]=(int)ceil(a.v[i]);
        return rounded;
    }

    template<unsigned int N, class T>
    inline Vec<N,T> fabs(const Vec<N,T> &a)
    {
        Vec<N,T> result;
        for(unsigned int i=0; i<N; ++i)
            result.v[i]=std::fabs(a.v[i]);
        return result;
    }

    template<unsigned int N, class T>
    inline void minmax(const Vec<N,T> &x0, const Vec<N,T> &x1, Vec<N,T> &xmin, Vec<N,T> &xmax)
    {
        for(unsigned int i=0; i<N; ++i)
            minmax(x0.v[i], x1.v[i], xmin.v[i], xmax.v[i]);
    }

    template<unsigned int N, class T>
    inline void minmax(const Vec<N,T> &x0, const Vec<N,T> &x1, const Vec<N,T> &x2, Vec<N,T> &xmin, Vec<N,T> &xmax)
    {
        for(unsigned int i=0; i<N; ++i)
            minmax(x0.v[i], x1.v[i], x2.v[i], xmin.v[i], xmax.v[i]);
    }

    template<unsigned int N, class T>
    inline void minmax(const Vec<N,T> &x0, const Vec<N,T> &x1, const Vec<N,T> &x2, const Vec<N,T> &x3,
                       Vec<N,T> &xmin, Vec<N,T> &xmax)
    {
        for(unsigned int i=0; i<N; ++i)
            minmax(x0.v[i], x1.v[i], x2.v[i], x3.v[i], xmin.v[i], xmax.v[i]);
    }

    template<unsigned int N, class T>
    inline void minmax(const Vec<N,T> &x0, const Vec<N,T> &x1, const Vec<N,T> &x2, const Vec<N,T> &x3, const Vec<N,T> &x4,
                       Vec<N,T> &xmin, Vec<N,T> &xmax)
    {
        for(unsigned int i=0; i<N; ++i)
            minmax(x0.v[i], x1.v[i], x2.v[i], x3.v[i], x4.v[i], xmin.v[i], xmax.v[i]);
    }

    template<unsigned int N, class T>
    inline void minmax(const Vec<N,T> &x0, const Vec<N,T> &x1, const Vec<N,T> &x2, const Vec<N,T> &x3, const Vec<N,T> &x4,
                       const Vec<N,T> &x5, Vec<N,T> &xmin, Vec<N,T> &xmax)
    {
        for(unsigned int i=0; i<N; ++i)
            minmax(x0.v[i], x1.v[i], x2.v[i], x3.v[i], x4.v[i], x5.v[i], xmin.v[i], xmax.v[i]);
    }

    template<unsigned int N, class T>
    inline void update_minmax(const Vec<N,T> &x, Vec<N,T> &xmin, Vec<N,T> &xmax)
    {
        for(unsigned int i=0; i<N; ++i) update_minmax(x[i], xmin[i], xmax[i]);
    }
}

#endif
