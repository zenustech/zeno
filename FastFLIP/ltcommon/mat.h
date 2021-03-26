#ifndef MAT_H
#define MAT_H

#include <cstring>
#include <vec.h>

namespace LosTopos {

  template<unsigned int M, unsigned int N, class T>
struct Mat
{
    T a[M*N]; // entries stored column by column (FORTRAN layout)
    
    Mat<M,N,T>(void)
    {}
    
    template<class S>
    Mat<M,N,T>(const S *source)
    {
        for(unsigned int i=0; i<M*N; ++i) a[i]=source[i];
    }
    
    Mat<M,N,T>(T a0, T a1, T a2, T a3)
    {
        assert(M*N==4);
        a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3;
    }
    
    Mat<M,N,T>(T a0, T a1, T a2, T a3, T a4, T a5)
    {
        assert(M*N==6);
        a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3; a[4]=a4; a[5]=a5;
    }
    
    Mat<M,N,T>(T a0, T a1, T a2, T a3, T a4, T a5, T a6, T a7, T a8)
    {
        assert(M*N==9);
        a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3; a[4]=a4; a[5]=a5; a[6]=a6; a[7]=a7; a[8]=a8;
    }
    
    Mat<M,N,T>(const Vec<M,T> &col0, const Vec<M,T> &col1)
    {
        assert(N==2);
        setcol(0,col0); setcol(1,col1);
    }
    
    Mat<M,N,T>(const Vec<M,T> &col0, const Vec<M,T> &col1, const Vec<M,T> &col2)
    {
        assert(N==3);
        setcol(0,col0); setcol(1,col1); setcol(2,col2);
    }
    
    Mat<M,N,T>(const Vec<M,T> &col0, const Vec<M,T> &col1, const Vec<M,T> &col2, const Vec<M,T> &col3)
    {
        assert(N==4);
        setcol(0,col0); setcol(1,col1); setcol(2,col2); setcol(3,col3);
    }
    
    T &operator()(int i, int j)
    {
        assert(0<=i && (unsigned int)i<M && 0<=j && (unsigned int)j<N);
        return a[i+M*j];
    }
    
    const T &operator()(int i, int j) const
    {
        assert(0<=i && (unsigned int)i<M && 0<=j && (unsigned int)j<N);
        return a[i+M*j];
    }
    
    Vec<M,T> col(int j) const
    {
        assert(0<=j && (unsigned int)j<N);
        return Vec<M,T>(a+j*M);
    }
    
    Vec<N,T> row(int i) const
    {
        assert(0<=i && i<M);
        Vec<N,T> v;
        for(unsigned int j=0; j<N; ++j) v[j]=a(i,j);
        return v;
    }
    
    Mat<M,N,T> operator+=(const Mat<M,N,T> &b)
    {
        for(unsigned int i=0; i<M*N; ++i) a[i]+=b.a[i];
        return *this;
    }
    
    Mat<M,N,T> operator+(const Mat<M,N,T> &b) const
    {
        Mat<M,N,T> sum(*this);
        sum+=b;
        return sum;
    }
    
    Mat<M,N,T> operator-=(const Mat<M,N,T> &b)
    {
        for(unsigned int i=0; i<M*N; ++i) a[i]-=b.a[i];
        return *this;
    }
    
    Mat<M,N,T> operator-(const Mat<M,N,T> &b) const
    {
        Mat<M,N,T> diff(*this);
        diff-=b;
        return diff;
    }
    
    Mat<M,N,T> operator*=(T scalar)
    {
        for(unsigned int i=0; i<M*N; ++i) a[i]*=scalar;
        return *this;
    }
    
    Mat<M,N,T> operator*(T scalar) const
    {
        Mat<M,N,T> b(*this);
        b*=scalar;
        return b;
    }
    
    Vec<M,T> operator*(const Vec<N,T> v) const
    {
        Vec<M,T> r;
        unsigned int i, j;
        const T *pa, *pv;
        T s, *pr=r.v;
        for(i=0; i<M; ++i, ++pr){
            pa=a+i;
            pv=v.v;
            s=0;
            for(j=0; j<N; ++j, pa+=M, ++pv)
                s+=*pa*(*pv);
            *pr=s;
        }
        return r;
    }
    
    template<unsigned int P>
    Mat<M,P,T> operator*(const Mat<N,P,T> b) const
    {
        Mat<M,P,T> c;
        unsigned int i, j, k;
        const T *pa, *pb;
        T s, *pc=c.a;
        for(k=0; k<P; ++k){
            for(i=0; i<M; ++i, ++pc){
                pa=a+i;
                pb=b.a+N*k;
                s=0;
                for(j=0; j<N; ++j, pa+=M, ++pb)
                    s+=*pa*(*pb);
                *pc=s;
            }
        }
        return c;
    }
    
    Mat<M,N,T> operator/=(T scalar)
    {
        for(unsigned int i=0; i<M*N; ++i) a[i]/=scalar;
        return *this;
    }
    
    Mat<M,N,T> operator/(T scalar) const
    {
        Mat<M,N,T> b(*this);
        b/=scalar;
        return b;
    }
    
    Mat<N,M,T> transpose() const {
        Mat<N,M,T> result;
        
        for(unsigned int i = 0; i < M; ++i) {
            for(unsigned int j = 0; j < N; ++j) {
                result(j,i) = (*this)(i,j);
            }
        }
        return result;
    }
};

typedef Mat<2,2,double> Mat22d;
typedef Mat<2,2,float>  Mat22f;
typedef Mat<2,2,int>    Mat22i;
typedef Mat<3,2,double> Mat32d;
typedef Mat<3,2,float>  Mat32f;
typedef Mat<3,2,int>    Mat32i;
typedef Mat<2,3,double> Mat23d;
typedef Mat<2,3,float>  Mat23f;
typedef Mat<2,3,int>    Mat23i;
typedef Mat<3,3,double> Mat33d;
typedef Mat<3,3,float>  Mat33f;
typedef Mat<3,3,int>    Mat33i;
typedef Mat<4,4,double> Mat44d;
typedef Mat<4,4,float>  Mat44f;
typedef Mat<4,4,int>    Mat44i;

// more for human eyes than a good machine-readable format
template<unsigned int M, unsigned int N, class T>
std::ostream &operator<<(std::ostream &out, const Mat<M,N,T> &a)
{
    for(unsigned int i=0; i<M; ++i){
        out<<(i==0 ? '[' : ' ');
        for(unsigned int j=0; j<N-1; ++j)
            out<<a(i,j)<<',';
        out<<a(i,N-1);
        if(i<M-1) out<<';'<<std::endl;
        else out<<']';
    }
    return out;
}

template<unsigned int M, unsigned int N, class T>
inline Mat<M,N,T> operator*(T scalar, const Mat<M,N,T> &a)
{
    Mat<M,N,T> b(a);
    b*=scalar;
    return b;
}

template<unsigned int M, unsigned int N, class T>
inline Mat<M,N,T> outer(const Vec<M,T> &x, const Vec<N,T> &y)
{
    Mat<M,N,T> r;
    T *pr=r.a;
    for(unsigned int j=0; j<N; ++j)
        for(unsigned int i=0; i<M; ++i, ++pr)
            *pr=x[i]*y[j];
    return r;
}

template<unsigned int M, unsigned int N, class T>
inline void zero(Mat<M,N,T>& mat) {
    std::memset(mat.a, 0, N*M*sizeof(T));
}

template<unsigned int N, class T>
inline void make_identity(Mat<N,N,T>& mat)
{
    std::memset(mat.a, 0, N*N*sizeof(T));
    for(unsigned int i=0; i<N; ++i)
        mat.a[(N+1)*i]=1;
}

template<unsigned int N, class T>
inline T trace(Mat<N,N,T>& mat)
{
    T t=0;
    for(unsigned int i=0; i<N; ++i)
        t+=mat.a[(N+1)*i];
    return t;
}

template<class T>
inline Mat<3,3,T> star_matrix(const Vec<3,T> &w)
{
    return Mat<3,3,T>(0, -w.v[2], w.v[1],
                      w.v[2], 0, -w.v[0],
                      -w.v[1], w.v[0], 0);
}

// determine rotation Q and symmetrix matrix A so that Q*A=F
template<class T>
void signed_polar_decomposition(const Mat<2,2,T>& F, Mat<2,2,T>& Q, Mat<2,2,T>& A)
{
    T s=F(0,1)-F(1,0);
    if(s){
        T c=F(0,0)+F(1,1), hyp=std::sqrt(c*c+s*s);
        if(c>0){ c/=hyp; s/=hyp; }
        else{ c/=-hyp; s/=-hyp; }
        Q(0,0)=c; Q(0,1)=s; Q(1,0)=-s; Q(1,1)=c;
        A(0,0)=c*F(0,0)-s*F(1,0);
        A(1,0)=A(0,1)=s*F(0,0)+c*F(1,0);
        A(1,1)=s*F(0,1)+c*F(1,1);
    }else{ // F is already symmetric: no rotation needed
        Q(0,0)=1; Q(0,1)=0; Q(1,0)=0; Q(1,1)=1;
        A=F;
    }
}

// for symmetric A, determine c and s for rotation Q and diagonal matrix D so that Q'*A*Q=D
// (D sorted, with |D[0]|>=|D[1]|; the upper triangular entry of A is ignored)
template<class T>
void symmetric_eigenproblem(const Mat<2,2,T>& A, T& c, T& s, Vec<2,T>& D)
{
    s=2*A(1,0);
    if(s==0){ // already diagonal
        c=1;
        D[0]=A(0,0); D[1]=A(1,1);
    }else{
        T d=A(0,0)-A(1,1);
        T disc=std::sqrt(d*d+s*s);
        c=d+(d>0 ? disc : -disc);
        T hyp=std::sqrt(c*c+s*s);
        c/=hyp; s/=hyp;
        // there is probably a better way of computing these (more stable etc.)
        D[0]=c*c*A(0,0)+2*c*s*A(1,0)+s*s*A(1,1);
        D[1]=s*s*A(0,0)-2*c*s*A(1,0)+c*c*A(1,1);
    }
    if(std::fabs(D[0])<std::fabs(D[1])){ // if D is in the wrong order, fix it
        std::swap(D[0], D[1]);
        c=-c;
        std::swap(c, s);
    }
    if(c<0){ c=-c; s=-s; }
}

template<class T>
inline T determinant(const Mat<3,3,T> &mat)
{
    return mat(0,0)*(mat(2,2)*mat(1,1)-mat(2,1)*mat(1,2))-
    mat(1,0)*(mat(2,2)*mat(0,1)-mat(2,1)*mat(0,2))+
    mat(2,0)*(mat(1,2)*mat(0,1)-mat(1,1)*mat(0,2));
}

// for symmetric A, determine rotation Q and diagonal matrix D so that Q'*A*Q=D
// (D not sorted! and the upper triangular entry is ignored)
template<class T>
void symmetric_eigenproblem(const Mat<2,2,T>& A, Mat<2,2,T>& Q, Vec<2,T>& D)
{
    T c, s;
    symmetric_eigenproblem(A, c, s, D);
    Q(0,0)=c; Q(1,0)=s; Q(0,1)=-s; Q(1,1)=c;
}

// figures out A=U*S*V' with U and V rotations, S diagonal
// (S is sorted by magnitude, and may contain both positive and negative entries)
template<class T>
void signed_svd(const Mat<2,2,T>& A, Mat<2,2,T>& U, Mat<2,2,T>& V, Vec<2,T>& S)
{
    Vec<2,T> D;
    symmetric_eigenproblem(Mat<2,2,T>(sqr(A(0,0))+sqr(A(0,1)),
                                      A(0,0)*A(1,0)+A(0,1)*A(1,1),
                                      0, // ignored by routine, so we don't need to fill it in
                                      sqr(A(1,0))+sqr(A(1,1))),
                           U, D);
    // form F=A'*U 
    T f00=A(0,0)*U(0,0)+A(1,0)*U(1,0),
    f10=A(0,1)*U(0,0)+A(1,1)*U(1,0),
    f01=A(0,0)*U(0,1)+A(1,0)*U(1,1),
    f11=A(0,1)*U(0,1)+A(1,1)*U(1,1);
    // do signed polar decomposition of F to get V
    T s=f01-f10, c;
    if(s){
        c=f00+f11;
        T hyp=std::sqrt(c*c+s*s);
        if(c>0){ c/=hyp; s/=hyp; }
        else{ c/=-hyp; s/=-hyp; }
    }else
        c=1;
    V(0,0)=c; V(0,1)=s; V(1,0)=-s; V(1,1)=c;
    // and finally grab the singular values from direct computation (maybe there's a better way?)
    S[0]=(U(0,0)*A(0,0)+U(1,0)*A(1,0))*V(0,0) + (U(0,0)*A(0,1)+U(1,0)*A(1,1))*V(1,0);
    S[1]=(U(0,1)*A(0,0)+U(1,1)*A(1,0))*V(0,1) + (U(0,1)*A(0,1)+U(1,1)*A(1,1))*V(1,1);
}

// 3x3 version: Get the A=QR decomposition using Householder reflections
template<class T>
void find_QR(const Mat<3,3,T>& A, Mat<3,3,T>& Q, Mat<3,3,T>& R)
{
    Q(0,0)=1; Q(0,1)=0; Q(0,2)=0;
    Q(1,0)=0; Q(1,1)=1; Q(1,2)=0;
    Q(2,0)=0; Q(2,1)=0; Q(2,2)=1;
    R=A;
    Vec<3,T> u;
    T n2, u2, n, c, d;
    
    // first column
    n2=sqr(R(0,0))+sqr(R(1,0))+sqr(R(2,0));
    u[0]=R(0,0); u[1]=R(1,0); u[2]=R(2,0); // will change u[0] in a moment
    u2=sqr(u[1])+sqr(u[2]); // will add in sqr(u[0]) when we know it
    R(1,0)=0; R(2,0)=0;
    if(u2){ // if there are entries to annihilate below the diagonal in this column
        n=std::sqrt(n2);
        u[0]+=(u[0]>0 ? n : -n);
        u2+=sqr(u[0]);
        c=2/u2;
        // update diagonal entrt of R
        R(0,0)=(u[0]>0 ? -n : n);
        // update rest of R with reflection
        // second column
        d=c*(u[0]*R(0,1)+u[1]*R(1,1)+u[2]*R(2,1));
        R(0,1)-=d*u[0]; R(1,1)-=d*u[1]; R(2,1)-=d*u[2];
        // third column
        d=c*(u[0]*R(0,2)+u[1]*R(1,2)+u[2]*R(2,2));
        R(0,2)-=d*u[0]; R(1,2)-=d*u[1]; R(2,2)-=d*u[2];
        // set Q to this symmetric reflection
        Q(0,0)-=c*u[0]*u[0]; Q(0,1)-=c*u[0]*u[1]; Q(0,2)-=c*u[0]*u[2];
        Q(1,0)=Q(0,1);       Q(1,1)-=c*u[1]*u[1]; Q(1,2)-=c*u[1]*u[2];
        Q(2,0)=Q(0,2);       Q(2,1)=Q(1,2);       Q(2,2)-=c*u[2]*u[2];
    }else{
        // still do a reflection around (1,0,0), since we want there to be exactly two reflections (to get back to rotation)
        R(0,0)=-R(0,0); R(0,1)=-R(0,1); R(0,2)=-R(0,2);
        Q(0,0)=-1;
    }
    
    // second column
    n2=sqr(R(1,1))+sqr(R(2,1));
    u[1]=R(1,1); u[2]=R(2,1); // will change u[1] in a moment
    u2=sqr(u[2]); // will add in sqr(u[1]) when we know it
    R(2,1)=0;
    if(u2){ // if there are entries to annihilate below the diagonal in this column
        n=std::sqrt(n2);
        u[1]+=(u[1]>0 ? n : -n);
        u2+=sqr(u[1]);
        c=2/u2;
        // update diagonal entrt of R
        R(1,1)=(u[1]>0 ? -n : n);
        // update rest of R with reflection
        // third column
        d=c*(u[1]*R(1,2)+u[2]*R(2,2));
        R(1,2)-=d*u[1]; R(2,2)-=d*u[2];
        // update Q by right multiplication with the reflection
        d=c*(Q(0,1)*u[1]+Q(0,2)*u[2]); Q(0,1)-=d*u[1]; Q(0,2)-=d*u[2]; // row 0
        d=c*(Q(1,1)*u[1]+Q(1,2)*u[2]); Q(1,1)-=d*u[1]; Q(1,2)-=d*u[2]; // row 1
        d=c*(Q(2,1)*u[1]+Q(2,2)*u[2]); Q(2,1)-=d*u[1]; Q(2,2)-=d*u[2]; // row 2
    }else{
        // still need to multiply in a reflection around (0,1,0), to get Q back to being a rotation
        R(1,1)=-R(1,1); R(1,2)=-R(1,2);
        Q(0,1)=-Q(0,1); Q(1,1)=-Q(1,1); Q(2,1)=-Q(2,1);
    }
    
    // flip back some signs to be closer to A
    R(0,0)=-R(0,0); R(0,1)=-R(0,1); R(0,2)=-R(0,2);
    R(1,1)=-R(1,1); R(1,2)=-R(1,2);
    Q(0,0)=-Q(0,0); Q(0,1)=-Q(0,1);
    Q(1,0)=-Q(1,0); Q(1,1)=-Q(1,1);
    Q(2,0)=-Q(2,0); Q(2,1)=-Q(2,1);
}

// Specialization for 2xN case: use a Givens rotation
template<unsigned int N, class T>
void find_QR(const Mat<2,N,T>& A, Mat<2,N,T>& Q, Mat<2,N,T>& R)
{
    T c=A(0,0), s=A(1,0), hyp=std::sqrt(c*c+s*s);
    if(hyp){
        c/=hyp;
        s/=hyp;
        Q(0,0)=c; Q(1,0)=s; Q(0,1)=-s; Q(1,1)=c;
        R(0,0)=hyp; R(1,0)=0;
        for(unsigned int j=1; j<N; ++j){
            R(0,j)=c*A(0,j)+s*A(1,j);
            R(1,j)=-s*A(0,j)+c*A(1,j);
        }
    }else{
        Q(0,0)=1; Q(1,0)=0; Q(0,1)=0; Q(1,1)=1;
        R=A;
    }
}


template<class T>
inline Mat<2,2,T> inverse(const Mat<2,2,T> &mat)
{
    const T& a = mat(0,0);
    const T& b = mat(0,1);
    const T& c = mat(1,0);
    const T& d = mat(1,1);
    
    T invdet = 1.0 / ( a*d - b*c );
    return Mat<2,2,T>( invdet*d, -invdet*c, -invdet*b, invdet*a );
}


template<class T>
inline Mat<3,3,T> inverse(const Mat<3,3,T> &mat)
{
    T invdet = 1.0 / determinant(mat);
    return Mat<3,3,T>(
                      invdet*(mat(2,2)*mat(1,1)-mat(2,1)*mat(1,2)), -invdet*(mat(2,2)*mat(0,1)-mat(2,1)*mat(0,2)),  invdet*(mat(1,2)*mat(0,1)-mat(1,1)*mat(0,2)),
                      -invdet*(mat(2,2)*mat(1,0)-mat(2,0)*mat(1,2)),  invdet*(mat(2,2)*mat(0,0)-mat(2,0)*mat(0,2)), -invdet*(mat(1,2)*mat(0,0)-mat(1,0)*mat(0,2)),
                      invdet*(mat(2,1)*mat(1,0)-mat(2,0)*mat(1,1)), -invdet*(mat(2,1)*mat(0,0)-mat(2,0)*mat(0,1)),  invdet*(mat(1,1)*mat(0,0)-mat(1,0)*mat(0,1))
                      );
}

}

#endif
