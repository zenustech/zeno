#include <sparse_matrix.h>
#include <util.h>

namespace LosTopos {
//============================================================================
void SparseMatrixDynamicCSR::
clear(void)
{
    m=n=0;
    row.clear();
}

void SparseMatrixDynamicCSR::
set_zero(void)
{
    for(int i=0; i<m; ++i) row[i].clear();
}

void SparseMatrixDynamicCSR::
resize(int m_, int n_)
{
    m=m_;
    n=n_;
    row.resize(m);
}

const double &SparseMatrixDynamicCSR::
operator()(int i, int j) const
{
    assert(i>=0 && i<m && j>=0 && j<n);
    const DynamicSparseVector &r=row[i];
    for(DynamicSparseVector::const_iterator p=r.begin(); p!=r.end(); ++p){
        if(p->index==j) return p->value;
        else if(p->index>j) break;
    }
    return zero;
}

double &SparseMatrixDynamicCSR::
operator()(int i, int j)
{
    assert(i>=0 && i<m && j>=0 && j<n);
    DynamicSparseVector &r=row[i];
    DynamicSparseVector::iterator p;
    for(p=r.begin(); p!=r.end(); ++p){
        if(p->index==j) return p->value;
        else if(p->index>j) break;
    }
    return (r.insert(p, SparseEntry(j, 0)))->value;
}

void SparseMatrixDynamicCSR::
add_sparse_row(int i, const DynamicSparseVector &x, double multiplier)
{
    assert(i>=0 && i<m);
    DynamicSparseVector &r=row[i];
    DynamicSparseVector::iterator p=r.begin();
    DynamicSparseVector::const_iterator q=x.begin();
    while(p!=r.end() && q!=x.end()){
        if(p->index<q->index) ++p;
        else if(p->index>q->index){
            r.insert(p, SparseEntry(q->index, multiplier*q->value));
            ++q;
        }else{
            p->value+=multiplier*q->value;
            ++p;
            ++q;
        }
    }
    for(; q!=x.end(); ++q) r.push_back(SparseEntry(q->index, multiplier*q->value));
}

void SparseMatrixDynamicCSR::
apply(const double *x, double *y) const
{
    assert(x && y);
    if(x!= NULL && y != NULL) {
       for(int i=0; i<m; ++i){
           double d=0;
           const DynamicSparseVector &r=row[i];
           for(DynamicSparseVector::const_iterator p=r.begin(); p!=r.end(); ++p)
               d+=p->value*x[p->index];
           y[i]=d;
       }
    }
}

void SparseMatrixDynamicCSR::
apply_and_subtract(const double *x, const double *y, double *z) const
{
    assert(x && y && z);
    if(x!= NULL && y != NULL && z != NULL) {
       for(int i=0; i<m; ++i){
          double d=0;
          const DynamicSparseVector &r=row[i];
          for(DynamicSparseVector::const_iterator p=r.begin(); p!=r.end(); ++p)
             d+=p->value*x[p->index];
          z[i]=y[i]-d;
       }
    }
}

void SparseMatrixDynamicCSR::
apply_transpose(const double *x, double *y) const
{
    assert(x && y);
    if(x!= NULL && y != NULL) {
       BLAS::set_zero(n, y);
       for(int i=0; i<m; ++i){
          const DynamicSparseVector &r=row[i];
          double xi=x[i];
          for(DynamicSparseVector::const_iterator p=r.begin(); p!=r.end(); ++p)
             y[p->index]+=p->value*xi;
       }
    }
}

void SparseMatrixDynamicCSR::
apply_transpose_and_subtract(const double *x, const double *y, double *z) const
{
    assert(x && y && z);
    if(x!= NULL && y != NULL && z != NULL) {
       if(y!=z) BLAS::copy(n, y, z);
       for(int i=0; i<m; ++i){
          const DynamicSparseVector &r=row[i];
          double xi=x[i];
          for(DynamicSparseVector::const_iterator p=r.begin(); p!=r.end(); ++p)
             z[p->index]-=p->value*xi;
       }
    }
}

void SparseMatrixDynamicCSR::
write_matlab(std::ostream &output, const char *variable_name) const
{
    output<<variable_name<<"=sparse([";
    for(int i=0; i<m; ++i) for(DynamicSparseVector::const_iterator p=row[i].begin(); p!=row[i].end(); ++p)
        output<<i+1<<" ";
    output<<"],...\n  [";
    for(int i=0; i<m; ++i) for(DynamicSparseVector::const_iterator p=row[i].begin(); p!=row[i].end(); ++p)
        output<<p->index+1<<" ";
    output<<"],...\n  [";
    for(int i=0; i<m; ++i) for(DynamicSparseVector::const_iterator p=row[i].begin(); p!=row[i].end(); ++p)
        output<<p->value<<" ";
    output<<"], "<<m<<", "<<n<<");"<<std::endl;
}

//============================================================================

SparseMatrixStaticCSR::
SparseMatrixStaticCSR(const SparseMatrixDynamicCSR &matrix)
: LinearOperator(matrix.m, matrix.n), rowstart(matrix.m+1, 0),
colindex(0), value(0)
{
    int nnz=0;
    for(int i=0; i<m; ++i){
        nnz+=(int)matrix.row[i].size();
        rowstart[i+1]=static_cast<int>(nnz);
    }
    colindex.resize(nnz);
    value.resize(nnz);
    for(int i=0, k=0; i<m; ++i){
        const DynamicSparseVector &r=matrix.row[i];
        for(DynamicSparseVector::const_iterator p=r.begin(); p!=r.end(); ++p){
            colindex[k]=p->index;
            value[k]=p->value;
            ++k;
        }
    }
}

void SparseMatrixStaticCSR::
clear(void)
{
    m=n=0;
    rowstart.clear();
    rowstart.push_back(0);
    colindex.clear();
    value.clear();
}

void SparseMatrixStaticCSR::
set_zero(void)
{
    std::memset(&rowstart[0], 0, (m+1)*sizeof(int));
    colindex.resize(0);
    value.resize(0);
}

void SparseMatrixStaticCSR::
resize(int m_, int n_)
{
    assert(m_>=0 && n_>=0);
    n=n_;
    if(m_>m){ // extra rows?
        m=m_;
        rowstart.resize(m+1,rowstart.back());
    }else if(m_<m){ // deleting rows?
        m=m_;
        rowstart.resize(m+1);
        colindex.resize(rowstart.back());
        value.resize(rowstart.back());
    }
}

double SparseMatrixStaticCSR::
operator()(int i, int j) const
{
    assert(i>=0 && i<m && j>=0 && j<n);
    // linear search for now - could be accelerated if needed!
    for(int k=rowstart[i]; k<rowstart[k+1]; ++k){
        if(colindex[k]==j) return value[k];
        else if(colindex[k]>j) break;
    }
    return 0;
}

void SparseMatrixStaticCSR::
apply_and_subtract(const double *x, const double *y, double *z) const
{
    assert(x && y && z);
    if(x!= NULL && y != NULL && z != NULL) {
       for(int i=0, k=rowstart[0]; i<m; ++i){
          double d=0;
          for(; k<rowstart[i+1]; ++k) d+=value[k]*x[colindex[k]];
          z[i]=y[i]-d;
       }
    }
}


void SparseMatrixStaticCSR::
apply_transpose_and_subtract(const double *x, const double *y, double *z) const
{
    assert(x && y && z);
    if(x != NULL && y != NULL && z != NULL) {
       if(y!=z) BLAS::copy(n, y, z);
       for(int i=0, k=rowstart[0]; i<m; ++i){
           double xi=x[i];
           for(; k<rowstart[i+1]; ++k) z[colindex[k]]-=value[k]*xi;
       }
    }
}

void SparseMatrixStaticCSR::
write_matlab(std::ostream &output, const char *variable_name) const
{
    output<<variable_name<<"=sparse([";
    for(int i=0, k=rowstart[0]; i<m; ++i) for(; k<rowstart[i+1]; ++k)
        output<<i+1<<" ";
    output<<"],...\n  [";
    for(int i=0, k=rowstart[0]; i<m; ++i) for(; k<rowstart[i+1]; ++k)
        output<<colindex[k]+1<<" ";
    output<<"],...\n  [";
    for(int i=0, k=rowstart[0]; i<m; ++i) for(; k<rowstart[i+1]; ++k)
        output<<value[k]<<" ";
    output<<"], "<<m<<", "<<n<<");"<<std::endl;
}

}