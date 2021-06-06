// Released into the public-domain by Robert Bridson, 2009.
// Modified by Tyson Brochu, 2011.

#include <expansion.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include "commonoptions.h"

namespace {
    
    //==============================================================================
    void
    two_sum(double a,
            double b,
            double& x,
            double& y)
    {
        x=a+b;
        double z=x-a;
        y=(a-(x-z))+(b-z);
    }
    
    //==============================================================================
    // requires that |a|>=|b|
    void
    fast_two_sum(double a,
                 double b,
                 double& x,
                 double& y)
    {
        assert( a == a && b == b );
        assert(std::fabs(a)>=std::fabs(b));
        x=a+b;
        y=(a-x)+b;
    }
    
    //==============================================================================
    void
    split(double a,
          double& x,
          double& y)
    {
        double c=134217729*a;
        x=c-(c-a);
        y=a-x;
    }
    
    //==============================================================================
    void
    two_product(double a,
                double b,
                double& x,
                double& y)
    {
        x=a*b;
        double a1, a2, b1, b2;
        split(a, a1, a2);
        split(b, b1, b2);
        y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2);
    }
    
}  // namespace


//==============================================================================
bool
is_zero( const expansion& a )
{
    return ( a.v.size() == 0 );
}


//==============================================================================
int 
sign( const expansion& a )
{
    if ( a.v.size() == 0 ) { return 0; }
    
    // REVIEW: I'm assuming we can get the sign of the expansion by the sign of its leading term (i.e. the sum of all other terms < leading term )
    // This true if the expansion if increasing and nonoverlapping
    if ( a.v.back() > 0 )
    {
        return 1;
    }
    return -1;
}


//==============================================================================
void
add(double a, double b, expansion& sum)
{
    sum.resize(2);
    two_sum(a, b, sum.v[1], sum.v[0]);
}

//==============================================================================
// a and sum may be aliased to the same expansion for in-place addition
void
add(const expansion& a, double b, expansion& sum)
{
    size_t m=a.v.size();
    sum.v.reserve(m+1);
    double s;
    for(size_t i=0; i<m; ++i){
        two_sum(b, a.v[i], b, s);
        if(s) sum.v.push_back(s);
    }
    sum.v.push_back(b);
}

//==============================================================================
// aliasing a, b and sum is safe
void
add(const expansion& a, const expansion& b, expansion& sum)
{
    
    if(a.v.empty())
    {
        sum=b;
        return;
    }else if(b.v.empty())
    {
        sum=a;
        return;
    }
    
    // Shewchuk's fast-expansion-sum
    expansion merge(a.v.size()+b.v.size(), 0);
    unsigned int i=0, j=0, k=0;
    for(;;){
        if(std::fabs(a.v[i])<std::fabs(b.v[j])){
            merge.v[k++]=a.v[i++];
            if(i==a.v.size()){
                while(j<b.v.size()) merge.v[k++]=b.v[j++];
                break;
            }
        }else{
            merge.v[k++]=b.v[j++];
            if(j==b.v.size()){
                while(i<a.v.size()) merge.v[k++]=a.v[i++];
                break;
            }
        }
    }
    sum.v.reserve(merge.v.size());
    sum.v.resize(0);
    double q, r;
    fast_two_sum(merge.v[1], merge.v[0], q, r);
    if(r) sum.v.push_back(r);
    for(i=2; i<merge.v.size(); ++i){
        two_sum(q, merge.v[i], q, r);    
        if(r) sum.v.push_back(r);
    }
    if(q) sum.v.push_back(q);
    
}

//==============================================================================
void
subtract( const double& a, const double& b, expansion& difference)
{
    add( a, -b, difference );
}

//==============================================================================
void
subtract(const expansion& a, const expansion& b, expansion& difference)
{
    // could improve this a bit!
    expansion c;
    negative(b, c);
    add(a, c, difference);
}

//==============================================================================
void
negative(const expansion& input, expansion& output)
{
    output.resize(input.v.size());
    for(unsigned int i=0; i<input.v.size(); ++i)
        output.v[i]=-input.v[i];
}

//==============================================================================
void
multiply(double a, double b, expansion& product)
{
    product.resize(2);
    two_product(a, b, product.v[1], product.v[0]);
}

//==============================================================================
void
multiply(double a, double b, double c, expansion& product)
{
    expansion ab;
    multiply(a, b, ab);
    multiply(ab, c, product);
}

//==============================================================================
void
multiply(double a, double b, double c, double d, expansion& product)
{
    multiply(a, b, product);
    expansion abc;
    multiply(product, c, abc);
    multiply(abc, d, product);
}

//==============================================================================
void
multiply(const expansion& a, double b, expansion& product)
{
    
    // basic idea:
    // multiply each entry in a by b (producing two new entries), then
    // two_sum them in such a way to guarantee increasing/non-overlapping output
    product.resize(2*a.v.size());
    if(a.v.empty()) return;
    two_product(a.v[0], b, product.v[1], product.v[0]); // finalize product[0]
    double x, y, z;
    for(unsigned int i=1; i<a.v.size(); ++i){
        two_product(a.v[i], b, x, y);
        // finalize product[2*i-1]
        two_sum(product.v[2*i-1], y, z, product.v[2*i-1]);
        // finalize product[2*i], could be fast_two_sum instead
        fast_two_sum(x, z, product.v[2*i+1], product.v[2*i]);
    }
    // multiplication is a prime candidate for producing spurious zeros, so
    // remove them by default
    remove_zeros(product);
    
} 

//==============================================================================
void
multiply(const expansion& a, const expansion& b, expansion& product)
{
    // most stupid way of doing it:
    // multiply a by each entry in b, add each to product
    product.resize(0);
    expansion term;
    for(unsigned int i=0; i<b.v.size(); ++i){
        multiply(a, b.v[i], term);
        add(product, term, product);
    }
}


//==============================================================================


void compress( const expansion& e, expansion& h )
{
    if ( is_zero( e ) )
    {
        make_zero( h );
        return;
    }
    
    expansion g( e.v.size(), 0 );
    
    size_t bottom = e.v.size() - 1;
    double q = e.v[bottom];
    
    for ( ssize_t i = e.v.size() - 2; i >= 0; --i )
    {
        double new_q, small_q;
        fast_two_sum( q, e.v[i], new_q, small_q );
        if ( small_q != 0 )
        {
            g.v[bottom--] = new_q;
            q = small_q;
        }
        else
        {
            q = new_q;
        }
    }
    g.v[bottom] = q;
    
    h.v.resize( e.v.size(), 0 );
    
    unsigned int top = 0;
    
    for ( size_t i = bottom+1; i < e.v.size(); ++i )
    {
        double new_q, small_q;
        fast_two_sum( g.v[i], q, new_q, small_q );
        if ( small_q != 0 )
        {
            h.v[top++] = small_q;
        }
        q = new_q;
    }
    h.v[top] = q;
    h.resize( top+1 );
    
}


//==============================================================================

bool divide( const expansion& x, const expansion& y, expansion& q )
{
    
    assert( !is_zero( y ) );
    
    if ( is_zero( x ) ) 
    {
        // 0 / y = 0
        make_expansion( 0, q );
        return true;
    }
    
    const double divisor = estimate(y);
    
    // q is the quotient, built by repeatedly dividing the remainder
    // Initially, q = estimate(x) / estimate(y)
    
    make_expansion( estimate(x) / divisor, q );
    
    expansion qy;
    multiply( q, y, qy );
    expansion r;
    subtract( x, qy, r );  
    
    while ( !is_zero(r) )
    {
        // s is the next term in the quotient q:
        // s = estimate(r) / estimate(y)
        expansion s;
        make_expansion( estimate(r) / divisor, s );
        
        if ( is_zero(s) )
        {
            assert ( !is_zero(y) );
            std::cout << "underflow, s == 0" << std::endl;
            std::cout << "divisor: " << divisor << std::endl;
            return false;         
        }
        
        // q += s
        add( q, s, q );
        
        // r -= s*y
        expansion sy;
        multiply( s, y, sy );
        
        // underflow, quotient not representable by an expansion
        if ( is_zero(sy) )
        {
            assert ( !is_zero(s) && !is_zero(y) );
            std::cout << "underflow, sy == 0" << std::endl;
            return false;
        }
        
        subtract( r, sy, r );     
        
        expansion compressed_r;
        compress( r, compressed_r );
        r = compressed_r;
        
    }
    
    remove_zeros( q );
    return true;
    
}

//==============================================================================
void
remove_zeros(expansion& a)
{
    
    unsigned int i, j;
    
    for ( i = 0, j = 0; i < a.v.size(); ++i )
    {
        if ( a.v[i] )
        {
            a.v[j++] = a.v[i];
        }
    }
    
    a.resize(j);
    
}

//==============================================================================
double
estimate(const expansion& a)
{
    double x=0;
    for(unsigned int i=0; i<a.v.size(); ++i)
        x+=a.v[i];
    return x;
}

//==============================================================================

bool equals( const expansion& a, const expansion& b )
{
    bool same = (a.v.size() == b.v.size());
    
    if (!same) { return false; }
    
    for ( unsigned int i = 0; i < a.v.size(); ++i )
    {
        same &= (a.v[i] == b.v[i]);
    }
    
    return same;
    
}

//==============================================================================

void print_full( const expansion& e )
{
    if ( e.v.size() == 0 ) 
    { 
        std::cout << "0" << std::endl;
        return; 
    }
    
    for ( unsigned int j = 0; j < e.v.size(); ++j )
    {
        std::cout << e.v[j] << " ";
    }
    std::cout << std::endl;   
}




