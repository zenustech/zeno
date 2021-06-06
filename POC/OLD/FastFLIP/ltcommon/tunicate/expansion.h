#ifndef EXPANSION_H
#define EXPANSION_H

// Released into the public-domain by Robert Bridson, 2009.
// Modified by Tyson Brochu, 2011.
// Simple functions for manipulating multiprecision floating-point
// expansions, with simplicity favoured over speed.

#include <commonoptions.h>
#include <vector>

// The basic type is essentially a vector of *increasing* and 
// *nonoverlapping* doubles, apart from allowed zeroes anywhere.

class expansion;

void negative(const expansion& input, expansion& output);

int sign( const expansion& a );

bool
is_zero( const expansion& a );

void
add(double a, double b, expansion& sum);

// a and sum may be aliased to the same expansion for in-place addition
void
add(const expansion& a, double b, expansion& sum);

inline void
add(double a, const expansion& b, expansion& sum)
{ add(b, a, sum); }

// aliasing a, b and sum is safe
void
add(const expansion& a, const expansion& b, expansion& sum);

void
subtract( const double& a, const double& b, expansion& difference);

// aliasing a, b and difference is safe
void
subtract(const expansion& a, const expansion& b, expansion& difference);

// aliasing input and output is safe
void
negative(const expansion& input, expansion& output);

void
multiply(double a, double b, expansion& product);

void
multiply(double a, double b, double c, expansion& product);

void
multiply(double a, double b, double c, double d, expansion& product);

void
multiply(const expansion& a, double b, expansion& product);

inline void
multiply(double a, const expansion& b, expansion& product)
{ multiply(b, a, product); }

// Aliasing NOT safe
void
multiply(const expansion& a, const expansion& b, expansion& product);

void compress( const expansion& e, expansion& h );

// Aliasing NOT safe
bool divide( const expansion& x, const expansion& y, expansion& q );

void
remove_zeros(expansion& a);

double
estimate(const expansion& a);

bool equals( const expansion& a, const expansion& b );

void
print_full( const expansion& e );


// ----------------------------------------------------

class expansion
{
    
public:
    
    std::vector<double> v;
    
    expansion()
    : v(0)
    {}
    
    explicit expansion( double val )
    : v(1, val)
    {}
    
    expansion( std::size_t n, double val )
    : v(n,val)
    {}
    
    virtual ~expansion() {}
    
    expansion& operator+=(const expansion &rhs)
    {
        add( *this, rhs, *this );
        return *this;
    }
    
    expansion& operator-=(const expansion &rhs)
    {
        subtract( *this, rhs, *this );
        return *this;
    }
    
    expansion& operator*=(const expansion &rhs)
    {
        expansion p;
        multiply( *this, rhs, p );
        *this = p;
        return *this;
    }
    
    inline expansion operator+(const expansion &other) const 
    {
        expansion result = *this;     
        result += other;  
        return result;              
    }
    
    inline expansion operator-(const expansion &other) const 
    {
        expansion result = *this;    
        result -= other;  
        return result;              
    }
    
    
    inline expansion operator*(const expansion &other) const
    {
        expansion result = *this;    
        result *= other;  
        return result;              
    }
    
    inline expansion operator-( ) const
    {
        expansion result;
        negative( *this, result );
        return result;
    }
    
    inline double estimate() const
    {
        return ::estimate( *this );
    }
    
    inline bool indefinite_sign() const
    {
        return false;
    }
    
    static void begin_special_arithmetic()
    {}
    
    static void end_special_arithmetic()
    {}
    
    inline void clear()
    {
        v.clear();
    }
    
    inline void resize( size_t new_size )
    {
        v.resize(new_size);
    }
    
};


inline void make_expansion( double a, expansion& e )
{ 
    if(a) 
    {
        e = expansion(1, a); 
    }
    else
    {
        e.clear();
    }
}

inline void
make_zero(expansion& e)
{ e.resize(0); }

inline void create_from_double( double a, expansion& out )
{
    make_expansion( a, out );
}

inline bool certainly_opposite_sign( const expansion& a, const expansion& b )
{
    return ( sign(a) > 0 && sign(b) < 0 ) || ( sign(a) < 0 && sign(b) > 0 );
}


#endif
