#include <stdio.h>


int main()
{
    /* Local variables */
    double base, emin, prec, emax, rmin, rmax, t, sfmin;
    extern double dmach(char *);
    double rnd, eps;

    eps = dmach("Epsilon");
    sfmin = dmach("Safe minimum");
    base = dmach("Base");
    prec = dmach("Precision");
    t = dmach("Number of digits in mantissa");
    rnd = dmach("Rounding mode");
    emin = dmach("Minnimum exponent");
    rmin = dmach("Underflow threshold");
    emax = dmach("Largest exponent");
    rmax = dmach("Overflow threshold");

    printf(" Epsilon                      = %e\n", eps);
    printf(" Safe minimum                 = %e\n", sfmin);
    printf(" Base                         = %.0f\n", base);
    printf(" Precision                    = %e\n", prec);
    printf(" Number of digits in mantissa = %.0f\n", t);
    printf(" Rounding mode                = %.0f\n", rnd);
    printf(" Minimum exponent             = %.0f\n", emin);
    printf(" Underflow threshold          = %e\n", rmin);
    printf(" Largest exponent             = %.0f\n", emax);
    printf(" Overflow threshold           = %e\n", rmax);
    printf(" Reciprocal of safe minimum   = %e\n", 1./sfmin);
    return 0;
}
