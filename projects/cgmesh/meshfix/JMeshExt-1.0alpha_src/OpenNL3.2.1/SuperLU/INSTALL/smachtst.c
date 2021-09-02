#include <stdio.h>


int main()
{
    /* Local variables */
    float base, emin, prec, emax, rmin, rmax, t, sfmin;
    extern float smach(char *);
    double rnd, eps;

    eps = smach("Epsilon");
    sfmin = smach("Safe minimum");
    base = smach("Base");
    prec = smach("Precision");
    t = smach("Number of digits in mantissa");
    rnd = smach("Rounding mode");
    emin = smach("Minnimum exponent");
    rmin = smach("Underflow threshold");
    emax = smach("Largest exponent");
    rmax = smach("Overflow threshold");

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
