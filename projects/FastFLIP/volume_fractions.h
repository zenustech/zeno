#ifndef VOLUME_FRACTIONS_H
#define VOLUME_FRACTIONS_H

// Given a triangle with level set values, use linear interpolation to
// estimate the fraction of the triangle occupied by the phi<0 part
float area_fraction(float phi0, float phi1, float phi2);
double area_fraction(double phi0, double phi1, double phi2);

// Given a rectangle with level set values, estimate fraction occupied
// by the phi<0 part
float area_fraction(float phi00, float phi10, float phi01, float phi11);
double area_fraction(double phi00, double phi10, double phi01, double phi11);

// Given a tetrahedron with level set values, use linear interpolation to
// estimate the fraction of the tetrahedron occupied by the phi<0 part
float volume_fraction(float phi0, float phi1, float phi2, float phi3);
double volume_fraction(double phi0, double phi1, double phi2, double phi3);

// Given a parallelepiped (e.g. cube) with level set values, estimate
// fraction occupied by the phi<0 part
float volume_fraction(float phi000, float phi100,
                      float phi010, float phi110,
                      float phi001, float phi101,
                      float phi011, float phi111);
double volume_fraction(double phi000, double phi100,
                       double phi010, double phi110,
                       double phi001, double phi101,
                       double phi011, double phi111);

#endif
