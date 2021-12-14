// Source file for the RING matrix class 




// Matrix concatenation order is designed to be compatible with OpenGL.
// The idea is that matrixs can be applied (matrices multiplied)
// in the same order as OpenGL calls would be made.  Make more global
// matrixs first (i.e., ones at the top of the scene hierachy).
// Vectors are column vectors on the right of matrices.  Transformation
// matrices are post-multiplied.
// 
// Matrix storage is NOT COMPATIBLE with OpenGL.  OpenGL represents
// matrices in COLUMN MAJOR ORDER.  I find manipulating matrices in this
// format both inconvenient (operator[]) and confusing.  Therefore, this
// package represents matices in ROW MAJOR ORDER.  In order to pass
// matrices to OpenGL, a transpose is required.  Look into this re
// efficiency ???



// Include files 

#include "R3.h"



// Public variables 

const R3Matrix R3null_matrix (
                              0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0
                              );

const R3Matrix R3identity_matrix (
                                  1.0, 0.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0, 0.0,
                                  0.0, 0.0, 0.0, 1.0
                                  );



// Public functions 

int 
R3InitMatrix()
{
  // Return success 
  return true;
}



void 
R3StopMatrix()
{
}



R3Matrix::
R3Matrix(void)
{
}



R3Matrix::
R3Matrix(const R3Matrix& matrix)
{
  // Assign matrix entries
  m[0][0] = matrix.m[0][0]; m[0][1] = matrix.m[0][1]; m[0][2] = matrix.m[0][2]; m[0][3] = matrix.m[0][3];
  m[1][0] = matrix.m[1][0]; m[1][1] = matrix.m[1][1]; m[1][2] = matrix.m[1][2]; m[1][3] = matrix.m[1][3];
  m[2][0] = matrix.m[2][0]; m[2][1] = matrix.m[2][1]; m[2][2] = matrix.m[2][2]; m[2][3] = matrix.m[2][3];
  m[3][0] = matrix.m[3][0]; m[3][1] = matrix.m[3][1]; m[3][2] = matrix.m[3][2]; m[3][3] = matrix.m[3][3];
}



R3Matrix::
R3Matrix(double a00, double a01, double a02, double a03,
         double a10, double a11, double a12, double a13,
	 double a20, double a21, double a22, double a23,
	 double a30, double a31, double a32, double a33)
{
  // Assign matrix entries
  m[0][0] = a00; m[0][1] = a01; m[0][2] = a02; m[0][3] = a03;
  m[1][0] = a10; m[1][1] = a11; m[1][2] = a12; m[1][3] = a13;
  m[2][0] = a20; m[2][1] = a21; m[2][2] = a22; m[2][3] = a23;
  m[3][0] = a30; m[3][1] = a31; m[3][2] = a32; m[3][3] = a33;
}



R3Matrix::
R3Matrix(const double *a)
{
  // Assign matrix entries
  m[0][0] = a[0]; m[0][1] = a[1]; m[0][2] = a[2]; m[0][3] = a[3];
  m[1][0] = a[4]; m[1][1] = a[5]; m[1][2] = a[6]; m[1][3] = a[7];
  m[2][0] = a[8]; m[2][1] = a[9]; m[2][2] = a[10]; m[2][3] = a[11];
  m[3][0] = a[12]; m[3][1] = a[13]; m[3][2] = a[14]; m[3][3] = a[15];
}



const bool R3Matrix::
IsZero (void) const
{
  // Return whether matrix is zero
  return (*this == R3null_matrix);
}



const bool R3Matrix::
IsIdentity (void) const
{
  // Return whether matrix is identity
  return (*this == R3identity_matrix);
}



const bool R3Matrix::
IsIsotropic(void) const
{
  // Return whether matrix matrix is isotropic
  double d0 = m[0][0]*m[0][0] + m[1][0]*m[1][0] + m[2][0]*m[2][0];
  double d1 = m[0][1]*m[0][1] + m[1][1]*m[1][1] + m[2][1]*m[2][1];
  if (d0 != d1) return false;
  double d2 = m[0][2]*m[0][2] + m[1][2]*m[1][2] + m[2][2]*m[2][2];
  if (d0 != d2) return false;
  return true;
}



const bool R3Matrix::
HasTranslation(void) const
{
  // Return whether matrix matrix has translation
  if (m[0][3] != 0) return true;
  if (m[1][3] != 0) return true;
  if (m[2][3] != 0) return true;
  return false;
}



const bool R3Matrix::
HasScale(void) const
{
  // Return whether matrix matrix has scale
  double d0 = m[0][0]*m[0][0] + m[1][0]*m[1][0] + m[2][0]*m[2][0];
  if (d0 != 1.0) return true;
  double d1 = m[0][1]*m[0][1] + m[1][1]*m[1][1] + m[2][1]*m[2][1];
  if (d1 != 1.0) return true;
  double d2 = m[0][2]*m[0][2] + m[1][2]*m[1][2] + m[2][2]*m[2][2];
  if (d2 != 1.0) return true;
  return false;
}



const bool R3Matrix::
HasRotation(void) const
{
  // Return whether matrix matrix has rotation
  if (m[0][1] != 0) return true;
  if (m[0][2] != 0) return true;
  if (m[1][0] != 0) return true;
  if (m[1][2] != 0) return true;
  if (m[2][0] != 0) return true;
  if (m[2][1] != 0) return true;
  return false;
}



const bool R3Matrix::
HasMirror(void) const
{
  // Return whether matrix matrix has mirror operator
  R3Vector vx(m[0][0], m[1][0], m[2][0]);
  R3Vector vy(m[0][1], m[1][1], m[2][1]);
  R3Vector vz(m[0][2], m[1][2], m[2][2]);
  return (vz.Dot(vx % vy) < 0);
}



const double R3Matrix::
Determinant(void) const
{
  // Return matrix determinant
  return R3MatrixDet4(m[0][0], m[0][1], m[0][2], m[0][3],
                      m[1][0], m[1][1], m[1][2], m[1][3],
                      m[2][0], m[2][1], m[2][2], m[2][3],
                      m[3][0], m[3][1], m[3][2], m[3][3]);
}



const R3Matrix R3Matrix::
Transpose(void) const
{
  // Return transpose of matrix
  return R3Matrix(m[0][0], m[1][0], m[2][0], m[3][0],
                  m[0][1], m[1][1], m[2][1], m[3][1],
                  m[0][2], m[1][2], m[2][2], m[3][2],
                  m[0][3], m[1][3], m[2][3], m[3][3]);
}



const R3Matrix R3Matrix::
Inverse(void) const
{
  // Return inverse of matrix
  R3Matrix inverse(*this);
  inverse.Invert();
  return inverse;
}



void R3Matrix::
Flip(void)
{
  // Transpose matrix
  double tmp;
  tmp = m[1][0]; m[1][0] = m[0][1]; m[0][1] = tmp;
  tmp = m[2][0]; m[2][0] = m[0][2]; m[0][2] = tmp;
  tmp = m[3][0]; m[3][0] = m[0][3]; m[0][3] = tmp;
  tmp = m[1][2]; m[1][2] = m[2][1]; m[2][1] = tmp;
  tmp = m[1][3]; m[1][3] = m[3][1]; m[3][1] = tmp;
  tmp = m[2][3]; m[2][3] = m[3][2]; m[3][2] = tmp;
}



void R3Matrix::
Invert(void)
{
  // Copy matrix into local variables
  double Ma, Mb, Mc, Md, Me, Mf, Mg, Mh, Mi, Mj, Mk, Ml, Mm, Mn, Mo, Mp;
  Ma = m[0][0]; Mb = m[0][1]; Mc = m[0][2]; Md = m[0][3];
  Me = m[1][0]; Mf = m[1][1]; Mg = m[1][2]; Mh = m[1][3];
  Mi = m[2][0]; Mj = m[2][1]; Mk = m[2][2]; Ml = m[2][3];
  Mm = m[3][0]; Mn = m[3][1]; Mo = m[3][2]; Mp = m[3][3];

  // Compute sub-determinants and determinant
  double a1 = R3MatrixDet3(Mf, Mg, Mh, Mj, Mk, Ml, Mn, Mo, Mp);
  double a2 = R3MatrixDet3(Me, Mg, Mh, Mi, Mk, Ml, Mm, Mo, Mp);  
  double a3 = R3MatrixDet3(Me, Mf, Mh, Mi, Mj, Ml, Mm, Mn, Mp);
  double a4 = R3MatrixDet3(Me, Mf, Mg, Mi, Mj, Mk, Mm, Mn, Mo);
  double det = Ma*a1 - Mb*a2 + Mc*a3 - Md*a4;
  if (det == 0) {
    fprintf(stderr, "Unable to invert matrix with zero determinant");
    return;
  }

  // Compute inverse matrix
  m[0][0] = (a1)/det;
  m[1][0] = -(a2)/det;
  m[2][0] = (a3)/det;
  m[3][0] = -(a4)/det;

  m[0][1] = -(R3MatrixDet3(Mb, Mc, Md, Mj, Mk, Ml, Mn, Mo, Mp))/det;
  m[1][1] = (R3MatrixDet3(Ma, Mc, Md, Mi, Mk, Ml, Mm, Mo, Mp))/det;
  m[2][1] = -(R3MatrixDet3(Ma, Mb, Md, Mi, Mj, Ml, Mm, Mn, Mp))/det;
  m[3][1] = (R3MatrixDet3(Ma, Mb, Mc, Mi, Mj, Mk, Mm, Mn, Mo))/det;

  m[0][2] = (R3MatrixDet3(Mb, Mc, Md, Mf, Mg, Mh, Mn, Mo, Mp))/det;
  m[1][2] = -(R3MatrixDet3(Ma, Mc, Md, Me, Mg, Mh, Mm, Mo, Mp))/det;
  m[2][2] = (R3MatrixDet3(Ma, Mb, Md, Me, Mf, Mh, Mm, Mn, Mp))/det;
  m[3][2] = -(R3MatrixDet3(Ma, Mb, Mc, Me, Mf, Mg, Mm, Mn, Mo))/det;

  m[0][3] = -(R3MatrixDet3(Mb, Mc, Md, Mf, Mg, Mh, Mj, Mk, Ml))/det;
  m[1][3] = (R3MatrixDet3(Ma, Mc, Md, Me, Mg, Mh, Mi, Mk, Ml))/det;
  m[2][3] = -(R3MatrixDet3(Ma, Mb, Md, Me, Mf, Mh, Mi, Mj, Ml))/det;
  m[3][3] = (R3MatrixDet3(Ma, Mb, Mc, Me, Mf, Mg, Mi, Mj, Mk))/det;
}



void R3Matrix:: 
XTranslate(double offset)
{
  // Translate matrix -- post-multiply by: 
  //   [ 1 0 0 tx ]
  //   [ 0 1 0 0  ]
  //   [ 0 0 1 0  ]
  //   [ 0 0 0 1  ]
  m[0][3] += m[0][0] * offset;
  m[1][3] += m[1][0] * offset;
  m[2][3] += m[2][0] * offset;
}



void R3Matrix:: 
YTranslate(double offset)
{
  // Translate matrix -- post-multiply by: 
  //   [ 1 0 0 0  ]
  //   [ 0 1 0 ty ]
  //   [ 0 0 1 0  ]
  //   [ 0 0 0 1  ]
  m[0][3] += m[0][1] * offset;
  m[1][3] += m[1][1] * offset;
  m[2][3] += m[2][1] * offset;
}



void R3Matrix:: 
ZTranslate(double offset)
{
  // Translate matrix -- post-multiply by: 
  //   [ 1 0 0 0  ]
  //   [ 0 1 0 0  ]
  //   [ 0 0 1 tz ]
  //   [ 0 0 0 1  ]
  m[0][3] += m[0][2] * offset;
  m[1][3] += m[1][2] * offset;
  m[2][3] += m[2][2] * offset;
}



void R3Matrix:: 
Translate(int axis, double offset)
{
  // Translate matrix along axis
  switch (axis) {
  case R3_X: 
    XTranslate(offset); 
    break;

  case R3_Y: 
    YTranslate(offset); 
    break;

  case R3_Z: 
    ZTranslate(offset); 
    break;

  default: 
    fprintf(stderr, "Matrix translate along undefined axis");
    break;
  }
}



void R3Matrix::
Translate(double offset)
{
  // Translate matrix
  XTranslate(offset);
  YTranslate(offset);
  ZTranslate(offset);
}



void R3Matrix:: 
Translate(const R3Vector& offset)
{
  // Translate matrix
  XTranslate(offset.X());
  YTranslate(offset.Y());
  ZTranslate(offset.Z());
}



void R3Matrix:: 
XScale(double scale)
{
  // Scale matrix -- post-multiply by: 
  //   [ sx 0 0 0 ]
  //   [ 0  1 0 0 ]
  //   [ 0  0 1 0 ]
  //   [ 0  0 0 1 ]
  m[0][0] *= scale;
  m[1][0] *= scale;
  m[2][0] *= scale;
}



void R3Matrix:: 
YScale(double scale)
{
  // Scale matrix -- post-multiply by: 
  //   [ 1 0  0 0 ]
  //   [ 0 sy 0 0 ]
  //   [ 0 0  1 0 ]
  //   [ 0 0  0 1 ]
  m[0][1] *= scale;
  m[1][1] *= scale;
  m[2][1] *= scale;
}



void R3Matrix:: 
ZScale(double scale)
{
  // Scale matrix -- post-multiply by: 
  //   [ 1 0 0  0 ]
  //   [ 0 1 0  0 ]
  //   [ 0 0 sz 0 ]
  //   [ 0 0 0  1 ]
  m[0][2] *= scale;
  m[1][2] *= scale;
  m[2][2] *= scale;
}



void R3Matrix:: 
Scale(int axis, double scale)
{
  // Scale matrix along axis
  switch (axis) {
  case R3_X: 
    XScale(scale); 
    break;

  case R3_Y: 
    YScale(scale); 
    break;

  case R3_Z: 
    ZScale(scale); 
    break;

  default: 
    fprintf(stderr, "Matrix scale along undefined axis");
    break;
  }
}



void R3Matrix:: 
Scale(double scale)
{
  // Scale matrix 
  XScale(scale);
  YScale(scale);
  ZScale(scale);
}



void R3Matrix:: 
Scale(const R3Vector& scale)
{
  // Scale matrix
  XScale(scale.X());
  YScale(scale.Y());
  ZScale(scale.Z());
}



void R3Matrix:: 
XRotate(double radians)
{
  // rotate matrix around X axis counterclockwise
  double c = cos(radians);
  double s = sin(radians);
  R3Matrix rotation(
                    1.0, 0.0, 0.0, 0.0,
                    0.0, c,   -s,  0.0,
                    0.0, s,   c,   0.0,
                    0.0, 0.0, 0.0, 1.0 );
  *this *= rotation;
}



void R3Matrix:: 
YRotate(double radians)
{
  // rotate matrix around Y axis counterclockwise
  double c = cos(radians);
  double s = sin(radians);
  R3Matrix rotation(c,   0.0, s,   0.0,
                    0.0, 1.0, 0.0, 0.0,
                    -s,  0.0, c,   0.0,
                    0.0, 0.0, 0.0, 1.0 );
  *this *= rotation;
}



void R3Matrix:: 
ZRotate(double radians)
{
  // rotate matrix around Z axis counterclockwise 
  double c = cos(radians);
  double s = sin(radians);
  R3Matrix rotation(c,   -s,  0.0, 0.0,
                    s,   c,   0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0 );
  *this *= rotation;
}



void R3Matrix:: 
Rotate(int axis, double radians)
{
  // rotate matrix around an axis counterclockwise
  switch (axis) {
  case R3_X: 
    XRotate(radians); 
    break;

  case R3_Y: 
    YRotate(radians); 
    break;

  case R3_Z: 
    ZRotate(radians); 
    break;

  default: 
    fprintf(stderr, "Matrix rotation around undefined axis");
    break;
  }
}



void R3Matrix:: 
Rotate(const R3Vector& radians)
{
  XRotate(radians.X());
  YRotate(radians.Y());
  ZRotate(radians.Z());
}



void R3Matrix:: 
Rotate(const R3Vector& axis, double radians)
{
  // rotate matrix for arbitrary axis counterclockwise
  // From Graphics Gems I, p. 466
  double x = axis.X();
  double y = axis.Y();
  double z = axis.Z();
  double c = cos(radians);
  double s = sin(radians);
  double t = 1.0 - c;
  R3Matrix rotation(t*x*x + c, t*x*y - s*z, t*x*z + s*y, 0.0,
                    t*x*y + s*z, t*y*y + c, t*y*z - s*x, 0.0,
                    t*x*z - s*y, t*y*z + s*x, t*z*z + c, 0.0,
                    0.0, 0.0, 0.0, 1.0);
  *this *= rotation;
}



void R3Matrix:: 
Rotate(const R3Vector& from, const R3Vector& to)
{
  // rotate matrix that takes direction of vector "from" -> "to"
  // This is a quickie hack -- there's got to be a better way
  double d1 = from.Length();
  if (d1 == 0) return;
  double d2 = to.Length();
  if (d2 == 0) return;
  double cosine = from.Dot(to) / (d1 * d2);
  double radians = acos(cosine);
  R3Vector axis = from % to;
  axis.Normalize();
  Rotate(axis, radians);
}



void R3Matrix:: 
Add(const R3Matrix& a)
{
  // Add matrix entry-by-entry
  m[0][0] += a.m[0][0]; m[0][1] += a.m[0][1]; m[0][2] += a.m[0][2]; m[0][3] += a.m[0][3]; 
  m[1][0] += a.m[1][0]; m[1][1] += a.m[1][1]; m[1][2] += a.m[1][2]; m[1][3] += a.m[1][3]; 
  m[2][0] += a.m[2][0]; m[2][1] += a.m[2][1]; m[2][2] += a.m[2][2]; m[2][3] += a.m[2][3]; 
  m[3][0] += a.m[3][0]; m[3][1] += a.m[3][1]; m[3][2] += a.m[3][2]; m[3][3] += a.m[3][3];
}



void R3Matrix::
Subtract(const R3Matrix& a)
{
  // Subtract matrix entry-by-entry
  m[0][0] -= a.m[0][0]; m[0][1] -= a.m[0][1]; m[0][2] -= a.m[0][2]; m[0][3] -= a.m[0][3]; 
  m[1][0] -= a.m[1][0]; m[1][1] -= a.m[1][1]; m[1][2] -= a.m[1][2]; m[1][3] -= a.m[1][3]; 
  m[2][0] -= a.m[2][0]; m[2][1] -= a.m[2][1]; m[2][2] -= a.m[2][2]; m[2][3] -= a.m[2][3]; 
  m[3][0] -= a.m[3][0]; m[3][1] -= a.m[3][1]; m[3][2] -= a.m[3][2]; m[3][3] -= a.m[3][3];
}



R3Matrix& R3Matrix::
operator=(const R3Matrix& a)
{
  // Assign matrix entry-by-entry
  m[0][0] = a.m[0][0]; m[0][1] = a.m[0][1]; m[0][2] = a.m[0][2]; m[0][3] = a.m[0][3]; 
  m[1][0] = a.m[1][0]; m[1][1] = a.m[1][1]; m[1][2] = a.m[1][2]; m[1][3] = a.m[1][3]; 
  m[2][0] = a.m[2][0]; m[2][1] = a.m[2][1]; m[2][2] = a.m[2][2]; m[2][3] = a.m[2][3]; 
  m[3][0] = a.m[3][0]; m[3][1] = a.m[3][1]; m[3][2] = a.m[3][2]; m[3][3] = a.m[3][3];
  return *this;
}



R3Matrix& R3Matrix::
operator*=(double a)
{
  // Scale matrix entry-by-entry
  m[0][0] *= a; m[0][1] *= a; m[0][2] *= a; m[0][3] *= a; 
  m[1][0] *= a; m[1][1] *= a; m[1][2] *= a; m[1][3] *= a; 
  m[2][0] *= a; m[2][1] *= a; m[2][2] *= a; m[2][3] *= a; 
  m[3][0] *= a; m[3][1] *= a; m[3][2] *= a; m[3][3] *= a;
  return *this;
}



R3Matrix& R3Matrix::
operator/=(double a)
{
  // Scale matrix entry-by-entry
  m[0][0] /= a; m[0][1] /= a; m[0][2] /= a; m[0][3] /= a; 
  m[1][0] /= a; m[1][1] /= a; m[1][2] /= a; m[1][3] /= a; 
  m[2][0] /= a; m[2][1] /= a; m[2][2] /= a; m[2][3] /= a; 
  m[3][0] /= a; m[3][1] /= a; m[3][2] /= a; m[3][3] /= a;
  return *this;
}



R3Matrix 
operator-(const R3Matrix& a)
{
  // Negate matrix
  return R3Matrix(-a.m[0][0], -a.m[0][1], -a.m[0][2], -a.m[0][3], 
                  -a.m[1][0], -a.m[1][1], -a.m[1][2], -a.m[1][3], 
                  -a.m[2][0], -a.m[2][1], -a.m[2][2], -a.m[2][3], 
                  -a.m[3][0], -a.m[3][1], -a.m[3][2], -a.m[3][3]);
}



R3Matrix 
operator+(const R3Matrix& a, const R3Matrix& b) 
{
  // Sum matrix
  return R3Matrix(a.m[0][0]+b.m[0][0], a.m[0][1]+b.m[0][1], a.m[0][2]+b.m[0][2], a.m[0][3]+b.m[0][3], 
                  a.m[1][0]+b.m[1][0], a.m[1][1]+b.m[1][1], a.m[1][2]+b.m[1][2], a.m[1][3]+b.m[1][3], 
                  a.m[2][0]+b.m[2][0], a.m[2][1]+b.m[2][1], a.m[2][2]+b.m[2][2], a.m[2][3]+b.m[2][3], 
                  a.m[3][0]+b.m[3][0], a.m[3][1]+b.m[3][1], a.m[3][2]+b.m[3][2], a.m[3][3]+b.m[3][3]);
}



R3Matrix 
operator-(const R3Matrix& a, const R3Matrix& b) 
{
  // Subtract matrix
  return R3Matrix(a.m[0][0]-b.m[0][0], a.m[0][1]-b.m[0][1], a.m[0][2]-b.m[0][2], a.m[0][3]-b.m[0][3], 
                  a.m[1][0]-b.m[1][0], a.m[1][1]-b.m[1][1], a.m[1][2]-b.m[1][2], a.m[1][3]-b.m[1][3], 
                  a.m[2][0]-b.m[2][0], a.m[2][1]-b.m[2][1], a.m[2][2]-b.m[2][2], a.m[2][3]-b.m[2][3], 
                  a.m[3][0]-b.m[3][0], a.m[3][1]-b.m[3][1], a.m[3][2]-b.m[3][2], a.m[3][3]-b.m[3][3]);
}



R3Matrix 
operator*(const R3Matrix& a, double b)
{
  // Scale matrix
  return R3Matrix(a.m[0][0]*b, a.m[0][1]*b, a.m[0][2]*b, a.m[0][3]*b, 
                  a.m[1][0]*b, a.m[1][1]*b, a.m[1][2]*b, a.m[1][3]*b, 
                  a.m[2][0]*b, a.m[2][1]*b, a.m[2][2]*b, a.m[2][3]*b, 
                  a.m[3][0]*b, a.m[3][1]*b, a.m[3][2]*b, a.m[3][3]*b);
}



inline R3Matrix 
operator/(const R3Matrix& a, double b) 
{
  // Scale matrix
  assert(b != 0.0);
  return R3Matrix(a.m[0][0]*b, a.m[0][1]*b, a.m[0][2]*b, a.m[0][3]*b, 
                  a.m[1][0]*b, a.m[1][1]*b, a.m[1][2]*b, a.m[1][3]*b, 
                  a.m[2][0]*b, a.m[2][1]*b, a.m[2][2]*b, a.m[2][3]*b, 
                  a.m[3][0]*b, a.m[3][1]*b, a.m[3][2]*b, a.m[3][3]*b);
}



R3Matrix 
operator*(const R3Matrix& a, const R3Matrix& b) 
{
  R3Matrix result;
  int r, c;

  // Multiply matrix
  for (r=0; r<4; r++)
    for (c=0; c<4; c++)
      result.m[r][c] = a.m[r][0] * b.m[0][c] + a.m[r][1] * b.m[1][c] +
        a.m[r][2] * b.m[2][c] + a.m[r][3] * b.m[3][c];

  // Return result
  return result;
}



R3Vector
operator*(const R3Matrix& a, const R3Vector& v)
{
  // Multiply matrix by vector
  double x = a.m[0][0] * v.X() + a.m[0][1] * v.Y() + a.m[0][2] * v.Z();
  double y = a.m[1][0] * v.X() + a.m[1][1] * v.Y() + a.m[1][2] * v.Z();
  double z = a.m[2][0] * v.X() + a.m[2][1] * v.Y() + a.m[2][2] * v.Z();
  return R3Vector(x, y, z);
}



R3Point 
operator*(const R3Matrix& a, const R3Point& p)
{
  // Multiply matrix by point
  double x = a.m[0][0] * p.X() + a.m[0][1] * p.Y() + a.m[0][2] * p.Z() + a.m[0][3];
  double y = a.m[1][0] * p.X() + a.m[1][1] * p.Y() + a.m[1][2] * p.Z() + a.m[1][3];
  double z = a.m[2][0] * p.X() + a.m[2][1] * p.Y() + a.m[2][2] * p.Z() + a.m[2][3];
  return R3Point(x, y, z);
}



double R3MatrixDet2(double a, double b,
                    double c, double d)
{
  // Return determinant of 2x2 matrix 
  return (a * d - b * c);
}



double R3MatrixDet3 (double a, double b, double c, 
                     double d, double e, double f, 
                     double g, double h, double i)
{
  // Return determinant of 3x3 matrix 
  return (a * (e * i - h * f) - b * (d * i - g * f) + c * (d * h - g * e));
}



double R3MatrixDet4 (double a, double b, double c, double d, 
                     double e, double f, double g, double h, 
                     double i, double j, double k, double l, 
                     double m, double n, double o, double p)
{
  // Return determinant of 4x4 matrix 
  double det = 0.0;
  det += a * R3MatrixDet3(f,g,h,j,k,l,n,o,p);
  det -= b * R3MatrixDet3(e,g,h,i,k,l,m,o,p);
  det += c * R3MatrixDet3(e,f,h,i,j,l,m,n,p);
  det -= d * R3MatrixDet3(e,f,g,i,j,k,m,n,o);
  return (det);
}




