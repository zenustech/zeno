// Include file for the RING matrix class 



// Class definition 

class R3Matrix {
 public:
  // Constructor functions
  R3Matrix(void);
  R3Matrix(const R3Matrix& matrix);
  R3Matrix(double a00, double a01, double a02, double a03,
           double a10, double a11, double a12, double a13,
           double a20, double a21, double a22, double a23,
           double a30, double a31, double a32, double a33);
  R3Matrix(const double* array);

  // Entry acccess
  double *operator[](int i);
  const double *operator[](int i) const;

  // Properties
  const bool IsZero(void) const;
  const bool IsIdentity(void) const;
  const bool IsIsotropic(void) const;
  const bool HasTranslation(void) const;
  const bool HasScale(void) const;
  const bool HasRotation(void) const;
  const bool HasMirror(void) const;
  const double Determinant(void) const;
  const R3Matrix Transpose(void) const;
  const R3Matrix Inverse(void) const;
  const bool operator==(const R3Matrix& matrix) const;
  const bool operator!=(const R3Matrix& matrix) const;

  // Manipulation functions/operators
  void Flip(void);
  void Invert(void);
  void XTranslate(double offset);
  void YTranslate(double offset);
  void ZTranslate(double offset);
  void Translate(double offset);
  void Translate(int axis, double offset);
  void Translate(const R3Vector& offset);
  void XScale(double scale);
  void YScale(double scale);
  void ZScale(double scale);
  void Scale(double scale);
  void Scale(int axis, double scale);
  void Scale(const R3Vector& scale);
  void XRotate(double radians);
  void YRotate(double radians);
  void ZRotate(double radians);
  void Rotate(const R3Vector& radians);
  void Rotate(int axis, double radians);
  void Rotate(const R3Vector& vector, double radians);
  void Rotate(const R3Vector& from, const R3Vector& to);
  void Transform(const R3Matrix& matrix);
  void Multiply(const R3Matrix& matrix);
  void Add(const R3Matrix& matrix);
  void Subtract(const R3Matrix& matrix);

  // Assignment operators
  R3Matrix& operator=(const R3Matrix& matrix);
  R3Matrix& operator+=(const R3Matrix& matrix);
  R3Matrix& operator-=(const R3Matrix& matrix);
  R3Matrix& operator*=(double a);
  R3Matrix& operator*=(const R3Matrix& matrix);
  R3Matrix& operator/=(double a);

  // Arithmetic operators
  friend R3Matrix operator-(const R3Matrix& matrix);
  friend R3Matrix operator+(const R3Matrix& matrix1, const R3Matrix& matrix2);
  friend R3Matrix operator-(const R3Matrix& matrix1, const R3Matrix& matrix2);
  friend R3Matrix operator*(double a, const R3Matrix& matrix);
  friend R3Matrix operator*(const R3Matrix& matrix, double a);
  friend R3Matrix operator*(const R3Matrix& matrix1, const R3Matrix& matrix2);
  friend R3Matrix operator/(const R3Matrix& matrix, double scale);
  friend R3Vector operator*(const R3Matrix& matrix, const R3Vector& vector);
  friend R3Point operator*(const R3Matrix& matrix, const R3Point& point);

  // Draw functions/operators
  void Load(void) const;
  void Draw(void) const;
  void Push(void) const;
  void Pop(void) const;

 private:
  double m[4][4];
};



// Public variables 

extern const R3Matrix R3null_matrix;
extern const R3Matrix R3identity_matrix;



// Utility functions 

double R3MatrixDet2 (
                     double a, double b,
                     double c, double d);

double R3MatrixDet3 (
                     double a, double b, double c, 
                     double d, double e, double f, 
                     double g, double h, double i);

double R3MatrixDet4 (
                     double a, double b, double c, double d, 
                     double e, double f, double g, double h, 
                     double i, double j, double k, double l, 
                     double m, double n, double o, double p);



// Inline functions 

inline const double *R3Matrix::
operator[] (int i) const
{
  assert ((i>=0)&&(i<=3));
  return m[i];
}



inline const bool R3Matrix::
operator==(const R3Matrix& matrix) const
{
  // Return whether or not another matrix is the same
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (m[i][j] != matrix.m[i][j]) return false;
    }
  }
  return true;
}



inline const bool R3Matrix::
operator!=(const R3Matrix& matrix) const
{
  // Return whether matrix is not equal
  return (!(*this == matrix));
}



inline void R3Matrix:: 
Transform(const R3Matrix& a)
{
  // Post-multiply transform
  *this = *this * a;
}



inline void R3Matrix:: 
Multiply(const R3Matrix& a)
{
  // Multiply matrix
  *this = *this * a;
}



inline R3Matrix& R3Matrix::
operator+=(const R3Matrix& a)
{
  // Add matrix entry-by-entry
  Add(a);
  return *this;
}



inline R3Matrix& R3Matrix::
operator-=(const R3Matrix& a)
{
  // Subtract matrix entry-by-entry
  Subtract(a);
  return *this;
}



inline R3Matrix& R3Matrix::
operator*=(const R3Matrix& a)
{
  // Multiply matrix
  Multiply(a);
  return *this;
}



inline double *R3Matrix::
operator[] (int i) 
{
  assert ((i>=0)&&(i<=3));
  return m[i];
}



