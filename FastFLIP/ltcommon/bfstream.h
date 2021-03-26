#ifndef BFSTREAM_H
#define BFSTREAM_H
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <cassert>
#include <cstdarg>
#include <cstdlib>
#include <fstream>

#undef __LITTLE_ENDIAN__
#undef __BIG_ENDIAN__
#define __LITTLE_ENDIAN__

#if defined(__BIG_ENDIAN__) && defined(__LITTLE_ENDIAN__)
#if __BIG_ENDIAN__
#undef __LITTLE_ENDIAN__
#else
#undef __BIG_ENDIAN__
#endif
#endif

#ifdef __BIG_ENDIAN__
#ifdef __LITTLE_ENDIAN__
#error Cannot be both big and little endian
#endif
#else
#ifndef __LITTLE_ENDIAN__
#error Need to define either big or little endian
#endif
#endif


//=================================================================================
template<class T> inline void swap_endianity(T &x)
{
   assert(sizeof(T)<=8); // should not be called on composite types: instead specialize swap_endianity if needed.
   T old=x;
   for(unsigned int k=0; k<sizeof(T); ++k)
      ((char*)&x)[k] = ((char*)&old)[sizeof(T)-1-k];
}


//=================================================================================
struct bifstream
{
   std::ifstream input;
   bool big_endian;

   bifstream(void) :
      input(),
#ifdef __BIG_ENDIAN__
   big_endian(true)
#else
   big_endian(false)
#endif
   {
      assert_correct_endianity();
   }

   bifstream(const char *filename_format, ...) :
      input(),
#ifdef __BIG_ENDIAN__
      big_endian(true)
#else
      big_endian(false)
#endif
   {
#ifdef _MSC_VER
      va_list ap;
      va_start(ap, filename_format);
      int len=_vscprintf(filename_format, ap) // _vscprintf doesn't count
                                          +1; // terminating '\0'
      char *filename=new char[len];
      vsprintf(filename, filename_format, ap);
      input.open(filename, std::ifstream::binary);
      delete[] filename;
      va_end(ap);
#else
      va_list ap;
      va_start(ap, filename_format);
      char *filename;
      vasprintf(&filename, filename_format, ap);
      input.open(filename, std::ifstream::binary);
      std::free(filename);
      va_end(ap);
#endif
   }

   void assert_correct_endianity(void)
   {
      int test=1;
#ifdef __BIG_ENDIAN__
      assert(*(char*)&test == 0); // if this fails, you should have defined __LITTLE_ENDIAN__ instead
#else
      assert(*(char*)&test == 1); // if this fails, you should have defined __BIG_ENDIAN__ instead
#endif
   }

   void open(const char *filename_format, ...)
   {
#ifdef _MSC_VER
      va_list ap;
      va_start(ap, filename_format);
      int len=_vscprintf(filename_format, ap) // _vscprintf doesn't count
                                          +1; // terminating '\0'
      char *filename=new char[len];
      vsprintf(filename, filename_format, ap);
      input.open(filename, std::ifstream::binary);
      delete[] filename;
      va_end(ap);
#else
      va_list ap;
      va_start(ap, filename_format);
      char *filename;
      vasprintf(&filename, filename_format, ap);
      input.open(filename, std::ifstream::binary);
      std::free(filename);
      va_end(ap);
#endif
   }

   void vopen(const char *filename_format, va_list ap)
   {
#ifdef _MSC_VER
      int len=_vscprintf(filename_format, ap) // _vscprintf doesn't count
                                          +1; // terminating '\0'
      char *filename=new char[len];
      vsprintf(filename, filename_format, ap);
      input.open(filename, std::ifstream::binary);
      delete[] filename;
#else
      char *filename;
      vasprintf(&filename, filename_format, ap);
      input.open(filename, std::ifstream::binary);
      std::free(filename);
#endif
   }

   bool good(void)
   { return input.good(); }

   bool fail(void)
   { return input.fail(); }

   void close(void)
   { input.close(); }

   void set_big_endian(void)
   { big_endian=true; }

   void set_little_endian(void)
   { big_endian=false; }

   void read_endianity(void)
   { (*this)>>big_endian; }

   void skip(long numbytes)
   { input.seekg(numbytes, std::ios_base::cur); }

   void seek(long position)
   { input.seekg(position); }

   int get(void)
   { return input.get(); }

private: // don't expose dangerous template
   template<class T>
   bifstream &templated_read(T &d)
   {
      input.read((char*)&d, sizeof(T));
#ifdef __BIG_ENDIAN__
      if(!big_endian)
#else
      if(big_endian)
#endif
         swap_endianity(d);
      return *this;
   }
public:

   template<class T>
   void read(T *d, unsigned int num)
   {
      assert(d!=0);
      for(unsigned int i=0; i<num; ++i) (*this)>>d[i];
   }

   friend bifstream &operator>>(bifstream &, bool &);
   friend bifstream &operator>>(bifstream &, unsigned char &);
   friend bifstream &operator>>(bifstream &, short int &);
   friend bifstream &operator>>(bifstream &, unsigned short int &);
   friend bifstream &operator>>(bifstream &, int &);
   friend bifstream &operator>>(bifstream &, unsigned int &);
   friend bifstream &operator>>(bifstream &, long int &);
   friend bifstream &operator>>(bifstream &, unsigned long int &);
   friend bifstream &operator>>(bifstream &, float &);
   friend bifstream &operator>>(bifstream &, double &);
};


bifstream &operator>>(bifstream &input, bool &d);
bifstream &operator>>(bifstream &input, char &d);
bifstream &operator>>(bifstream &input, signed char &d);
bifstream &operator>>(bifstream &input, unsigned char &d);
bifstream &operator>>(bifstream &input, short int &d);
bifstream &operator>>(bifstream &input, unsigned short int &d);
bifstream &operator>>(bifstream &input, int &d);
bifstream &operator>>(bifstream &input, unsigned int &d);
bifstream &operator>>(bifstream &input, long int &d);
bifstream &operator>>(bifstream &input, unsigned long int &d);
bifstream &operator>>(bifstream &input, float &d);
bifstream &operator>>(bifstream &input, double &d);


//=================================================================================
struct bofstream
{
   std::ofstream output;
   bool big_endian;

   bofstream(void) :
      output(),
#ifdef __BIG_ENDIAN__
   big_endian(true)
#else
   big_endian(false)
#endif
   {
      assert_correct_endianity();
   }

   bofstream(const char *filename_format, ...) :
      output(),
#ifdef __BIG_ENDIAN__
      big_endian(true)
#else
      big_endian(false)
#endif
   {
      assert_correct_endianity();
#ifdef _MSC_VER
      va_list ap;
      va_start(ap, filename_format);
      int len=_vscprintf(filename_format, ap) // _vscprintf doesn't count
                                          +1; // terminating '\0'
      char *filename=new char[len];
      vsprintf(filename, filename_format, ap);
      output.open(filename, std::ofstream::binary);
      delete[] filename;
      va_end(ap);
#else
      va_list ap;
      va_start(ap, filename_format);
      char *filename;
      vasprintf(&filename, filename_format, ap);
      output.open(filename, std::ofstream::binary);
      std::free(filename);
      va_end(ap);
#endif
   }

   void assert_correct_endianity(void)
   {
      int test=1;
#ifdef __BIG_ENDIAN__
      assert(*(char*)&test == 0); // if this fails, you should have defined __LITTLE_ENDIAN__ instead
#else
      assert(*(char*)&test == 1); // if this fails, you should have defined __BIG_ENDIAN__ instead
#endif
   }

   void open(const char *filename_format, ...)
   {
#ifdef _MSC_VER
      va_list ap;
      va_start(ap, filename_format);
      int len=_vscprintf(filename_format, ap) // _vscprintf doesn't count
                                          +1; // terminating '\0'
      char *filename=new char[len];
      vsprintf(filename, filename_format, ap);
      output.open(filename, std::ofstream::binary);
      delete[] filename;
      va_end(ap);
#else
      va_list ap;
      va_start(ap, filename_format);
      char *filename;
      vasprintf(&filename, filename_format, ap);
      output.open(filename, std::ofstream::binary);
      std::free(filename);
      va_end(ap);
#endif
   }

   void vopen(const char *filename_format, va_list ap)
   {

#ifdef _MSC_VER
      int len=_vscprintf(filename_format, ap) // _vscprintf doesn't count
                                          +1; // terminating '\0'
      char *filename=new char[len];
      vsprintf(filename, filename_format, ap);
      output.open(filename, std::ofstream::binary);
      delete[] filename;
#else
      char *filename;
      vasprintf(&filename, filename_format, ap);
      output.open(filename, std::ofstream::binary);
      std::free(filename);
#endif
   }

   bool good(void)
   { return output.good(); }

   bool fail(void)
   { return output.fail(); }

   void close(void)
   { output.close(); }

   void set_big_endian(void)
   { big_endian=true; }

   void set_little_endian(void)
   { big_endian=false; }

   void write_endianity(void)
   { (*this)<<big_endian; }

   void write_zero(unsigned int numbytes)
   { for(unsigned int i=0; i<numbytes; ++i) output.put(0); }

   void put(char byte)
   { output.put(byte); }

private: // don't expose dangerous templates
   template<class T>
   bofstream &templated_write(const T &d)
   {
#ifdef __BIG_ENDIAN__
      if(!big_endian)
#else
      if(big_endian)
#endif
      {
         T swapped_copy=d;
         swap_endianity(swapped_copy);
         output.write((const char*)&swapped_copy, sizeof(T));
      }else
         output.write((const char*)&d, sizeof(T));
      return *this;
   }
public:

   template<class T>
   void write(const T *d, unsigned int num)
   {
      assert(d!=0);
      for(unsigned int i=0; i<num; ++i) (*this)<<d[i];
   }

   friend bofstream &operator<<(bofstream &, const bool &);
   friend bofstream &operator<<(bofstream &, const short int &);
   friend bofstream &operator<<(bofstream &, const unsigned short int &);
   friend bofstream &operator<<(bofstream &, const int &);
   friend bofstream &operator<<(bofstream &, const unsigned int &);
   friend bofstream &operator<<(bofstream &, const long int &);
   friend bofstream &operator<<(bofstream &, const unsigned long int &);
   friend bofstream &operator<<(bofstream &, const float &);
   friend bofstream &operator<<(bofstream &, const double &);
};

bofstream &operator<<(bofstream &output, const bool &d);
bofstream &operator<<(bofstream &output, const char &d);
bofstream &operator<<(bofstream &output, const signed char &d);
bofstream &operator<<(bofstream &output, const unsigned char &d);
bofstream &operator<<(bofstream &output, const short int &d);
bofstream &operator<<(bofstream &output, const unsigned short int &d);
bofstream &operator<<(bofstream &output, const int &d);
bofstream &operator<<(bofstream &output, const unsigned int &d);
bofstream &operator<<(bofstream &output, const long int &d);
bofstream &operator<<(bofstream &output, const unsigned long int &d);
bofstream &operator<<(bofstream &output, const float &d);
bofstream &operator<<(bofstream &output, const double &d);


#ifdef _MSC_VER
#undef _CRT_SECURE_NO_WARNINGS
#endif
#endif
