#include "bfstream.h"

bifstream &operator>>(bifstream &input, bool &d)
{ d=input.get()?true:false; return input; } // note: on some platforms sizeof(bool)!=1
   
bifstream &operator>>(bifstream &input, char &d)
{ d=(char)input.get(); return input; }

bifstream &operator>>(bifstream &input, signed char &d)
{ d=(signed char)input.get(); return input; }

bifstream &operator>>(bifstream &input, unsigned char &d)
{ d=(unsigned char)input.get(); return input; }

bifstream &operator>>(bifstream &input, short int &d)
{ return input.templated_read(d); }

bifstream &operator>>(bifstream &input, unsigned short int &d)
{ return input.templated_read(d); }

bifstream &operator>>(bifstream &input, int &d)
{ return input.templated_read(d); }

bifstream &operator>>(bifstream &input, unsigned int &d)
{ return input.templated_read(d); }

bifstream &operator>>(bifstream &input, long int &d)
{ return input.templated_read(d); }

bifstream &operator>>(bifstream &input, unsigned long int &d)
{ return input.templated_read(d); }

bifstream &operator>>(bifstream &input, float &d)
{ return input.templated_read(d); }

bifstream &operator>>(bifstream &input, double &d)
{ return input.templated_read(d); }

//=============================================================================

bofstream &operator<<(bofstream &output, const bool &d)
{ output.put((char)d); return output; }

bofstream &operator<<(bofstream &output, const char &d)
{ output.put(d); return output; }

bofstream &operator<<(bofstream &output, const signed char &d)
{ output.put((char)d); return output; }

bofstream &operator<<(bofstream &output, const unsigned char &d)
{ output.put((char)d); return output; }

bofstream &operator<<(bofstream &output, const short int &d)
{ return output.templated_write(d); }

bofstream &operator<<(bofstream &output, const unsigned short int &d)
{ return output.templated_write(d); }

bofstream &operator<<(bofstream &output, const int &d)
{ return output.templated_write(d); }

bofstream &operator<<(bofstream &output, const unsigned int &d)
{ return output.templated_write(d); }

bofstream &operator<<(bofstream &output, const long int &d)
{ return output.templated_write(d); }

bofstream &operator<<(bofstream &output, const unsigned long int &d)
{ return output.templated_write(d); }

bofstream &operator<<(bofstream &output, const float &d)
{ return output.templated_write(d); }

bofstream &operator<<(bofstream &output, const double &d)
{ return output.templated_write(d); }

