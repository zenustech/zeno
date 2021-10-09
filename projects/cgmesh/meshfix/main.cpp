#include "meshfix.h"
#include <cstring>
#include <cstdlib>

//#define DISCLAIMER

void usage()
{
 printf("\nMeshFix V1.0 - by Marco Attene\n------\n");
 printf("Usage: MeshFix meshfile [-a epsilon_angle] [-w] [-n]\n");
 printf("  Processes 'meshfile' and saves the result to 'meshfile_fixed.off'\n");
 printf("  By default, epsilon_angle is 0.\n  If specified, it must be in the range (0 - 2) degrees.\n");
 printf("  With '-w', the result is saved in VRML1.0 format instead of OFF.\n");
 printf("  With '-n', only the biggest input component is kept.\n");
 printf("  Accepted input formats are OFF, PLY and STL.\n  Other formats are supported only partially.\n");
 printf("  See http://jmeshlib.sourceforge.net for details on supported formats.\n");
 printf("\nIf MeshFix is used for research purposes, please cite the following paper:\n");
 printf("\n   M. Attene.\n   A lightweight approach to repairing digitized polygon meshes.\n   The Visual Computer, 2010. (c) Springer.\n");
 printf("\nHIT ENTER TO EXIT.\n");
 getchar();
 exit(0);
}

char *createFilename(const char *iname, const char *subext, const char *newextension)
{
 static char tname[2048];
 char *oname = (char *)malloc(strlen(iname)+strlen(subext)+strlen(newextension)+1);
 strcpy(tname, iname);
 for (int n=strlen(tname)-1; n>0; n--) if (tname[n]=='.') {tname[n] = '\0'; break;}
 sprintf(oname,"%s%s%s",tname,subext,newextension);
 return oname;
}

int main(int argc, char *argv[])
{
 char subext[128]="_fixed";
 JMesh::init();
 JMesh::app_name = "MeshFix";
 JMesh::app_version = "1.0";
 JMesh::app_year = "2010";
 JMesh::app_authors = "Marco Attene";
 JMesh::app_maillist = "attene@ge.imati.cnr.it";

 ExtTriMesh tin;

#ifdef DISCLAIMER
 printf("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");
 printf("This software can be used ONLY with an explicit authorization of the author.\n");
 printf("If you do not have such an authorization, you must delete this software.\n");
 printf("In no event this version of MeshFix can be redistributed.\n");
 printf("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");
#endif

 if (argc < 2) usage();

 bool keep_all_components = true;
 const char *input_filename = argv[1];
 double epsilon_angle = 0.0;

 bool save_vrml = false;
 float par;
 for (int i=2; i<argc; i++)
 {
  if (i<argc-1) par = (float)atof(argv[i+1]); else par = 0;
  if      (!strcmp(argv[i], "-a"))
  {
   if (par < 0) JMesh::error("Epsilon angle must be > 0.\n");
   if (par > 2) JMesh::error("Epsilon angle must be < 2 degrees.\n");
   epsilon_angle = par;
    if (epsilon_angle)
    {
      JMesh::acos_tolerance = asin((M_PI*epsilon_angle)/180.0);
#ifdef MESHFIX_VERBOSE
      printf("Fixing asin tolerance to %e\n",JMesh::acos_tolerance);
#endif
    }
  }
  else if (!strcmp(argv[i], "-n")) keep_all_components = false;
  else if (!strcmp(argv[i], "-w")) save_vrml = true;
  else if (argv[i][0] == '-') JMesh::warning("%s - Unknown operation.\n",argv[i]);

  if (par) i++;
 }

#ifndef MESHFIX_VERBOSE
  JMesh::quiet = true;
#endif
 // The loader performs the conversion to a set of oriented manifolds
 if (tin.load(argv[1]) != 0) JMesh::error("Can't open file.\n");

 meshfix(epsilon_angle,keep_all_components,tin);

 char *fname = createFilename(argv[1], subext, (save_vrml)?(".wrl"):(".off"));
#ifdef MESHFIX_VERBOSE
 printf("Saving output mesh to '%s'\n",fname);
#endif
 if (save_vrml) tin.saveVRML1(fname); else tin.saveOFF(fname);

 return 0;
}



