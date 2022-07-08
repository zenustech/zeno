#pragma
#include <cassert>
#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "zensim/geometry/Mesh.hpp"
#include "zensim/math/Vec.h"
#include "zensim/types/Optional.h"

namespace zs {
    
    // some supported cell types
    // constexpr int VTK_VERTEEX = 1;
    // constexpr int VTK_POLY_VERTEX = 2;
    // constexpr int VTK_LINE = 3;
    // constexpr int VTK_POLY_LINE = 4;
    constexpr int VTK_TRIANGLE = 5;
    // constexpr int VTK_TRIANGLE_STRIP = 6;
    constexpr int VTK_POLYGON = 7;
    // constexpr int VTK_PIXEL = 8;
    constexpr int VTK_QUAD = 9;
    constexpr int VTK_TETRA = 10;
    constexpr int VTK_VOXEL = 11;
    constexpr int VTK_HEXAHEDRON = 12;
    // constexpr int VTK_WEDGE = 13;
    // constexpr int PYRAMID = 14;
    // constexpr int VTK_PENTAGONAL_PRISM = 15;
    // constexpr int VTK_HEXAGONAL_PRISM = 16;
    // // some other vtk cell types

    constexpr int FILENAMESIZE = 1024;
    constexpr int INPUTLINESIZE = 2048;

    void swap_bytes(unsigned char* var, int size)
    {
        int i = 0;
        int j = size - 1;
        char c;

        while (i < j) {
            c = var[i]; var[i] = var[j]; var[j] = c;
            i++, j--;
        }
    }

    bool test_is_big_endian()
    {
        short word = 0x4321;
        if((*(char *)& word) != 0x21)
            return true;
        else 
            return false;
    }

    char* readline(char *string, FILE *infile, int *linenumber)
    {
        constexpr int FILENAMESIZE = 1024;
        constexpr int INPUTLINESIZE = 2048;
        char *result;

        // Search for a non-empty line.
        do {
            result = fgets(string, INPUTLINESIZE - 1, infile);
            if (linenumber) (*linenumber)++;
            if (result == (char *) NULL) {
            return (char *) NULL;
            }
            // Skip white spaces.
            while ((*result == ' ') || (*result == '\t')) result++;
            // If it's end of line, read another line and try again.
        } while ((*result == '\0') || (*result == '\r') || (*result == '\n'));
        return result;
    }


    char* findnextnumber(char *string){
        char *result;

        result = string;
        // Skip the current field.  Stop upon reaching whitespace or a comma.
        while ((*result != '\0') && (*result != '#') && (*result != ' ') && 
                (*result != '\t') && (*result != ',')) {
            result++;
        }
        // Now skip the whitespace and anything else that doesn't look like a
        //   number, a comment, or the end of a line. 
        while ((*result != '\0') && (*result != '#')
                && (*result != '.') && (*result != '+') && (*result != '-')
                && ((*result < '0') || (*result > '9'))) {
            result++;
        }
        // Check for a comment (prefixed with `#').
        if (*result == '#') {
            *result = '\0';
        }
        return result;
    }



    template <typename CELL_TYPE,typename T,typenname Tn>
    bool load_vtk(const std::string& file,Mesh<T,3,Tn,4>& mesh,int verbose = 0) {
        static_assert(CELL_TYPE == VTK_TRIANGLE || CELL_TYPE = VTK_TETRA,"only vtk_triangles and vtk tetrahedron are supported");
        using MESH = typename Mesh<T, 3, Tn, CELL_TYPE == VTK_TRIANGLE ? 3 : 4>;
        using Node = typename MESH::Node;
        using Elem = typename MESH::Elem;
        auto &X = mesh.nodes;
        auto &indices = mesh.elems;
        bool ImALittleEndian = !test_is_big_endian();

        FILE *fp;
        char line[INPUTLINESIZE];
        char mode[128],id[256],fmt[64];

        auto infilename = file.c_str();
        if (!(fp = fopen(file.c_str(), "r"))) {
            printf("Error:  Unable to open file %s\n", infilename);
            return false;
        }

        int first_number = 0;

        char *bufferp;
        int line_count = 0;

        int numberofpoints = 0;
        int smallestidx = 0;

        int ncells = 0;
        int dummy;

        while((bufferp = readline(line,fp,&line_count)) != NULL) {
            if(strlen(line) == 0) continue;
            if(line[0] == '#' || line[0]=='\n' || line[0] == 10 || line[0] == 13 || line[0] == 32) continue;
            // check the tag
            sscanf(line, "%s", id);
            // by default we use BINARY MODE
            if(!strcmp(id,"BINARY")) {
                printf("the binary file format is not currently supported\n");
                fclose(fp);
                return false;
            }
            // handling all the points
            if(!strcmp(id,"POINTS")) {
                sscanf(line,"%s %d %s",id,&nverts,fmt);
                if(nverts > 0) {
                    numberofpoints = nverts;
                    X.resize(nverts);
                    smallestidx = nverts + 1;
                }
                for(int i = 0;i < nverts;++i){
                    bufferp = readline(line,fp,&line_count);
                    if(bufferp == NULL) {
                        printf("Unexpected end of file on line %d in file %s\n",
                            line_count,infilename);
                        fclose(fp);
                        return false;
                    }
                    for(int j = 0;j < 3;++j){
                        if(*bufferp == '\0') {
                            printf("Syntax error reading vertex coords on line");
                            printf("%d in file %s\n",line_count,infilename);
                            fclose(fp);
                            return false;
                        }
                        X[i][j] = (T) strtod(bufferp,&bufferp);
                        bufferp = findnextnumber(bufferp);
                    }
                }
                continue;
            }
            // the polygons is also treated as a special form of cell, polygon, currently only triangulated surface mesh is supported
            if(!strcmp(id,"POLYGONS")) {
                if constexpr (CELL_TYPES != VTK_TRIANGLE){
                    printf("we only support triangle mesh reading polygon mode mesh\n");
                    fclose(fp);
                    return false;
                }
                sscanf(line,"%s %d %d",id,&nfaces,&dummy);
                if(nfaces > 0) {
                    faces.resize(nfaces);
                }
                for(int i = 0;i < nfaces;++i) {
                    bufferp = readline(line,fp,&line_count);
                    // the size of input polygon
                    int nn = (int) strtol(bufferp,&bufferp,0);
                    if(nn != 3){
                        printf("only triangle surface mesh is supported\n");
                        printf("%d in file %s\n",line_count,infilename);
                        fclose(fp);
                        return false;
                    }

                    if(*bufferp == '\0'){
                        printf("Systax error reading polygon indices on line ");
                        printf("%d in file %s\n",line_count,infilename);
                        fclose(fp);
                        return false;
                    }
                    for(int j = 0;j < 3;++j){
                        bufferp = findnextnumber(bufferp);
                        faces[i][j] = (int) strtol(bufferp,&bufferp,0);   
                    }
                    for(int j = 0;j < 3;++j)
                        smallestidx = smallestidx > faces[i][j] ? faces[i][j] : smallestidx;
                }
                fclose(fp);  
                if(smallestidx == 0){
                    firstnumber = 0;
                }else if(smallestidx == 1){
                    firstnumber = 1;
                }else {
                    print("A wrong smallest index (%d) was detected in file %s\n",smallestidx,infilename);
                    return false;
                }

                continue;
            }

            if(!strcmp(id,"CELL")){
                sscanf(line,"%s %d %d",id,ncells,dummy);
                if(ncells > 0){
                    indices.resize(ncells);
                }
                for(int i = 0;i < ncells;++i){
                    bufferp = readline(line,fp,&line_count);
                    int nn = strtol(bufferp,&bufferp,0);
                    if(nn != 4){
                        printf("only tetrahedron mesh is supported\n");
                        fclose(fp);
                        return false;
                    }
                    if(*bufferp == '\0'){
                        printf("Systax error reading cell indices on line ");
                        printf("%d in file %s\n",line_count,infilename);
                        fclose(fp);
                        return false;                        
                    }
                    for(int j = 0;j < 4;++j){
                        bufferp = findnextnumber(bufferp);
                        indices[i][j] = (int) strtol(bufferp,&bufferp,0);    
                    }
                }
            } 

            if(!strcmp(id,"CELL_TYPES")) {
                // check the cell types is actually tet not quad
                int _ncell;
                sscanf(line,"%s %d",id,_ncell);
                if(_ncell != ncell){
                    print("the number of cell and the number of cell types does not match\n");
                    return false;
                }

                for(int i = 0;i < ncells;++i) {
                    bufferp = readling(line,fp,&line_count);
                    int type_id = strtol(bufferp,&bufferp);
                    if(type_id != CELL_TYPE){
                        printf("invalid cell type detected at %d line\n",line_count);
                        return false;
                    }

                }
            }

            if(!strcmp(id,"CELL_DATA")){

            }

            if(!strcmp(id,"POINTS_DATA")){

            }



        } // while()

        return true;
    }
}