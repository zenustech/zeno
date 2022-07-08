#pragma once

#include <cassert>
#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>

#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>


#include <type_traits>

namespace zeno {
    
    // some supported cell types
    constexpr int VTK_VERTEEX = 1;
    constexpr int VTK_TRIANGLE = 5;
    constexpr int VTK_TRIANGLE_STRIP = 6;
    constexpr int VTK_POLYGON = 7;
    constexpr int VTK_PIXEL = 8;
    constexpr int VTK_QUAD = 9;
    constexpr int VTK_TETRA = 10;
    constexpr int VTK_VOXEL = 11;
    constexpr int VTK_HEXAHEDRON = 12;
    constexpr int VTK_WEDGE = 13;
    constexpr int PYRAMID = 14;
    constexpr int VTK_PENTAGONAL_PRISM = 15;
    constexpr int VTK_HEXAGONAL_PRISM = 16;
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

    char* readline(char *buffer, FILE *infile, int *linenumber)
    {
        char *result;
        // Search for a non-empty line.
        do {
            result = fgets(buffer, INPUTLINESIZE - 1, infile);
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

    char* find_next_numeric(char *string){
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

    char* find_next_numeric(char* seek,char line[INPUTLINESIZE],FILE* infile,int *linenumber){
        while(true){
            seek = find_next_numeric(seek);
            if(*seek == '\0'){
                seek = readline(line,infile,linenumber);
                if(seek == NULL){
                    return NULL;
                }
                continue;
            }
            break;
        }
        return seek;
    }

    bool parsing_verts_coord(FILE *fp,zeno::AttrVector<zeno::vec3f>& pos,int numberofpoints,int& line_count){
        char *bufferp;
        char buffer[INPUTLINESIZE];

        pos.resize(numberofpoints);
        auto& verts = pos.values;
        int nm_points_read = 0;
        bufferp = readline(buffer,fp,&line_count);
        if(bufferp == NULL){
            printf("reach unexpected end of file on line %d\n",line_count);
            fclose(fp);
            return false;
        }
        while(nm_points_read < numberofpoints){
            for(int j = 0;j != 3;++j){
                if(*bufferp == '\0') {
                    printf("syntax error reading vertex coords on line %d while reading coords\n",line_count);
                    fclose(fp);
                    return false;
                }
                bufferp = find_next_numeric(bufferp,buffer,fp,&line_count);
                if(bufferp == NULL) {
                    printf("reach unexpected end of file on line %d\n",line_count);
                    fclose(fp);
                    return false;
                }
                verts[nm_points_read][j] = (float)strtod(bufferp,&bufferp);
            }
            nm_points_read++;
        }
        return true;
    }

    template<int CELL_SIZE>
    bool parsing_cells_topology(FILE* fp,typename zeno::AttrVector<zeno::vec<CELL_SIZE,int>>& cells_attrv,int numberofcells,int &line_count) {
        char *bufferp;
        char buffer[INPUTLINESIZE];

        cells_attrv.resize(numberofcells);
        auto& cells = cells_attrv.values;

        int nm_cells_read = 0;

        while(nm_cells_read < numberofcells){
            bufferp = readline(buffer,fp,&line_count);
            if(bufferp == NULL) {
                printf("reach unexpected end of file on line %d\n",line_count);
                fclose(fp);
                return false;
            }
            int nn = strtol(bufferp,&bufferp,0);
            if(nn != CELL_SIZE){
                printf("the size of cell(%d) does not match the target memory type(%d)\n",nn,CELL_SIZE);
                fclose(fp);
                return false;
            }
            for(int j = 0;j != CELL_SIZE;++j){
                if(*bufferp == '\0') {
                    printf("syntax error reading vertex coords on line %d parsing cells(%d)\n",line_count,CELL_SIZE);
                    fclose(fp);
                    return false;
                }
                bufferp = find_next_numeric(bufferp,buffer,fp,&line_count);
                cells[nm_cells_read][j] = (int)strtol(bufferp,&bufferp,0);
            }
        }   
        return true;
    }

    bool parsing_attribs(FILE *fp,typename std::vector<zeno::vec3f>& attr,int rows,int cols,int& line_count){
        char *bufferp;
        char buffer[INPUTLINESIZE];

        bufferp = readline(buffer,fp,&line_count);
        if(bufferp == NULL){
            printf("reach unexpected end of file on line %d\n",line_count);
            fclose(fp);
            return false;
        }
        int nm_field_read = 0;
        while(nm_field_read < rows){
            // if(cols == 1){
            //     bufferp = find_next_numeric(bufferp,buffer,fp,&line_count);
            //     attr[nm_field_read] = (T)strtod(bufferp,&bufferp);
            // }else{ // vector
                for(int i = 0;i != 3;++i){
                    bufferp = find_next_numeric(bufferp,buffer,fp,&line_count);
                    attr[nm_field_read][i] = (float)strtod(bufferp,&bufferp);
                }
            // }
            nm_field_read++;
        }
    }

    template<typename T>
    bool parsing_attributes_data(FILE *fp,zeno::AttrVector<T>& attrv,int& line_count) {
        char *bufferp;
        char buffer[INPUTLINESIZE];
        char id[256],dummy_str[64],data_name[64],array_name[64];
        int dummy;

        int buffer_size = attrv.size();

        while((bufferp = readline(buffer,fp,&line_count)) != NULL) {
            if(strlen(bufferp) == 0) continue;
            if(bufferp[0] == '#' || bufferp[0]=='\n' || bufferp[0] == 10 || bufferp[0] == 13 || bufferp[0] == 32) continue;   

            sscanf(bufferp,"%s",id);
            // currently we don't support lookup table
            // if(!strcmp(id,"SCALERS")){
                // int numberofpoints = 0;
                // sscanf(line,"%s %d %s",id,numberofpoints,dummy_str);
                // parsing_verts_coord(fp,prim,numberofpoints,line_count);
                // continue;
            // }

            if(!strcmp(id,"COLOR_SCALARS")){    
                int numberofchannels = 0;
                sscanf(bufferp,"%s %s %d",id,dummy_str,&numberofchannels);
                if(numberofchannels != 3){
                    printf("currently only rgb color is supported\n");
                    fclose(fp);
                    return false;
                }
                auto& data = attrv.add_attr("clr",zeno::vec3f{0,0,0});
                parsing_attribs(fp,data,buffer_size,3,line_count);
                continue;
            }            
            if(!strcmp(id,"VECTORS") || !strcmp(id,"NORMALS")) {
                sscanf(bufferp,"%s %s %s",id,data_name,dummy_str);
                auto& data = attrv.add_attr(data_name,zeno::vec3f{1,0,0});
                parsing_attribs(fp,data,buffer_size,3,line_count);
                continue;
            }
            if(!strcmp(id,"TEXTURE_COORDINATES")) {
                int tex_dim;
                sscanf(bufferp,"%s %s %d %s",id,data_name,&tex_dim,dummy_str);
                if(tex_dim != 2 && tex_dim != 3){
                    printf("only 2d and 3d texture are supported\n");
                    fclose(fp);
                    return false;
                }
                auto& data = attrv.add_attr(data_name,zeno::vec3f{0,0,0});
                parsing_attribs(fp,data,buffer_size,tex_dim,line_count);
                continue;                
            }
            if(!strcmp(id,"TENSORS")) {
                printf("the tensor data is not currently supported, ignore it\n");
                continue;
            }
            if(!strcmp(id,"FIELD")) {
                int nm_arrays = 0;
                sscanf(bufferp,"%s %s %d",id,data_name,&nm_arrays);
                for(int array_id = 0;array_id != nm_arrays;++array_id){
                    int nm_components,nm_tuples;
                    bufferp = readline(buffer,fp,&line_count);
                    sscanf(bufferp,"%s %d %d %s",array_name,&nm_components,&nm_tuples,dummy_str);

                    if(nm_tuples != buffer_size){
                        printf("the number of tuples[%d] in the field(%s) should match the number of cells[%d]\n",
                            nm_tuples,array_name,buffer_size);
                        fclose(fp);
                        return false;
                    }

                    if(nm_components < 1 && nm_components > 3){
                        printf("invalid nm_components(%d) in the field(%s)\n",nm_components,array_name);
                        fclose(fp);
                        return false;
                    }
                    // for nm_channels == 1,2,3, we use unified 3d vector to store the field data.
                    auto& data = attrv.add_attr(array_name,zeno::vec3f{0,0,0});
                    parsing_attribs(fp,data,buffer_size,nm_components,line_count);
                    continue;
                }
            }

            if(!strcmp(id,"POINT_DATA")){
                printf("detected point-wise data while parsing cell data\n");
                fclose(fp);
                return false;
            }
        }
    }

    // define the dataset type

    // DATASET STRUCTURED_POINTS
    // DIMENSIONS nx ny nz
    // ORIGIN x y z
    // SPACING sx sy yz
    constexpr char structure_points[] = "STRUCTURED_POINTS";
    bool read_structure_points(FILE *fp,std::shared_ptr<zeno::PrimitiveObject>& prim,int& line_count){
        // char *bufferp;
        // char line[INPUTLINESIZE];
    }

    // DATASET STRUCTURED_GRID
    // DIMENSIONS nx ny nz
    // POINTS n dataType
    // p0x p0y p0z
    // p1x p1y p1z
    // ...
    // p(n-1)x p(n-1)y p(n-1)z
    constexpr char structured_grids[] = "STRUCTURE_GRIDS";
    bool read_structured_grid(FILE *fp,std::shared_ptr<zeno::PrimitiveObject>& prim,int& line_count){
        // char *bufferp;
        // char line[INPUTLINESIZE];
    }

    // DATASET RECTILINEAR_GRID
    // DIMENSIONS nx ny nz
    // X_COORDINATES nx dataType
    // x0 x1 ... x(nx-1)
    // Y_COORDINATES ny dataType
    // y0 y1 ... y(ny-1)
    // Z_COORDINATES nz dataType
    // z0 z1 ... z(nz-1)
    constexpr char rectilinear_grid[] = "RECTILINEAR_GRID";
    bool read_rectilinear_grid(FILE *fp,std::shared_ptr<zeno::PrimitiveObject>& prim,int& line_count){
        // char *bufferp;
        // char line[INPUTLINESIZE];
    }

    // DATASET POLYDATA
    // POINTS n dataType
    // p0x p0y p0z
    // p1x p1y p1z
    // ...
    // p(n-1)x p(n-1)y p(n-1)z

    // VERTICES n size
    // numPoints0, i0, j0, k0, ...
    // numPoints1, i1, j1, k1, ...
    // ...
    // numPointsn-1, in-1, jn-1, kn-1, ...

    // LINES n size
    // numPoints0, i0, j0, k0, ...
    // numPoints1, i1, j1, k1, ...
    // ...
    // numPointsn-1, in-1, jn-1, kn-1, ...

    // POLYGONS n size
    // numPoints0, i0, j0, k0, ...
    // numPoints1, i1, j1, k1, ...
    // ...    //         if(strlen(line) == 0) continue;
    //         if(line[0] == '#' || line[0]=='\n' || line[0] == 10 || line[0] == 13 || line[0] == 32) continue;
    constexpr char polydata[] = "POLYDATA";
    bool read_polydata(FILE *fp,std::shared_ptr<zeno::PrimitiveObject>& prim,int& line_count){
        char *bufferp;
        char line[INPUTLINESIZE];
        char id[256],dummy_str[64];
        int dummy;

        while((bufferp = readline(line,fp,&line_count)) != NULL) {
            if(strlen(line) == 0) continue;
            if(line[0] == '#' || line[0]=='\n' || line[0] == 10 || line[0] == 13 || line[0] == 32) continue;   

            sscanf(line,"%s",id);
            // reading the points
            if(!strcmp(id,"POINTS")){
                int numberofpoints = 0;
                sscanf(line,"%s %d %s",id,numberofpoints,dummy_str);
                auto& verts = prim->verts;
                if(!parsing_verts_coord(fp,verts,numberofpoints,line_count))
                    return false;
                continue;
            }
            // skip handling the verts topology
            // handle lines topology
            if(!strcmp(id,"LINES")){
                int numberoflines = 0;
                sscanf(line,"%s %d %d",id,&numberoflines,&dummy);
                parsing_cells_topology<2>(fp,prim->lines,numberoflines,line_count);
                continue;
            }

            if(!strcmp(id,"POLYGONS")) {
                int numberofpolys = 0;
                sscanf(line,"%s %d %d",id,&numberofpolys,&dummy);
                parsing_cells_topology<3>(fp,prim->tris,numberofpolys,line_count);
                continue;
            }

            // parsing the cell data
            if(!strcmp(id,"CELL_DATA")){
                int numberofcells = 0;
                sscanf(line,"%s %d",id,numberofcells);
                parsing_attributes_data(fp,prim->tris,line_count);
                continue;
            }
            // parsing the points data
            if(!strcmp(id,"POINT_DATA")){
                int numberofpoints = 0;
                sscanf(line,"%s %d",id,numberofpoints);
                parsing_attributes_data(fp,prim->verts,line_count);
                continue;
            }
        }

        return true;
    }

    // DATASET UNSTRUCTURED_GRID
    // POINTS n dataType
    // p0x p0y p0z
    // p1x p1y p1z
    // ...
    // p(n-1)x p(n-1)y p(n-1)z

    // CELLS n size
    // numPoints0, i0, j0, k0, ...
    // numPoints1, i1, j1, k1, ...
    // numPoints2, i2, j2, k2, ...
    // ...
    // numPointsn-1, in-1, jn-1, kn-1, ...

    // CELL_TYPES n
    // type0
    // type1
    // type2
    // ...
    // typen-1
    constexpr char unstructured_grid[] = "UNSTRUCTURED_GRID";
    bool read_unstructured_grid(FILE *fp,std::shared_ptr<zeno::PrimitiveObject>& prim,int& line_count) {
        char *bufferp;
        char line[INPUTLINESIZE];
        char id[256],dummy_str[64];
        int dummy;

        while((bufferp = readline(line,fp,&line_count)) != NULL) {
            if(strlen(line) == 0) continue;
            if(line[0] == '#' || line[0]=='\n' || line[0] == 10 || line[0] == 13 || line[0] == 32) continue;   
            sscanf(line,"%s",id);
            // reading the points
            if(!strcmp(id,"POINTS")){
                int numberofpoints = 0;
                sscanf(line,"%s %d %s",id,numberofpoints,dummy_str);
                parsing_verts_coord(fp,prim->verts,numberofpoints,line_count);
                continue;
            }
            if(!strcmp(id,"CELLS")){
                int numberofcells = 0;
                sscanf(line,"%s %d %d",id,&numberofcells,&dummy);
                parsing_cells_topology<4>(fp,prim->quads,numberofcells,line_count);
                continue;
            }            
            if(!strcmp(id,"CELL_TYPES")){
                int numberofcells = 0;
                sscanf(line,"%s %d",&numberofcells);
                if(numberofcells > 0){
                    bufferp = readline(line,fp,&line_count);
                    int type = strtoll(bufferp,&bufferp,0);
                    if(type != VTK_TETRA){
                        printf("non-tetra cell detected on line %d parsing cell types\n",line_count);
                        fclose(fp);
                        return false;
                    }
                }
                continue;
            }

            // parsing the cell data
            if(!strcmp(id,"CELL_DATA")){
                int numberofcells = 0;
                sscanf(line,"%s %d",id,numberofcells);
                parsing_attributes_data(fp,prim->tris,line_count);
                continue;
            }
            // parsing the points data
            if(!strcmp(id,"POINT_DATA")){
                int numberofpoints = 0;
                sscanf(line,"%s %d",id,numberofpoints);
                parsing_attributes_data(fp,prim->verts,line_count);
                continue;
            }
        }
    }


    template <typename T,typename Tn,int SPACE_DIM,int SIMPLEX_SIZE>
    bool load_vtk_data(const std::string& file,std::shared_ptr<zeno::PrimitiveObject>& prim,int verbose = 0) {
        static_assert(SPACE_DIM == 3,"only 3d model is currently supported");
        static_assert(SIMPLEX_SIZE == 4 || SIMPLEX_SIZE == 3,"only tet and tri mesh is currently supported");

        // auto &X = mesh.nodes;
        // auto &eles = mesh.elms;

        auto infilename = file.c_str();
        FILE *fp;
        if (!(fp = fopen(file.c_str(), "r"))) {
            printf("Error:  Unable to open file %s\n", infilename);
            return false;
        }

        char mode[128],id[256],fmt[64],dummy[64];
        int line_count;
        char *bufferp;
        char line[INPUTLINESIZE];

        while((bufferp = readline(line,fp,&line_count)) != NULL) {
            if(strlen(line) == 0) continue;
            if(line[0] == '#' || line[0]=='\n' || line[0] == 10 || line[0] == 13 || line[0] == 32) continue;            
            sscanf(line,"%s",id);
            
            // currently the binary format is not supported
            if(!strcmp(id,"BINARY")){
                sscanf(line,"%s",mode);
                printf("the binary file is not currently supported");
                fclose(fp);
                return false;
            }
            if(!strcmp(id,"DATASET")){
                sscanf(line,"%s %s",id,fmt);
                if(!strcmp(fmt,structure_points))
                    return read_structure_points(fp,prim,line_count);
                else if(!strcmp(fmt,structured_grids))
                    return read_structured_grid(fp,prim,line_count);
                else if(!strcmp(fmt,rectilinear_grid))
                    return read_rectilinear_grid(fp,prim,line_count);
                else if(!strcmp(fmt,polydata))
                    return read_polydata(fp,prim,line_count);
                else if(!strcmp(fmt,unstructured_grid))
                    return read_unstructured_grid(fp,prim,line_count);
                else{
                    printf("unrecoginized dataset type(%s) detected at %d line\n",fmt,line);
                    fclose(fp);
                    return false;
                }
            }

        }

    }

}