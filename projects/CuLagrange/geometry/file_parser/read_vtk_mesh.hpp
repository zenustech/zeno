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
#include "vtk_types.hpp"

namespace zeno {
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

    bool test_is_big_endian(){
        short word = 0x4321;
        if((*(char *)& word) != 0x21)
            return true;
        else 
            return false;
    }

    char* readline(char *buffer, FILE *infile, int *linenumber,int verbose = 0){
        char *result;
        // Search for a non-empty line.
        do {
            result = fgets(buffer, INPUTLINESIZE - 1, infile);
            // if(*linenumber < 100)
            //     printf("read %p %d %p %d\n",buffer,*linenumber,result,verbose);
            if (linenumber) (*linenumber)++;
            if (!result) {
                // if(verbose > 0 && *linenumber < 100)
                //     printf("return null\n");
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
        // while(true){
            seek = find_next_numeric(seek);
            if(*seek == '\0'){
                seek = readline(line,infile,linenumber);
                // printf("read new lines : %s\n",seek);
                if(seek == NULL){
                    return NULL;
                }
                // continue;
            }
            // break;
        // }
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

        // printf("read line : %s\n",bufferp);

        while(nm_points_read < numberofpoints){
            // printf("reading point<%d> : ",nm_points_read);
            for(int j = 0;j != 3;++j){
                if(*bufferp == '\0') {
                    printf("syntax error reading vertex coords on line %d while reading coords\n",line_count);
                    fclose(fp);
                    return false;
                }
                verts[nm_points_read][j] = (float)strtod(bufferp,&bufferp);
                // printf("%f\t",verts[nm_points_read][j]);
                // if(j != 2)
                if(j != 2 || nm_points_read != (numberofpoints - 1))
                    bufferp = find_next_numeric(bufferp,buffer,fp,&line_count);

            }
            // printf("\n");
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

        // printf("numberofcells : %d\n",numberofcells);

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
                printf("line : %s\n",bufferp);
                fclose(fp);
                return false;
            }
            // printf("reading cell<%d> : ",nm_cells_read);
            for(int j = 0;j != CELL_SIZE;++j){
                if(*bufferp == '\0') {
                    printf("syntax error reading vertex coords on line %d parsing cells(%d)\n",line_count,CELL_SIZE);
                    fclose(fp);
                    return false;
                }
                bufferp = find_next_numeric(bufferp);// skip the header line idx, different index packs in different lines
                cells[nm_cells_read][j] = (int)strtol(bufferp,&bufferp,0);
                // printf("%d\t",cells[nm_cells_read][j]);
            }
            // printf("\n");
            nm_cells_read++;
        }   
        return true;
    }

    bool parsing_attribs(FILE *fp,typename std::vector<zeno::vec3f>& attr,int rows,int cols,int& line_count){
        char *bufferp;
        char buffer[INPUTLINESIZE];

        // printf("parsing attribs \n");

        bufferp = readline(buffer,fp,&line_count);
        if(bufferp == NULL){
            printf("reach unexpected end of file on line %d\n",line_count);
            fclose(fp);
            return false;
        }
        // if(attr.size() != rows) {
            // printf("the buffer size %d does not match the input size %d and cols %d\n",attr.size(),rows,cols);
        // }

        int nm_field_read = 0;
        while(nm_field_read < rows){
            // if(cols == 1){
            //     bufferp = find_next_numeric(bufferp,buffer,fp,&line_count);
            //     attr[nm_field_read] = (T)strtod(bufferp,&bufferp);
            // }else{ // vector
                for(int i = 0;i != cols;++i){
                    attr[nm_field_read][i] = (float)strtod(bufferp,&bufferp);
                    // printf("read in<%d,%d,%d> : %f\n",nm_field_read,rows,i,attr[nm_field_read][i]);
                    bufferp = find_next_numeric(bufferp,buffer,fp,&line_count);
                }
                // printf("read lines : <%d %d> %d \n",nm_field_read,rows,nm_field_read < rows);
            // }
            nm_field_read++;
        }

        return true;
    }

    bool parsing_attribs(FILE *fp,typename std::vector<float>& attr,int rows,int& line_count){
        char *bufferp;
        char buffer[INPUTLINESIZE];

        // printf("parsing attribs \n");

        bufferp = readline(buffer,fp,&line_count);
        if(bufferp == NULL){
            printf("reach unexpected end of file on line %d\n",line_count);
            fclose(fp);
            return false;
        }
        // if(attr.size() != rows) {
            // printf("the buffer size %d does not match the input size %d and cols %d\n",attr.size(),rows,cols);
        // }

        int nm_field_read = 0;
        while(nm_field_read < rows){
            attr[nm_field_read] = (float)strtod(bufferp,&bufferp);
            bufferp = find_next_numeric(bufferp,buffer,fp,&line_count);
            nm_field_read++;
        }

        return true;
    }


    template<typename AttrVec>
    bool parsing_attributes_data(FILE *fp,std::shared_ptr<zeno::PrimitiveObject>& prim,AttrVec& attrv,int& line_count,int cell_type) {
        char *bufferp;
        char buffer[INPUTLINESIZE];
        char id[256],dummy_str[64],data_name[64],array_name[64],lookup_dummy[64];
        int dummy;

        // using EleIds = std::variant<std::monostate, AttrVector<vec3f>&, AttrVector<vec2i>&, AttrVector<vec3i>&, AttrVector<vec4i>&>;
        // EleIds attrv;
        // attrv = (!cell_wise) ? EleIds(prim->verts) : ((cell_type == 2) ? EleIds(prim->lines) : ((cell_type) == 3 ? EleIds(prim->tris) : EleIds(prim->quads)));

        // zs::match([]() {})(attrv);

        int buffer_size = attrv.size();

        while((bufferp = readline(buffer,fp,&line_count)) != NULL) {
            if(strlen(bufferp) == 0) continue;
            if(bufferp[0] == '#' || bufferp[0]=='\n' || bufferp[0] == 10 || bufferp[0] == 13 || bufferp[0] == 32) continue;   

            sscanf(bufferp,"%s",id);
            // currently we don't support lookup table
            if(!strcmp(id,"SCALARS")){
                // int numberofpoints = 0;
                sscanf(bufferp,"%s %s %s",id,data_name,dummy_str);
                printf("parsing scalers: %s\n",data_name);
                // skip the defination of lookup table
                bufferp = readline(buffer,fp,&line_count);
                sscanf(bufferp,"%s %s",lookup_dummy,dummy_str);
                if(!strcmp(lookup_dummy,"LOOKUP_TABLE"))
                    printf("the SCALERS defination should follow by LOOKUP_TABLE defination\n");
                auto& data = attrv.add_attr(data_name,(float)0.0);
                printf("parsing scalers\n");
                parsing_attribs(fp,data,buffer_size,line_count);
                continue;
            }
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
                    printf("array_name : %s | nm_components  %d | nm_tuples : %d | type : %s\n",
                        array_name,nm_components,nm_tuples,dummy_str);

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
                    if(nm_components > 1){
                        auto& data = attrv.add_attr(array_name,zeno::vec3f{0,0,0});
                        printf("add channel  %s <%d,%d>\n",array_name,data.size(),nm_components);
                        parsing_attribs(fp,data,buffer_size,nm_components,line_count);
                    }else{
                        auto& data = attrv.add_attr(array_name,0.f);
                        printf("add channel  %s <%d,%d>\n",array_name,data.size(),nm_components);
                        parsing_attribs(fp,data,buffer_size,line_count);
                    }
                    continue;
                }
            }

            // return true;
            if(!strcmp(id,"POINT_DATA")){
                printf("parsing point-wise attributes\n");
                return parsing_attributes_data(fp,prim,prim->verts,line_count,cell_type);
            }
            if(!strcmp(id,"CELL_DATA")){
                printf("parsing cell-wise attributes\n");
                if(cell_type == 2)
                    return parsing_attributes_data(fp,prim,prim->lines,line_count,cell_type);
                if(cell_type == 3)
                    return parsing_attributes_data(fp,prim,prim->tris,line_count,cell_type);
                if(cell_type == 4)
                    return parsing_attributes_data(fp,prim,prim->quads,line_count,cell_type);
            }
        }

        return true;
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
        return false;
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
        return false;
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
        return false;
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
            printf("tag : %s\n",id);
            // reading the points
            if(!strcmp(id,"POINTS")){
                int numberofpoints = 0;
                printf("reading points\n");
                sscanf(line,"%s %d %s",id,&numberofpoints,dummy_str);
                auto& verts = prim->verts;
                if(!parsing_verts_coord(fp,verts,numberofpoints,line_count))
                    return false;
                printf("finish reading points\n");
                continue;
            }
            // skip handling the verts topology
            // handle lines topology
            if(!strcmp(id,"LINES")){
                int numberoflines = 0;
                printf("reading lines\n");
                sscanf(line,"%s %d %d",id,&numberoflines,&dummy);
                if(!parsing_cells_topology<2>(fp,prim->lines,numberoflines,line_count))
                    return false;
                printf("finish reading lines\n");
                continue;
            }

            if(!strcmp(id,"POLYGONS")) {
                int numberofpolys = 0;

                sscanf(line,"%s %d %d",id,&numberofpolys,&dummy);
                printf("readling polygons %s %d %d\n",id,numberofpolys,dummy);
                if(!parsing_cells_topology<3>(fp,prim->tris,numberofpolys,line_count))
                    return false;
                printf("finish reading polygons\n");
                continue;
            }
            // parsing the cell data
            if(!strcmp(id,"CELL_DATA")){
                int numberofpolys = 0;
                printf("parsing cell-wise attributes\n");
                sscanf(line,"%s %d",id,&numberofpolys);
                if(!parsing_attributes_data(fp,prim,prim->tris,line_count,3))
                    return false;
                printf("finish parsing cell-wise attributes\n");
                continue;
            }
            // parsing the points data
            if(!strcmp(id,"POINT_DATA")){
                int numberofpoints = 0;
                printf("parsing points data\n");
                sscanf(line,"%s %d",id,&numberofpoints);
                if(!parsing_attributes_data(fp,prim,prim->verts,line_count,3))
                    return false;
                printf("fnish parsing points data\n");
                continue;
            }
        }

        printf("finish reading polydata\n");
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

        while((bufferp = readline(line,fp,&line_count,1)) != nullptr) {
            // if(line_count < 100)
            
            // if(bufferp == nullptr){
            //     printf("reach the end of file\n");
            //     break;
            // }

            // if(line_count < 100)
            //     printf("read lines[%d] %d\n",line_count,(bufferp == (char*)NULL));

            if(strlen(line) == 0) continue;
            if(line[0] == '#' || line[0]=='\n' || line[0] == 10 || line[0] == 13 || line[0] == 32) continue;   
            sscanf(line,"%s",id);
            // reading the points
            if(!strcmp(id,"POINTS")){
                printf("reading points\n");
                int numberofpoints = 0;
                sscanf(line,"%s %d %s",id,&numberofpoints,dummy_str);
                printf("number of points %d\n",numberofpoints);
                parsing_verts_coord(fp,prim->verts,numberofpoints,line_count);
                continue;
            }
            if(!strcmp(id,"CELLS")){
                printf("reading cells\n");
                int numberofcells = 0;
                sscanf(line,"%s %d %d",id,&numberofcells,&dummy);
                parsing_cells_topology<4>(fp,prim->quads,numberofcells,line_count);
                continue;
            }            
            if(!strcmp(id,"CELL_TYPES")){
                printf("reading cell types\n");
                int numberofcells = 0;
                sscanf(line,"%s %d",id,&numberofcells);
                printf("number of cell types : %d\n",numberofcells);
                bufferp = readline(line,fp,&line_count);
                if(numberofcells > 0){
                    int type = strtol(bufferp,&bufferp,0);
                    if(type != VTK_TETRA){
                        printf("non-tetra cell detected on line %d parsing cell types\n",line_count);
                        fclose(fp);
                        return false;
                    }
                    bufferp = find_next_numeric(bufferp,line,fp,&line_count);
                }
                printf("finish cell type check\n");
                continue;
            }

            // parsing the cell data
            if(!strcmp(id,"CELL_DATA")){
                printf("reading cell data\n");
                int numberofcells = 0;
                sscanf(line,"%s %d",id,&numberofcells);
                parsing_attributes_data(fp,prim,prim->quads,line_count,4);
                continue;
            }
            // parsing the points data
            if(!strcmp(id,"POINT_DATA")){
                printf("reading point data\n");
                int numberofpoints = 0;
                sscanf(line,"%s %d",id,&numberofpoints);
                parsing_attributes_data(fp,prim,prim->verts,line_count,4);
                continue;
            }

            // printf("read lines[%d] %s\n",line_count,line);
            // if(line_count > 100)
        }

        fclose(fp);
        return true;
    }


    bool load_vtk_data(const std::string& file,std::shared_ptr<zeno::PrimitiveObject>& prim,int verbose = 0) {
        // static_assert(SPACE_DIM == 3,"only 3d model is currently supported");
        // static_assert(SIMPLEX_SIZE == 4 || SIMPLEX_SIZE == 3,"only tet and tri mesh is currently supported");

        // auto &X = mesh.nodes;
        // auto &eles = mesh.elms;

        auto infilename = file.c_str();
        FILE *fp;
        if (!(fp = fopen(file.c_str(), "r"))) {
            printf("Error:  Unable to open file %s\n", infilename);
            return false;
        }

        char mode[128],id[256],fmt[64],dummy[64];
        int line_count = 0;
        char *bufferp;
        char line[INPUTLINESIZE];

        bool ret;

        while((bufferp = readline(line,fp,&line_count,1)) != (char*)NULL) {
            if(strlen(bufferp) == 0){ 
                printf("zeno-length bufferp\n");
                continue;
            }

            if(bufferp[0] == '#' || bufferp[0]=='\n' || bufferp[0] == 10 || bufferp[0] == 13 || bufferp[0] == 32) continue;            
            sscanf(bufferp,"%s",id);
            
            // currently the binary format is not supported
            if(!strcmp(id,"BINARY")){
                sscanf(bufferp,"%s",mode);
                printf("the binary file is not currently supported");
                fclose(fp);
                ret = false;
                break;
            }

            if(!strcmp(id,"DATASET")){
                sscanf(bufferp,"%s %s",id,fmt);
                if(!strcmp(fmt,structure_points)){
                    printf("reading structure points\n");
                    ret = read_structure_points(fp,prim,line_count);
                }
                else if(!strcmp(fmt,structured_grids)){
                    printf("reading structure grids\n");
                    ret = read_structured_grid(fp,prim,line_count);
                }
                else if(!strcmp(fmt,rectilinear_grid)){
                    printf("reading rectilinear grid\n");
                    ret = read_rectilinear_grid(fp,prim,line_count);
                }
                else if(!strcmp(fmt,polydata)){
                    printf("reading polydata\n");
                    ret = read_polydata(fp,prim,line_count);
                }
                else if(!strcmp(fmt,unstructured_grid)){
                    printf("reading unstructured grid at line %d\n",line_count);
                    ret = read_unstructured_grid(fp,prim,line_count);
                }else{
                    printf("unrecoginized dataset type(%s) detected at %d line\n",fmt,line);
                    fclose(fp);
                    ret = false;
                }
                break;
            }

        }

        return ret;
    }

}