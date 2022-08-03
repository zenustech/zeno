#pragma once 

#include <cassert>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

#include "../../Structures.hpp"
#include "../../Utils.hpp"

#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>


#include <type_traits>
#include "vtk_types.hpp"

// #include <zensim/types/Polymorphism.h>

// header

namespace zeno {
    bool write_verts_coord(FILE *fp,const std::vector<zeno::vec3f>& pos,bool use_double) {
        int numberofverts = pos.size();
        fprintf(fp,"POINTS %d %s\n",numberofverts,use_double ? "double" : "float");
        for(int i = 0;i < numberofverts;++i){
            if(use_double)
                fprintf(fp,"%lf %lf %lf\n",(double)pos[i][0],(double)pos[i][1],(double)pos[i][2]);
            else
                fprintf(fp,"%f %f %f\n",(float)pos[i][0],(float)pos[i][1],(float)pos[i][2]);
        }
        return true;
    } 

    template<int CELL_SIZE>
    bool write_cells_topology(FILE* fp,const std::vector<vec<CELL_SIZE,int>>& cells_topo){
        int numberofcells = cells_topo.size();
        fprintf(fp,"CELLS %d %d\n",numberofcells,numberofcells * (CELL_SIZE + 1));
        for(int i = 0;i < numberofcells;++i){
            fprintf(fp,"%d",CELL_SIZE);
            for(int j = 0;j < CELL_SIZE;++j)
                fprintf(fp," %d",cells_topo[i][j]);
            fprintf(fp,"\n");
        }

        return true;
    }

    bool write_vectors(FILE* fp,const std::string& data_name,const std::vector<zeno::vec3f>& vecs,bool use_double) {
        int numberofvecs = vecs.size();
        fprintf(fp,"VECTORS %s %s\n",data_name.c_str(),use_double ? "double" : "float");
        for(int i = 0;i < numberofvecs;++i){
            if(use_double)
                fprintf(fp,"%lf %lf %lf\n",(double)vecs[i][0],(double)vecs[i][1],(double)vecs[i][2]);
            else
                fprintf(fp,"%f %f %f\n",(float)vecs[i][0],(float)vecs[i][1],(float)vecs[i][2]);
        }

        return true;
    }

    bool write_normals(FILE* fp,const std::string& data_name,const std::vector<zeno::vec3f>& nrm,bool use_double) {
        int numberofnormals = nrm.size();
        fprintf(fp,"NORMALS %s %s\n",data_name.c_str(),use_double ? "double" : "float");
        for(int i = 0;i < numberofnormals;++i){
            if(use_double)
                fprintf(fp,"%lf %lf %lf\n",(double)nrm[i][0],(double)nrm[i][1],(double)nrm[i][2]);
            else
                fprintf(fp,"%f %f %f\n",(float)nrm[i][0],(float)nrm[i][1],(float)nrm[i][2]);
        }

        return true;
    }

    bool write_scalars(FILE *fp,const std::string& data_name,const std::vector<float>& scalars,bool use_double) {
        int numberofscalars = scalars.size();
        fprintf(fp,"SCALARS %s %s %d\n",data_name.c_str(),use_double ? "double" : "float",numberofscalars);
        fprintf(fp,"LOOKUP_TABLE %s_LOOKUP_TABLE\n",data_name.c_str());

        for(int i = 0;i < numberofscalars;++i){
            if(use_double)
                fprintf(fp,"%lf\n",(double)scalars[i]);
            else
                fprintf(fp,"%f\n",(float)scalars[i]);
        }

        return true;
    }

    // template<int COLOR_DIM = 3>
    // bool write_colors(FILE* fp,const std::string& data_name,const std::vector<zeno::vec<COLOR_DIM,float>>& clr) {
    //     int numberofcolors = clr.size();
    //     fprintf(fp,"COLORS_SCALARS %s %d\n",data_name.c_str(),numberofcolors);
    //     for(int i = 0;i < numberofcolors;++i){
    //         for(int j = 0;j < COLOR_DIM;++j){
    //             auto d = round(clr[i][j]*255.) % 256;
    //             fprintf(fp,"%d ",d);
    //         }
    //         fprintf(fp,"\n");
    //     }

    //     return true;
    // }

    template<int TEX_DIM = 2>
    bool write_tex_coords(FILE* fp,const std::string& data_name,const std::vector<zeno::vec<TEX_DIM,float>>& tex,bool use_double) {
        int numberoftexcoords = tex.size();
        fprintf(fp,"TEXTURE_COORDINATES %s %d %s\n",data_name.c_str(),TEX_DIM,use_double ? "double" : "float");
        for(int i = 0;i < numberoftexcoords;++i){
            for(int j = 0;j < TEX_DIM;++j){
                if(use_double)
                    fprintf(fp,"%lf ",(double)tex[i][j]);
                else
                    fprintf(fp,"%f ",(float)tex[i][j]);
            }
            fprintf(fp,"\n");
        }

        return true;
    }

    bool write_field_header(FILE* fp,const std::string& data_name,int num_arrays) {
        fprintf(fp,"FIELD %s %d\n",data_name.c_str(),num_arrays);
        return true;
    }

    template<int ARRAY_DIM>
    bool write_array(FILE* fp,const std::string& array_name,const std::vector<zeno::vec<ARRAY_DIM,float>>& array,bool use_double) {
        int numberofarrays = array.size();
        fprintf(fp,"%s %d %d %s\n",array_name.c_str(),ARRAY_DIM,numberofarrays,use_double ? "double" : "float");
        for(int i = 0;i < numberofarrays;++i){
            for(int j = 0;j < ARRAY_DIM;++j){
                if(use_double){
                    fprintf(fp,"%lf ",(double)array[i][j]);
                }else{
                    fprintf(fp,"%f ",(float)array[i][j]);
                }
            }
            fprintf(fp,"\n");
        }

        return true;
    }

    bool write_array(FILE* fp,const std::string& array_name,const std::vector<float>& array,bool use_double) {
        int numberofarrays = array.size();
        fprintf(fp,"%s %d %d %s\n",array_name.c_str(),1,numberofarrays,use_double ? "double" : "float");
        for(int i = 0;i < numberofarrays;++i){
            if(use_double){
                fprintf(fp,"%lf\n",(double)array[i]);
            }else{
                fprintf(fp,"%f\n",(float)array[i]);
            }
        }

        return true;
    }    

    template<int TYPE>
    bool write_cell_types(FILE* fp,int numberofcells) {
        fprintf(fp,"CELL_TYPES %d\n",numberofcells);
        for(int i = 0;i < numberofcells;++i)
            fprintf(fp,"%d\n",TYPE);
        return true;
    }

    bool return_and_close_file(FILE *fp,bool ret) {
        fclose(fp);
        return ret;
    }

    bool write_vtk_data(const std::string& file,std::shared_ptr<zeno::PrimitiveObject>& prim,
            bool out_customed_nodal_attributes,
            bool out_customed_cell_attributes,
            bool use_double = false,int verbose = 0) {
        auto outfilename = file.c_str();
        FILE* fp;
        if(!(fp = fopen(outfilename,"w"))) {
            printf("Error: Unable to open file %s for writing\n",outfilename);
            return false;
        }

    // const char* header = "# vtk DataFile Version 2.0\n"
    //                      "From zeno node vtk output module\n"
    //                      "ASCII\n"
    //                      "DATASET UNSTRUCTURED_GRID\n"   
        // write the headers
        fprintf(fp,"# vtk DataFile Version 2.0\n");
        fprintf(fp,"From zeno node vtk output module\n");
        fprintf(fp,"ASCII\n");
        fprintf(fp,"DATASET UNSTRUCTURED_GRID\n"  );

        if(!write_verts_coord(fp,prim->attr<zeno::vec3f>("pos"),use_double)){
            printf("Failed writing verts to %s\n",outfilename);
            return return_and_close_file(fp,false);
        }
        bool has_quads = prim->quads.size() > 0;
        bool has_tris = prim->tris.size() > 0;
        if(has_quads){
            bool success = write_cells_topology<4>(fp,prim->quads.values);
            if(!success){
                printf("Failed writing quad topos to %s\n",outfilename);
                return return_and_close_file(fp,false);
            }
            success = write_cell_types<VTK_TETRA>(fp,prim->quads.size());
            if(!success) {
                printf("Failed writing cell types to %s\n",outfilename);
                return return_and_close_file(fp,false);
            }
        }
        if(!has_quads && has_tris){
            bool success = write_cells_topology<3>(fp,prim->tris.values);
            if(!success){
                printf("Failed writing tris topos %s\n",outfilename);
                return return_and_close_file(fp,false);
            }
            success = write_cell_types<VTK_TRIANGLE>(fp,prim->quads.size());
            if(!success) {
                printf("Failed writing cell types to %s\n",outfilename);
                return return_and_close_file(fp,false);
            }            
        }
        if(!has_quads && !has_tris){
            printf("the primitive has no tris or tets topo\n");
            return return_and_close_file(fp,false);
        }


        // WRITING POINT DATA

        // if(prim->has_attr("nrm")){
        //     bool success = write_normals(fp,"nodal_normals",prim->attr<zeno::vec3f>("nrm"),use_double);
        //     if(!success){
        //         printf("Failed writing nodal normals to %s\n",outfilename);
        //         return return_and_close_file(fp,false);
        //     }
        // }
        // if(prim->has_attr("clr")){
        //     bool success = write_colors<3>(fp,"nodal_colors",prim->attr<zeno::vec3f>("clr"));
        //     if(!success){
        //         printf("Failed writing nodal colors to %s\n",outfilename);
        //         return return_and_close_file(fp,false);
        //     }
        // }

        if(out_customed_nodal_attributes) {
            int numberoffielddata = 0;
            for(auto &&[key,attr] : prim->verts.attrs) {
                if(key == "pos"/* || key == "nrm" || key == "clr"*/)
                    continue;
                numberoffielddata++;
                printf("detect nodal attribute : %s\n",key.c_str());
            }
            printf("number of customed nodal attributes %d \n",numberoffielddata);
            if(numberoffielddata > 0){
                fprintf(fp,"POINT_DATA %d\n",prim->size());
                write_field_header(fp,"nodal_field",numberoffielddata);
                for(auto &&[key,arr] : prim->verts.attrs) {
                    if(key == "pos"/* || key == "nrm" || key == "clr"*/)
                        continue;
                    const auto& k{key};
                    zs::match(
                        [&k,&fp,&use_double] (const std::vector<zeno::vec3f>& vals) {
                            write_array<3>(fp,k,vals,use_double);
                        },
                        [&k,&fp,&use_double] (const std::vector<zeno::vec2f>& vals) {
                            write_array<2>(fp,k,vals,use_double);
                        },
                        [&k,&fp,&use_double] (const std::vector<float>& vals) {
                            write_array(fp,k,vals,use_double);
                        },
                        [](...) {
                            throw std::runtime_error("what the heck is this type of attribute!");
                        })(arr);
                }
            }


        }

        // WRITING QUAD-WISE ATTRIBUTES
        if(has_quads){
            // if(prim->quads.has_attr("nrm")){
            //     bool success = write_normals(fp,"elm_normals",prim->quads.attr<zeno::vec3f>("nrm"),use_double);
            //     if(!success){
            //         printf("Failed writing nodal normals to %s\n",outfilename);
            //         return return_and_close_file(fp,false);
            //     }
            // }
            // if(prim->quads.has_attr("clr")){
            //     bool success = write_colors<3>(fp,"elm_colors",prim->quads.attr<zeno::vec3f>("clr"));
            //     if(!success){
            //         printf("Failed writing nodal colors to %s\n",outfilename);
            //         return return_and_close_file(fp,false);
            //     }
            // }

            if(out_customed_cell_attributes) {
                int numberoffielddata = 0;
                for(auto &&[key,attr] : prim->verts.attrs) {
                    // if(key == "nrm" || key == "clr")
                    //     continue;
                    numberoffielddata++;
                }
                printf("number of customed cell attributes %d \n",numberoffielddata);
                if(numberoffielddata > 0){
                    fprintf(fp,"CELL_DATA %d\n",prim->quads.size());
                    write_field_header(fp,"elm_field",numberoffielddata);
                    for(auto &&[key,arr] : prim->quads.attrs) {
                        // if(key == "nrm" || key == "clr")
                        //     continue;
                        const auto& k{key};
                        zs::match(
                            [&k,&fp,&use_double] (const std::vector<zeno::vec3f>& vals) {
                                write_array<3>(fp,k,vals,use_double);
                            },
                            [&k,&fp,&use_double] (const std::vector<zeno::vec2f>& vals) {
                                write_array<2>(fp,k,vals,use_double);
                            },
                            [&k,&fp,&use_double] (const std::vector<float>& vals) {
                                write_array(fp,k,vals,use_double);
                            },
                            [](...) {
                                throw std::runtime_error("what the heck is this type of attribute!");
                            })(arr);
                    }
                }
            }   
        }else if(has_tris) {
            fprintf(fp,"CELL_DATA %d\n",prim->tris.size());
            // if(prim->tris.has_attr("nrm")){
            //     bool success = write_normals(fp,"elm_normals",prim->tris.attr<zeno::vec3f>("nrm"),use_double);
            //     if(!success){
            //         printf("Failed writing nodal normals to %s\n",outfilename);
            //         return return_and_close_file(fp,false);
            //     }
            // }
            // if(prim->tris.has_attr("clr")){
            //     bool success = write_colors<3>(fp,"elm_colors",prim->tris.attr<zeno::vec3f>("clr"));
            //     if(!success){
            //         printf("Failed writing nodal colors to %s\n",outfilename);
            //         return return_and_close_file(fp,false);
            //     }
            // }

            if(out_customed_cell_attributes) {
                int numberoffielddata = 0;
                for(auto &&[key,attr] : prim->verts.attrs) {
                    // if(key == "nrm" || key == "clr")
                    //     continue;
                    numberoffielddata++;
                }
                printf("number of customed cell attributes %d \n",numberoffielddata);
                if(numberoffielddata > 0){
                    write_field_header(fp,"elm_field",numberoffielddata);
                    for(auto &&[key,arr] : prim->quads.attrs) {
                        // if(key == "nrm" || key == "clr")
                        //     continue;
                        const auto& k{key};
                        zs::match(
                            [&k,&fp,&use_double] (const std::vector<zeno::vec3f>& vals) {
                                write_array<3>(fp,k,vals,use_double);
                            },
                            [&k,&fp,&use_double] (const std::vector<zeno::vec2f>& vals) {
                                write_array<2>(fp,k,vals,use_double);
                            },
                            [&k,&fp,&use_double] (const std::vector<float>& vals) {
                                write_array(fp,k,vals,use_double);
                            },
                            [](...) {
                                throw std::runtime_error("what the heck is this type of attribute!");
                            })(arr);
                    }
                }
            }              
        }
             
        return return_and_close_file(fp,true);
    }
    

}