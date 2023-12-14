# Open a .blender file and export abc mesh and material infos.
# 
# To use this script, you need install bpy module first or build blender python module from source.
# python -m pip install --user bpy
#
# https://pypi.org/project/bpy/#history
# https://builder.blender.org/download/bpy/

import gc
import os
import bpy
import json
import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="blender file full path")
parser.add_argument("-o", "--output", help="output alembic full path")
args = parser.parse_args()

print("---- Input Path", args.input)
print("---- Output Path", args.output)

assert os.path.exists(args.input)


blender_file = args.input
output_directory = args.output
json_output = os.path.join(output_directory, "info.json")
abc_output = os.path.join(output_directory, "output.abc")

material_info = {}




# Open the .blend file
def BlenderToZenoMap():
    mat_map = {}
    mat_map["Base Color"] = "basecolor"
    mat_map["Subsurface"] = "subsurface"
    mat_map["Metallic"] = "metallic"
    mat_map["Specular"] = "specular"
    mat_map["Roughness"] = "roughness"
    mat_map["Anisotropic"] = "anisotropic"
    mat_map["Anisotropic Rotation"] = "anisoRotation"
    mat_map["IOR"] = "ior"
    mat_map["Sheen"] = "sheen"
    mat_map["Sheen Tint"] = "sheenTint"
    mat_map["Alpha"] = "opacity"
    mat_map["Normal"] = "normal"
    mat_map["Transmission"] = "specTrans"
    mat_map["Emission"] = "emission"
    mat_map["Emission Strength"] = "emissionIntensity"
    mat_map["Clearcoat"] = "clearcoat"
    mat_map["Clearcoat Roughness"] = "clearcoatRoughness"
    return mat_map

  
def FindNodeToSocket(socket,node_bl_label):
    if (not socket.is_linked):
        return None,None
    else:
        links = socket.links
        for link in links:
            if(link.from_node.bl_label == node_bl_label):
                return link.from_node, link
        
        return None,None

def EvalTextureForSocket(socket):
    path = ""
    postprocess = "srgb"
    channel = "vec3"
    scale = [1.0,1.0]
    PP_node=None
    image_node = None
    link = None

    P_node,link = FindNodeToSocket(socket,"Separate Color")
    if (not(P_node is None)):
        channel = link.from_socket.name
        PP_node = P_node
    
    P_node,link = FindNodeToSocket(socket,"Normal Map")
    if (not(P_node is None)):
        postprocess = "normal_map"
        PP_node = P_node
    
    if (PP_node is None):
        image_node,link = FindNodeToSocket(socket,"Image Texture")
    else:
        image_node,link = FindNodeToSocket(PP_node.inputs["Color"],"Image Texture")
    
    if (not (image_node is None)):
        if (not (image_node.image is None)):
            path = image_node.image.filepath
    
    return path,postprocess,channel,scale
    


def EvalBSDFNode(bsdf_node):
    mat = {}
    for socket in bsdf_node.inputs:
        mat[socket.name]={}
        if isinstance(socket.default_value,(float,int)):    #it is not a bpy_prop_array
            mat[socket.name]["value"] =  socket.default_value
        else:
            mat[socket.name]["value"] =  socket.default_value[:]
        
        if(socket.is_linked):
            path,postpro,chennal,scale = EvalTextureForSocket(socket)
            mat[socket.name]["path"] = path
            mat[socket.name]["postpro"] = postpro
            mat[socket.name]["chennal"] = chennal
            mat[socket.name]["scale"] = scale
        
    return mat


def Mat2Json(material):
    mat_name = material.name
    print("Material Name is {}".format(mat_name))
    if (mat.use_nodes):
        mat_ouput_node = None 
        mat_info = None

        for node in mat.node_tree.nodes:
            if (node.bl_label=="Material Output"):
                mat_ouput_node = node
        
        if (mat_ouput_node is None):
            print("Can not found \"Materail Ouput\" node in {}".format(mat_name))
            return None,None
        
        bsdf_node,link = FindNodeToSocket(mat_ouput_node.inputs["Surface"],"Principled BSDF")
            
        if (bsdf_node is None):
            print("Can not found \"Principled BSDF\" node in {}".format(mat_name) )
        else:
            mat_info = EvalBSDFNode(bsdf_node)
        
        return mat_name,mat_info

    return None,None

bpy.ops.wm.open_mainfile(filepath=blender_file)

for mat in bpy.data.materials:
    mat_name,mat_info = Mat2Json(mat)
    if(not (mat_name is None)):
        material_info[mat_name] = mat_info

js = json.dumps(material_info,indent=4)
print(js)



