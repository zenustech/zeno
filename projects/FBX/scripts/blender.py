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

export_all = False

input = args.input
output = args.output
output_json = os.path.join(output, "info.json")
abc_output_dir = os.path.join(output, "abc")
if not os.path.exists(abc_output_dir):
    os.makedirs(abc_output_dir)

"""
{
    "<mesh_unique_name>": {          # name corresponding .abc file 
                                     # value - float,vec3
        "base_color":           { name: "Base Color", value: "<value>" }
        "subsurface":           { name: "Subsurface", value: "<value>" }
        "subsurface_radius":    { name: "Subsurface Radius", value: "<value>" }
        "subsurface_color":     { name: "Subsurface Color", value: "<value>" }
        "subsurface_ior":       { name: "Subsurface IOR", value: "<value>" }
        "subsurface_anisotropy":{ name: "Subsurface Anisotropy", value: "<value>" }
        "metallic":             { name: "Metallic", value: "<value>" }
        "specular":             { name: "Specular", value: "<value>" }
        "specular_tint":        { name: "Specular Tint", value: "<value>" }
        "roughness":            { name: "Roughness", value: "<value>" }
        "anisotropic":          { name: "Anisotropic", value: "<value>" }
        "anisotropic_rotation": { name: "Anisotropic Rotation", value: "<value>" }
        "sheen":                { name: "Sheen", value: "<value>" }
        "sheen_roughness":      { name: "Sheen Roughness", value: "<value>" }
        "sheen_tint":           { name: "Sheen Tint", value: "<value>" }
        "clearcoat":            { name: "Clearcoat", value: "<value>" }
        "clearcoat_roughness":  { name: "Clearcoat Roughness", value: "<value>" }
        "ior":                  { name: "IOR", value: "<value>" }
        "transmission":         { name: "Transmission", value: "<value>" }
        "emission":             { name: "Emission", value: "<value>" }
        "emission_strength":    { name: "Emission Strength", value: "<value>" }
        "alpha":                { name: "Alpha", value: "<value>" }
        "normal":               { name: "Normal", value: "<value>" }
        "clearcoat_normal":     { name: "Clearcoat Normal", value: "<value>" }
        "tangent":              { name: "Tangent", value: "<value>" }
        "_textures": {
            "<mat_param_name>": {
                "<tex_full_path>": {
                    "channel_name": <name>    # material parameter name
                    "separate_name": "<name>" # Red Green Blue
                    "mapping_scale": <value>  # vec3
                }
            }
        }
    }
}
"""
info = {}
obj2mat = {}

# Replace 'your_file_path.blend' with the path to your .blend file
blend_file_path = input

# Open the .blend file
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

# Replace 'your_output_path.abc' with the desired output Alembic file path
export_path = output

# Export Alembic with animation and hierarchy
if export_all:
    bpy.ops.wm.alembic_export(filepath=export_path)

for obj in bpy.data.objects:
        obj.select_set(False)
# Iterate through all objects in the scene and find meshes
for obj in bpy.context.scene.objects:
    gc.collect()
    if obj.type == 'MESH':
        material_params = {}
        material_textures = {}

        # In blender, every object has a uniqe id
        obj_name = obj.name_full
        print(" Mesh Full:", obj_name)

        # Select the current object and deselect all others
        bpy.context.view_layer.objects.active = obj
        #for other_obj in bpy.data.objects:
        #    other_obj.select_set(other_obj == obj)
        obj.select_set(True)
        abc_file_path = os.path.join(abc_output_dir, f"{obj_name}.abc")
        bpy.ops.wm.alembic_export(filepath=abc_file_path, selected=True)
        obj.select_set(True)

        # Check if the object has a material
        if obj.material_slots:

            # Iterate through material slots
            for material_slot in obj.material_slots:
                material = material_slot.material

                # Material name
                print("  Material Name:", material.name)

                # Iterate through material nodes
                if material.use_nodes:
                    for node in material.node_tree.nodes:
                        # Connections: Ref - https://cdn.staticaly.com/gh/aaronmack/image-hosting@master/e/image.66eprfiya4w0.webp
                        #
                        #  Texture Coordinate (UV) -> (Vector) Mapping (Vector) -> (Vector) Image Texture (Color) -> (Color) Separate (Red, Green, Blue) -> (Color, Float) Principled BSDF
                        #
                        #                                                                   [Start From Here]     -> (Color) Principled BSDF
                        #
                        #                                                                                         -> (Color) Normal Map (Normal) -> (Color) Principled BSDF
                        #
                        # Check if the node is an image texture node
                        if node.type == 'TEX_IMAGE':
                            texture = node.image
                            texture_record = []

                            # Image texture filepath
                            img_full_path = bpy.path.abspath(texture.filepath).replace("\\", "/")
                            print("   Texture Filepath:", texture.filepath, "Full:", img_full_path)

                            for node_out in node.outputs:
                                if node_out.name == "Color":
                                    for link in node_out.links:
                                        if link.to_node.type == "BSDF_PRINCIPLED": # .name = "Principled BSDF":
                                            channel_name = link.to_socket.name
                                            print("    --> channel:", channel_name)
                                            texture_detail = {}
                                            texture_detail[img_full_path] = { "channel_name": channel_name }
                                            texture_record.append(channel_name)
                                            material_textures[channel_name] = texture_detail

                                        if link.to_node.type == "SEPARATE_COLOR": # .name = "Separate Color":
                                            for sep_color_out in link.to_node.outputs:
                                                for sep_link in sep_color_out.links:
                                                    if sep_link.to_node.type == "BSDF_PRINCIPLED":
                                                        channel_name = sep_link.to_socket.name
                                                        print("    --> sep channel:", channel_name, sep_color_out.name)
                                                        texture_detail = {}
                                                        texture_detail[img_full_path] = {"channel_name": channel_name, "separate_name": sep_color_out.name}
                                                        texture_record.append(channel_name)
                                                        material_textures[channel_name] = texture_detail

                                        if link.to_node.type == "NORMAL_MAP": # .name = "Normal Map":
                                            for normal_map_out in link.to_node.outputs:
                                                for normal_link in normal_map_out.links:
                                                    channel_name = normal_link.to_socket.name
                                                    print("    --> normal channel:", normal_link.to_socket.name)
                                                    texture_detail = {}
                                                    texture_detail[img_full_path] = {"channel_name": channel_name}
                                                    texture_record.append(channel_name)
                                                    material_textures[channel_name] = texture_detail

                            for node_in in node.inputs:
                                if node_in.name == "Vector":
                                    for link in node_in.links:
                                        if link.from_node.type == "MAPPING": # .name = "Mapping"
                                            mapping_scale = link.from_node.inputs['Scale'].default_value
                                            print("    --> mapping scale:", mapping_scale)
                                            for record in texture_record:
                                                material_textures[record][img_full_path]["mapping_scale"] = mapping_scale[:]

                            # fill default value
                            for key in material_textures:
                                for _key in material_textures[key]:
                                    value = material_textures[key][_key]
                                    if not "separate_name" in value:
                                        value["separate_name"] = ""
                                    if not "mapping_scale" in value:
                                        value["mapping_scale"] = [1.0, 1.0, 1.0]


                            material_params["_textures"] = material_textures

                        if node.type == 'BSDF_PRINCIPLED':
                            base_color = node.inputs['Base Color'].default_value
                            subsurface = node.inputs['Subsurface'].default_value
                            subsurface_radius = node.inputs['Subsurface Radius'].default_value
                            subsurface_color = node.inputs['Subsurface Color'].default_value
                            subsurface_ior = node.inputs['Subsurface IOR'].default_value
                            subsurface_anisotropy = node.inputs['Subsurface Anisotropy'].default_value
                            metallic = node.inputs['Metallic'].default_value
                            specular = node.inputs['Specular'].default_value
                            specular_tint = node.inputs['Specular Tint'].default_value
                            roughness = node.inputs['Roughness'].default_value
                            anisotropic = node.inputs['Anisotropic'].default_value
                            anisotropic_rotation = node.inputs['Anisotropic Rotation'].default_value
                            sheen = node.inputs['Sheen'].default_value
                            #sheen_roughness = node.inputs['Sheen Roughness'].default_value  # KeyError: 'bpy_prop_collection[key]: key "Sheen Roughness" not found' for blender 3.6
                            sheen_roughness = 0.5
                            sheen_tint = node.inputs['Sheen Tint'].default_value
                            clearcoat = node.inputs['Clearcoat'].default_value
                            clearcoat_roughness = node.inputs['Clearcoat Roughness'].default_value
                            ior = node.inputs['IOR'].default_value
                            transmission = node.inputs['Transmission'].default_value
                            emission = node.inputs['Emission'].default_value
                            emission_strength = node.inputs['Emission Strength'].default_value
                            alpha = node.inputs['Alpha'].default_value
                            normal = node.inputs['Normal'].default_value
                            clearcoat_normal = node.inputs['Clearcoat Normal'].default_value
                            tangent = node.inputs['Tangent'].default_value
                            #weight = node.inputs['Weight'].default_value
                            print("   Base Color:", base_color[:])

                            material_params["base_color"] =             {"value": base_color[:],        "name": "Base Color"}
                            material_params["subsurface"] =             {"value": subsurface,           "name": "Subsurface"}
                            material_params["subsurface_radius"] =      {"value": subsurface_radius[:], "name": "Subsurface Radius"}
                            material_params["subsurface_color"] =       {"value": subsurface_color[:],  "name": "Subsurface Color"}
                            material_params["subsurface_ior"] =         {"value": subsurface_ior,       "name": "Subsurface IOR"}
                            material_params["subsurface_anisotropy"] =  {"value": subsurface_anisotropy,"name": "Subsurface Anisotropy"}
                            material_params["metallic"] =               {"value": metallic,             "name": "Metallic"}
                            material_params["specular"] =               {"value": specular,             "name": "Specular"}
                            material_params["specular_tint"] =          {"value": specular_tint,        "name": "Specular Tint"}
                            material_params["roughness"] =              {"value": roughness,            "name": "Roughness"}
                            material_params["anisotropic"] =            {"value": anisotropic,          "name": "Anisotropic"}
                            material_params["anisotropic_rotation"] =   {"value": anisotropic_rotation, "name": "Anisotropic Rotation"}
                            material_params["sheen"] =                  {"value": sheen,                "name": "Sheen"}
                            material_params["sheen_tint"] =             {"value": sheen_tint,           "name": "Sheen Roughness"}
                            material_params["sheen_roughness"] =        {"value": sheen_roughness,      "name": "Sheen Tint"}
                            material_params["clearcoat"] =              {"value": clearcoat,            "name": "Clearcoat"}
                            material_params["clearcoat_roughness"] =    {"value": clearcoat_roughness,  "name": "Clearcoat Roughness"}
                            material_params["ior"] =                    {"value": ior,                  "name": "IOR"}
                            material_params["transmission"] =           {"value": transmission,         "name": "Transmission"}
                            material_params["emission"] =               {"value": emission[:],          "name": "Emission"}
                            material_params["emission_strength"] =      {"value": emission_strength,    "name": "Emission Strength"}
                            material_params["alpha"] =                  {"value": alpha,                "name": "Alpha"}
                            material_params["normal"] =                 {"value": normal[:],            "name": "Normal"}
                            material_params["clearcoat_normal"] =       {"value": clearcoat_normal[:],  "name": "Clearcoat Normal"}
                            material_params["tangent"] =                {"value": tangent[:],           "name": "Tangent"}
        info[obj_name] = material_params
    print("-----")


# Serializing json
json_object = json.dumps(info, indent=4)

# Writing to sample.json
with open(output_json, "w") as outfile:
    outfile.write(json_object)
