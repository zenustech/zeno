#version 3.6;
#include "colors.inc"
#include "metals.inc"
#include "textures.inc"

camera {
	location <5,-50,0>
	sky z
	right -0.62*x*image_width/image_height
	up 0.62*y
	look_at <5,0,0>
}

background{rgb 1}

light_source{<-16,-30,30> color rgb <0.77,0.75,0.75>}
light_source{<25,-16,8> color rgb <0.43,0.45,0.45>}

#declare r=0.05;
#declare s=0.5;

union {
#include "finite_sys_out.pov"
	pigment{rgb <0.9,0.9,0.3>} finish{reflection 0.2 specular 0.25 ambient 0.28 metallic}
}

union{
#include "finite_sys_in.pov"
	pigment{rgb <0.3,0.5,0.9>} finish{reflection 0.2 specular 0.25 ambient 0.28 metallic}
}
