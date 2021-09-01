#version 3.6;

#include "colors.inc"
#include "metals.inc"
#include "textures.inc"

// Right-handed coordinate system in which the z axis points upwards
camera {
	location <20,-50,20>
	sky z
	right -0.34*x*image_width/image_height
	up 0.34*z
	look_at <0,0,0>
}

// White background
background{rgb 1}

// Two lights with slightly different colors
light_source{<-8,-20,30> color rgb <0.77,0.75,0.75>}
light_source{<20,-15,5> color rgb <0.38,0.40,0.40>}

// Radius of the Voronoi cell network, and the particle radius
#declare r=0.06;
#declare s=0.48;

// Particles
union{
#include "cylinder_inv_p.pov"
	pigment{rgb <0.5,0.8,0.7>} finish{reflection 0.12 specular 0.3 ambient 0.42}
}

// Voronoi cells
union{
#include "cylinder_inv_v.pov"
	pigment{rgb <0.1,0.3,0.9>} finish{specular 0.3 ambient 0.42}
}
