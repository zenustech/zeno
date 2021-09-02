#version 3.6;

// Right-handed coordinate system in which the z-axis points upwards
camera {
	orthographic
	location <8,11,-50>
	right 25*x*image_width/image_height
	up 25*y
	look_at <8,11,0>
}

// White background
background{rgb 1}

// Two lights with slightly different colors
light_source{<-8,30,-20> color rgb <0.79,0.75,0.75>}
light_source{<25,12,-12> color rgb <0.36,0.40,0.40>}

// Radius of the Voronoi cell network
#declare r=0.2;

// Radius of the particles
#declare s=2.8;

union {
	sphere{<0,0,0>,1.4}
	pigment{rgbft <0.9,0.82,0.4>} finish{reflection 0.1 specular 0.3 ambient 0.42}
}


// Particles
/*#declare s=2.8;
union{
#include "doe_diagram_p.pov"
	pigment{rgbft <0.9,0.82,0.4,0,0.5>} finish{reflection 0.1 specular 0.3 ambient 0.42}
}


#declare s=1.4;
union{
#include "doe_diagram_p.pov"
	pigment{rgbft <0.4,0.6,0.3>} finish{reflection 0.1 specular 0.3 ambient 0.42}
}*/

// Voronoi cells
/*intersection{
	union{
		#include "doe_diagram_v.pov"
	}
	box{<-2,-1,-4>,<18,23,4>}
	pigment{rgb <0.8,0.59,0.9>} finish{specular 0.5 ambient 0.42}
}*/
