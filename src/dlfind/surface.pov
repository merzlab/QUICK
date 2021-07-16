//This is a comment
global_settings { max_trace_level 8 }
#include "colors.inc"
//#include "stones.inc"
#include "textures.inc"

#declare Bond_Thick=0.2;

/*
// Coordinate System:
cylinder{<0,0,0><0.5,0,0>,0.02   texture {pigment {color rgb <   1.00000,   0.00000,   0.00000>}}}
cylinder{<0,0,0><0,5,0>,0.02   texture {pigment {color rgb <   0.00000,   1.00000,   0.00000>}}}
cylinder{<0,0,0><0,0,0.5>,0.02   texture {pigment {color rgb <   0.00000,   0.00000,   1.00000>}}}
*/

#declare blueval = -0.69;
#declare redval  = 1.3;
#declare line_thick=0.007;

//sphere { <-2,0,0>,0.2 texture {
//    pigment {color rgb <   1.00000,   0.00000,   0.00000>}}}

////////////////////////////////////////////////////////////////////////
// Optimization path
////////////////////////////////////////////////////////////////////////
union{
#include "path.inc"
  texture {pigment {color rgb <   1.00000,   0.00000,   0.00000>}}
//no_shadow
translate <0,0.5,0>
}

union{
#include "axis.inc"
  texture {pigment {color rgb <   0.00000,   0.00000,   1.00000>}}
//no_shadow
translate <0,0.5,0>
}

////////////////////////////////////////////////////////////////////////
// Unbiased Surface
////////////////////////////////////////////////////////////////////////
mesh{
#include "surface.inc"
 texture {
   pigment {
    gradient y translate -0.48 scale 1/0.03*(redval-blueval) translate blueval
    color_map {
      [0.00 color Blue]
      [0.48 color Blue]
      [0.49  color Green]
      [0.50 color Yellow]
      [0.51 color Red]
      [1.0 color Red]
              }
            }
   finish {ambient 0.5 diffuse 0.6 brilliance 8.0 phong 0.5
  phong_size 80.0 reflection 0.0 }
         }
clipped_by{box{<-10,-1,-10>,<10,1.07,2.>}}
//scale<1,1.3,1>
}

// Equipotential lines
mesh{
#include "surface.inc"
 texture {
   pigment {
    gradient y scale 0.2
    color_map {
      [0.00 color rgbt<1,1,1,1>]
      [0.30 color rgbt<1,1,1,1>]
      [0.30 color Black]
      [0.35 color Black]
      [0.35 color rgbt<1,1,1,1>]
              }
            }
         }
clipped_by{box{<-10,-1,-10>,<10,1.07,2.>}}
translate<0.0,0.0001,0.0>
  //scale<1,1.3,1>
}

// geht
camera {
       orthographic
       location <10.,0., 0.> 
       look_at  <0 , 0.0 , 0. >
        angle 20
	 rotate <0,0,90>
	 rotate <0,90,0>
	 translate < 0,0,0.7>
       }

/*
// versuch
camera {
       orthographic
       location <10.,0., 0.> 
       look_at  <0 , 0.0 , 0. >
        angle 15
	 rotate <0,0,30> 
	 rotate <0,50,0> // rotate around vertical
       }
*/


#declare Intensity = 1.50;
light_source {
             <20, 150, 0> color rgb Intensity*1.000 shadowless
             }
light_source {
             <20, 50, 30> color rgb Intensity*1.000 shadowless
             }
light_source {
             <0, 50, 0> color rgb Intensity*1.000 //shadowless
             }

/*
plane {<-5,5,5>,   -3.0 texture{ pigment{ color <1,1,1> }
        finish {ambient 0.5 diffuse 0.5 brilliance 2.5 phong 0.0
  phong_size 80.0 reflection 0.0 }
}}
*/

background { White }
