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
#declare redval  = 0.6;
#declare line_thick=0.007;

//sphere { <-2,0,0>,0.2 texture {
//    pigment {color rgb <   1.00000,   0.00000,   0.00000>}}}

////////////////////////////////////////////////////////////////////////
// Optimization path
////////////////////////////////////////////////////////////////////////
union{
 union{
#include "path_62.inc"
//#include "VAR_FILE"
  texture {pigment {color rgb <   1.00000,   0.00000,   0.00000>}}}
union{
sphere{<     -0.5582,     -0.6920,      1.4417> 0.045}
sphere{<      0.6235,     -0.5102,      0.0280> 0.045}
  texture {pigment {color rgb <   0.00000,   1.00000,   0.00000>}}}
union{
sphere{<     -0.8219,     -0.1918,      0.6244> 0.045}
  texture {pigment {color rgb <   0.00000,   0.00000,   1.00000>}}}
//no_shadow
translate <0,0.05,0>
}

////////////////////////////////////////////////////////////////////////
// Unbiased Surface
////////////////////////////////////////////////////////////////////////
mesh{
#declare trp=0.0;
#include "surface.inc"
 texture {
   pigment {
    gradient y translate -0.48 scale 1/0.03*(redval-blueval) translate blueval
    color_map {
      [0.00 color rgbt <0,0,1,trp>]
      [0.48 color rgbt <0,0,1,trp>]
      [0.49  color rgbt <0,1,0,trp> ]
      [0.50 color rgbt <1,1,0,trp>]
      [0.51 color rgbt <1,0,0,trp>]
      [1.0 color rgbt <1,0,0,trp>]
              }
            }
   finish {ambient 0.4 diffuse 0.7 brilliance 8.0 phong 0.5
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
    gradient y translate -0.6  scale 0.15 
    color_map {
      [0.00 color rgbt<1,1,1,1>]
      [0.40 color rgbt<1,1,1,1>]
      [0.40 color Black]
      [0.43 color Black]
      [0.43 color rgbt<1,1,1,1>]
              }
            }
         }
clipped_by{box{<-10,-1,-10>,<10,1.07,2.>}}
translate<0.0,0.0001,0.0>
  //scale<1,1.3,1>
}

/*
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
*/

// versuch
camera {
       orthographic
       location <10.,0., -0.7> 
       look_at  <0 , 0.0 , -0.7 >
        angle 15
	 rotate <0,0,40> 
	 rotate <0,155,0> // rotate around vertical
       }



#declare Intensity = 1.50;
light_source {
             <20, 150, 0> color rgb Intensity*0.000 shadowless
             }
light_source {
             <20, 50, 30> color rgb Intensity*0.000 shadowless
             }
light_source {
             <-20, 40, 30> color rgb Intensity*1.000 //shadowless
             }
light_source {
             <-20, 30, 0> color rgb Intensity*1.000 //shadowless
             }

/*
plane {<-5,5,5>,   -3.0 texture{ pigment{ color <1,1,1> }
        finish {ambient 0.5 diffuse 0.5 brilliance 2.5 phong 0.0
  phong_size 80.0 reflection 0.0 }
}}
*/

background { White }
