/* 
 Copyright (C) 2013 Rolf Wuerdemann, M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
 
#include "util.h" 
 
#define XC_GGA_X_SFAT 530 /* short-range recipe for exchange GGA functionals */ 
/* see 
   Savin, Flad, Int. J. Quant. Chem. Vol. 56, 327-332 (1995) 
   Akinaga, Ten-no, Chem. Phys. Lett. 462 (2008) 348-351 
*/ 
 
typedef struct{ 
  int func_id; 
  xc_gga_enhancement_t enhancement_factor; 
} gga_x_sfat_params; 
 
static void 
gga_x_sfat_init(xc_func_type *p) 
{ 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(gga_x_sfat_params)); 
#ifdef CUDA 
  p->params_byte_size = sizeof(gga_x_sfat_params); 
#endif 
 
  /* random functional, mainly intended for testing */ 
  ((gga_x_sfat_params *) (p->params))->func_id = -1; 
  xc_gga_x_sfat_set_params(p, XC_GGA_X_B88, 0.44); 
} 
 
void  
xc_gga_x_sfat_set_params(xc_func_type *p, int func_id, double omega) 
{ 
  gga_x_sfat_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (gga_x_sfat_params *) (p->params); 
 
  p->cam_omega = omega; 
 
  /* if func_id == -1 do nothing */ 
  if(func_id != -1 && params->func_id == -1){ /* intialize stuff */ 
    p->n_func_aux  = 1; 
    p->func_aux    = (xc_func_type **) malloc(sizeof(xc_func_type *)); 
    p->func_aux[0] = (xc_func_type  *) malloc(sizeof(xc_func_type  )); 
  } 
 
  if(func_id != -1 && params->func_id != func_id){ 
    if(params->func_id != -1) 
      xc_func_end (p->func_aux[0]); 
 
    params->func_id = func_id; 
    xc_func_init (p->func_aux[0], params->func_id, p->nspin); 
 
    params->enhancement_factor = xc_get_gga_enhancement_factor(func_id); 
  } 
} 
 
 
static void  
func(const xc_func_type *pt, xc_gga_work_c_t *r) 
{ 
  int is, js; 
  gga_x_sfat_params *params; 
  xc_gga_work_x_t aux; 
  const double sign[2] = {1.0, -1.0}; 
 
  double opz, opz13, rss, x2, drssdrs, drssdz, d2rssdrsz, d2rssdz2; 
  double ex, dexdrss, dexdz, d2exdrss2, d2exdrssz, d2exdz2; 
  double k_GGA, dk_GGAdrss, dk_GGAdxs, d2k_GGAdrss2, d2k_GGAdrssxs, d2k_GGAdxs2; 
  double aa, daadrss, daadxs, d2aadrss2, d2aadrssxs, d2aadxs2; 
  double f_aa, df_aa, d2f_aa, d3f_aa; 
  double dftdrss, dftdz, d2ftdrss2, d2ftdrssz; 
 
  assert(pt != NULL && pt->params != NULL); 
  params = (gga_x_sfat_params *) (pt->params); 
 
  r->f = 0.0; 
  if(r->order >= 1) 
    r->dfdrs = r->dfdz = r->dfdxt = r->dfdxs[0] = r->dfdxs[1] = 0.0; 
  if(r->order >= 2){ 
    r->d2fdrs2 = r->d2fdrsz = r->d2fdrsxt = r->d2fdrsxs[0] = r->d2fdrsxs[1] = r->d2fdz2 = 0.0; 
    r->d2fdzxt = r->d2fdzxs[0] = r->d2fdzxs[1] = r->d2fdxt2 = r->d2fdxtxs[0] = r->d2fdxtxs[1] = 0.0; 
    r->d2fdxs2[0] = r->d2fdxs2[1] = r->d2fdxs2[2] = 0.0; 
  } 
  for(is = 0; is < 2; is++){ 
    opz   = 1.0 + sign[is]*r->z; 
    if(opz < MIN_ZETA) continue; 
     
    opz13 = CBRT(opz); 
    rss   = r->rs*M_CBRT2/opz13; 
 
    /* call enhancement factor */ 
    aux.x     = r->xs[is]; 
    aux.order = r->order; 
    params->enhancement_factor(pt->func_aux[0], &aux); 
 
    k_GGA = sqrt(9.0*M_PI/(2.0*X_FACTOR_C*aux.f))*RS_FACTOR/rss; 
    aa = pt->cam_omega/(2.0*k_GGA); 
 
    xc_lda_x_attenuation_function_yukawa(r->order, aa, &f_aa, &df_aa, &d2f_aa, &d3f_aa); 
 
    ex    = -X_FACTOR_C*RS_FACTOR*opz/(2.0*rss); 
    r->f += ex*aux.f*f_aa; 
 
    if(r->order < 1) continue; 
 
    drssdrs = M_CBRT2/opz13; 
    drssdz  = -sign[is]*rss/(3.0*opz); 
 
    dk_GGAdrss = -k_GGA/rss; 
    dk_GGAdxs  = -k_GGA*aux.dfdx/(2.0*aux.f); 
 
    daadrss = -aa*dk_GGAdrss/k_GGA; 
    daadxs  = -aa*dk_GGAdxs /k_GGA; 
 
    dexdrss = -ex/rss; 
    dexdz   = sign[is]*ex/opz + dexdrss*drssdz; /* total derivative */ 
 
    r->dfdrs    += aux.f*(dexdrss*f_aa + ex*df_aa*daadrss)*drssdrs; 
    r->dfdz     += aux.f*(dexdz*f_aa + ex*df_aa*daadrss*drssdz); 
    r->dfdxs[is] = ex*(aux.dfdx*f_aa + aux.f*df_aa*daadxs); 
 
    if(r->order < 2) continue; 
 
    js = (is == 0) ? 0 : 2; 
 
    d2rssdrsz = drssdz/r->rs; 
    d2rssdz2  = -4.0*sign[is]*drssdz/(3.0*opz); 
 
    d2k_GGAdrss2  = -2.0*dk_GGAdrss/rss; 
    d2k_GGAdrssxs = -dk_GGAdxs/rss; 
    d2k_GGAdxs2   = -k_GGA/(2.0*aux.f) * (aux.d2fdx2 - 3.0*aux.dfdx*aux.dfdx/(2.0*aux.f)); 
 
    d2aadrss2  = -aa/k_GGA * (d2k_GGAdrss2  - 2.0*dk_GGAdrss*dk_GGAdrss/k_GGA); 
    d2aadrssxs = -aa/k_GGA * (d2k_GGAdrssxs - 2.0*dk_GGAdrss*dk_GGAdxs /k_GGA); 
    d2aadxs2   = -aa/k_GGA * (d2k_GGAdxs2   - 2.0*dk_GGAdxs *dk_GGAdxs /k_GGA); 
 
    d2exdrss2  = -2.0*dexdrss/rss; 
    d2exdrssz  = -dexdz/rss + ex*drssdz/(rss*rss); /* total derivative */ 
    d2exdz2    = sign[is]*dexdrss*drssdz/opz + d2exdrssz*drssdz + dexdrss*d2rssdz2; /* total derivative */ 
 
    r->d2fdrs2 += aux.f*(d2exdrss2*f_aa + 2.0*dexdrss*df_aa*daadrss +  
		       ex*d2f_aa*daadrss*daadrss + ex*df_aa*d2aadrss2)*drssdrs*drssdrs; 
 
    r->d2fdrsz += aux.f*((d2exdrssz*f_aa + dexdz*df_aa*daadrss +  
			dexdrss*df_aa*daadrss*drssdz + ex*d2f_aa*daadrss*daadrss*drssdz)*drssdrs +  
		       (dexdrss*f_aa + ex*df_aa*daadrss)*d2rssdrsz); 
 
    r->d2fdz2  += aux.f*(d2exdz2*f_aa + 2.0*dexdz*df_aa*daadrss*drssdz + ex*df_aa*daadrss*d2rssdz2 + 
		       ex*(d2f_aa*daadrss*daadrss + df_aa*d2aadrss2)*drssdz*drssdz); 
 
    r->d2fdrsxs[is] = aux.dfdx*(dexdrss*f_aa + ex*df_aa*daadrss)*drssdrs + 
      aux.f*(dexdrss*df_aa*daadxs + ex*d2f_aa*daadrss*daadxs + ex*df_aa*d2aadrssxs)*drssdrs; 
 
    r->d2fdzxs[is]  = aux.dfdx*(dexdz*f_aa + ex*df_aa*daadrss*drssdz) + 
      aux.f*(dexdz*df_aa*daadxs + ex*(d2f_aa*daadrss*daadxs + df_aa*d2aadrssxs)*drssdz); 
 
    r->d2fdxs2[js] = ex*(aux.d2fdx2*f_aa + 2.0*aux.dfdx*df_aa*daadxs + 
			 aux.f*d2f_aa*daadxs*daadxs + aux.f*df_aa*d2aadxs2); 
  } 
} 
 
#ifdef CUDA
#define kernel_id -1
#endif 
#include "work_gga_c.c" 
 
const xc_func_info_type xc_func_info_gga_x_sfat = { 
  XC_GGA_X_SFAT, 
  XC_EXCHANGE, 
  "Short-range recipe for exchange GGA functionals - Yukawa", 
  XC_FAMILY_GGA, 
  {&xc_ref_Savin1995_327, &xc_ref_Akinaga2008_348, NULL, NULL, NULL}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC, 
  1e-19, 
  0, NULL, NULL, 
  gga_x_sfat_init, 
  NULL, NULL,  
  work_gga_c, 
  NULL 
}; 
 
