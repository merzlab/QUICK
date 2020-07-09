/* 
 Copyright (C) 2006-2007 M.A.L. Marques 
 
 This Source Code Form is subject to the terms of the Mozilla Public 
 License, v. 2.0. If a copy of the MPL was not distributed with this 
 file, You can obtain one at http://mozilla.org/MPL/2.0/. 
*/ 
 
#include "util.h" 
 
#define XC_GGA_X_WPBEH 524 /* short-range version of the PBE */ 
 
typedef struct{ 
  double omega; 
} gga_x_wpbeh_params; 
 
static void 
gga_x_wpbeh_init(xc_func_type *p) 
{ 
  assert(p->params == NULL); 
  p->params = malloc(sizeof(gga_x_wpbeh_params)); 
#if defined CUDA || defined CUDA_MPIV 
  p->params_byte_size = sizeof(gga_x_wpbeh_params); 
#endif 
 
  /* The default value is actually PBEh */ 
  xc_gga_x_wpbeh_set_params(p, 0.0); 
} 
 
void  
xc_gga_x_wpbeh_set_params(xc_func_type *p, double omega) 
{ 
  gga_x_wpbeh_params *params; 
 
  assert(p != NULL && p->params != NULL); 
  params = (gga_x_wpbeh_params *) (p->params); 
 
  params->omega = omega; 
} 
 
 
/* This implementation follows the one from espresso, that, in turn, 
   follows the one of the thesis of Jochen Heyd. Analytic derivatives 
   are only implemented in espresso though. These implementations can 
   be found in: 
 
   vasp: xclib_grad.F, MODULE wpbe, and in particular SUBROUTINE EXCHWPBE_R 
   espresso: flib/functionals.f90, SUBROUTINE wpbe_analy_erfc_approx_grad 
 
   very important details can be found in references: 
 
   *) J Heyd, GE Scuseria, and M Ernzerhof, J. Chem. Phys. 118, 8207 (2003) 
      Erratum: J. Chem. Phys. 124, 219906 (2006). 
   *) M Ernzerhof and JP Perdew, J. Chem. Phys. 109, 3313 (1998) 
   *) J Heyd and GE Scuseria, J. Chem. Phys. 120, 7274 (2004) 
 
   Also the whole mess with the rescaling of s is explained in 
 
   *) TM Henderson, AF Izmaylov, G Scalmani, and GE Scuseria, J. Chem. Phys. 131, 044108 (2009) 
*/ 
 
static inline void 
s_scaling(int version, int order, double s1, double *s2, double *ds2ds1) 
{ 
  /* parameters for the re-scaling of s */ 
  static const double strans=8.3, smax=8.572844, sconst=18.79622316; 
  static const double s0=8.572844, p4=0.615482, p5=1.136921, p6=-0.449154, 
    q4=1.229195, q5=-0.0269253, q6=0.313417, q7=-0.0508314, q8=0.0175739; 
 
  double expms1, expmsmax, s12, s14, num, den, dnum, dden; 
 
  switch(version){ 
  case 0: /* no scaling */ 
    *s2 = s1; 
    break; 
 
  case 1: /* original scaling of Heyd */ 
    *s2  = (s1 < strans) ? s1 : smax - sconst/(s1*s1); 
    break; 
 
  case 2: /* first version of the scaling by TM Henderson, apparently used by Gaussian */ 
    if(s1 < 1.0) 
      *s2 = s1; 
    else if(s1 > 15.0) 
      *s2 = smax; 
    else{ 
      expms1   = exp(-s1); 
      expmsmax = exp(-smax); 
      *s2  = s1 - log(1.0 + expmsmax/expms1); 
    } 
    break; 
 
  case 3: /* second version of the scaling by TM Henderson */ 
    expms1   = exp(-s1); 
    expmsmax = exp(-smax); 
    *s2 = s1 - (1.0 - expms1)*log(1.0 + expmsmax/expms1); 
    break; 
 
  case 4: /* appendix of JCP 128, 194105 (2008) */ 
    s12 = s1*s1; 
    s14 = s12*s12; 
 
    num = s1*(1.0 + s14*(p4 + s1*(p5 + s1*(p6 + s1*q8*s0)))); 
    den = 1.0 + s14*(q4 + s1*(q5 + s1*(q6 + s1*(q7 + s1*q8)))); 
 
    *s2 = num/den; 
    break; 
 
  default: 
    fprintf(stderr, "Internal error in gga_x_wpbeh\n"); 
    exit(1); 
  } 
 
  if(order < 1) return; 
 
  switch(version){ 
  case 0: 
    *ds2ds1 = 1.0; 
    break; 
 
  case 1: 
    *ds2ds1 = (s1 < strans) ? 1.0 : 2.0*sconst/(s1*s1*s1); 
    break; 
 
  case 2: 
    if(s1 < 1.0) 
      *ds2ds1 = 1.0; 
    else if(s1 > 15.0) 
      *ds2ds1 = 0.0; 
    else 
      *ds2ds1 = expms1/(expms1 + expmsmax); 
    break; 
 
  case 3: 
    *ds2ds1 = expms1*(1.0 + expmsmax)/(expms1 + expmsmax) - expms1*log(1.0 + expmsmax/expms1); 
 
  case 4: /* appendix of JCP 128, 194105 (2008) */ 
    dnum = 1.0 + s14*(5.0*p4 + s1*(6.0*p5 + s1*(7.0*p6 + s1*8.0*q8*s0))); 
    dden = s12*s1*(4.0*q4 + s1*(5.0*q5 + s1*(6.0*q6 + s1*(7.0*q7 + s1*8.0*q8)))); 
 
    *ds2ds1 = (dnum*den - num*dden)/(den*den); 
  } 
} 
 
static inline void  
func_3(const xc_func_type *p, int order, double x, double ds, 
     double *f, double *dfdx, double *lvrho) 
{ 
  static const double AA=1.0161144, BB=-0.37170836, CC=-0.077215461, DD=0.57786348, EE=-0.051955731; 
  static const double m89=-8.0/9.0; 
 
  /* Cutoff criterion below which to use polynomial expansion */ 
  static const double EGscut=0.08, wcutoff=14, expfcutoff=700.0; 
 
  double omega, kF, ww, ww2, ww3, ww4, ww5, ww6, ww7, ww8, dwdrho; 
  double ss, ss2, ss3, ss4, ss5, ss6, dssdx; 
  double AA2, AA3, AA12, AA32, AA52; 
  double DHs, DHs2, DHs3, DHs4, DHs72, DHs92; 
  double eb1, f94Hs2_A, DHsw, DHsw2, DHsw52, DHsw72; 
  double Hsbw, Hsbw2, Hsbw3, Hsbw4, Hsbw12, Hsbw32, Hsbw52, Hsbw72; 
  double DHsbw, DHsbw2, DHsbw3, DHsbw4, DHsbw5, DHsbw12, DHsbw32, DHsbw52, DHsbw72, DHsbw92; 
  double HsbwA94, HsbwA942, HsbwA943, HsbwA945, HsbwA9412; 
  double H, F, EG, dHds, dFds, dEGds, dDHsds, dDHswdw, dHsbwds, dHsbwdw; 
  double term1, term2, term3, term4, term5, t10, piexperf, expei; 
  double dterm1ds, dterm2ds, dterm3ds, dterm4ds, dterm5ds, dterm1dw, dterm3dw, dterm4dw, dterm5dw; 
  double dt10ds, dt10dw, dpiexperfds, dpiexperfdw, dexpeids, dexpeidw; 
 
  assert(p->params != NULL); 
  omega = ((gga_x_wpbeh_params *)(p->params))->omega; 
 
  /* Note that kF has a 6 and not a 3 as it should in principle 
     be. This is because the HSE formula, if one would take the papers 
     seriously, does not fulfill the spin sum-rule. This is probably 
     an oversight from them. So, we have to choose, either a 6 or a 3. 
      
     Nwchem seems to have the factor of 6, but VASP and espresso have 
     a 3. This would amount to rescaling omega by a factor of 
     cbrt(2). We follow the quantum chemistry community and put the 6. 
  */ 
  kF  = pow(6.0*M_PI*M_PI*ds, 1.0/3.0); 
  ww  = omega/kF; 
  ww2 = ww*ww; ww3 = ww*ww2; ww4 = ww*ww3; ww5 = ww*ww4; ww6 = ww*ww5; ww7 = ww*ww6; ww8 = ww*ww7; 
 
  /*  Rescaling the s values to ensure the Lieb-Oxford bound */ 
  s_scaling(2, order, X2S*x, &ss, &dssdx); 
  ss2 = ss*ss;  
  ss3 = ss*ss2;  
  ss4 = ss*ss3;  
  ss5 = ss*ss4;  
  ss6 = ss*ss5; 
 
  if(order >= 1){ 
    dwdrho  = -ww/(3.0*ds); 
    dssdx  *= X2S; 
  } 
 
  AA2  = AA*AA; 
  AA3  = AA2*AA; 
  AA12 = sqrt(AA); 
  AA32 = AA12*AA; 
  AA52 = AA32*AA; 
 
  /* first let us calculate H(s) */ 
  { 
    static const double Ha1=0.00979681, Ha2=0.0410834, Ha3=0.187440, Ha4=0.00120824, Ha5=0.0347188; 
    double Hnum, Hden, dHnum, dHden; 
 
    Hnum = Ha1*ss2 + Ha2*ss4; 
    Hden = 1.0 + Ha3*ss4 + Ha4*ss5 + Ha5*ss6; 
 
    H = Hnum/Hden; 
 
    if(order >= 1){ 
      dHnum = 2.0*Ha1*ss  + 4.0*Ha2*ss3; 
      dHden = 4.0*Ha3*ss3 + 5.0*Ha4*ss4 + 6.0*Ha5*ss5; 
 
      dHds  = (Hden*dHnum - Hnum*dHden)/(Hden*Hden); 
    } 
  } 
 
  /* now we calculate F(s) */ 
  { 
    double Fc1, Fc2; 
 
    //Fc1 = 4.0*AA*AA/(9.0*CC) + (BB - AA*DD)/CC; 
    //Fc2 = -4.0/(3.0*36.0*CC); 
 
    Fc1 = 6.4753871; 
    Fc2 = 0.47965830; 
 
    F = Fc1*H + Fc2; 
 
    if(order >= 1) 
      dFds = Fc1*dHds; 
  } 
 
  /* useful variables for what comes next */ 
  DHs   = DD + ss2*H;  
  DHs2  = DHs*DHs;  
  DHs3  = DHs2*DHs;  
  DHs4  = DHs3*DHs; 
  DHs72 = DHs3*sqrt(DHs);  
  DHs92 = DHs72*DHs; 
 
  f94Hs2_A = 9.0*H*ss2/(4.0*AA); 
 
  DHsw   = DHs + ww2; 
  DHsw2  = DHsw*DHsw;  
  DHsw52 = sqrt(DHsw)*DHsw2;  
  DHsw72 = DHsw52*DHsw; 
 
  eb1 = (ww < wcutoff) ? 1.455915450052607 : 2.0; 
 
  Hsbw   = ss2*H + eb1*ww2;  
  Hsbw2  = Hsbw*Hsbw;  
  Hsbw3  = Hsbw2*Hsbw;  
  Hsbw4  = Hsbw3*Hsbw; 
  Hsbw12 = sqrt(Hsbw);  
  Hsbw32 = Hsbw12*Hsbw;  
  Hsbw52 = Hsbw32*Hsbw;  
  Hsbw72 = Hsbw52*Hsbw; 
 
  if(order >= 1){ 
    dDHsds  = 2.0*ss*H + ss2*dHds; 
    dDHswdw = 2.0*ww; 
    dHsbwds = ss2*dHds + 2.0*ss*H; 
    dHsbwdw = 2.0*eb1*ww; 
  } 
 
  DHsbw   = DD + Hsbw; /* derivatives of DHsbw are equal to the ones of Hsbw */ 
  DHsbw2  = DHsbw*DHsbw;  
  DHsbw3  = DHsbw2*DHsbw;  
  DHsbw4  = DHsbw3*DHsbw;  
  DHsbw5  = DHsbw4*DHsbw; 
  DHsbw12 = sqrt(DHsbw);  
  DHsbw32 = DHsbw12*DHsbw;  
  DHsbw52 = DHsbw32*DHsbw;  
  DHsbw72 = DHsbw52*DHsbw; 
  DHsbw92 = DHsbw72*DHsbw; 
 
  HsbwA94   = 9.0*Hsbw/(4.0*AA); 
  HsbwA942  = HsbwA94*HsbwA94; 
  HsbwA943  = HsbwA942*HsbwA94; 
  HsbwA945  = HsbwA943*HsbwA942; 
  HsbwA9412 = sqrt(HsbwA94); 
 
  /* and now G(s) */ 
  if(ss > EGscut){ 
    double Ga, Gb, dGa, dGb; 
 
    Ga = M_SQRTPI*(15.0*EE + 6.0*CC*(1.0 + F*ss2)*DHs + 4.0*BB*DHs2 + 8.0*AA*DHs3)/(16.0*DHs72) 
      - (3.0*M_PI/4.0)*sqrt(AA)*exp(f94Hs2_A)*(1.0 - erf(sqrt(f94Hs2_A))); 
    Gb = 15.0*M_SQRTPI*ss2/(16.0*DHs72); 
 
    EG = -(3.0*M_PI/4.0 + Ga)/Gb; 
 
    if(order >= 1){ 
      dGa = (M_SQRTPI/32.0) * 
	((36.0*(2.0*H + dHds*ss)/(AA12*sqrt(H/AA)) +  
	  (1.0/DHs92) *(-8.0*AA*dDHsds*DHs3 - 105.0*dDHsds*EE - 30.0*CC*dDHsds*DHs*(1.0 + ss2*F) + 
			12.0*DHs2*(-BB*dDHsds + CC*ss*(dFds*ss + 2.0*F))) -  
	  ((54.0*exp(f94Hs2_A)*M_SQRTPI*ss*(2.0*H + dHds*ss)*erfc(sqrt(f94Hs2_A)))/AA12))); 
 
      dGb = (15.0*M_SQRTPI*ss*(4.0*DHs - 7.0*dDHsds*ss))/(32.0*DHs92); 
 
      dEGds = (-4.0*dGa*Gb + dGb*(4.0*Ga + 3.0*M_PI))/(4.0*Gb*Gb); 
    } 
  }else{ 
    static const double EGa1=-0.02628417880, EGa2=-0.07117647788, EGa3=0.08534541323; 
 
    EG = EGa1 + EGa2*ss2 + EGa3*ss4; 
 
    if(order >= 1){ 
      dEGds = 2.0*EGa2*ss + 4.0*EGa3*ss3; 
    } 
  } 
 
  /* Calculate the terms needed in any case */ 
  term2 = (DHs2*BB + DHs*CC + 2.0*EE + DHs*ss2*CC*F + 2.0*ss2*EG)/(2.0*DHs3); 
  term3 = -ww*(4.0*DHsw2*BB + 6.0*DHsw*CC + 15.0*EE + 6.0*DHsw*ss2*CC*F + 15.0*ss2*EG)/(8.0*DHs*DHsw52); 
  term4 = -ww3*(DHsw*CC + 5.0*EE + DHsw*ss2*CC*F + 5.0*ss2*EG)/(2.0*DHs2*DHsw52); 
  term5 = -ww5*(EE + ss2*EG)/(DHs3*DHsw52); 
   
  if(order >=1){ 
    dterm2ds = (-6.0*dDHsds*(EG*ss2 + EE) 
		+ DHs2*(-dDHsds*BB + ss*CC*(dFds*ss + 2.0*F)) 
		+ 2.0*DHs*(2.0*EG*ss - dDHsds*CC + ss2*(dEGds - dDHsds*CC*F))) 
      /(2.0*DHs4); 
 
    dterm3ds = ww*(2.0*dDHsds*DHsw*(4.0*DHsw2*BB + 6.0*DHsw*CC + 15.0*EE + 3.0*ss2*(5.0*EG + 2.0*DHsw*CC*F)) 
		   + DHs*(75.0*dDHsds*(EG*ss2 + EE) + 4.0*DHsw2*(dDHsds*BB - 3.0*ss*CC*(dFds*ss + 2.0*F)) 
			  - 6.0*DHsw*(-3.0*dDHsds*CC + ss*(10.0*EG + 5.0*dEGds*ss - 3.0*dDHsds*ss*CC*F)))) 
      /(16.0*DHs2*DHsw72); 
 
    dterm3dw = (-2.0*DHsw*(4.0*DHsw2*BB + 6.0*DHsw*CC + 15.0*EE + 3.0*ss2*(5.0*EG + 2.0*DHsw*CC*F)) 
		+ ww*dDHswdw*(75.0*(EG*ss2 + EE) + 2.0*DHsw*(2.0*DHsw*BB + 9.0*CC + 9.0*ss2*CC*F))) 
      /(16.0*DHs*DHsw72); 
 
    dterm4ds = ww3*(4.0*dDHsds*DHsw*(DHsw*CC + 5.0*EE + ss2*(5.0*EG + DHsw*CC*F)) 
		   + DHs*(25.0*dDHsds*(EG*ss2 + EE) - 2.0*DHsw2*ss*CC*(dFds*ss + 2.0*F) 
			  + DHsw*(3.0*dDHsds*CC + ss*(-20.0*EG - 10.0*dEGds*ss + 3.0*dDHsds*ss*CC*F)))) 
      /(4.0*DHs3*DHsw72); 
 
    dterm4dw = ww2*(-6.0*DHsw*(DHsw*CC + 5.0*EE + ss2*(5.0*EG + DHsw*CC*F)) 
		    + ww*dDHswdw*(25.0*(EG*ss2 + EE) + 3.0*DHsw*CC*(1.0 + ss2*F))) 
      /(4.0*DHs2*DHsw72); 
 
    dterm5ds = ww5*(6.0*dDHsds*DHsw*(EG*ss2 + EE) +  
		    DHs*(-2.0*DHsw*ss*(2.0*EG + dEGds*ss) + 5.0*dDHsds*(EG*ss2 + EE))) 
      /(2.0*DHs4*DHsw72); 
 
    dterm5dw = ww4*5.0*(EG*ss2 + EE)*(-2.0*DHsw + dDHswdw*ww) 
      /(2.0*DHs3*DHsw72); 
  } 
 
  if((ss > 0.0) || (ww > 0.0)){ 
    double dt10; 
 
    t10 = 0.5*AA*log(Hsbw/DHsbw); 
 
    if(order >= 1){ 
      dt10 = 0.5*AA*(1.0/Hsbw - 1.0/DHsbw); 
 
      dt10ds = dt10*dHsbwds; 
      dt10dw = dt10*dHsbwdw; 
    } 
  } 
 
  /* Calculate exp(x)*f(x) depending on size of x */ 
  if(HsbwA94 < expfcutoff){ 
    piexperf = M_PI*exp(HsbwA94)*erfc(HsbwA9412); 
    expei    = exp(HsbwA94)*(-expint_e1(HsbwA94)); 
 
  }else{ 
    static const double expei1=4.03640, expei2=1.15198, expei3=5.03627, expei4=4.19160; 
 
    piexperf = M_PI*(1.0/(M_SQRTPI*HsbwA9412) - 1.0/(2.0*sqrt(M_PI*HsbwA943))+ 3.0/(4.0*sqrt(M_PI*HsbwA945))); 
    expei  = - (1.0/HsbwA94)*(HsbwA942 + expei1*HsbwA94 + expei2)/(HsbwA942 + expei3*HsbwA94 + expei4); 
  } 
 
  if(order >= 1){ 
    double dpiexperf, dexpei; 
 
    dpiexperf   = -(3.0*M_SQRTPI*sqrt(Hsbw/AA))/(2.0*Hsbw) + (9.0*piexperf)/(4.0*AA); 
    dpiexperfds = dpiexperf*dHsbwds; 
    dpiexperfdw = dpiexperf*dHsbwdw; 
 
    dexpei  = 1.0/Hsbw + 9.0*expei/(4.0*AA); 
    dexpeids = dexpei*dHsbwds; 
    dexpeidw = dexpei*dHsbwdw; 
  } 
 
  if (ww == 0.0){ /* Fall back to original expression for the PBE hole */ 
    double t1, dt1ds, dt1dw; 
 
    if(ss > MIN_GRAD){ 
      t1    = -0.5*AA*expei; 
      *f    = m89*(t1 + t10 + term2); 
    }else{ 
      *f = 1.0; 
    } 
 
    if(order >= 1){ 
      if(ss > MIN_GRAD){ 
	dt1ds  = -0.5*AA*dexpeids; 
	dt1dw  = -0.5*AA*dexpeidw; 
	 
	*dfdx  = m89*(dt1ds + dt10ds + dterm2ds); 
	*lvrho = m89*(dt1dw + dt10dw); 
      }else{ 
	*dfdx  = 0.0; 
	*lvrho = 0.0; 
      } 
    } 
 
  }else if(ww > wcutoff){ /* Use simple gaussian approximation for large w */ 
    double dterm1; 
 
    term1   = -0.5*AA*(expei + log(DHsbw) - log(Hsbw)); 
    *f = m89*(term1 + term2 + term3 + term4 + term5); 
 
    if(order >= 1){ 
      dterm1   =  -AA/(2.0*DHsbw) + m89*expei; 
 
      dterm1ds = dterm1*dHsbwds; 
      dterm1dw = dterm1*dHsbwdw; 
 
      *dfdx  = m89*(dterm1ds + dterm2ds + dterm3ds + dterm4ds + dterm5ds); 
      *lvrho = m89*(dterm1dw + dterm3dw + dterm4dw + dterm5dw); 
    } 
 
  }else{ /*  For everything else use the full blown expression */ 
 
    static const double ea1=-1.128223946706117, ea2=1.452736265762971, ea3=-1.243162299390327, 
      ea4=0.971824836115601, ea5=-0.568861079687373, ea6=0.246880514820192, ea7=-0.065032363850763, 
      ea8=0.008401793031216; 
 
    double np1, np2, t1, f2, f3, f4, f5, f6, f7, f8, f9, t2t9; 
    double dnp1dw, dnp2dw, dt1ds, dt1dw, df2, df2ds, df2dw, df3, df3ds, df3dw; 
    double df4, df4ds, df4dw, df5, df5ds, df5dw, df6, df6ds, df6dw, df7ds, df7dw; 
    double df8, df8ds, df8dw, df9, df9ds, df9dw, dt2t9ds, dt2t9dw; 
 
    np1 = -1.5*ea1*AA12*ww + 27.0*ea3*ww3/(8.0*AA12) - 243.0*ea5*ww5/(32.0*AA32) + 2187.0*ea7*ww7/(128.0*AA52); 
    np2 = -AA + 9.0*ea2*ww2/4.0 - 81.0*ea4*ww4/(16.0*AA) + 729.0*ea6*ww6/(64.0*AA2) - 6561.0*ea8*ww8/(256.0*AA3); 
 
    t1 = 0.5*(np1*piexperf + np2*expei); 
 
    f2 = 0.5*ea1*M_SQRTPI*AA/DHsbw12; 
    f3 = 0.5*ea2*AA/DHsbw; 
    f4 = ea3*M_SQRTPI*(-9.0/(8.0*Hsbw12) + 0.25*AA/DHsbw32); 
    f5 = (ea4/128.0)*(-144.0/Hsbw + 64.0*AA/DHsbw2); 
    f6 = ea5*(3.0*M_SQRTPI*(3.0*DHsbw52*(9.0*Hsbw - 2.0*AA) 
			    + 4.0*Hsbw32*AA2))/(32.0*DHsbw52*Hsbw32*AA); 
    f7 = ea6*((32.0*AA/DHsbw3 + (-36.0 + 81.0*ss2*H/AA)/Hsbw2))/32.0; 
    f8 = ea7*(-3.0*M_SQRTPI*(-40.0*Hsbw52*AA3 + 9.0*DHsbw72*(27.0*Hsbw2 - 6.0*Hsbw*AA + 4.0*AA2)))/(128.0*DHsbw72*Hsbw52*AA2); 
    f9 = (324.0*ea6*eb1*DHsbw4*Hsbw*AA + ea8*(384.0*Hsbw3*AA3 + DHsbw4*(-729.0*Hsbw2 + 324.0*Hsbw*AA - 288.0*AA2)))/(128.0*DHsbw4*Hsbw3*AA2); 
 
    t2t9  = f2*ww + f3*ww2 + f4*ww3 + f5*ww4 + f6*ww5 + f7*ww6 + f8*ww7 + f9*ww8; 
 
    term1 = t1 + t2t9 + t10; 
 
    *f = m89*(term1 + term2 + term3 + term4 + term5); 
 
    if(order >= 1){ 
      dnp1dw = -1.5*ea1*AA12 + 81.0*ea3*ww2/(8.0*AA12) - 1215.0*ea5*ww4/(32.0*AA32) 
	+ (15309.0*ea7*ww6)/(128.0*AA52); 
 
      dnp2dw = 0.5*9.0*ea2*ww - 81.0*ea4*ww3/(4.0*AA) + 2187.0*ea6*ww5/(32.0*AA2) 
	- 6561.0*ea8*ww7/(32.0*AA3); 
 
      dt1ds = 0.5*(dpiexperfds*np1 + dexpeids*np2); 
      dt1dw = 0.5*(dnp2dw*expei + dpiexperfdw*np1 + dexpeidw*np2 + dnp1dw*piexperf); 
 
      df2   = -ea1*M_SQRTPI*AA/(4.0*DHsbw32); 
      df2ds = df2*dHsbwds; 
      df2dw = df2*dHsbwdw; 
 
      df3   = -ea2*AA/(2.0*DHsbw2); 
      df3ds = df3*dHsbwds; 
      df3dw = df3*dHsbwdw; 
 
      df4   = ea3*M_SQRTPI*(9.0/(16.0*Hsbw32)- 3.0*AA/(8.0*DHsbw52)); 
      df4ds = df4*dHsbwds; 
      df4dw = df4*dHsbwdw; 
 
      df5   = ea4*(9.0/(8.0*Hsbw2) - AA/DHsbw3); 
      df5ds = df5*dHsbwds; 
      df5dw = df5*dHsbwdw; 
 
      df6   = ea5*M_SQRTPI*(27.0/(32.0*Hsbw52)- 81.0/(64.0*Hsbw32*AA) - 15.0*AA/(16.0*DHsbw72)); 
      df6ds = df6*dHsbwds; 
      df6dw = df6*dHsbwdw; 
 
      df7ds = ea6*(3.0*(27.0*dHds*DHsbw4*Hsbw*ss2 + 8.0*dHsbwds*AA*(3.0*DHsbw4 - 4.0*Hsbw3*AA) + 
			54.0*DHsbw4*ss*(Hsbw - dHsbwds*ss)*H))/(32.0*DHsbw4*Hsbw3*AA); 
      df7dw = ea6*dHsbwdw*(9.0/(4.0*Hsbw3) - 3.0*AA/DHsbw4 - 81.0*ss2*H/(16.0*Hsbw3*AA)); 
 
      df8   = ea7*M_SQRTPI*(135.0/(64.0*Hsbw72) + 729.0/(256.0*Hsbw32*AA2) 
			    - 243.0/(128.0*Hsbw52*AA) - 105.0*AA/(32.0*DHsbw92)); 
      df8ds = df8*dHsbwds; 
      df8dw = df8*dHsbwdw; 
 
      df9   =  -81.0*ea6*eb1/(16.0*Hsbw3*AA) + 
	ea8*(27.0/(4.0*Hsbw4) + 729.0/(128.0*Hsbw2*AA2) - 81.0/(16.0*Hsbw3*AA) - 12.0*AA/DHsbw5); 
      df9ds = df9*dHsbwds; 
      df9dw = df9*dHsbwdw; 
 
      dt2t9ds = df2ds*ww + df3ds*ww2 + df4ds*ww3 + df5ds*ww4 + 
	df6ds*ww5 + df7ds*ww6 + df8ds*ww7 + df9ds*ww8; 
      dt2t9dw = f2 + df2dw*ww  + 2.0*f3*ww + df3dw*ww2 + 
	3.0*f4*ww2 + df4dw*ww3 + 4.0*f5*ww3 + df5dw*ww4 +  
	5.0*f6*ww4 + df6dw*ww5 + 6.0*f7*ww5 + df7dw*ww6 +  
	7.0*f8*ww6 + df8dw*ww7 + 8.0*f9*ww7 + df9dw*ww8; 
 
      dterm1ds = dt1ds + dt2t9ds + dt10ds; 
      dterm1dw = dt1dw + dt2t9dw + dt10dw; 
 
      *dfdx  = m89*(dterm1ds + dterm2ds + dterm3ds + dterm4ds + dterm5ds); 
      *lvrho = m89*(dterm1dw + dterm3dw + dterm4dw + dterm5dw);       
    } 
 
  } 
 
  /* scale and convert to the right variables */ 
  if(order >= 1){ 
    *dfdx  *= dssdx; 
    *lvrho *= dwdrho; 
  } 
} 
 
/* convert into work_gga_c_ variables */ 
static inline void  
func(const xc_func_type *p, xc_gga_work_c_t *r) 
{ 
  int i; 
  double ds, ex, f, lvrho, dexdrs, ddsdrs, dexdz, ddsdz; 
  double sign[2] = {1.0, -1.0}; 
 
  r->f     = 0.0; 
  r->dfdrs = 0.0; 
  r->dfdz  = 0.0; 
 
  for(i=0; i<2; i++){ 
    ds = pow(RS_FACTOR/r->rs, 3.0)*(1.0 + sign[i]*r->z)/2.0; 
    func_3(p, r->order, r->xs[i], ds, &f, &(r->dfdxs[i]), &lvrho); 
 
    ex = -X_FACTOR_C*RS_FACTOR*pow((1.0 + sign[i]*r->z)/2.0, 4.0/3.0)/r->rs; 
 
    r->f += ex*f; 
 
    if(r->order < 1) continue; 
 
    ddsdrs = -3.0*ds/r->rs; 
    dexdrs = -ex/r->rs; 
 
    ddsdz  = pow(RS_FACTOR/r->rs, 3.0)*sign[i]*r->z/2.0; 
    dexdz  = -4.0/6.0*sign[i]*X_FACTOR_C*RS_FACTOR*pow((1.0 + sign[i]*r->z)/2.0, 1.0/3.0)/r->rs; 
 
    r->dfdrs    += dexdrs*f + ex*lvrho*ddsdrs; 
    r->dfdz     += dexdz *f + ex*lvrho*ddsdz; 
    r->dfdxs[i] *= ex; 
  } 
} 
 
#if defined CUDA || defined CUDA_MPIV
#define kernel_id -1
#endif 
#include "work_gga_c.c" 
 
const xc_func_info_type xc_func_info_gga_x_wpbeh = { 
  XC_GGA_X_WPBEH, 
  XC_EXCHANGE, 
  "short-range part of the PBE (default w=0 gives PBEh)", 
  XC_FAMILY_GGA, 
  {&xc_ref_Heyd2003_8207, &xc_ref_Heyd2003_8207_err, &xc_ref_Ernzerhof1998_3313, &xc_ref_Heyd2004_7274, &xc_ref_Henderson2009_044108}, 
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC, 
  1e-32, 
  0, NULL, NULL, 
  gga_x_wpbeh_init, NULL,  
  NULL, work_gga_c, NULL 
}; 
