typedef struct{  
  int interaction;  /* 0: exponentially screened; 1: soft-Coulomb */  
  double bb;         /* screening parameter */  
  
  const double *para, *ferro;  
} lda_c_1d_csc_params;  
typedef struct{  
  double N;  
  double c;  
} lda_c_2d_prm_params;  

typedef struct{  
  int interaction;  /* 0: exponentially screened; 1: soft-Coulomb */  
  double bb;         /* screening parameter beta */  
} lda_x_1d_params;  

typedef struct{ 
  double alpha;       /* parameter for Xalpha functional */ 
} lda_x_params; 

typedef struct{  
  double T;    
  double b[2][5], c[2][3], d[2][5],  e[2][5];  
} lda_xc_ksdt_params;  

typedef struct{  
  double css, copp;  
} mgga_c_bc95_params;  

typedef struct{  
  double gamma_ss, gamma_ab;  
  const double css[5], cab[5];  
} mgga_c_m05_params;  

typedef struct{  
  double gamma_ss, gamma_ab, alpha_ss, alpha_ab;  
  const double css[5], cab[5], dss[6], dab[6];  
} mgga_c_m06l_params;  

typedef struct{  
  const double m08_a[12], m08_b[12];  
} mgga_c_m08_params;  

typedef struct{  
  double beta, d;  
  double C0_c[4];  
} mgga_c_tpss_params;  

typedef struct{  
  const double alpha_ss, alpha_ab;  
  const double dss[6], dab[6];  
} mgga_c_vsxc_params;  

typedef struct{  
  double c;  
} mgga_x_tb09_params;  

typedef struct{  
  const double *a, *d;  
} mgga_x_m06l_params;  

typedef struct{  
  const double a[12], b[12];  
} mgga_x_m08_params;  

typedef struct{  
  const double a[12], b[12];  
} mgga_x_m11_params;  

typedef struct{  
  const double a[12], b[21], c[12], d[12];  
} mgga_x_m11_l_params;  

typedef struct{  
  int legorder;  
  const double *coefs;  
} mgga_x_mbeef_params;  

typedef struct{  
  const double c[40];  
} mgga_x_mn12_params;  

typedef struct{  
  double kappa, c, b;  
} mgga_x_ms_params;  

typedef struct{  
  double c1, c2, d, k1;  
} mgga_x_scan_params;  

typedef struct{  
  const double *cx_local;  
  const double *cx_nlocal;  
} mgga_x_tau_hcth_params;  

typedef struct{  
  double b, c, e, kappa, mu;  
  double BLOC_a, BLOC_b;  
} mgga_x_tpss_params;  

typedef struct{
  double csi_HF;
  const double a[12];
} mgga_x_m05_params;

