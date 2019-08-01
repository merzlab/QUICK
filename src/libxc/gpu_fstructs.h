
typedef struct{ 
  double A, B, c, d; 
} gga_c_lyp_params; 

typedef struct{  
  double beta, gamma, BB;  
} gga_c_pbe_params;  

typedef struct{  
  double beta, alpha;  
} gga_c_zpbeint_params;  

typedef struct{  
  double beta, alpha, omega;  
} gga_c_zvpbeint_params;  

typedef struct {
  double c_ss[5], c_ab[5];
} gga_c_bmk_params;

typedef struct {
  double sogga11_a[6], sogga11_b[6];
} gga_c_sogga11_params;

typedef struct {
  double a, b, c, d, k;
} gga_c_wi_params;

typedef struct{  
  double aa[5], bb[5];  
} gga_k_dk_params;  

typedef struct{  
  double aa, bb, cc;  
} gga_k_ol2_params;  

typedef struct{  
  double gamma, lambda;  
} gga_k_tflw_params;  

typedef struct{  
  double beta, gamma, omega;  
} gga_x_b86_params;  

typedef struct{ 
  double beta, gamma; 
} gga_x_b88_params; 

typedef struct{  
  const double *omega;  
} gga_xc_th1_params;  

typedef struct{  
  double *omega;  
} gga_xc_th3_params;  

typedef struct{  
  double beta0, beta1, beta2;  
} gga_x_ft97_params;  

typedef struct{  
  double omega;  
} gga_x_hjs_b88_v2_params;  

typedef struct{  
  double omega;  
  
  const double *a, *b; /* pointers to the a and b parameters */  
} gga_x_hjs_params;  

typedef struct{  
  int func_id;  
  xc_gga_enhancement_t enhancement_factor;  
} gga_x_ityh_params;  

typedef struct{  
  double gamma, delta;  
} gga_x_kt_params;  

typedef struct{  
  int    modified; /* shall we use a modified version */  
  double threshold; /* when to start using the analytic form */  
  double ip;        /* ionization potential of the species */  
  double qtot;      /* total charge in the region */  
  
  double aa;     /* the parameters of LB94 */  
  double gamm;  
  
  double alpha;  
  double beta;  
} xc_gga_x_lb_params;  

typedef struct{  
  double a;  
  double c1, c2, c3;  
} gga_x_mpbe_params;  

typedef struct{  
  const double (*CC)[4];  
} gga_x_n12_params;  

typedef struct{  
  double a, b, gamma;  
} gga_x_optx_params;  

typedef struct{  
  double kappa, mu;  
  
  /* parameter used in the Odashima & Capelle versions */  
  double lambda;  
} gga_x_pbe_params;  

typedef struct{  
  double kappa, alpha, muPBE, muGE;  
} gga_x_pbeint_params;  

typedef struct{  
  double aa, bb, cc;  
} gga_x_pw86_params;  

typedef struct{  
  double a, b, c, d, f, alpha, expo;  
} gga_x_pw91_params;  

typedef struct{  
  double rpbe_kappa, rpbe_mu;  
} gga_x_rpbe_params;  

typedef struct{  
  int func_id;  
  xc_gga_enhancement_t enhancement_factor;  
} gga_x_sfat_params;  

typedef struct{  
  double kappa, mu, a[6], b[6];  
} gga_x_sogga11_params;  

typedef struct{  
  double A, B, C, D, E;  
} gga_x_ssb_sw_params;  

typedef struct{  
  double mu;  
  double alpha;  
} gga_x_vmt84_params;  

typedef struct{  
  double mu;  
  double alpha;  
} gga_x_vmt_params;  

typedef struct{  
  double omega;  
} gga_x_wpbeh_params;  

typedef struct {
  double a1, b1, alpha;
} gga_x_dk87_params;

typedef struct {
  double c_x[5], c_ss[5], c_ab[5];
} gga_xc_wb97_params;

typedef struct {
  double c_x[5], c_ss[5], c_ab[5];
} gga_xc_b97_params;

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

typedef struct {
  double ap, bp, af, bf;
} lda_c_chachiyo_params;

typedef struct {
  double r[2], c[2];
} lda_c_hl_params;

typedef struct {
  double C1, C2, C3;
} lda_c_lp96_params;

typedef struct {
  double fc, q;
} lda_c_ml1_params;

typedef struct {
  double pp[3], a[3], alpha1[3];
  double beta1[3], beta2[3], beta3[3], beta4[3];
  double fz20;
} lda_c_pw_params;

typedef struct {
  double gamma[2];
  double beta1[2];
  double beta2[2];
  double a[2], b[2], c[2], d[2];
} lda_c_pz_params;

typedef struct {
  double a, b;
} lda_c_wigner_params;

typedef struct {
  double ax;
} lda_k_tf_params;

typedef struct {
  double alpha;
  double a1, a2, a3;
} lda_xc_1d_ehwlrg_params;

/*typedef struct{  
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
*/
