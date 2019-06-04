/* C. Kallidonis: Header file for the qlua-interface
 * lib/interface_qlua.cpp
 */

#ifndef INTERFACE_QLUA_H__
#define INTERFACE_QLUA_H__

#ifndef LONG_T
#define LONG_T long long
#endif
typedef double QUDA_REAL;

#ifdef __cplusplus

#define XTRN_CPLX std::complex<QUDA_REAL>

#define EXTRN_C extern "C"

#define delete_not_null(p) do { if (NULL != (p)) delete (p) ; } while(0)

#else /*!defined(__cplusplus)*/
#include <complex.h>
#define EXTRN_C
#define XTRN_CPLX double complex
#endif/*__cplusplus*/

#define QUDA_Nc 3
#define QUDA_Ns 4
#define QUDA_DIM 4
#define QUDA_MAX_RANK 6
#define QUDA_MAX_NVEC 32
#define QUDA_NVEC_PROP 12
#define QUDA_NTYPE_CONTRACT 14
#define QUDA_TIME_AXIS 3
#define QUDA_LEN_G (QUDA_Ns*QUDA_Ns)

#define PI 2*asin(1.0)
#define THREADS_PER_BLOCK 64

extern const char *qcContractTypeStr[QUDA_NTYPE_CONTRACT];

typedef enum qluaCntrQQ_Id_s{
  cntr12 = 12,
  cntr13 = 13,
  cntr14 = 14,
  cntr23 = 23,
  cntr24 = 24,
  cntr34 = 34,
  cntr_INVALID = 0
} qluaCntrQQ_Id;

typedef enum {
  what_none           = 0,
  what_qbarq_g_F_B    = 1,
  what_qbarq_g_F_aB   ,   /*  2 */
  what_qbarq_g_F_hB   ,   /*  3 */
  what_qbarq_g_vD_vD  ,   /*  4 */
  what_qbarq_g_vD_avD ,   /*  5 */
  what_qbarq_g_vD_hvD ,   /*  6 */
  what_meson_F_B      ,   /*  7 */
  what_meson_F_aB     ,   /*  8 */
  what_meson_F_hB     ,   /*  9 */
  what_baryon_sigma_UUS,  /* 10 */
  what_qpdf_g_F_B     ,   /* 11 */
  what_tmd_g_F_B      ,   /* 12 */
  what_bb_g_F_B           /* 13 */
} qluaCntr_Type;

struct qudaLattice_s { 
  int node;
  int rank;
  int net[QUDA_MAX_RANK];
  int net_coord[QUDA_MAX_RANK];
  /* local volume : lo[mu] <= x[mu] < hi[mu] */
  int site_coord_lo[QUDA_MAX_RANK];
  int site_coord_hi[QUDA_MAX_RANK];
  LONG_T locvol;
  LONG_T *ind_qdp2quda;
};
typedef struct qudaLattice_s qudaLattice;

typedef struct {
  QUDA_REAL alpha[QUDA_MAX_RANK];
  QUDA_REAL beta;
  int Nstep;
  QudaVerbosity verbosity;
} wuppertalParam;

typedef struct {
  QUDA_REAL alpha;
  int Nstep;
  QUDA_REAL tol;
} APESmearParam;

typedef struct {
  int nVec;
  qluaCntrQQ_Id cntrID;
} QQParam;

typedef struct {
  int nVec;
  int QsqMax;
  int Nmoms;
  int Ndata;
  int expSgn;
  int GPU_phaseMatrix;
  int momDim;
  LONG_T V3;
  LONG_T totV3;
  int tAxis;
  int Tdim;
  int locT;
  double bc_t;
  int csrc[QUDA_DIM];
  LONG_T locvol;
  int push_res;
  qluaCntr_Type cntrType;  
  int localL[QUDA_DIM];
  int totalL[QUDA_DIM];
  int bb_max_depth;
} cntrParam;

typedef struct {
  QudaVerbosity verbosity;
  wuppertalParam wParam;
  APESmearParam apeParam;
  QQParam cQQParam;
  cntrParam mpParam;
  int preserveBasis;
  char shfFlag[512];
  char shfType[512];
  int qdp2quda;
} qudaAPI_Param;

typedef struct{
  QUDA_REAL re;
  QUDA_REAL im;
} QLUA_COMPLEX;

typedef enum RotateType_s {
  QLUA_qdp2quda = 1,
  QLUA_quda2qdp = 2
} RotateType;
//----------------------------------------------------------------



//---------- Functions in interface_qlua.cpp ----------//
//------- These functions are called within Qlua ------//

EXTRN_C
QudaVerbosity parseVerbosity(const char *v);

EXTRN_C
qluaCntrQQ_Id parseContractIdx(const char *v);

EXTRN_C
qluaCntr_Type parse_qcContractType(const char *s);

EXTRN_C
void Qlua_printMemInfo();

EXTRN_C
int QluaCheckMemoryStatus(long long nElem);


EXTRN_C int
doQQ_contract_Quda(
		   QUDA_REAL *hprop_out,
		   QUDA_REAL *hprop_in1,
		   QUDA_REAL *hprop_in2,
		   const qudaLattice *qS,
		   int nColor, int nSpin,
		   const qudaAPI_Param qAparam);

EXTRN_C int
momentumProjectionPropagator_Quda(
				  QUDA_REAL *corrOut,
				  QUDA_REAL *corrIn,
				  const qudaLattice *qS,
				  qudaAPI_Param paramAPI);

EXTRN_C int
laplacianQuda(
	      QUDA_REAL *hv_out,
	      QUDA_REAL *hv_in,
	      QUDA_REAL *h_gauge[],
	      const qudaLattice *qS,
	      int nColor, int nSpin,
	      const qudaAPI_Param qAparam);


EXTRN_C int
Qlua_invertQuda(
		QUDA_REAL *hv_out,
		QUDA_REAL *hv_in,
		QUDA_REAL *h_gauge[],
		const qudaLattice *qS,
		int nColor, int nSpin,
		qudaAPI_Param qAparam);

EXTRN_C int
QuarkContract_momProj_Quda(XTRN_CPLX *momproj_buf, XTRN_CPLX *corrPosSpc, const qudaLattice *qS, const int *momlist,
			   QUDA_REAL *hprop1, QUDA_REAL *hprop2, QUDA_REAL *hprop3, QUDA_REAL *h_gauge[],
			   XTRN_CPLX *S2, XTRN_CPLX *S1,
			   int Nc, int Ns, qudaAPI_Param paramAPI);



//----- TMD related functions -----//

EXTRN_C int
TMD_QPDF_initState_Quda(void **Vqcs, const qudaLattice *qS,
			const int *momlist,
			void *qluaPropFrw_host, void *qluaPropBkw_host,
			void *qluaGauge_host[],
			qudaAPI_Param paramAPI);

EXTRN_C int
TMD_QPDF_freeState_Quda(void **Vqcs);


EXTRN_C int
TMDstep_momProj_Quda(void *Vqcs,
		     XTRN_CPLX *momproj_buf,     /* output in Pspace */
		     XTRN_CPLX *corrQuda,        /* output in Xspace if push_res */
		     const char *b_lpath, const char *v_lpath);

EXTRN_C int
QPDFstep_momProj_Quda(void *Vqcs,
		      XTRN_CPLX *momproj_buf,     /* output in Pspace */
		      XTRN_CPLX *corrQuda,        /* output in Xspace if push_res */
		      const char *b_lpath);

EXTRN_C int
BBstep_momProj_Quda(void *Vqcs,
		    XTRN_CPLX *momproj_buf,     /* output in Pspace */
		    XTRN_CPLX *corrQuda,        /* output in Xspace if push_res */
		    const char *b_lpath);

#endif/*INTERFACE_QLUA_H__*/
