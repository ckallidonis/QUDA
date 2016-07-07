#include <comm_quda.h>
#include <quda_internal.h>
#include <quda.h>
#include <iostream>
#include <complex>
#include <cuda.h>
#include <color_spinor_field.h>
#include <enum_quda.h>
#include <typeinfo>

#ifndef _QUDAQKXTM_KEPLER_H
#define _QUDAQKXTM_KEPLER_H


#define QUDAQKXTM_DIM 4
#define MAX_NSOURCES 1000
#define MAX_NMOMENTA 5000
#define MAX_TSINK 10
#define MAX_DEFLSTEPS 10
#define MAX_PROJS 5

#define LEXIC(it,iz,iy,ix,L) ( (it)*L[0]*L[1]*L[2] + (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_TZY(it,iz,iy,L) ( (it)*L[1]*L[2] + (iz)*L[1] + (iy) )
#define LEXIC_TZX(it,iz,ix,L) ( (it)*L[0]*L[2] + (iz)*L[0] + (ix) )
#define LEXIC_TYX(it,iy,ix,L) ( (it)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_ZYX(iz,iy,ix,L) ( (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )




namespace quda {

  enum SOURCE_T{UNITY,RANDOM};
  enum CORR_SPACE{POSITION_SPACE,MOMENTUM_SPACE};
  enum FILE_WRITE_FORMAT{ASCII_FORM,HDF5_FORM};

  typedef struct {
    int nsmearAPE;
    int nsmearGauss;
    double alphaAPE;
    double alphaGauss;
    int lL[QUDAQKXTM_DIM];
    int Nsources;
    int sourcePosition[MAX_NSOURCES][QUDAQKXTM_DIM];
    QudaPrecision Precision;
    int Q_sq;
    int Q_sq_loop;
    int Ntsink;
    int Nproj[MAX_TSINK];
    int traj;
    bool check_files;
    char *thrp_type[3];
    char *thrp_proj_type[5];
    char *baryon_type[10];
    char *meson_type[10];
    int tsinkSource[MAX_TSINK];
    int proj_list[MAX_TSINK][MAX_PROJS];
    int run3pt_src[MAX_NSOURCES];
    FILE_WRITE_FORMAT CorrFileFormat;
    SOURCE_T source_type;
    CORR_SPACE CorrSpace;
  } qudaQKXTMinfo_Kepler;



  enum WHICHSPECTRUM{SR,LR,SM,LM,SI,LI};

  typedef struct{
    int PolyDeg;     // degree of the Chebysev polynomial
    int nEv;         // Number of the eigenvectors we want
    int nKv;         // total size of Krylov space
    WHICHSPECTRUM spectrumPart; // for which part of the spectrum we want to solve
    bool isACC;            
    double tolArpack;
    int maxIterArpack;
    char arpack_logfile[512];
    double amin;
    double amax;
    bool isEven;
    bool isFullOp;
  }qudaQKXTM_arpackInfo;

  typedef struct{
    int Nstoch;
    unsigned long int seed;
    int Ndump;
    char loop_fname[512];
    int smethod;
    int nSteps_defl;
    int deflStep[MAX_DEFLSTEPS];
    int traj;
    int Nprint;
    int Nmoms;
    int Qsq;
    FILE_WRITE_FORMAT FileFormat;
    char *loop_type[6]; // = {"Scalar", "dOp", "Loops", "LoopsCv", "LpsDw", "LpsDwCv"}
    bool loop_oneD[6];  // = { false  , false,  true   , true    ,  true  ,  true}
    bool useTSM;
    bool fullOp_stochEO;
    int TSM_NHP;
    int TSM_NLP;
    int TSM_NdumpHP;
    int TSM_NdumpLP;
    int TSM_NprintHP;
    int TSM_NprintLP;
    long int TSM_maxiter;
    double TSM_tol;
  }qudaQKXTM_loopInfo;


  // forward declaration
template<typename Float>  class QKXTM_Field_Kepler;
template<typename Float>  class QKXTM_Gauge_Kepler;
template<typename Float>  class QKXTM_Vector_Kepler;
template<typename Float>  class QKXTM_Propagator_Kepler;
template<typename Float>  class QKXTM_Propagator3D_Kepler;
template<typename Float>  class QKXTM_Vector3D_Kepler;

  enum ALLOCATION_FLAG{NONE,HOST,DEVICE,BOTH,BOTH_EXTRA};
  enum CLASS_ENUM{FIELD,GAUGE,VECTOR,PROPAGATOR,PROPAGATOR3D,VECTOR3D};
  enum WHICHPARTICLE{PROTON,NEUTRON};
  enum WHICHPROJECTOR{G4,G5G123,G5G1,G5G2,G5G3};
  enum THRP_TYPE{THRP_LOCAL,THRP_NOETHER,THRP_ONED};

  enum APEDIM{D3,D4};

  //////////////////////////////////////// functions /////////////////////////////////////////////

  void init_qudaQKXTM_Kepler(qudaQKXTMinfo_Kepler *info);
  void printf_qudaQKXTM_Kepler();
  void run_calculatePlaq_kernel(cudaTextureObject_t gaugeTexPlaq, int precision);
  void run_GaussianSmearing(void* out, cudaTextureObject_t vecTex, cudaTextureObject_t gaugeTex, int precision);
  void run_UploadToCuda(void* in,cudaColorSpinorField &qudaVec, int precision, bool isEven);
  void run_DownloadFromCuda(void* out,cudaColorSpinorField &qudaVec, int precision, bool isEven);
  void run_ScaleVector(double a, void* inOut, int precision);
  void run_contractMesons (cudaTextureObject_t texProp1,cudaTextureObject_t texProp2,void* corr, int it, int isource, int precision, CORR_SPACE CorrSpace);
  void run_contractBaryons(cudaTextureObject_t texProp1,cudaTextureObject_t texProp2,void* corr, int it, int isource, int precision, CORR_SPACE CorrSpace);
  void run_rotateToPhysicalBase(void* inOut, int sign, int precision);
  void run_castDoubleToFloat(void *out, void *in);
  void run_castFloatToDouble(void *out, void *in);
  void run_conjugate_vector(void *inOut, int precision);
  void run_apply_gamma5_vector(void *inOut, int precision);
  void run_conjugate_propagator(void *inOut, int precision);
  void run_apply_gamma5_propagator(void *inOut, int precision);
  void run_seqSourceFixSinkPart1(void* out, int timeslice, cudaTextureObject_t tex1, cudaTextureObject_t tex2, int c_nu, int c_c2, WHICHPROJECTOR PID, WHICHPARTICLE PARTICLE, int precision);
  void run_seqSourceFixSinkPart2(void* out, int timeslice, cudaTextureObject_t tex, int c_nu, int c_c2, WHICHPROJECTOR PID, WHICHPARTICLE PARTICLE, int precision);
  void run_fixSinkContractions(void* corrThp_local, void* corrThp_noether, void* corrThp_oneD, cudaTextureObject_t fwdTex, cudaTextureObject_t seqTex, cudaTextureObject_t gaugeTex,
			       WHICHPARTICLE PARTICLE, int partflag, int it, int isource, int precision, CORR_SPACE CorrSpace);
  void invertWritePropsNoApe_SL_v2_Kepler(void **gauge, void **gaugeAPE ,QudaInvertParam *param ,QudaGaugeParam *gauge_param,int *sourcePosition, char *prop_path);
  void invertWritePropsNoApe_SS_v2_Kepler(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,int *sourcePosition, char *prop_path);
  void deflateAndInvert_Kepler(void **gauge, void **gaugeAPE ,QudaInvertParam *param ,QudaGaugeParam *gauge_param, int NeV, char *filename_eigenValues, char *filename_eigenVectors);
  void checkEigenVector_Kepler(void **gauge, void **gaugeAPE, QudaInvertParam *param ,QudaGaugeParam *gauge_param,char *filename_eigenValues, char *filename_eigenVectors, char *filename_out);

  ///////////////////////////////////////////////////// class QKXTM_Field ////////////////////////////////////////////

template<typename Float>
  class QKXTM_Field_Kepler {           // base class use only for inheritance not polymorphism

  protected:

    int field_length;
    int total_length;        
    int ghost_length;
    int total_plus_ghost_length;

    size_t bytes_total_length;
    size_t bytes_ghost_length;
    size_t bytes_total_plus_ghost_length;

    Float *h_elem;
    Float *h_elem_backup;
    Float *d_elem;
    Float *h_ext_ghost;

    bool isAllocHost;
    bool isAllocDevice;
    bool isAllocHostBackup;

    void create_host();
    void create_host_backup();
    void destroy_host();
    void destroy_host_backup();
    void create_device();
    void destroy_device();

  public:
    QKXTM_Field_Kepler(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT);
    virtual ~QKXTM_Field_Kepler();
    void zero_host();
    void zero_host_backup();
    void zero_device();
    void createTexObject(cudaTextureObject_t *tex);
    void destroyTexObject(cudaTextureObject_t tex);

    Float* H_elem() const { return h_elem; }
    Float* D_elem() const { return d_elem; }

    size_t Bytes_total() const { return bytes_total_length; }
    size_t Bytes_ghost() const { return bytes_ghost_length; }
    size_t Bytes_total_plus_ghost() const { return bytes_total_plus_ghost_length; }

    int Precision() const{
      if( typeid(Float) == typeid(float) )
	return 4;
      else if( typeid(Float) == typeid(double) )
	return 8;
      else
	return 0;
    } 
    void printInfo();
  };

  ///////////////////////////////////////////////////////// end QKXTM_Field //////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////// class QKXTM_Gauge /////////////////////////////////////////////////////////////////
 template<typename Float>
   class QKXTM_Gauge_Kepler : public QKXTM_Field_Kepler<Float> {
 public:
  QKXTM_Gauge_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT);
  ~QKXTM_Gauge_Kepler(){;}                       // class destructor
  
  void packGauge(void **gauge);
  void packGaugeToBackup(void **gauge);
  void loadGaugeFromBackup();
  void justDownloadGauge();
  void loadGauge();
  
  void ghostToHost();
  void cpuExchangeGhost();
  void ghostToDevice();
  void calculatePlaq();
  
};


  //////////////////////////////////////////////////////////////////////////// end QKXTM_Gauge //////////////////////////


  ////////////////////////////////////////////////////////////////////// class vector ///////////////////////////////////////////
 template<typename Float>
   class QKXTM_Vector_Kepler : public QKXTM_Field_Kepler<Float> {
  public:
  QKXTM_Vector_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT); // class constructor
  ~QKXTM_Vector_Kepler(){;}                       // class destructor

    void packVector(Float *vector);
    void unpackVector();
    void unpackVector(Float *vector);
    void loadVector();
    void unloadVector();
    void ghostToHost();
    void cpuExchangeGhost();
    void ghostToDevice();

    void download(); //////////////////// take the vector from device to host
    void uploadToCuda(cudaColorSpinorField *cudaVector, bool isEv = false);
    void downloadFromCuda(cudaColorSpinorField *cudaVector, bool isEv = false);
    void gaussianSmearing(QKXTM_Vector_Kepler<Float> &vecIn,QKXTM_Gauge_Kepler<Float> &gaugeAPE);
    void scaleVector(double a);
    void castDoubleToFloat(QKXTM_Vector_Kepler<double> &vecIn);
    void castFloatToDouble(QKXTM_Vector_Kepler<float> &vecIn);
    void norm2Host();
    void norm2Device();
    void copyPropagator3D(QKXTM_Propagator3D_Kepler<Float> &prop, int timeslice, int nu , int c2);
    void copyPropagator(QKXTM_Propagator_Kepler<Float> &prop, int nu , int c2);
    void write(char* filename);
    void conjugate();
    void apply_gamma5();
 };




///////////////////////////////////////////////////////////////////////// end QKXTM_Propagator ///////////////////////////

/////////////////////////////////////////////////////////////////////// class propagator ////////////////////////////////////
 template<typename Float>
  class QKXTM_Propagator_Kepler : public QKXTM_Field_Kepler<Float> {

  public:
  QKXTM_Propagator_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT);
    ~QKXTM_Propagator_Kepler(){;}

    void ghostToHost();
    void cpuExchangeGhost();
    void ghostToDevice();

    void conjugate();
    void apply_gamma5();

    void absorbVectorToHost(QKXTM_Vector_Kepler<Float> &vec, int nu, int c2);
    void absorbVectorToDevice(QKXTM_Vector_Kepler<Float> &vec, int nu, int c2);
    void rotateToPhysicalBase_host(int sign);
    void rotateToPhysicalBase_device(int sign);
  };




  ///////////////////////////////////////////////////////
 template<typename Float>
   class QKXTM_Propagator3D_Kepler : public QKXTM_Field_Kepler<Float> {

  public:
  QKXTM_Propagator3D_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT);
    ~QKXTM_Propagator3D_Kepler(){;}
    
    void absorbTimeSliceFromHost(QKXTM_Propagator_Kepler<Float> &prop, int timeslice);
    void absorbTimeSlice(QKXTM_Propagator_Kepler<Float> &prop, int timeslice);
    void absorbVectorTimeSlice(QKXTM_Vector_Kepler<Float> &vec, int timeslice, int nu, int c2);
    void broadcast(int tsink);
  };

  ///////////////////////////////////////////////////////
 template<typename Float>
   class QKXTM_Vector3D_Kepler : public QKXTM_Field_Kepler<Float> {
  public:
    QKXTM_Vector3D_Kepler();
    ~QKXTM_Vector3D_Kepler(){;}
  };

////////////////////////////////////////////////
 template<typename Float>
   class QKXTM_Contraction_Kepler {
 public:
   QKXTM_Contraction_Kepler(){;}
   ~QKXTM_Contraction_Kepler(){;}
   void contractMesons( QKXTM_Propagator_Kepler<Float> &prop1,QKXTM_Propagator_Kepler<Float> &prop2, char *filename_out, int isource);
   void contractBaryons(QKXTM_Propagator_Kepler<Float> &prop1,QKXTM_Propagator_Kepler<Float> &prop2, char *filename_out, int isource);
   void contractMesons( QKXTM_Propagator_Kepler<Float> &prop1,QKXTM_Propagator_Kepler<Float> &prop2, void *corrMesons , int isource, CORR_SPACE CorrSpace);
   void contractBaryons(QKXTM_Propagator_Kepler<Float> &prop1,QKXTM_Propagator_Kepler<Float> &prop2, void *corrBaryons, int isource, CORR_SPACE CorrSpace);

   void writeTwopBaryons_ASCII(void *corrBaryons, char *filename_out, int isource, CORR_SPACE CorrSpace);
   void writeTwopMesons_ASCII (void *corrMesons , char *filename_out, int isource, CORR_SPACE CorrSpace);

   void copyTwopBaryonsToHDF5_Buf(void *Twop_baryons_HDF5, void *corrBaryons, int isource, CORR_SPACE CorrSpace);
   void copyTwopMesonsToHDF5_Buf (void *Twop_mesons_HDF5 , void *corrMesons , CORR_SPACE CorrSpace);

   void writeTwopBaryonsHDF5(void *twopBaryons, char *filename, qudaQKXTMinfo_Kepler info, int isource);
   void writeTwopMesonsHDF5 (void *twopMesons , char *filename, qudaQKXTMinfo_Kepler info, int isource);

   void writeTwopBaryonsHDF5_MomSpace(void *twopBaryons, char *filename, qudaQKXTMinfo_Kepler info, int isource);
   void writeTwopBaryonsHDF5_PosSpace(void *twopBaryons, char *filename, qudaQKXTMinfo_Kepler info, int isource);
   void writeTwopMesonsHDF5_MomSpace (void *twopMesons , char *filename, qudaQKXTMinfo_Kepler info, int isource);
   void writeTwopMesonsHDF5_PosSpace (void *twopMesons , char *filename, qudaQKXTMinfo_Kepler info, int isource);

   void seqSourceFixSinkPart1(QKXTM_Vector_Kepler<Float> &vec, QKXTM_Propagator3D_Kepler<Float> &prop1, QKXTM_Propagator3D_Kepler<Float> &prop2, int timeslice,int nu,int c2, WHICHPROJECTOR typeProj, WHICHPARTICLE testParticle);
   void seqSourceFixSinkPart2(QKXTM_Vector_Kepler<Float> &vec, QKXTM_Propagator3D_Kepler<Float> &prop, int timeslice,int nu,int c2, WHICHPROJECTOR typeProj, WHICHPARTICLE testParticle);

   void contractFixSink(QKXTM_Propagator_Kepler<Float> &seqProp, QKXTM_Propagator_Kepler<Float> &prop , QKXTM_Gauge_Kepler<Float> &gauge, WHICHPROJECTOR typeProj,
			WHICHPARTICLE testParticle, int partFlag , char *filename_out, int isource, int tsinkMtsource);
   void contractFixSink(QKXTM_Propagator_Kepler<Float> &seqProp, QKXTM_Propagator_Kepler<Float> &prop , QKXTM_Gauge_Kepler<Float> &gauge, 
			void *corrThp_local, void *corrThp_noether, void *corrThp_oneD, WHICHPROJECTOR typeProj , WHICHPARTICLE testParticle, int partFlag, int isource, CORR_SPACE CorrSpace);

   void writeThrp_ASCII(void *corrThp_local, void *corrThp_noether, void *corrThp_oneD, WHICHPARTICLE testParticle, int partflag , char *filename_out, int isource, int tsinkMtsource, CORR_SPACE CorrSpace);
   void copyThrpToHDF5_Buf(void *Thrp_HDF5, void *corrThp,  int mu, int uORd, int its, int Nsink, int pr, int thrp_sign, THRP_TYPE type, CORR_SPACE CorrSpace);
   void writeThrpHDF5(void *Thrp_local_HDF5, void *Thrp_noether_HDF5, void **Thrp_oneD_HDF5, char *filename, qudaQKXTMinfo_Kepler info, int isource, WHICHPARTICLE NUCLEON);
   void writeThrpHDF5_MomSpace(void *Thrp_local_HDF5, void *Thrp_noether_HDF5, void **Thrp_oneD_HDF5, char *filename, qudaQKXTMinfo_Kepler info, int isource, WHICHPARTICLE NUCLEON);
   void writeThrpHDF5_PosSpace(void *Thrp_local_HDF5, void *Thrp_noether_HDF5, void **Thrp_oneD_HDF5, char *filename, qudaQKXTMinfo_Kepler info, int isource, WHICHPARTICLE NUCLEON);


 };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////// function outside quda namespace //////////////
void testPlaquette(void **gauge);
void testGaussSmearing(void **gauge);
void invertWritePropsNoApe_SL_v2_Kepler(void **gauge, void **gaugeAPE ,QudaInvertParam *param ,QudaGaugeParam *gauge_param,quda::qudaQKXTMinfo_Kepler info, char *prop_path);
void invertWritePropsNoApe_SL_v2_Kepler_single(void **gauge, void **gaugeAPE ,QudaInvertParam *param ,QudaGaugeParam *gauge_param,quda::qudaQKXTMinfo_Kepler info, char *prop_path);
void checkReadingEigenVectors(int N_eigenVectors, char* pathIn, char *pathOut, char* pathEigenValues);
void checkEigenVectorQuda(void **gauge,QudaInvertParam *param ,QudaGaugeParam *gauge_param,char *filename_eigenValues, char *filename_eigenVectors, char *filename_out, int NeV);
void checkDeflateVectorQuda(void **gauge,QudaInvertParam *param ,QudaGaugeParam *gauge_param,char *filename_eigenValues, char *filename_eigenVectors, char *filename_out,int NeV);
void checkDeflateAndInvert(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues, char *filename_eigenVectors, char *filename_out,int NeV );
void DeflateAndInvert_twop(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_up, char *filename_eigenVectors_up, char *filename_eigenValues_down, char *filename_eigenVectors_down, char *filename_out,int NeV, quda::qudaQKXTMinfo_Kepler info );
void DeflateAndInvert_loop(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_down, char *filename_eigenVectors_down,char *filename_out , int NeV , int Nstoch, int seed , int NdumpStep, quda::qudaQKXTMinfo_Kepler info);
void DeflateAndInvert_loop_w_One_Der(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_down, char *filename_eigenVectors_down,char *filename_out , int NeV , int Nstoch, int seed , int NdumpStep, quda::qudaQKXTMinfo_Kepler info);
void DeflateAndInvert_loop_w_One_Der_volumeSource(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param,char *filename_eigenValues_up, char *filename_eigenVectors_up, char *filename_eigenValues_down, char *filename_eigenVectors_down,char *filename_out , int NeV , int Nstoch, int seed , int NdumpStep, quda::qudaQKXTMinfo_Kepler info);
void DeflateAndInvert_threepTwop(void **gaugeSmeared, void **gauge, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_up, char *filename_eigenVectors_up, char *filename_eigenValues_down, char *filename_eigenVectors_down, char *filename_twop, char *filename_threep,int NeV, quda::qudaQKXTMinfo_Kepler info, quda::WHICHPARTICLE NUCLEON, quda::WHICHPROJECTOR PID );

void calcEigenVectors(QudaInvertParam *param , quda::qudaQKXTM_arpackInfo arpackInfo);
void calcEigenVectors_Check(QudaInvertParam *param , quda::qudaQKXTM_arpackInfo arpackInfo);

void calcEigenVectors_loop_wOneD_FullOp(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, quda::qudaQKXTM_arpackInfo arpackInfo, quda::qudaQKXTM_arpackInfo arpackInfoEO, quda::qudaQKXTM_loopInfo loopInfo, quda::qudaQKXTMinfo_Kepler info);
void calcEigenVectors_loop_wOneD_EvenOdd(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param,  quda::qudaQKXTM_arpackInfo arpackInfo, quda::qudaQKXTM_loopInfo loopInfo, quda::qudaQKXTMinfo_Kepler info);

void calcEigenVectors_threepTwop_EvenOdd(void **gaugeSmeared, void **gauge, QudaGaugeParam *gauge_param, QudaInvertParam *param, quda::qudaQKXTM_arpackInfo arpackInfo, quda::qudaQKXTMinfo_Kepler info,
					char *filename_twop, char *filename_threep, quda::WHICHPARTICLE NUCLEON);
void calcEigenVectors_threepTwop_FullOp(void **gaugeSmeared, void **gauge, QudaGaugeParam *gauge_param, QudaInvertParam *param, quda::qudaQKXTM_arpackInfo arpackInfo, quda::qudaQKXTMinfo_Kepler info,
					char *filename_twop, char *filename_threep, quda::WHICHPARTICLE NUCLEON, quda::WHICHPROJECTOR PID );


void calcEigenVectors_loop_wOneD_2pt3pt_EvenOdd(void **gaugeSmeared, void **gauge, QudaGaugeParam *gauge_param, QudaInvertParam *param, void **gauge_loop, QudaGaugeParam *gauge_param_loop, QudaInvertParam *param_loop,
						quda::qudaQKXTM_arpackInfo arpackInfo, quda::qudaQKXTM_loopInfo loopInfo,
						quda::qudaQKXTMinfo_Kepler info, char *filename_twop, char *filename_threep, quda::WHICHPARTICLE NUCLEON, quda::WHICHPROJECTOR PID );

void calcEigenVectors_loop_wOneD_EvenOdd_noDefl(double *eigVecs_d, double *eigVals_d, void **gaugeToPlaquette, QudaInvertParam *param, QudaGaugeParam *gauge_param,
						quda::qudaQKXTM_arpackInfo arpackInfo, quda::qudaQKXTM_loopInfo loopInfo, quda::qudaQKXTMinfo_Kepler info);
				 

#endif
