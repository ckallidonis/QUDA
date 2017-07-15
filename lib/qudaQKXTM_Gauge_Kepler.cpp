//-C.K. Interface for performing the loops and the correlation function 
//contractions, including the exact deflation using ARPACK
//#include <qudaQKXTM.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <qudaQKXTM_Kepler.h>
#include <qudaQKXTM_Kepler_utils.h>
#include <errno.h>
#include <mpi.h>
#include <limits>

#ifdef HAVE_MKL
#include <mkl.h>
#endif

#ifdef HAVE_OPENBLAS
#include <cblas.h>
#include <common.h>
#endif

#include <omp.h>
#include <hdf5.h>
 
#define PI 3.141592653589793
 
//using namespace quda;
extern Topology *default_topo;
 
/* Block for global variables */
extern float GK_deviceMemory;
extern int GK_nColor;
extern int GK_nSpin;
extern int GK_nDim;
extern int GK_strideFull;
extern double GK_alphaAPE;
extern double GK_alphaGauss;
extern int GK_localVolume;
extern int GK_totalVolume;
extern int GK_nsmearAPE;
extern int GK_nsmearGauss;
extern bool GK_dimBreak[QUDAQKXTM_DIM];
extern int GK_localL[QUDAQKXTM_DIM];
extern int GK_totalL[QUDAQKXTM_DIM];
extern int GK_nProc[QUDAQKXTM_DIM];
extern int GK_plusGhost[QUDAQKXTM_DIM];
extern int GK_minusGhost[QUDAQKXTM_DIM];
extern int GK_surface3D[QUDAQKXTM_DIM];
extern int GK_CSurf2D[QUDAQKXTM_DIM][QUDAQKXTM_DIM];
extern int GK_SurfDir[QUDAQKXTM_DIM][QUDAQKXTM_DIM][2];
extern int GK_procPP[QUDAQKXTM_DIM][QUDAQKXTM_DIM];
extern int GK_procPM[QUDAQKXTM_DIM][QUDAQKXTM_DIM];
extern int GK_procMP[QUDAQKXTM_DIM][QUDAQKXTM_DIM];
extern int GK_procMM[QUDAQKXTM_DIM][QUDAQKXTM_DIM];
extern bool GK_init_qudaQKXTM_Kepler_flag;
extern int GK_Nsources;
extern int GK_sourcePosition[MAX_NSOURCES][QUDAQKXTM_DIM];
extern int GK_Nmoms;
extern short int GK_moms[MAX_NMOMENTA][3];
// for mpi use global variables
extern MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
extern MPI_Comm GK_spaceComm , GK_timeComm;
extern int GK_localRank;
extern int GK_localSize;
extern int GK_timeRank;
extern int GK_timeSize;


//--------------------------//
// class QKXTM_Gauge_Kepler //
//--------------------------//

template<typename Float>
QKXTM_Gauge_Kepler<Float>::QKXTM_Gauge_Kepler(ALLOCATION_FLAG alloc_flag, 
					      CLASS_ENUM classT): 
  QKXTM_Field_Kepler<Float>(alloc_flag, classT){ ; }

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::packGauge(void **gauge){

  double **p_gauge = (double**) gauge;
  
  for(int dir = 0 ; dir < GK_nDim ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++)
	  for(int part = 0 ; part < 2 ; part++){
	    CC::h_elem[dir*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       iv*2 + part] = 
	      (Float) p_gauge[dir][iv*GK_nColor*GK_nColor*2 + 
				   c1*GK_nColor*2 + c2*2 + part];
	  }

  gaugePacked = true;
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::packGaugeToBackup(void **gauge){
  double **p_gauge = (double**) gauge;
  if(CC::h_elem_backup != NULL){
    for(int dir = 0 ; dir < GK_nDim ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++)
    for(int part = 0 ; part < 2 ; part++){
      CC::h_elem_backup[dir*GK_nColor*GK_nColor*GK_localVolume*2 + 
			c1*GK_nColor*GK_localVolume*2 + 
			c2*GK_localVolume*2 + 
			iv*2 + part] = 
	(Float) p_gauge[dir][iv*GK_nColor*GK_nColor*2 + 
			     c1*GK_nColor*2 + 
			     c2*2 + part];
    }
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }

}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::justDownloadGauge(){
  cudaMemcpy(CC::h_elem,CC::d_elem,CC::bytes_total_length, 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::loadGauge(){
  cudaMemcpy(CC::d_elem,CC::h_elem,CC::bytes_total_length, 
	     cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::loadGaugeFromBackup(){
  if(CC::h_elem_backup != NULL){
    cudaMemcpy(CC::d_elem,CC::h_elem_backup, CC::bytes_total_length, 
	       cudaMemcpyHostToDevice );
    checkCudaError();
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }
}

// gpu collect ghost and send it to host
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::ghostToHost(){   

  // direction x 
  if( GK_localL[0] < GK_totalL[0]){
    int position;
    // number of blocks that we need
    int height = GK_localL[1] * GK_localL[2] * GK_localL[3];
    size_t width = 2*sizeof(Float);
    size_t spitch = GK_localL[0]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    position = GK_localL[0]-1;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);
	  h_elem_offset = (CC::h_elem + 
			   GK_minusGhost[0]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
			   c1*GK_nColor*GK_surface3D[0]*2 + 
			   c2*GK_surface3D[0]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_plusGhost[0]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
			   c1*GK_nColor*GK_surface3D[0]*2 + 
			   c2*GK_surface3D[0]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction y 
  if( GK_localL[1] < GK_totalL[1]){
    int position;
    // number of blocks that we need
    int height = GK_localL[2] * GK_localL[3];
    size_t width = GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[1]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*(GK_localL[1]-1);
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_minusGhost[1]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
			   c1*GK_nColor*GK_surface3D[1]*2 + 
			   c2*GK_surface3D[1]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_plusGhost[1]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
			   c1*GK_nColor*GK_surface3D[1]*2 + 
			   c2*GK_surface3D[1]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  
  // direction z 
  if( GK_localL[2] < GK_totalL[2]){

    int position;
    // number of blocks that we need
    int height = GK_localL[3]; 
    size_t width = GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[2]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1);
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_minusGhost[2]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
			   c1*GK_nColor*GK_surface3D[2]*2 + 
			   c2*GK_surface3D[2]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_plusGhost[2]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
			   c1*GK_nColor*GK_surface3D[2]*2 + 
			   c2*GK_surface3D[2]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction t 
  if( GK_localL[3] < GK_totalL[3]){
    int position;
    int height = GK_nDim*GK_nColor*GK_nColor;
    size_t width = GK_localL[2]*GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[3]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*GK_localL[2]*(GK_localL[3]-1);
    d_elem_offset=CC::d_elem+position*2;
    h_elem_offset=CC::h_elem+GK_minusGhost[3]*GK_nDim*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
    // set minus points to plus area
    position = 0;
    d_elem_offset=CC::d_elem+position*2;
    h_elem_offset=CC::h_elem+GK_plusGhost[3]*GK_nDim*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
  }
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::cpuExchangeGhost(){
  if( comm_size() > 1 ){
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];

    Float *pointer_receive = NULL;
    Float *pointer_send = NULL;

    for(int idim = 0 ; idim < GK_nDim; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	size_t nbytes = 
	  GK_surface3D[idim]*GK_nColor*GK_nColor*GK_nDim*2*sizeof(Float);
	// send to plus
	pointer_receive = CC::h_ext_ghost + (GK_minusGhost[idim]-GK_localVolume)*GK_nColor*GK_nColor*GK_nDim*2;
	pointer_send = CC::h_elem + GK_minusGhost[idim]*GK_nColor*GK_nColor*GK_nDim*2;
	mh_from_back[idim] = comm_declare_receive_relative(pointer_receive,idim,-1,nbytes);
	mh_send_fwd[idim] = comm_declare_send_relative(pointer_send,idim,1,nbytes);
	comm_start(mh_from_back[idim]);
	comm_start(mh_send_fwd[idim]);
	comm_wait(mh_send_fwd[idim]);
	comm_wait(mh_from_back[idim]);
		
	// send to minus
	pointer_receive = CC::h_ext_ghost + (GK_plusGhost[idim]-GK_localVolume)*GK_nColor*GK_nColor*GK_nDim*2;
	pointer_send = CC::h_elem + GK_plusGhost[idim]*GK_nColor*GK_nColor*GK_nDim*2;
	mh_from_fwd[idim] = comm_declare_receive_relative(pointer_receive,idim,1,nbytes);
	mh_send_back[idim] = comm_declare_send_relative(pointer_send,idim,-1,nbytes);
	comm_start(mh_from_fwd[idim]);
	comm_start(mh_send_back[idim]);
	comm_wait(mh_send_back[idim]);
	comm_wait(mh_from_fwd[idim]);
		
	pointer_receive = NULL;
	pointer_send = NULL;

      }
    }

    for(int idim = 0 ; idim < GK_nDim ; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	comm_free(mh_send_fwd[idim]);
	comm_free(mh_from_fwd[idim]);
	comm_free(mh_send_back[idim]);
	comm_free(mh_from_back[idim]);
      }
    }
    
  }
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::ghostToDevice(){
  if(comm_size() > 1){
    Float *host = CC::h_ext_ghost;
    Float *device = CC::d_elem+GK_localVolume*GK_nColor*GK_nColor*GK_nDim*2;
    cudaMemcpy(device,host,CC::bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::calculatePlaq(){
  cudaTextureObject_t tex;

  ghostToHost();
  cpuExchangeGhost();
  ghostToDevice();
  CC::createTexObject(&tex);
  run_calculatePlaq_kernel(tex, sizeof(Float));
  CC::destroyTexObject(tex);

}

//-C.K. Allocate the host corner boundary buffers
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::initHostCornerBufs(){

  size_t buflgh = sizeof(Float)*GK_nDim*GK_nColor*GK_nColor*2;

  //-C.K. This is done on purpose, we want these to be empty!
  for(int mu=0;mu<GK_nDim;mu++){
    h_crnBufPP[mu][mu] = NULL;
    h_crnBufPM[mu][mu] = NULL;
    h_crnBufMP[mu][mu] = NULL;
    h_crnBufMM[mu][mu] = NULL;
  }

  double bytesUsed = 0;
  for(int mu=0;mu<GK_nDim;mu++){
    for(int nu=mu+1;nu<GK_nDim;nu++){
      h_crnBufPP[mu][nu] = (Float*) malloc(buflgh*GK_CSurf2D[mu][nu]);
      h_crnBufPM[mu][nu] = (Float*) malloc(buflgh*GK_CSurf2D[mu][nu]);
      h_crnBufMP[mu][nu] = (Float*) malloc(buflgh*GK_CSurf2D[mu][nu]); 
      h_crnBufMM[mu][nu] = (Float*) malloc(buflgh*GK_CSurf2D[mu][nu]);

      if(h_crnBufPP[mu][nu] == NULL) errorQuda("Cannot allocate Gauge field corner-buffer PP[%d][%d]\n",mu,nu);
      if(h_crnBufPM[mu][nu] == NULL) errorQuda("Cannot allocate Gauge field corner-buffer PM[%d][%d]\n",mu,nu);
      if(h_crnBufMP[mu][nu] == NULL) errorQuda("Cannot allocate Gauge field corner-buffer MP[%d][%d]\n",mu,nu);
      if(h_crnBufMM[mu][nu] == NULL) errorQuda("Cannot allocate Gauge field corner-buffer MM[%d][%d]\n",mu,nu);

      h_crnBufPP[nu][mu] = h_crnBufPP[mu][nu]; // These should contain
      h_crnBufPM[nu][mu] = h_crnBufMP[mu][nu]; // the same information
      h_crnBufMP[nu][mu] = h_crnBufPM[mu][nu]; // No need to
      h_crnBufMM[nu][mu] = h_crnBufMM[mu][nu]; // allocate again

      bytesUsed += 4 * buflgh*GK_CSurf2D[mu][nu];
    }
  }

  zeroHostCornerBufs();
  printfQuda("Host corner buffers for Gauge fields initialized properly! Memory used: %6.1lf MB\n",bytesUsed/(1024*1024));
}

//-C.K. Allocate the device corner boundary buffers
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::initDeviceCornerBufs(){

  size_t buflgh = sizeof(Float)*GK_nDim*GK_nColor*GK_nColor*2;

  //-C.K. This is done on purpose, we want these to be empty!
  for(int mu=0;mu<GK_nDim;mu++){
    d_crnBufPP[mu][mu] = NULL;
    d_crnBufPM[mu][mu] = NULL;
    d_crnBufMP[mu][mu] = NULL;
    d_crnBufMM[mu][mu] = NULL;
  }

  double bytesUsed = 0;
  for(int mu=0;mu<GK_nDim;mu++){
    for(int nu=mu+1;nu<GK_nDim;nu++){
      cudaMalloc((void**)&(d_crnBufPP[mu][nu]),buflgh*GK_CSurf2D[mu][nu]);
      checkCudaError();
      cudaMalloc((void**)&(d_crnBufPM[mu][nu]),buflgh*GK_CSurf2D[mu][nu]);
      checkCudaError();
      cudaMalloc((void**)&(d_crnBufMP[mu][nu]),buflgh*GK_CSurf2D[mu][nu]);
      checkCudaError();
      cudaMalloc((void**)&(d_crnBufMM[mu][nu]),buflgh*GK_CSurf2D[mu][nu]);
      checkCudaError();

      d_crnBufPP[nu][mu] = d_crnBufPP[mu][nu]; // These should contain
      d_crnBufPM[nu][mu] = d_crnBufMP[mu][nu]; // the same information
      d_crnBufMP[nu][mu] = d_crnBufPM[mu][nu]; // No need to
      d_crnBufMM[nu][mu] = d_crnBufMM[mu][nu]; // allocate again

      bytesUsed += 4 * buflgh*GK_CSurf2D[mu][nu];
    }
  }

  zeroDeviceCornerBufs();
  printfQuda("Device corner buffers for Gauge fields initialized properly! Memory used: %6.1lf MB\n",bytesUsed/(1024*1024));
}

//-C.K. Set the host corner boundary buffers to zero
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::zeroHostCornerBufs(){
  
  size_t buflgh = sizeof(Float)*GK_nDim*GK_nColor*GK_nColor*2;
  
  for(int mu=0;mu<GK_nDim;mu++){
    for(int nu=mu+1;nu<GK_nDim;nu++){
      memset(h_crnBufPP[mu][nu], 0, buflgh*GK_CSurf2D[mu][nu]);
      memset(h_crnBufPM[mu][nu], 0, buflgh*GK_CSurf2D[mu][nu]);
      memset(h_crnBufMP[mu][nu], 0, buflgh*GK_CSurf2D[mu][nu]);
      memset(h_crnBufMM[mu][nu], 0, buflgh*GK_CSurf2D[mu][nu]);
    }
  }
}

//-C.K. Set the device corner boundary buffers to zero
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::zeroDeviceCornerBufs(){
  
  size_t buflgh = sizeof(Float)*GK_nDim*GK_nColor*GK_nColor*2;

  for(int mu=0;mu<GK_nDim;mu++){
    for(int nu=mu+1;nu<GK_nDim;nu++){
      cudaMemset(d_crnBufPP[mu][nu], 0, buflgh*GK_CSurf2D[mu][nu]);
      cudaMemset(d_crnBufPM[mu][nu], 0, buflgh*GK_CSurf2D[mu][nu]);
      cudaMemset(d_crnBufMP[mu][nu], 0, buflgh*GK_CSurf2D[mu][nu]);
      cudaMemset(d_crnBufMM[mu][nu], 0, buflgh*GK_CSurf2D[mu][nu]);
    }
  }
  checkCudaError();
}

//-C.K. Free the host corner boundary buffers
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::freeHostCornerBufs(){

  for(int mu=0;mu<GK_nDim;mu++){
    for(int nu=mu+1;nu<GK_nDim;nu++){
      free(h_crnBufPP[mu][nu]); h_crnBufPP[mu][nu] = NULL;
      free(h_crnBufPM[mu][nu]); h_crnBufPM[mu][nu] = NULL;
      free(h_crnBufMP[mu][nu]);	h_crnBufMP[mu][nu] = NULL;
      free(h_crnBufMM[mu][nu]);	h_crnBufMM[mu][nu] = NULL;
    }
  }

}

//-C.K. Free the device corner boundary buffers
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::freeDeviceCornerBufs(){

  for(int mu=0;mu<GK_nDim;mu++){
    for(int nu=mu+1;nu<GK_nDim;nu++){
      cudaFree(d_crnBufPP[mu][nu]);
      cudaFree(d_crnBufPM[mu][nu]);
      cudaFree(d_crnBufMP[mu][nu]);
      cudaFree(d_crnBufMM[mu][nu]);
      checkCudaError();
      
      d_crnBufPP[mu][nu] = NULL;
      d_crnBufPM[mu][nu] = NULL;
      d_crnBufMP[mu][nu] = NULL;
      d_crnBufMM[mu][nu] = NULL;
    }
  }

}


//-C.K. Communicate corner buffers on host (with MPI)
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::commCornersHost(){

  if(!gaugePacked) errorQuda("Gauge commCornerHost: Must pack gauge before calling this function! Exiting.\n");

  int Nc = GK_nColor;
  int Nd = GK_nDim;
  int buflgh = Nd*Nc*Nc*2;
  int dr[2],Lgh[2];
  int vecPP[Nd],vecPM[Nd],vecMP[Nd],vecMM[Nd];
  int Lv = GK_localVolume;

  Float *sendBufPP,*sendBufPM,*sendBufMP,*sendBufMM;

  MPI_Datatype DATATYPE = -1;
  if( typeid(Float) == typeid(float))  DATATYPE = MPI_FLOAT;
  if( typeid(Float) == typeid(double)) DATATYPE = MPI_DOUBLE;

  for(int mu=0;mu<Nd;mu++){
    for(int nu=mu+1;nu<Nd;nu++){
      double t1 = MPI_Wtime();
      vecPP[mu] = GK_localL[mu]-1;
      vecPP[nu] = GK_localL[nu]-1;
      vecPM[mu] = GK_localL[mu]-1;
      vecPM[nu] = 0;
      vecMP[mu] = 0;
      vecMP[nu] = GK_localL[nu]-1;
      vecMM[mu] = 0;
      vecMM[nu] = 0;
      
      for(int i=0;i<2;i++){
	dr[i]  = GK_SurfDir[mu][nu][i];  //- Determine which
	Lgh[i] = GK_localL[dr[i]];       //- directions will run!
      }

      sendBufPP = (Float*) malloc(sizeof(Float)*buflgh*GK_CSurf2D[mu][nu]);
      sendBufPM = (Float*) malloc(sizeof(Float)*buflgh*GK_CSurf2D[mu][nu]);
      sendBufMP = (Float*) malloc(sizeof(Float)*buflgh*GK_CSurf2D[mu][nu]);
      sendBufMM = (Float*) malloc(sizeof(Float)*buflgh*GK_CSurf2D[mu][nu]);
      if(sendBufPP == NULL || sendBufPM == NULL || sendBufMP == NULL || sendBufMM == NULL)
	errorQuda("Gauge comms: Cannot allocate temporary corner buffers for mu = %d, nu = %d. Exiting.\n",mu,nu);

      for(int d0=0;d0<Lgh[0];d0++){    //-Loop over
	for(int d1=0;d1<Lgh[1];d1++){  //-the running directions
	  vecPP[dr[0]] = d0;
	  vecPP[dr[1]] = d1;
	  vecPM[dr[0]] = d0;
	  vecPM[dr[1]] = d1;
	  vecMP[dr[0]] = d0;
	  vecMP[dr[1]] = d1;
	  vecMM[dr[0]] = d0;
	  vecMM[dr[1]] = d1;
	  int ivPP = LEXIC(vecPP[3],vecPP[2],vecPP[1],vecPP[0],GK_localL);
	  int ivPM = LEXIC(vecPM[3],vecPM[2],vecPM[1],vecPM[0],GK_localL);
	  int ivMP = LEXIC(vecMP[3],vecMP[2],vecMP[1],vecMP[0],GK_localL);
	  int ivMM = LEXIC(vecMM[3],vecMM[2],vecMM[1],vecMM[0],GK_localL);

	  for(int dir=0;dir<Nd;dir++){
	    for(int c1=0;c1<Nc;c1++){
	      for(int c2=0;c2<Nc;c2++){
		for(int im=0;im<2;im++){
		  sendBufPP[im + 2*d0 + 2*Lgh[0]*d1 + 2*Lgh[0]*Lgh[1]*c2 + 2*Lgh[0]*Lgh[1]*Nc*c1 + 2*Lgh[0]*Lgh[1]*Nc*Nc*dir] = CC::h_elem[im + 2*ivPP + 2*Lv*c2 + 2*Lv*Nc*c1 + 2*Lv*Nc*Nc*dir];
		  sendBufPM[im + 2*d0 + 2*Lgh[0]*d1 + 2*Lgh[0]*Lgh[1]*c2 + 2*Lgh[0]*Lgh[1]*Nc*c1 + 2*Lgh[0]*Lgh[1]*Nc*Nc*dir] = CC::h_elem[im + 2*ivPM + 2*Lv*c2 + 2*Lv*Nc*c1 + 2*Lv*Nc*Nc*dir];
		  sendBufMP[im + 2*d0 + 2*Lgh[0]*d1 + 2*Lgh[0]*Lgh[1]*c2 + 2*Lgh[0]*Lgh[1]*Nc*c1 + 2*Lgh[0]*Lgh[1]*Nc*Nc*dir] = CC::h_elem[im + 2*ivMP + 2*Lv*c2 + 2*Lv*Nc*c1 + 2*Lv*Nc*Nc*dir];
		  sendBufMM[im + 2*d0 + 2*Lgh[0]*d1 + 2*Lgh[0]*Lgh[1]*c2 + 2*Lgh[0]*Lgh[1]*Nc*c1 + 2*Lgh[0]*Lgh[1]*Nc*Nc*dir] = CC::h_elem[im + 2*ivMM + 2*Lv*c2 + 2*Lv*Nc*c1 + 2*Lv*Nc*Nc*dir];
		}}}
	  }//-dir
	}//-d1
      }//-d0
      double t2 = MPI_Wtime();
      printfQuda("Gauge comms: Corner send-buffers filled, for mu = %d, nu = %d in %f sec\n",mu, nu, t2-t1);


      //- Communicate plus-plus corners
      int statPP =  MPI_Sendrecv((void*) sendBufPP , buflgh*GK_CSurf2D[mu][nu], DATATYPE, GK_procPP[mu][nu], 0,
				 h_crnBufMM[mu][nu], buflgh*GK_CSurf2D[mu][nu], DATATYPE, GK_procMM[mu][nu], 0,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if(statPP!=MPI_SUCCESS)
	errorQuda("Gauge comms: Communication plus-plus did not complete successfully for mu = %d, nu = %d. Exiting.\n",mu,nu);
      

      //- Communicate minus-minus corners
      int statMM =  MPI_Sendrecv((void*) sendBufMM , buflgh*GK_CSurf2D[mu][nu], DATATYPE, GK_procMM[mu][nu], 1,
				 h_crnBufPP[mu][nu], buflgh*GK_CSurf2D[mu][nu], DATATYPE, GK_procPP[mu][nu], 1,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if(statMM!=MPI_SUCCESS)
	errorQuda("Gauge comms: Communication minus-minus did not complete successfully for mu = %d, nu = %d. Exiting.\n",mu,nu);


      //- Communicate plus-minus corners
      int statPM =  MPI_Sendrecv((void*) sendBufPM , buflgh*GK_CSurf2D[mu][nu], DATATYPE, GK_procPM[mu][nu], 2,
				 h_crnBufMP[mu][nu], buflgh*GK_CSurf2D[mu][nu], DATATYPE, GK_procMP[mu][nu], 2,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if(statPM!=MPI_SUCCESS)
	errorQuda("Gauge comms: Communication plus-minus did not complete successfully for mu = %d, nu = %d. Exiting.\n",mu,nu);
      

      //- Communicate minus-plus corners
      int statMP =  MPI_Sendrecv((void*) sendBufMP , buflgh*GK_CSurf2D[mu][nu], DATATYPE, GK_procMP[mu][nu], 3,
				 h_crnBufPM[mu][nu], buflgh*GK_CSurf2D[mu][nu], DATATYPE, GK_procPM[mu][nu], 3,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if(statMP!=MPI_SUCCESS)
	errorQuda("Gauge comms: Communication minus-plus did not complete successfully for mu = %d, nu = %d. Exiting.\n",mu,nu);
      
      free(sendBufPP);
      free(sendBufPM);
      free(sendBufMP);
      free(sendBufMM);

      double t3 = MPI_Wtime();
      printfQuda("Gauge comms: Communication between host corner buffers completed successfully in %f sec.\n",t3-t1);
    }//-nu
  }//-mu

}


//-C.K. Copy host corner buffers to Device
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::copyHostCornersToDevice(){

  size_t buflgh = sizeof(Float)*GK_nDim*GK_nColor*GK_nColor*2;
  int Nd = GK_nDim;

  double t1 = MPI_Wtime();
  for(int mu=0;mu<Nd;mu++){
    for(int nu=mu+1;nu<Nd;nu++){
      cudaMemcpy(d_crnBufPP[mu][nu], h_crnBufPP[mu][nu],buflgh*GK_CSurf2D[mu][nu],cudaMemcpyHostToDevice);
      checkCudaError();
      cudaMemcpy(d_crnBufPM[mu][nu], h_crnBufPM[mu][nu],buflgh*GK_CSurf2D[mu][nu],cudaMemcpyHostToDevice);
      checkCudaError();
      cudaMemcpy(d_crnBufMP[mu][nu], h_crnBufMP[mu][nu],buflgh*GK_CSurf2D[mu][nu],cudaMemcpyHostToDevice);
      checkCudaError();
      cudaMemcpy(d_crnBufMM[mu][nu], h_crnBufMM[mu][nu],buflgh*GK_CSurf2D[mu][nu],cudaMemcpyHostToDevice);
      checkCudaError();
    }
  }

  double t2 = MPI_Wtime();
  printfQuda("Gauge comms: Host corner buffers copied to device successfully in %f sec.\n",t2-t1);
}


//-C.K.Function to create texture objects from the corner buffers
template<typename Float>
void QKXTM_Gauge_Kepler<Float>::createCornerTexObject(cudaTextureObject_t *tex, int mu, int nu, int dd){
  cudaChannelFormatDesc desc;
  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
  int precision = CC::Precision();
  if(precision == 4) desc.f = cudaChannelFormatKindFloat;
  else desc.f = cudaChannelFormatKindSigned;

  if(precision == 4){
    desc.x = 8*precision;
    desc.y = 8*precision;
    desc.z = 0;
    desc.w = 0;
  }
  else if(precision == 8){
    desc.x = 8*precision/2;
    desc.y = 8*precision/2;
    desc.z = 8*precision/2;
    desc.w = 8*precision/2;
  }

  size_t buflgh = sizeof(Float)*GK_nDim*GK_nColor*GK_nColor*2*GK_CSurf2D[mu][nu];

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  if(dd==0) resDesc.res.linear.devPtr = d_crnBufPP[mu][nu];
  if(dd==1) resDesc.res.linear.devPtr = d_crnBufPM[mu][nu];
  if(dd==2) resDesc.res.linear.devPtr = d_crnBufMP[mu][nu];
  if(dd==3) resDesc.res.linear.devPtr = d_crnBufMM[mu][nu];
  resDesc.res.linear.desc = desc;
  resDesc.res.linear.sizeInBytes = buflgh;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::destroyCornerTexObject(cudaTextureObject_t tex){
  cudaDestroyTextureObject(tex);
}
