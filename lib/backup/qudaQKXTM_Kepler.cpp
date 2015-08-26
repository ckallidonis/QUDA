#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <qudaQKXTM_Kepler.h>
#include <errno.h>
#include <mpi.h>
#include <limits>
#include <mkl.h>
#include <omp.h>

#define PI 3.141592653589793

using namespace quda;
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
extern bool GK_init_qudaQKXTM_Kepler_flag;
extern int GK_Nsources;
extern int GK_sourcePosition[MAX_NSOURCES][QUDAQKXTM_DIM];
extern int GK_Nmoms;
extern short int GK_moms[MAX_NMOMENTA][3];
// for mpi use global  variables                                                                                                                                                                       
extern MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
extern MPI_Comm GK_spaceComm , GK_timeComm;
extern int GK_localRank;
extern int GK_localSize;
extern int GK_timeRank;
extern int GK_timeSize;

//////////////////////////////////////////////////  

//////////////////////////////////// CLASSESS //////////////////////////////
#define CC QKXTM_Field_Kepler<Float>
#define DEVICE_MEMORY_REPORT
#define CMPLX_FLOAT std::complex<Float>
////////////////////////////////// class QKXTM_Field_Kepler ////////////////////
//////////////////////////////////////////////////////////////////////////
template<typename Float>
QKXTM_Field_Kepler<Float>::QKXTM_Field_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT):
  h_elem(NULL) , d_elem(NULL) , h_ext_ghost(NULL) , isAllocHost(false) , isAllocDevice(false)
{
  if(GK_init_qudaQKXTM_Kepler_flag == false) errorQuda("You must initialize init_qudaQKXTM_Kepler first");

  switch(classT){
  case FIELD:
    field_length = 1;
    total_length = GK_localVolume;
    break;
  case GAUGE:
    field_length = GK_nDim * GK_nColor * GK_nSpin;
    total_length = GK_localVolume;
    break;
  case VECTOR:
    field_length = GK_nSpin * GK_nColor;
    total_length = GK_localVolume;
    break;
  case PROPAGATOR:
    field_length = GK_nSpin * GK_nColor * GK_nSpin * GK_nColor;
    total_length = GK_localVolume;
    break;
  case PROPAGATOR3D:
    field_length = GK_nSpin * GK_nColor * GK_nSpin * GK_nColor;
    total_length = GK_localVolume/GK_localL[3];
    break;
  case VECTOR3D:
    field_length = GK_nSpin * GK_nColor;
    total_length = GK_localVolume/GK_localL[3];
    break;
  }

  ghost_length = 0;

  for(int i = 0 ; i < GK_nDim ; i++)
    ghost_length += 2*GK_surface3D[i];

  total_plus_ghost_length = total_length + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(Float);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(Float);
  bytes_total_plus_ghost_length = total_plus_ghost_length*field_length*2*sizeof(Float);

  if( alloc_flag == BOTH ){
    create_host();
    create_device();
  }
  else if (alloc_flag == HOST){
    create_host();
  }
  else if (alloc_flag == DEVICE){
    create_device();
  }
}

template<typename Float>
QKXTM_Field_Kepler<Float>::~QKXTM_Field_Kepler(){
  if(h_elem != NULL) destroy_host();
  if(d_elem != NULL)destroy_device();
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::create_host(){
  h_elem = (Float*) malloc(bytes_total_plus_ghost_length);
  h_ext_ghost = (Float*) malloc(bytes_ghost_length);
  if(h_elem == NULL || h_ext_ghost == NULL)errorQuda("Error with allocation host memory");
  isAllocHost = true;
  zero_host();
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::create_device(){
  cudaMalloc((void**)&d_elem,bytes_total_plus_ghost_length);
  checkCudaError();
#ifdef DEVICE_MEMORY_REPORT
  GK_deviceMemory += bytes_total_length/(1024.*1024.);               // device memory in MB         
  printfQuda("Device memory in used is %f MB A \n",GK_deviceMemory);
#endif
  isAllocDevice = true;
  zero_device();
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::destroy_host(){
  free(h_elem);
  free(h_ext_ghost);
  h_elem=NULL;
  h_ext_ghost = NULL;
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::destroy_device(){
  cudaFree(d_elem);
  checkCudaError();
  d_elem = NULL;
#ifdef DEVICE_MEMORY_REPORT
  GK_deviceMemory -= bytes_total_length/(1024.*1024.);
  printfQuda("Device memory in used is %f MB D \n",GK_deviceMemory);
#endif
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::zero_host(){
  memset(h_elem,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::zero_device(){
  cudaMemset(d_elem,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::createTexObject(cudaTextureObject_t *tex){
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

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_elem;
  resDesc.res.linear.desc = desc;
  resDesc.res.linear.sizeInBytes = bytes_total_plus_ghost_length;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
  checkCudaError();
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::destroyTexObject(cudaTextureObject_t tex){
  cudaDestroyTextureObject(tex);
}

template<typename Float>
void QKXTM_Field_Kepler<Float>::printInfo(){
  printfQuda("This object has precision %d\n",Precision());
  printfQuda("This object needs %f Mb\n",bytes_total_plus_ghost_length/(1024.*1024.));
  printfQuda("The flag for the host allocation is %d\n",(int) isAllocHost);
  printfQuda("The flag for the device allocation is %d\n",(int) isAllocDevice);
}

///////////////////// Class Gauge //////////////////////////////////////////////////////

template<typename Float>
QKXTM_Gauge_Kepler<Float>::QKXTM_Gauge_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT): QKXTM_Field_Kepler<Float>(alloc_flag, classT){ ; }

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::packGauge(void **gauge){
  double **p_gauge = (double**) gauge;

  for(int dir = 0 ; dir < GK_nDim ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++)
	  for(int part = 0 ; part < 2 ; part++){
	    CC::h_elem[dir*GK_nColor*GK_nColor*GK_localVolume*2 + c1*GK_nColor*GK_localVolume*2 + c2*GK_localVolume*2 + iv*2 + part] = (Float) p_gauge[dir][iv*GK_nColor*GK_nColor*2 + c1*GK_nColor*2 + c2*2 + part];
	  }
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::justDownloadGauge(){
  cudaMemcpy(CC::h_elem,CC::d_elem,CC::bytes_total_length, cudaMemcpyDeviceToHost);
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge_Kepler<Float>::loadGauge(){
  cudaMemcpy(CC::d_elem,CC::h_elem,CC::bytes_total_length, cudaMemcpyHostToDevice );
  checkCudaError();
}


template<typename Float>
void QKXTM_Gauge_Kepler<Float>::ghostToHost(){   // gpu collect ghost and send it to host

  // direction x ////////////////////////////////////
  if( GK_localL[0] < GK_totalL[0]){
    int position;
    int height = GK_localL[1] * GK_localL[2] * GK_localL[3]; // number of blocks that we need
    size_t width = 2*sizeof(Float);
    size_t spitch = GK_localL[0]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    position = GK_localL[0]-1;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = CC::d_elem + i*GK_nColor*GK_nColor*GK_localVolume*2 + c1*GK_nColor*GK_localVolume*2 + c2*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_minusGhost[0]*GK_nDim*GK_nColor*GK_nColor*2 + i*GK_nColor*GK_nColor*GK_surface3D[0]*2 + c1*GK_nColor*GK_surface3D[0]*2 + c2*GK_surface3D[0]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = CC::d_elem + i*GK_nColor*GK_nColor*GK_localVolume*2 + c1*GK_nColor*GK_localVolume*2 + c2*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_plusGhost[0]*GK_nDim*GK_nColor*GK_nColor*2 + i*GK_nColor*GK_nColor*GK_surface3D[0]*2 + c1*GK_nColor*GK_surface3D[0]*2 + c2*GK_surface3D[0]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction y ///////////////////////////////////
  if( GK_localL[1] < GK_totalL[1]){
    int position;
    int height = GK_localL[2] * GK_localL[3]; // number of blocks that we need
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
	  d_elem_offset = CC::d_elem + i*GK_nColor*GK_nColor*GK_localVolume*2 + c1*GK_nColor*GK_localVolume*2 + c2*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_minusGhost[1]*GK_nDim*GK_nColor*GK_nColor*2 + i*GK_nColor*GK_nColor*GK_surface3D[1]*2 + c1*GK_nColor*GK_surface3D[1]*2 + c2*GK_surface3D[1]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = CC::d_elem + i*GK_nColor*GK_nColor*GK_localVolume*2 + c1*GK_nColor*GK_localVolume*2 + c2*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_plusGhost[1]*GK_nDim*GK_nColor*GK_nColor*2 + i*GK_nColor*GK_nColor*GK_surface3D[1]*2 + c1*GK_nColor*GK_surface3D[1]*2 + c2*GK_surface3D[1]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  
  // direction z //////////////////////////////////
  if( GK_localL[2] < GK_totalL[2]){

    int position;
    int height = GK_localL[3]; // number of blocks that we need
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
	  d_elem_offset = CC::d_elem + i*GK_nColor*GK_nColor*GK_localVolume*2 + c1*GK_nColor*GK_localVolume*2 + c2*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_minusGhost[2]*GK_nDim*GK_nColor*GK_nColor*2 + i*GK_nColor*GK_nColor*GK_surface3D[2]*2 + c1*GK_nColor*GK_surface3D[2]*2 + c2*GK_surface3D[2]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = CC::d_elem + i*GK_nColor*GK_nColor*GK_localVolume*2 + c1*GK_nColor*GK_localVolume*2 + c2*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_plusGhost[2]*GK_nDim*GK_nColor*GK_nColor*2 + i*GK_nColor*GK_nColor*GK_surface3D[2]*2 + c1*GK_nColor*GK_surface3D[2]*2 + c2*GK_surface3D[2]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction t /////////////////////////////////////
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
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_minusGhost[3]*GK_nDim*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
  // set minus points to plus area
    position = 0;
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_plusGhost[3]*GK_nDim*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
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
	size_t nbytes = GK_surface3D[idim]*GK_nColor*GK_nColor*GK_nDim*2*sizeof(Float);
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
    Float *device = CC::d_elem + GK_localVolume*GK_nColor*GK_nColor*GK_nDim*2;
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
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Vector Class ///////////////////////////////////////////////////////////////////////////

template<typename Float>
QKXTM_Vector_Kepler<Float>::QKXTM_Vector_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT): QKXTM_Field_Kepler<Float>(alloc_flag, classT){ ; }

template<typename Float>
void QKXTM_Vector_Kepler<Float>::packVector(Float *vector){
      for(int iv = 0 ; iv < GK_localVolume ; iv++)
	for(int mu = 0 ; mu < GK_nSpin ; mu++)                // always work with format colors inside spins
	  for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	    for(int part = 0 ; part < 2 ; part++){
	      CC::h_elem[mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2 + iv*2 + part] = vector[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2 + c1*2 + part];
	    }
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::loadVector(){
  cudaMemcpy(CC::d_elem,CC::h_elem,CC::bytes_total_length, cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::castDoubleToFloat(QKXTM_Vector_Kepler<double> &vecIn){
  if(typeid(Float) != typeid(float) )errorQuda("This method works only to convert double to single precision\n");
  run_castDoubleToFloat((void*)d_elem, (void*)vecIn.D_elem());
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::ghostToHost(){
  // direction x ////////////////////////////////////
  if( GK_localL[0] < GK_totalL[0]){
    int position;
    int height = GK_localL[1] * GK_localL[2] * GK_localL[3]; // number of blocks that we need
    size_t width = 2*sizeof(Float);
    size_t spitch = GK_localL[0]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
  // set plus points to minus area
    position = (GK_localL[0]-1);
      for(int mu = 0 ; mu < GK_nSpin ; mu++)
	for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	  d_elem_offset = CC::d_elem + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_minusGhost[0]*GK_nSpin*GK_nColor*2 + mu*GK_nColor*GK_surface3D[0]*2 + c1*GK_surface3D[0]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;
      for(int mu = 0 ; mu < GK_nSpin ; mu++)
	for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	  d_elem_offset = CC::d_elem + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_plusGhost[0]*GK_nSpin*GK_nColor*2 + mu*GK_nColor*GK_surface3D[0]*2 + c1*GK_surface3D[0]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction y ///////////////////////////////////
  if( GK_localL[1] < GK_totalL[1]){
    int position;
    int height = GK_localL[2] * GK_localL[3]; // number of blocks that we need
    size_t width = GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[1]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
  // set plus points to minus area
    position = GK_localL[0]*(GK_localL[1]-1);
      for(int mu = 0 ; mu < GK_nSpin ; mu++)
	for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	  d_elem_offset = CC::d_elem + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_minusGhost[1]*GK_nSpin*GK_nColor*2 + mu*GK_nColor*GK_surface3D[1]*2 + c1*GK_surface3D[1]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;
      for(int mu = 0 ; mu < GK_nSpin ; mu++)
	for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	  d_elem_offset = CC::d_elem + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_plusGhost[1]*GK_nSpin*GK_nColor*2 + mu*GK_nColor*GK_surface3D[1]*2 + c1*GK_surface3D[1]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction z //////////////////////////////////
  if( GK_localL[2] < GK_totalL[2]){
    int position;
    int height = GK_localL[3]; // number of blocks that we need
    size_t width = GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[2]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
  // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1);
      for(int mu = 0 ; mu < GK_nSpin ; mu++)
	for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	  d_elem_offset = CC::d_elem + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_minusGhost[2]*GK_nSpin*GK_nColor*2 + mu*GK_nColor*GK_surface3D[2]*2 + c1*GK_surface3D[2]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;
      for(int mu = 0 ; mu < GK_nSpin ; mu++)
	for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	  d_elem_offset = CC::d_elem + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2 + position*2;  
	  h_elem_offset = CC::h_elem + GK_plusGhost[2]*GK_nSpin*GK_nColor*2 + mu*GK_nColor*GK_surface3D[2]*2 + c1*GK_surface3D[2]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction t /////////////////////////////////////
  if( GK_localL[3] < GK_totalL[3]){
    int position;
    int height = GK_nSpin*GK_nColor;
    size_t width = GK_localL[2]*GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[3]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
  // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*GK_localL[2]*(GK_localL[3]-1);
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_minusGhost[3]*GK_nSpin*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
  // set minus points to plus area
    position = 0;
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_plusGhost[3]*GK_nSpin*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
  }
}


template<typename Float>
void QKXTM_Vector_Kepler<Float>::cpuExchangeGhost(){
  if( comm_size() > 1 ){
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];

    Float *pointer_receive = NULL;
    Float *pointer_send = NULL;

    for(int idim = 0 ; idim < GK_nDim; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	size_t nbytes = GK_surface3D[idim]*GK_nSpin*GK_nColor*2*sizeof(Float);
	// send to plus
	pointer_receive = CC::h_ext_ghost + (GK_minusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_minusGhost[idim]*GK_nSpin*GK_nColor*2;

	mh_from_back[idim] = comm_declare_receive_relative(pointer_receive,idim,-1,nbytes);
	mh_send_fwd[idim] = comm_declare_send_relative(pointer_send,idim,1,nbytes);
	comm_start(mh_from_back[idim]);
	comm_start(mh_send_fwd[idim]);
	comm_wait(mh_send_fwd[idim]);
	comm_wait(mh_from_back[idim]);
		
	// send to minus
	pointer_receive = CC::h_ext_ghost + (GK_plusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_plusGhost[idim]*GK_nSpin*GK_nColor*2;

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
void QKXTM_Vector_Kepler<Float>::ghostToDevice(){ 
  if(comm_size() > 1){
    Float *host = CC::h_ext_ghost;
    Float *device = CC::d_elem + GK_localVolume*GK_nSpin*GK_nColor*2;
    cudaMemcpy(device,host,CC::bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::download(){

  cudaMemcpy(CC::h_elem,CC::d_elem,bytes_total_length, cudaMemcpyDeviceToHost);
  checkCudaError();

  Float *vector_tmp = (Float*) malloc( bytes_total_length );
  if(vector_tmp == NULL)errorQuda("Error in allocate memory of tmp vector");

      for(int iv = 0 ; iv < GK_localVolume ; iv++)
	for(int mu = 0 ; mu < GK_nSpin ; mu++)                // always work with format colors inside spins
	  for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	    for(int part = 0 ; part < 2 ; part++){
	      vector_tmp[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2 + c1*2 + part] = CC::h_elem[mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2 + iv*2 + part];
	    }

      memcpy(h_elem,vector_tmp,bytes_total_length);

  free(vector_tmp);
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::gaussianSmearing(QKXTM_Vector_Kepler<Float> &vecIn,QKXTM_Gauge_Kepler<Float> &gaugeAPE){
  gaugeAPE.ghostToHost();
  gaugeAPE.cpuExchangeGhost();
  gaugeAPE.ghostToDevice();

  vecIn.ghostToHost();
  vecIn.cpuExchangeGhost();
  vecIn.ghostToDevice();

  cudaTextureObject_t texVecIn,texVecOut,texGauge;
  this->createTexObject(&texVecOut);
  vecIn.createTexObject(&texVecIn);
  gaugeAPE.createTexObject(&texGauge);
  
  for(int i = 0 ; i < GK_nsmearGauss ; i++){
    if( (i%2) == 0){
      run_GaussianSmearing((void*)this->D_elem(),texVecIn,texGauge, sizeof(Float));
      this->ghostToHost();
      this->cpuExchangeGhost();
      this->ghostToDevice();
    }
    else{
      run_GaussianSmearing((void*)vecIn.D_elem(),texVecOut,texGauge, sizeof(Float));
      vecIn.ghostToHost();
      vecIn.cpuExchangeGhost();
      vecIn.ghostToDevice();
    }
  }

  if( (GK_nsmearGauss%2) == 0) cudaMemcpy(this->D_elem(),vecIn.D_elem(),bytes_total_length,cudaMemcpyDeviceToDevice);
  
  this->destroyTexObject(texVecOut);
  vecIn.destroyTexObject(texVecIn);
  gaugeAPE.destroyTexObject(texGauge);
  checkCudaError();
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::uploadToCuda(cudaColorSpinorField *qudaVector, bool isEv){
  run_UploadToCuda((void*) d_elem, *qudaVector, sizeof(Float), isEv);
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::downloadFromCuda(cudaColorSpinorField *qudaVector, bool isEv){
  run_DownloadFromCuda((void*) d_elem, *qudaVector, sizeof(Float), isEv);
}

template<typename Float>
void  QKXTM_Vector_Kepler<Float>::scaleVector(double a){
  run_ScaleVector(a,(void*)d_elem,sizeof(Float));
}

template<typename Float>
void QKXTM_Vector_Kepler<Float>::norm2Host(){
  Float res = 0.;
  Float globalRes;

  for(int i = 0 ; i < GK_nSpin*GK_nColor*GK_localVolume ; i++){
    res += h_elem[i*2 + 0]*h_elem[i*2 + 0] + h_elem[i*2 + 1]*h_elem[i*2 + 1];
  }

  int rc = MPI_Allreduce(&res , &globalRes , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  printfQuda("Vector norm2 is %e\n",globalRes);
}

// function for writting
extern "C"{
#include <lime.h>
}

static void qcd_swap_4(float *Rd, size_t N)
{
  register char *i,*j,*k;
  char swap;
  char *max;
  char *R =(char*) Rd;

  max = R+(N<<2);
  for(i=R;i<max;i+=4)
    {
      j=i; k=j+3;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
    }
}


static void qcd_swap_8(double *Rd, int N)
{
   register char *i,*j,*k;
   char swap;
   char *max;
   char *R = (char*) Rd;

   max = R+(N<<3);
   for(i=R;i<max;i+=8)
   {
      j=i; k=j+7;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
   }
}

static int qcd_isBigEndian()
{
   union{
     char C[4];
     int  R   ;
        }word;
   word.R=1;
   if(word.C[3]==1) return 1;
   if(word.C[0]==1) return 0;

   return -1;
}

static char* qcd_getParam(char token[],char* params,int len)
{
  int i,token_len=strlen(token);

  for(i=0;i<len-token_len;i++)
    {
      if(memcmp(token,params+i,token_len)==0)
	{
          i+=token_len;
          *(strchr(params+i,'<'))='\0';
          break;
        }
    }
  return params+i;
}


template<typename Float>
void QKXTM_Vector_Kepler<Float>::write(char *filename){
  FILE *fid;
  int error_in_header=0;
  LimeWriter *limewriter;
  LimeRecordHeader *limeheader = NULL;
  int ME_flag=0, MB_flag=0, limeStatus;
  u_int64_t message_length;
  MPI_Offset offset;
  MPI_Datatype subblock;  //MPI-type, 5d subarray  
  MPI_File mpifid;
  MPI_Status status;
  int sizes[5], lsizes[5], starts[5];
  long int i;
  int chunksize,mu,c1;
  char *buffer;
  int x,y,z,t;
  char tmp_string[2048];

  if(comm_rank() == 0){ // master will write the lime header
    fid = fopen(filename,"w");
    if(fid == NULL){
      fprintf(stderr,"Error open file to write propagator in %s \n",__func__);
      comm_abort(-1);
    }
    else{
      limewriter = limeCreateWriter(fid);
      if(limewriter == (LimeWriter*)NULL) {
	fprintf(stderr, "Error in %s. LIME error in file for writing!\n", __func__);
	error_in_header=1;
	comm_abort(-1);
      }
      else
	{
	  sprintf(tmp_string, "DiracFermion_Sink");
	  message_length=(long int) strlen(tmp_string);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "propagator-type", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
            {
              fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_abort(-1);
            }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
              error_in_header=1;
	      comm_abort(-1);
            }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
              error_in_header=1;
	      comm_abort(-1);
            }

	  if( typeid(Float) == typeid(double) )
	    sprintf(tmp_string, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n\t<field>diracFermion</field>\n\t<precision>64</precision>\n\t<flavours>1</flavours>\n\t<lx>%d</lx>\n\t<ly>%d</ly>\n\t<lz>%d</lz>\n\t<lt>%d</lt>\n\t<spin>4</spin>\n\t<colour>3</colour>\n</etmcFormat>", GK_totalL[0], GK_totalL[1], GK_totalL[2], GK_totalL[3]);
	  else
	    sprintf(tmp_string, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n\t<field>diracFermion</field>\n\t<precision>32</precision>\n\t<flavours>1</flavours>\n\t<lx>%d</lx>\n\t<ly>%d</ly>\n\t<lz>%d</lz>\n\t<lt>%d</lt>\n\t<spin>4</spin>\n\t<colour>3</colour>\n</etmcFormat>", GK_totalL[0], GK_totalL[1], GK_totalL[2], GK_totalL[3]);

	  message_length=(long int) strlen(tmp_string); 
	  MB_flag=1; ME_flag=1;

	  limeheader = limeCreateHeader(MB_flag, ME_flag, "quda-propagator-format", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
            {
              fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_abort(-1);
            }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
              error_in_header=1;
	      comm_abort(-1);
            }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
              error_in_header=1;
	      comm_abort(-1);
            }
	  
	  message_length = GK_totalVolume*4*3*2*sizeof(Float);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", message_length);
	  limeStatus = limeWriteRecordHeader( limeheader, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
              error_in_header=1;
            }
	  limeDestroyHeader( limeheader );
	}
      message_length=1;
      limeWriteRecordData(tmp_string, &message_length, limewriter);
      limeDestroyWriter(limewriter);
      offset = ftell(fid)-1;
      fclose(fid);
    }
  }

  MPI_Bcast(&offset,sizeof(MPI_Offset),MPI_BYTE,0,MPI_COMM_WORLD);
  
  sizes[0]=GK_totalL[3];
  sizes[1]=GK_totalL[2];
  sizes[2]=GK_totalL[1];
  sizes[3]=GK_totalL[0];
  sizes[4]=4*3*2;
  lsizes[0]=GK_localL[3];
  lsizes[1]=GK_localL[2];
  lsizes[2]=GK_localL[1];
  lsizes[3]=GK_localL[0];
  lsizes[4]=sizes[4];
  starts[0]=comm_coords(default_topo)[3]*GK_localL[3];
  starts[1]=comm_coords(default_topo)[2]*GK_localL[2];
  starts[2]=comm_coords(default_topo)[1]*GK_localL[1];
  starts[3]=comm_coords(default_topo)[0]*GK_localL[0];
  starts[4]=0;  

  if( typeid(Float) == typeid(double) )
    MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&subblock);
  else
    MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,MPI_FLOAT,&subblock);

  MPI_Type_commit(&subblock);
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY, MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_FLOAT, subblock, "native", MPI_INFO_NULL);

  chunksize=4*3*2*sizeof(Float);
  buffer = (char*) malloc(chunksize*GK_localVolume);

  if(buffer==NULL)  
    {
      fprintf(stderr,"Error in %s! Out of memory\n", __func__);
      comm_abort(-1);
    }

  i=0;
                        
  for(t=0; t<GK_localL[3];t++)
    for(z=0; z<GK_localL[2];z++)
      for(y=0; y<GK_localL[1];y++)
	for(x=0; x<GK_localL[0];x++)
	  for(mu=0; mu<4; mu++)
	    for(c1=0; c1<3; c1++) // works only for QUDA_DIRAC_ORDER (color inside spin)
	      {
		((Float *)buffer)[i] = h_elem[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 0];
		((Float *)buffer)[i+1] = h_elem[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 1];
		i+=2;
	      }
  if(!qcd_isBigEndian()){
    if( typeid(Float) == typeid(double) ) qcd_swap_8((double*) buffer,2*4*3*GK_localVolume);
    else qcd_swap_4((float*) buffer,2*4*3*GK_localVolume);
  }
  if( typeid(Float) == typeid(double) )
    MPI_File_write_all(mpifid, buffer, 4*3*2*GK_localVolume, MPI_DOUBLE, &status);
  else
    MPI_File_write_all(mpifid, buffer, 4*3*2*GK_localVolume, MPI_FLOAT, &status);

  free(buffer);
  MPI_File_close(&mpifid);
  MPI_Type_free(&subblock);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Propagator class

template<typename Float>
QKXTM_Propagator_Kepler<Float>::QKXTM_Propagator_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT): QKXTM_Field_Kepler<Float>(alloc_flag, classT){;}

template <typename Float>
void QKXTM_Propagator_Kepler<Float>::absorbVectorToHost(QKXTM_Vector_Kepler<Float> &vec, int nu, int c2){
  Float *pointProp_host;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++){
      pointProp_host = CC::h_elem + mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + nu*GK_nColor*GK_nColor*GK_localVolume*2 + c1*GK_nColor*GK_localVolume*2 + c2*GK_localVolume*2;
      pointVec_dev = vec.D_elem() + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2;
      cudaMemcpy(pointProp_host,pointVec_dev,GK_localVolume*2*sizeof(Float),cudaMemcpyDeviceToHost); 
    }
  checkCudaError();
}

template <typename Float>
void QKXTM_Propagator_Kepler<Float>::absorbVectorToDevice(QKXTM_Vector_Kepler<Float> &vec, int nu, int c2){
  Float *pointProp_dev;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++){
      pointProp_dev = CC::d_elem + mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + nu*GK_nColor*GK_nColor*GK_localVolume*2 + c1*GK_nColor*GK_localVolume*2 + c2*GK_localVolume*2;
      pointVec_dev = vec.D_elem() + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2;
      cudaMemcpy(pointProp_dev,pointVec_dev,GK_localVolume*2*sizeof(Float),cudaMemcpyDeviceToDevice); 
    }
  checkCudaError();
}

template<typename Float>
void QKXTM_Propagator_Kepler<Float>::rotateToPhysicalBase_device(int sign){
  if( (sign != +1) && (sign != -1) ) errorQuda("The sign can be only +-1\n");
  run_rotateToPhysicalBase((void*) CC::d_elem, sign , sizeof(Float));
}

template <typename Float>
void QKXTM_Propagator_Kepler<Float>::rotateToPhysicalBase_host(int sign){
  if( (sign != +1) && (sign != -1) ) errorQuda("The sign can be only +-1\n");

  std::complex<Float> P[4][4];
  std::complex<Float> PT[4][4];
  std::complex<Float> imag_unit;
  imag_unit.real() = 0.;
  imag_unit.imag() = 1.;

  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	      
	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++){
	    P[mu][nu].real() = CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + nu*GK_nColor*GK_nColor*GK_localVolume + c1*GK_nColor*GK_localVolume + c2*GK_localVolume + iv)*2 + 0];
	    P[mu][nu].imag() = CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + nu*GK_nColor*GK_nColor*GK_localVolume + c1*GK_nColor*GK_localVolume + c2*GK_localVolume + iv)*2 + 1];
	  }
	
	PT[0][0] = 0.5 * (P[0][0] + sign * ( imag_unit * P[0][2] ) + sign * ( imag_unit * P[2][0] ) - P[2][2]);
	PT[0][1] = 0.5 * (P[0][1] + sign * ( imag_unit * P[0][3] ) + sign * ( imag_unit * P[2][1] ) - P[2][3]);
	PT[0][2] = 0.5 * (sign * ( imag_unit * P[0][0] ) + P[0][2] - P[2][0] + sign * ( imag_unit * P[2][2] ));
	PT[0][3] = 0.5 * (sign * ( imag_unit * P[0][1] ) + P[0][3] - P[2][1] + sign * ( imag_unit * P[2][3] ));
	
	PT[1][0] = 0.5 * (P[1][0] + sign * ( imag_unit * P[1][2] ) + sign * ( imag_unit * P[3][0] ) - P[3][2]);
	PT[1][1] = 0.5 * (P[1][1] + sign * ( imag_unit * P[1][3] ) + sign * ( imag_unit * P[3][1] ) - P[3][3]);
	PT[1][2] = 0.5 * (sign * ( imag_unit * P[1][0] ) + P[1][2] - P[3][0] + sign * ( imag_unit * P[3][2] ));
	PT[1][3] = 0.5 * (sign * ( imag_unit * P[1][1] ) + P[1][3] - P[3][1] + sign * ( imag_unit * P[3][3] ));
	
	PT[2][0] = 0.5 * (sign * ( imag_unit * P[0][0] ) - P[0][2] + P[2][0] + sign * ( imag_unit * P[2][2] ));
	PT[2][1] = 0.5 * (sign * ( imag_unit * P[0][1] ) - P[0][3] + P[2][1] + sign * ( imag_unit * P[2][3] ));
	PT[2][2] = 0.5 * (sign * ( imag_unit * P[0][2] ) - P[0][0] + sign * ( imag_unit * P[2][0] ) + P[2][2]);
	PT[2][3] = 0.5 * (sign * ( imag_unit * P[0][3] ) - P[0][1] + sign * ( imag_unit * P[2][1] ) + P[2][3]);

	PT[3][0] = 0.5 * (sign * ( imag_unit * P[1][0] ) - P[1][2] + P[3][0] + sign * ( imag_unit * P[3][2] ));
	PT[3][1] = 0.5 * (sign * ( imag_unit * P[1][1] ) - P[1][3] + P[3][1] + sign * ( imag_unit * P[3][3] ));
	PT[3][2] = 0.5 * (sign * ( imag_unit * P[1][2] ) - P[1][0] + sign * ( imag_unit * P[3][0] ) + P[3][2]);
	PT[3][3] = 0.5 * (sign * ( imag_unit * P[1][3] ) - P[1][1] + sign * ( imag_unit * P[3][1] ) + P[3][3]);

	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++){
	    CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + nu*GK_nColor*GK_nColor*GK_localVolume + c1*GK_nColor*GK_localVolume + c2*GK_localVolume + iv)*2 + 0] = PT[mu][nu].real();
	    CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + nu*GK_nColor*GK_nColor*GK_localVolume + c1*GK_nColor*GK_localVolume + c2*GK_localVolume + iv)*2 + 1] = PT[mu][nu].imag();
	  }
      }
}
/////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Class Contraction
#define N_MESONS 10
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::contractMesons(QKXTM_Propagator_Kepler<Float> &prop1,QKXTM_Propagator_Kepler<Float> &prop2, char *filename_out, int isource){
  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  Float (*corr_mom_local)[2][N_MESONS] =(Float(*)[2][N_MESONS]) calloc(GK_localL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
  Float (*corr_mom_local_reduced)[2][N_MESONS] =(Float(*)[2][N_MESONS]) calloc(GK_localL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
  Float (*corr_mom)[2][N_MESONS] = (Float(*)[2][N_MESONS]) calloc(GK_totalL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));

  if( corr_mom_local == NULL || corr_mom_local_reduced == NULL || corr_mom == NULL )errorQuda("Error problem to allocate memory");

  for(int it = 0 ; it < GK_localL[3] ; it++){
    run_contractMesons(texProp1,texProp2,(void*) corr_mom_local,it,isource,sizeof(Float));
  }

  int error;

  if( typeid(Float) == typeid(float) ){
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_FLOAT,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_FLOAT,corr_mom,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_FLOAT,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }
  else{
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_DOUBLE,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_DOUBLE,corr_mom,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_DOUBLE,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_MESONS ; ip++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
        for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  fprintf(ptr_out,"%d \t %d \t %+d %+d %+d \t %+e %+e \t %+e %+e\n",ip,it,GK_moms[imom][0],GK_moms[imom][1],GK_moms[imom][2],
		  corr_mom[it_shift*GK_Nmoms*2+imom*2+0][0][ip], corr_mom[it_shift*GK_Nmoms*2+imom*2+1][0][ip], corr_mom[it_shift*GK_Nmoms*2+imom*2+0][1][ip], corr_mom[it_shift*GK_Nmoms*2+imom*2+1][1][ip]);
	}
    fclose(ptr_out);
  }

  free(corr_mom_local);
  free(corr_mom_local_reduced);
  free(corr_mom);
  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

#define N_BARYONS 10
template<typename Float>
void QKXTM_Contraction_Kepler<Float>::contractBaryons(QKXTM_Propagator_Kepler<Float> &prop1,QKXTM_Propagator_Kepler<Float> &prop2, char *filename_out, int isource){
  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  Float (*corr_mom_local)[2][N_BARYONS][4][4] =(Float(*)[2][N_BARYONS][4][4]) calloc(GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  Float (*corr_mom_local_reduced)[2][N_BARYONS][4][4] =(Float(*)[2][N_BARYONS][4][4]) calloc(GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  Float (*corr_mom)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4]) calloc(GK_totalL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));

  if( corr_mom_local == NULL || corr_mom_local_reduced == NULL || corr_mom == NULL )errorQuda("Error problem to allocate memory");

  for(int it = 0 ; it < GK_localL[3] ; it++){
    run_contractBaryons(texProp1,texProp2,(void*) corr_mom_local,it,isource,sizeof(Float));
  }

  int error;

  if( typeid(Float) == typeid(float) ){
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_FLOAT,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_FLOAT,corr_mom,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_FLOAT,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }
  else{
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_DOUBLE,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_DOUBLE,corr_mom,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_DOUBLE,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_BARYONS ; ip++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
        for(int imom = 0 ; imom < GK_Nmoms ; imom++)
	  for(int gamma = 0 ; gamma < 4 ; gamma++)
	    for(int gammap = 0 ; gammap < 4 ; gammap++){
	      int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	      int sign = (it+GK_sourcePosition[isource][3]) > GK_totalL[3] ? -1 : +1;
	      fprintf(ptr_out,"%d \t %d \t %+d %+d %+d \t %d %d \t %+e %+e \t %+e %+e\n",ip,it,GK_moms[imom][0],GK_moms[imom][1],GK_moms[imom][2],gamma,gammap,
		      sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+0][0][ip][gamma][gammap], sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+1][0][ip][gamma][gammap], sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+0][1][ip][gamma][gammap], sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+1][1][ip][gamma][gammap]);
	    }
    fclose(ptr_out);
  }

  free(corr_mom_local);
  free(corr_mom_local_reduced);
  free(corr_mom);
  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

////////////////////////////////////////////////////////////////////////////////////
template<typename Float>
QKXTM_Propagator3D_Kepler<Float>::QKXTM_Propagator3D_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT): QKXTM_Field_Kepler<Float>(alloc_flag, classT){
  if(alloc_flag != BOTH)
    errorQuda("Propagator3D class is only implemented to allocate memory for both\n");
}

template<typename Float>
void QKXTM_Propagator3D_Kepler<Float>::absorbTimeSliceFromHost(QKXTM_Propagator_Kepler<Float> &prop, int timeslice){
  int V3 = GK_localVolume/GK_localL[3];

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++)
	  for(int iv3 = 0 ; iv3 < V3 ; iv3++)
	    for(int ipart = 0 ; ipart < 2 ; ipart++)
	      CC::h_elem[ (mu*GK_nSpin*GK_nColor*GK_nColor*V3 + nu*GK_nColor*GK_nColor*V3 + c1*GK_nColor*V3 + c2*V3 + iv3)*2 + ipart] = prop.H_elem()[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + nu*GK_nColor*GK_nColor*GK_localVolume + c1*GK_nColor*GK_localVolume + c2*GK_localVolume + timeslice*V3 + iv3)*2 + ipart];

  cudaMemcpy(CC::d_elem,CC::h_elem,GK_nSpin*GK_nSpin*GK_nColor*GK_nColor*V3*2*sizeof(Float),cudaMemcpyHostToDevice);
  checkCudaError();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////// Class for deflation

template<typename Float>
class QKXTM_Deflation_Kepler{

private:
  int field_length;
  long int total_length;
  long int total_length_per_NeV;
  bool isEv;
  int NeV;
  size_t bytes_total_length_per_NeV;
  size_t bytes_total_length;
  Float **h_elem;
  Float *eigenValues;
  void create_host();
  void destroy_host();

public:
  QKXTM_Deflation_Kepler(int,bool);
  ~QKXTM_Deflation_Kepler();

  void zero();
  Float** H_elem() const { return h_elem; }
  Float* EigenValues() const { return eigenValues; }
  size_t Bytes() const { return bytes_total_length; }
  size_t Bytes_Per_NeV() const { return bytes_total_length_per_NeV; }
  int NeVs() const { return NeV;}
  void printInfo();

  void readEigenVectors(char *filename);
  void writeEigenVectors_ASCI(char *filename);
  void readEigenValues(char *filename);
  void deflateGuessVector(QKXTM_Vector_Kepler<Float> &vec_guess, QKXTM_Vector_Kepler<Float> &vec_source);
  void copyEigenVectorToQKXTM_Vector_Kepler(int eigenVector_id, Float *vec);
  void copyEigenVectorFromQKXTM_Vector_Kepler(int eigenVector_id,Float *vec);
  void rotateFromChiralToUKQCD();
  void multiply_by_phase();
};

template<typename Float>
QKXTM_Deflation_Kepler<Float>::QKXTM_Deflation_Kepler(int N_EigenVectors,bool isEven): h_elem(NULL), eigenValues(NULL){
  if(GK_init_qudaQKXTM_Kepler_flag == false)errorQuda("You must initialize QKXTM library first\n");
  NeV=N_EigenVectors;
  if(NeV == 0){
    warningQuda("Warning you choose zero eigenVectors\n");
    return;
  }
  isEv=isEven;
  field_length = 4*3;

  total_length_per_NeV = (GK_localVolume/2)*field_length;
  bytes_total_length_per_NeV = total_length_per_NeV*2*sizeof(Float);
  total_length = NeV*(GK_localVolume/2)*field_length;
  bytes_total_length = total_length*2*sizeof(Float);

  h_elem = (Float**)malloc(NeV*sizeof(Float*));
  for(int i = 0 ; i < NeV ; i++){
    h_elem[i] = (Float*)malloc(bytes_total_length_per_NeV);
    if(h_elem[i] == NULL)errorQuda("Error with allocation host memory for deflation class for eigenVector %d\n",i);    
  }

  eigenValues = (Float*)malloc(NeV*sizeof(Float));
  if(eigenValues == NULL)errorQuda("Error with allocation host memory for deflation class\n");

}

template<typename Float>
QKXTM_Deflation_Kepler<Float>::~QKXTM_Deflation_Kepler(){
  if(NeV == 0)return;
  for(int i = 0 ; i < NeV ; i++) free(h_elem[i]);
  free(h_elem);
  free(eigenValues);
  eigenValues=NULL;
  h_elem=NULL;
}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::printInfo(){
  printfQuda("Number of EigenVectors is %d in precision %d\n",NeV,(int) sizeof(Float));
  if(isEv == true) printfQuda("The EigenVectors are for the even-even operator\n");
  if(isEv == false) printfQuda("The EigenVectors are for the odd-odd operator\n");
  printfQuda("Allocated Gb for the eigenVectors space for each node are %lf and the pointer is %p\n",NeV * ( (double)bytes_total_length_per_NeV/((double) 1024.*1024.*1024.) ),h_elem);
}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::rotateFromChiralToUKQCD(){
  if(NeV == 0) return;
  std::complex<Float> transMatrix[4][4];
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++){
      transMatrix[mu][nu].real();
      transMatrix[mu][nu].imag();
    }
  Float value = 1./sqrt(2.);

  transMatrix[0][0].real() = -value; // g4*g5*U
  transMatrix[1][1].real() = -value;
  transMatrix[2][2].real() = +value;
  transMatrix[3][3].real() = +value;

  transMatrix[0][2].real() = +value;
  transMatrix[1][3].real() = +value;
  transMatrix[2][0].real() = +value;
  transMatrix[3][1].real() = +value;

  std::complex<Float> tmp[4];
  std::complex<Float> *vec_cmlx = NULL;

  for(int i = 0 ; i < NeV ; i++){
    vec_cmlx = (std::complex<Float>*) h_elem[i];
    for(int iv = 0 ; iv < (GK_localVolume)/2 ; iv++)
      for(int ic = 0 ; ic < 3 ; ic++){
        memset(tmp,0,4*2*sizeof(Float));
        for(int mu = 0 ; mu < 4 ; mu++)
          for(int nu = 0 ; nu < 4 ; nu++)
            tmp[mu] = tmp[mu] + transMatrix[mu][nu] * vec_cmlx[iv*4*3+nu*3+ic];
        for(int mu = 0 ; mu < 4 ; mu++)
          vec_cmlx[iv*4*3+mu*3+ic] = tmp[mu];
      }
  }

}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::multiply_by_phase(){
  if(NeV == 0)return;
  Float phaseRe, phaseIm;
  Float tmp0,tmp1;
  for(int ivec = 0 ; ivec < NeV ; ivec++)
    for(int t=0; t<GK_localL[3];t++)
      for(int z=0; z<GK_localL[2];z++)
        for(int y=0; y<GK_localL[1];y++)
          for(int x=0; x<GK_localL[0];x++)
            for(int mu=0; mu<4; mu++)
              for(int c1=0; c1<3; c1++)
                {
                  int oddBit     = (x+y+z+t) & 1;
                  if(oddBit){
                    continue;
                  }
                  else{
                    phaseRe = cos(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
                    phaseIm = sin(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
                    int pos = ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+z*GK_localL[1]*GK_localL[0]+y*GK_localL[0]+x)/2)*4*3*2 + mu*3*2 + c1*2 ;
                    tmp0 = h_elem[ivec][ pos + 0] * phaseRe - h_elem[ivec][ pos + 1] * phaseIm;
                    tmp1 = h_elem[ivec][ pos + 0] * phaseIm + h_elem[ivec][ pos + 1] * phaseRe;
		    h_elem[ivec][ pos + 0] = tmp0;
		    h_elem[ivec][ pos + 1] = tmp1;
                  }
                }
}



template<typename Float>
void QKXTM_Deflation_Kepler<Float>::readEigenVectors(char *prefix_path){
  if(NeV == 0)return;
   LimeReader *limereader;
   FILE *fid;
   char *lime_type,*lime_data;
   unsigned long int lime_data_size;
   char dummy;
   MPI_Offset offset;
   MPI_Datatype subblock;  //MPI-type, 5d subarray
   MPI_File mpifid;
   MPI_Status status;
   int sizes[5], lsizes[5], starts[5];
   unsigned int i,j;
   unsigned short int chunksize,mu,c1;
   char *buffer;
   unsigned int x,y,z,t;
   int  isDouble; // default precision
   int error_occured=0;
   int next_rec_is_prop = 0;
   char filename[257];
   
   for(int nev = 0 ; nev < NeV ; nev++){
     sprintf(filename,"%s.%05d",prefix_path,nev);
     if(comm_rank() == 0)
       {
	 /* read lime header */
	 fid=fopen(filename,"r");
	 if(fid==NULL)
	   {
	     fprintf(stderr,"process 0: Error in %s Could not open %s for reading\n",__func__, filename);
	     error_occured=1;
	   }
	 if ((limereader = limeCreateReader(fid))==NULL)
	   {
	     fprintf(stderr,"process 0: Error in %s! Could not create limeReader\n", __func__);
	     error_occured=1;
	   }
	 if(!error_occured)
	   {
	     while(limeReaderNextRecord(limereader) != LIME_EOF )
	       {
		 lime_type = limeReaderType(limereader);
		 if(strcmp(lime_type,"propagator-type")==0)
		   {
		     lime_data_size = limeReaderBytes(limereader);
		     lime_data = (char * )malloc(lime_data_size);
		     limeReaderReadData((void *)lime_data,&lime_data_size, limereader);
		     
		     if (strncmp ("DiracFermion_Source_Sink_Pairs", lime_data, strlen ("DiracFermion_Source_Sink_Pairs"))!=0 &&
			 strncmp ("DiracFermion_Sink", lime_data, strlen ("DiracFermion_Sink"))!=0 )
		       {
			 fprintf (stderr, " process 0: Error in %s! Got %s for \"propagator-type\", expecting %s or %s\n", __func__, lime_data, 
				  "DiracFermion_Source_Sink_Pairs", 
				  "DiracFermion_Sink");
			 error_occured = 1;
			 break;
		       }
		     free(lime_data);
		   }
		 //lime_type="scidac-binary-data";
		 if((strcmp(lime_type,"etmc-propagator-format")==0) || (strcmp(lime_type,"etmc-source-format")==0)
		    || (strcmp(lime_type,"etmc-eigenvectors-format")==0) || (strcmp(lime_type,"eigenvector-info")==0))
		   {
		     lime_data_size = limeReaderBytes(limereader);
		     lime_data = (char * )malloc(lime_data_size);
		     limeReaderReadData((void *)lime_data,&lime_data_size, limereader);
		     sscanf(qcd_getParam("<precision>",lime_data, lime_data_size),"%i",&isDouble);    
		     //		     printf("got precision: %i\n",isDouble);
		     free(lime_data);
		     
		     next_rec_is_prop = 1;
		   }
		 if(strcmp(lime_type,"scidac-binary-data")==0 && next_rec_is_prop)
		   {	      
		     break;
		   }
	       }
	     /* read 1 byte to set file-pointer to start of binary data */
	     lime_data_size=1;
	     limeReaderReadData(&dummy,&lime_data_size,limereader);
	     offset = ftell(fid)-1;
	     limeDestroyReader(limereader);      
	     fclose(fid);
	   }     
       }//end myid==0 condition
     
     MPI_Bcast(&error_occured,1,MPI_INT,0,MPI_COMM_WORLD);
     if(error_occured) errorQuda("Error with reading eigenVectors\n");
     //     if(isDouble != 32 && isDouble != 64 )isDouble = 32;     
     MPI_Bcast(&isDouble,1,MPI_INT,0,MPI_COMM_WORLD);
     MPI_Bcast(&offset,sizeof(MPI_Offset),MPI_BYTE,0,MPI_COMM_WORLD);

     //     printfQuda("I have precision %d\n",isDouble);

     if( typeid(Float) == typeid(double) ){
       if( isDouble != 64 ) errorQuda("Your precisions does not agree");
     }
     else if(typeid(Float) == typeid(float) ){
       if( isDouble != 32 ) errorQuda("Your precisions does not agree");
     }
     else
       errorQuda("Problem with the precision\n");

     if(isDouble==64)
       isDouble=1;      
     else if(isDouble==32)
       isDouble=0; 
     else
       {
	 fprintf(stderr,"process %i: Error in %s! Unsupported precision\n",comm_rank(), __func__);
       }  
     
     if(isDouble)
       {

	 sizes[0] = GK_totalL[3];
	 sizes[1] = GK_totalL[2];
	 sizes[2] = GK_totalL[1];
	 sizes[3] = GK_totalL[0];
	 sizes[4] = (4*3*2);
	 
	 lsizes[0] = GK_localL[3];
	 lsizes[1] = GK_localL[2];
	 lsizes[2] = GK_localL[1];
	 lsizes[3] = GK_localL[0];
	 lsizes[4] = sizes[4];
	 
	 starts[0]      = comm_coords(default_topo)[3]*GK_localL[3];
	 starts[1]      = comm_coords(default_topo)[2]*GK_localL[2];
	 starts[2]      = comm_coords(default_topo)[1]*GK_localL[1];
	 starts[3]      = comm_coords(default_topo)[0]*GK_localL[0];
	 starts[4]      = 0;


	 
	 MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&subblock);
	 MPI_Type_commit(&subblock);
      
	 MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpifid);
	 MPI_File_set_view(mpifid, offset, MPI_DOUBLE, subblock, "native", MPI_INFO_NULL);
	 
	 //load time-slice by time-slice:
	 chunksize=4*3*2*sizeof(double);
	 buffer = (char*) malloc(chunksize*GK_localVolume);
	 if(buffer==NULL)
	   {
	     fprintf(stderr,"process %i: Error in %s! Out of memory\n",comm_rank(), __func__);
	     return;
	   }
	 MPI_File_read_all(mpifid, buffer, 4*3*2*GK_localVolume, MPI_DOUBLE, &status);
	 if(!qcd_isBigEndian())      
	   qcd_swap_8((double*) buffer,(size_t)(2*4*3)*(size_t)GK_localVolume);
	 i=0;
	 for(t=0; t<GK_localL[3];t++)
	   for(z=0; z<GK_localL[2];z++)
	     for(y=0; y<GK_localL[1];y++)
	       for(x=0; x<GK_localL[0];x++)
		 for(mu=0; mu<4; mu++)
		   for(c1=0; c1<3; c1++)
		     {

		       int oddBit     = (x+y+z+t) & 1;
		       if(oddBit){
			 h_elem[nev][ ( (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+z*GK_localL[1]*GK_localL[0]+y*GK_localL[0]+x)/2)*4*3*2 + mu*3*2 + c1*2 + 0 ] = ((double*)buffer)[i];
			 h_elem[nev][ ( (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+z*GK_localL[1]*GK_localL[0]+y*GK_localL[0]+x)/2)*4*3*2 + mu*3*2 + c1*2 + 1 ] = ((double*)buffer)[i+1];
			 i+=2;
		       }
		       else{
			 i+=2;
		       }
			 
		     }
	 free(buffer);
	 MPI_File_close(&mpifid);
	 MPI_Type_free(&subblock);
	 
	 continue;
       }//end isDouble condition
     else
       {
	 sizes[0] = GK_totalL[3];
	 sizes[1] = GK_totalL[2];
	 sizes[2] = GK_totalL[1];
	 sizes[3] = GK_totalL[0];
	 sizes[4] = (4*3*2);
	 
	 lsizes[0] = GK_localL[3];
	 lsizes[1] = GK_localL[2];
	 lsizes[2] = GK_localL[1];
	 lsizes[3] = GK_localL[0];
	 lsizes[4] = sizes[4];
	 
	 starts[0]      = comm_coords(default_topo)[3]*GK_localL[3];
	 starts[1]      = comm_coords(default_topo)[2]*GK_localL[2];
	 starts[2]      = comm_coords(default_topo)[1]*GK_localL[1];
	 starts[3]      = comm_coords(default_topo)[0]*GK_localL[0];
	 starts[4]      = 0;

	 //	 for(int ii = 0 ; ii < 5 ; ii++)
	 //  printf("%d %d %d %d\n",comm_rank(),sizes[ii],lsizes[ii],starts[ii]);

      MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,MPI_FLOAT,&subblock);
      MPI_Type_commit(&subblock);
      
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpifid);
      MPI_File_set_view(mpifid, offset, MPI_FLOAT, subblock, "native", MPI_INFO_NULL);
      
      //load time-slice by time-slice:
      chunksize=4*3*2*sizeof(float);
      buffer = (char*) malloc(chunksize*GK_localVolume);
      if(buffer==NULL)
      {
	fprintf(stderr,"process %i: Error in %s! Out of memory\n",comm_rank(), __func__);
         return;
      }
      MPI_File_read_all(mpifid, buffer, 4*3*2*GK_localVolume, MPI_FLOAT, &status);

      if(!qcd_isBigEndian())
	qcd_swap_4((float*) buffer,(size_t)(2*4*3)*(size_t)GK_localVolume);
      
      i=0;
      for(t=0; t<GK_localL[3];t++)
	for(z=0; z<GK_localL[2];z++)
	  for(y=0; y<GK_localL[1];y++)
	    for(x=0; x<GK_localL[0];x++)
	      for(mu=0; mu<4; mu++)
		for(c1=0; c1<3; c1++)
		  {
		       int oddBit     = (x+y+z+t) & 1;
		       if(oddBit){
			 h_elem[nev][ ( (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+z*GK_localL[1]*GK_localL[0]+y*GK_localL[0]+x)/2)*4*3*2 + mu*3*2 + c1*2 + 0  ] = *((float*)(buffer + i)); i+=4 ;
			 h_elem[nev][ ( (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+z*GK_localL[1]*GK_localL[0]+y*GK_localL[0]+x)/2)*4*3*2 + mu*3*2 + c1*2 + 1 ] = *((float*)(buffer + i)); i+=4 ;
		       }
		       else{
			 i+=8;
		       }
		  }      
      
      free(buffer);
      MPI_File_close(&mpifid);
      MPI_Type_free(&subblock);            
      
      continue;
       }//end isDouble condition
   }
   printfQuda("Eigenvectors loaded successfully\n");
}//end qcd_getVectorLime 


template<typename Float>
void QKXTM_Deflation_Kepler<Float>::readEigenValues(char *filename){
  if(NeV == 0)return;
  FILE *ptr;
  Float dummy;
  ptr = fopen(filename,"r");
  if(ptr == NULL)errorQuda("Error cannot open file to read eigenvalues\n");
  char stringFormat[257];
  if(typeid(Float) == typeid(double))
    strcpy(stringFormat,"%lf");
  else if(typeid(Float) == typeid(float))
    strcpy(stringFormat,"%f");

  for(int i = 0 ; i < NeV ; i++)
    fscanf(ptr,stringFormat,&(EigenValues()[i]),&dummy);
       

  printfQuda("Eigenvalues loaded successfully\n");
  fclose(ptr);
}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::copyEigenVectorToQKXTM_Vector_Kepler(int eigenVector_id, Float *vec){
  if(NeV == 0)return;
  for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
      for(int y=0; y<GK_localL[1];y++)
	for(int x=0; x<GK_localL[0];x++)
	  for(int mu=0; mu<4; mu++)
	    for(int c1=0; c1<3; c1++)
	      {
		int oddBit     = (x+y+z+t) & 1;
		if(oddBit){
		  if(isEv == false){
		    vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 0] = h_elem[eigenVector_id][ ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + 0];
		    vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 1] = h_elem[eigenVector_id][ ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + 1];
		  }
		  else{
		    vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 0] =0.;
		    vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 1] =0.; 
		  }
		} // if for odd
		else{
		  if(isEv == true){
		    vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 0] = h_elem[eigenVector_id][ ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + 0];
		    vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 1] = h_elem[eigenVector_id][ ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + 1];
		  }
		  else{
		    vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 0] =0.;
		    vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 1] =0.; 
		  }
		}
	      }
}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::copyEigenVectorFromQKXTM_Vector_Kepler(int eigenVector_id,Float *vec){
  if(NeV == 0)return;
  for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
      for(int y=0; y<GK_localL[1];y++)
	for(int x=0; x<GK_localL[0];x++)
	  for(int mu=0; mu<4; mu++)
	    for(int c1=0; c1<3; c1++)
	      {
		int oddBit     = (x+y+z+t) & 1;
		if(oddBit){
		  if(isEv == false){
		    h_elem[eigenVector_id][((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + 0] = vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 0];
		    h_elem[eigenVector_id][((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + 1] = vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 1];
		  }
		} // if for odd
		else{
		  if(isEv == true){
		    h_elem[eigenVector_id][((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + 0] = vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 0];
		    h_elem[eigenVector_id][((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + 1] = vec[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 1];
		  }
		}
	      }
}

template <typename Float>
void QKXTM_Deflation_Kepler<Float>::deflateGuessVector(QKXTM_Vector_Kepler<Float> &vec_guess, QKXTM_Vector_Kepler<Float> &vec_source){
  if(NeV == 0){
    vec_guess.zero_device();
    return;
  }

  Float *tmp_vec = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;
  Float *tmp_vec_lex = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;
  Float *out_vec = (Float*) calloc(NeV*2,sizeof(Float)) ;
  Float *out_vec_reduce = (Float*) calloc(NeV*2,sizeof(Float)) ;

  if(tmp_vec == NULL || tmp_vec_lex == NULL || out_vec == NULL || out_vec_reduce == NULL)errorQuda("Error with memory allocation in deflation method\n");
  
  Float *tmp_vec_even = tmp_vec;
  Float *tmp_vec_odd = tmp_vec + (GK_localVolume/2)*4*3*2;

  for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
      for(int y=0; y<GK_localL[1];y++)
	for(int x=0; x<GK_localL[0];x++)
	  for(int mu=0; mu<4; mu++)
	    for(int c1=0; c1<3; c1++)
	      {
		int oddBit     = (x+y+z+t) & 1;
		if(oddBit){
		  for(int ipart = 0 ; ipart < 2 ; ipart++)
		    tmp_vec_odd[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart] = (Float) vec_source.H_elem()[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + ipart];
		}
		else{
		  for(int ipart = 0 ; ipart < 2 ; ipart++)
		    tmp_vec_even[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart] = (Float) vec_source.H_elem()[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + ipart];
		}
	      }  
  
  Float alpha[2] = {1.,0.};
  Float beta[2] = {0.,0.};
  int incx = 1;
  int incy = 1;
  long int NN = (GK_localVolume/2)*4*3;

  Float *ptr_elem = NULL;


  if(isEv == true){
    ptr_elem = tmp_vec_even;

  }
  else{
    ptr_elem = tmp_vec_odd;

  }

  //  printf("I get %d num of omp threads and %d cores\n",mkl_get_max_threads(),omp_get_num_procs());
  //  omp_set_nested(false);
  if( typeid(Float) == typeid(float) ){
    //#pragma omp parallel
    // {
    //  int id = omp_get_thread_num();
    //  int icpu = sched_getcpu();
    //  printf("%d %d\n",id,icpu);
    //#pragma omp parallel for
    for(int i = 0 ; i < NeV ; i++)
      cblas_cdotc_sub(NN,H_elem()[i],incx,ptr_elem,incy,&(out_vec[2*i]));
    
    // }
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    for(int i = 0 ; i < NeV ; i++){
      out_vec_reduce[i*2+0] /= eigenValues[i];
      out_vec_reduce[i*2+1] /= eigenValues[i];
    }

    memset(ptr_elem,0,(GK_localVolume/2)*4*3*2*sizeof(Float));
    
    for(int i = 0 ; i < NeV ; i++)
      cblas_caxpy(NN,&(out_vec_reduce[2*i]),H_elem()[i],incx,ptr_elem,incy);
  }
  else if ( typeid(Float) == typeid(double) ){
    //#pragma omp parallel
    // {
    //  int id = omp_get_thread_num();
    //  int icpu = sched_getcpu();
    //  printf("%d %d\n",id,icpu);

    //#pragma omp parallel for
    for(int i = 0 ; i < NeV ; i++)
      cblas_zdotc_sub(NN,H_elem()[i],incx,ptr_elem,incy,&(out_vec[2*i]));

    //}
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    for(int i = 0 ; i < NeV ; i++){
      out_vec_reduce[i*2+0] /= eigenValues[i];
      out_vec_reduce[i*2+1] /= eigenValues[i];
    }
    memset(ptr_elem,0,(GK_localVolume/2)*4*3*2*sizeof(Float));
    for(int i = 0 ; i < NeV ; i++) cblas_zaxpy(NN,&(out_vec_reduce[2*i]),H_elem()[i],incx,ptr_elem,incy);
  }

  /*  
  if( typeid(Float) == typeid(float) ){
    cblas_cgemv(CblasColMajor, CblasConjTrans, NN, NeV, alpha, H_elem(), NN, ptr_elem, incx, beta, out_vec, incy );
    memset(ptr_elem,0,(GK_localVolume/2)*4*3*2*sizeof(Float));
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    for(int i = 0 ; i < NeV ; i++){
      out_vec_reduce[i*2+0] /= eigenValues[i];
      out_vec_reduce[i*2+1] /= eigenValues[i];
    }
    cblas_cgemv(CblasColMajor, CblasNoTrans, NN, NeV, alpha, H_elem(), NN, out_vec_reduce, incx, beta, ptr_elem, incy );
  }
  else if ( typeid(Float) == typeid(double) ){
    cblas_zgemv(CblasColMajor, CblasConjTrans, NN, NeV, alpha, H_elem(), NN, ptr_elem, incx, beta, out_vec, incy );
    memset(ptr_elem,0,(GK_localVolume/2)*4*3*2*sizeof(Float));
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    for(int i = 0 ; i < NeV ; i++){
      out_vec_reduce[i*2+0] /= eigenValues[i];
      out_vec_reduce[i*2+1] /= eigenValues[i];
    }
    cblas_zgemv(CblasColMajor, CblasNoTrans, NN, NeV, alpha, H_elem(), NN, out_vec_reduce, incx, beta, ptr_elem, incy );    
  }
  */

  for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
      for(int y=0; y<GK_localL[1];y++)
	for(int x=0; x<GK_localL[0];x++)
	  for(int mu=0; mu<4; mu++)
	    for(int c1=0; c1<3; c1++)
	      {
		int oddBit     = (x+y+z+t) & 1;
		if(oddBit){
		  for(int ipart = 0 ; ipart < 2 ; ipart++)
		    tmp_vec_lex[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + ipart] = tmp_vec_odd[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart];
		}
		else{
		  for(int ipart = 0 ; ipart < 2 ; ipart++)
		    tmp_vec_lex[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + z*GK_localL[1]*GK_localL[0]*4*3*2 + y*GK_localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + ipart] = tmp_vec_even[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + z*GK_localL[1]*GK_localL[0] + y*GK_localL[0] + x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart];
		}
	      }  


  vec_guess.packVector((Float*) tmp_vec_lex);
  vec_guess.loadVector();


  free(out_vec);
  free(out_vec_reduce);
  free(tmp_vec);
  free(tmp_vec_lex);

}


template<typename Float>
void QKXTM_Deflation_Kepler<Float>::writeEigenVectors_ASCI(char *prefix_path){
  if(NeV == 0)return;
  char filename[257];
  if(comm_rank() != 0) return;
  FILE *fid;
  for(int nev = 0 ; nev < NeV ; nev++){
    sprintf(filename,"%s.%05d.txt",prefix_path,nev);
    fid = fopen(filename,"w");		  
    for(int ir = 0 ; ir < (GK_localVolume/2)*4*3 ; ir++)
      fprintf(fid,"%+e %+e\n",h_elem[nev][ir*2 + 0], h_elem[nev][ir*2 + 1]);
		
    
    fclose(fid);
  }
}

#include <cufft.h>
#include <gsl/gsl_rng.h>
#include <contractQuda.h>

template<typename Float>
void oneEndTrick(cudaColorSpinorField &x,cudaColorSpinorField &tmp3, cudaColorSpinorField &tmp4,QudaInvertParam *param, void *cnRes_gv,void *cnRes_vv){
  void *h_ctrn, *ctrnS;

  if((cudaMallocHost(&h_ctrn, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in CPU.\n");
  cudaMemset(h_ctrn, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);
  if((cudaMalloc(&ctrnS, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in GPU.\n");
  cudaMemset(ctrnS, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);
  checkCudaError();
  
  DiracParam dWParam;
  dWParam.matpcType        = QUDA_MATPC_EVEN_EVEN;
  dWParam.dagger           = QUDA_DAG_NO;
  dWParam.gauge            = gaugePrecise;
  dWParam.kappa            = param->kappa;
  dWParam.mass             = 1./(2.*param->kappa) - 4.;
  dWParam.m5               = 0.;
  dWParam.mu               = 0.;
  for     (int i=0; i<4; i++)
    dWParam.commDim[i]       = 1;

  if(param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    dWParam.type           = QUDA_CLOVER_DIRAC;
    dWParam.clover                 = cloverPrecise;
    DiracClover   *dW      = new DiracClover(dWParam);
    dW->M(tmp4,x);
    delete  dW;
  } 
  else if (param->dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    dWParam.type           = QUDA_WILSON_DIRAC;
    DiracWilson   *dW      = new DiracWilson(dWParam);
    dW->M(tmp4,x);
    delete  dW;
  }
  else{
    errorQuda("Error one end trick works only for twisted mass fermions\n");
  }
  checkCudaError();

  gamma5Cuda(&(tmp3.Even()), &(tmp4.Even()));
  gamma5Cuda(&(tmp3.Odd()),  &(tmp4.Odd()));

  long int sizeBuffer;
  sizeBuffer = sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];

  contractCuda    (x.Even(), tmp3.Even(), ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_EVEN_PARITY);
  contractCuda    (x.Odd(),  tmp3.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_ODD_PARITY);

  cudaMemcpy      (h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
    ((Float*) cnRes_gv)[ix] += ((Float*)h_ctrn)[ix]; // generalized one end trick

  contractCuda    (x.Even(), x.Even(), ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_EVEN_PARITY);
  contractCuda    (x.Odd(),  x.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_ODD_PARITY);

  cudaMemcpy      (h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
    ((Float*) cnRes_vv)[ix] -= ((Float*)h_ctrn)[ix]; // standard one end trick
  cudaDeviceSynchronize();
  cudaFreeHost(h_ctrn);
  cudaFree(ctrnS);
  checkCudaError();
}


template<typename Float>
void oneEndTrick_w_One_Der(cudaColorSpinorField &x,cudaColorSpinorField &tmp3, cudaColorSpinorField &tmp4,QudaInvertParam *param, void *cnRes_gv,void *cnRes_vv, void **cnD_gv, void **cnD_vv, void **cnC_gv, void **cnC_vv){
  void *h_ctrn, *ctrnS, *ctrnC;

  if((cudaMallocHost(&h_ctrn, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in CPU.\n");
  cudaMemset(h_ctrn, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);

  if((cudaMalloc(&ctrnS, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in GPU.\n");
  cudaMemset(ctrnS, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);

  if((cudaMalloc(&ctrnC, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in GPU.\n");
  cudaMemset(ctrnC, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);

  checkCudaError();
  
  DiracParam dWParam;
  dWParam.matpcType        = QUDA_MATPC_EVEN_EVEN;
  dWParam.dagger           = QUDA_DAG_NO;
  dWParam.gauge            = gaugePrecise;
  dWParam.kappa            = param->kappa;
  dWParam.mass             = 1./(2.*param->kappa) - 4.;
  dWParam.m5               = 0.;
  dWParam.mu               = 0.;
  for     (int i=0; i<4; i++)
    dWParam.commDim[i]       = 1;

  if(param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    dWParam.type           = QUDA_CLOVER_DIRAC;
    dWParam.clover                 = cloverPrecise;
    DiracClover   *dW      = new DiracClover(dWParam);
    dW->M(tmp4,x);
    delete  dW;
  } 
  else if (param->dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    dWParam.type           = QUDA_WILSON_DIRAC;
    DiracWilson   *dW      = new DiracWilson(dWParam);
    dW->M(tmp4,x);
    delete  dW;
  }
  else{
    errorQuda("Error one end trick works only for twisted mass fermions\n");
  }
  checkCudaError();

  gamma5Cuda(&(tmp3.Even()), &(tmp4.Even()));
  gamma5Cuda(&(tmp3.Odd()),  &(tmp4.Odd()));

  long int sizeBuffer;
  sizeBuffer = sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];

  ///////////////// LOCAL ///////////////////////////
  contractCuda    (x.Even(), tmp3.Even(), ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_EVEN_PARITY);
  contractCuda    (x.Odd(),  tmp3.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_ODD_PARITY);

  cudaMemcpy      (h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
    ((Float*) cnRes_gv)[ix] += ((Float*)h_ctrn)[ix]; // generalized one end trick

  contractCuda    (x.Even(), x.Even(), ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_EVEN_PARITY);
  contractCuda    (x.Odd(),  x.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_ODD_PARITY);

  cudaMemcpy      (h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
    ((Float*) cnRes_vv)[ix] -= ((Float*)h_ctrn)[ix]; // standard one end trick
  cudaDeviceSynchronize();
  /*
  ////////////////// DERIVATIVES //////////////////////////////
  for(int mu=0; mu<4; mu++)	// for generalized one-end trick
    {
      covDev(&(tmp4.Odd()),  *gaugePrecise, &(tmp3.Even()), QUDA_ODD_PARITY,  mu, profileCovDev);
      covDev(&(tmp4.Even()), *gaugePrecise, &(tmp3.Odd()),  QUDA_EVEN_PARITY, mu, profileCovDev);
      
      contractCuda(x.Even(), tmp4.Even(), ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_EVEN_PARITY);
      contractCuda(x.Odd(),  tmp4.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_ODD_PARITY);                 // Term 0
      cudaDeviceSynchronize();
      cudaMemcpy(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);
      
      covDev(&(tmp4.Odd()),  *gaugePrecise, &(x.Even()), QUDA_ODD_PARITY,  mu, profileCovDev);
      covDev(&(tmp4.Even()), *gaugePrecise, &(x.Odd()),  QUDA_EVEN_PARITY, mu, profileCovDev);
      
      contractCuda(tmp4.Even(), tmp3.Even(), ctrnC, QUDA_CONTRACT_GAMMA5_PLUS, QUDA_EVEN_PARITY);        // Term 2 (C Sum)
      contractCuda(tmp4.Odd(),  tmp3.Odd(),  ctrnC, QUDA_CONTRACT_GAMMA5_PLUS, QUDA_ODD_PARITY);
      
      contractCuda(tmp4.Even(), tmp3.Even(), ctrnS, QUDA_CONTRACT_GAMMA5_MINUS, QUDA_EVEN_PARITY);       // Term 2 (D Diff)
      contractCuda(tmp4.Odd(),  tmp3.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5_MINUS, QUDA_ODD_PARITY);
      
      //      cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);
      
      // for(int ix = 0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
      //	((Float *) cnX_gv[mu])[ix] += ((Float*)h_ctrn)[ix];
      
      covDev(&(tmp4.Even()), *gaugePrecise, &(tmp3.Odd()),  QUDA_EVEN_PARITY, mu+4, profileCovDev);
      covDev(&(tmp4.Odd()),  *gaugePrecise, &(tmp3.Even()), QUDA_ODD_PARITY,  mu+4, profileCovDev);
      
      contractCuda(x.Even(), tmp4.Even(), ctrnS, QUDA_CONTRACT_GAMMA5_MINUS, QUDA_EVEN_PARITY);       // Term 1
      contractCuda(x.Odd(),  tmp4.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5_MINUS, QUDA_ODD_PARITY);
      
      covDev(&(tmp4.Even()), *gaugePrecise, &(x.Odd()),  QUDA_EVEN_PARITY, mu+4, profileCovDev);
      covDev(&(tmp4.Odd()),  *gaugePrecise, &(x.Even()), QUDA_ODD_PARITY,  mu+4, profileCovDev);
      
      contractCuda(tmp4.Even(), tmp3.Even(), ctrnS, QUDA_CONTRACT_GAMMA5_PLUS, QUDA_EVEN_PARITY);       // Term 3
      contractCuda(tmp4.Odd(),  tmp3.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5_PLUS, QUDA_ODD_PARITY);
      cudaDeviceSynchronize();
      
      cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);
      
      for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
	((Float *) cnD_gv[mu])[ix] += ((Float*)h_ctrn)[ix];
      
      cudaMemcpy(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);
      
      for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
	((Float *) cnC_gv[mu])[ix] += ((Float*)h_ctrn)[ix];
    }
  
  for(int mu=0; mu<4; mu++) // for standard one-end trick
    {
      covDev(&(tmp4.Odd()),  *gaugePrecise, &(x.Even()), QUDA_ODD_PARITY,  mu, profileCovDev);
      covDev(&(tmp4.Even()), *gaugePrecise, &(x.Odd()),  QUDA_EVEN_PARITY, mu, profileCovDev);
      
      contractCuda(x.Even(), tmp4.Even(), ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_EVEN_PARITY);       // Term 0
      contractCuda(x.Odd(),  tmp4.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5, QUDA_ODD_PARITY);
      cudaDeviceSynchronize();
      
      cudaMemcpy(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);
      
      contractCuda(tmp4.Even(), x.Even(), ctrnC, QUDA_CONTRACT_GAMMA5_PLUS, QUDA_EVEN_PARITY);       // Term 2
      contractCuda(tmp4.Odd(),  x.Odd(),  ctrnC, QUDA_CONTRACT_GAMMA5_PLUS, QUDA_ODD_PARITY);
      
      contractCuda(tmp4.Even(), x.Even(), ctrnS, QUDA_CONTRACT_GAMMA5_MINUS, QUDA_EVEN_PARITY);       // Term 2
      contractCuda(tmp4.Odd(),  x.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5_MINUS, QUDA_ODD_PARITY);
      
      // cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);
      
      // for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
      //	((Float *) cnX_vv[mu])[ix] -= ((Float*)h_ctrn)[ix];
      
      covDev(&(tmp4.Odd()),  *gaugePrecise, &(x.Even()), QUDA_ODD_PARITY,  mu+4, profileCovDev);
      covDev(&(tmp4.Even()), *gaugePrecise, &(x.Odd()),  QUDA_EVEN_PARITY, mu+4, profileCovDev);
      
      contractCuda(x.Even(), tmp4.Even(), ctrnS, QUDA_CONTRACT_GAMMA5_MINUS, QUDA_EVEN_PARITY);       // Term 1
      contractCuda(x.Odd(),  tmp4.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5_MINUS, QUDA_ODD_PARITY);
      
      contractCuda(tmp4.Even(), x.Even(), ctrnS, QUDA_CONTRACT_GAMMA5_PLUS, QUDA_EVEN_PARITY);       // Term 3
      contractCuda(tmp4.Odd(),  x.Odd(),  ctrnS, QUDA_CONTRACT_GAMMA5_PLUS, QUDA_ODD_PARITY);
      cudaDeviceSynchronize();
      
      cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);
      
      for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
	((Float *) cnD_vv[mu])[ix]  -= ((Float*)h_ctrn)[ix];
      
      cudaMemcpy(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);
      
      for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
	((Float *) cnC_vv[mu])[ix] -= ((Float*)h_ctrn)[ix];
    }
///////////////
*/

  cudaFreeHost(h_ctrn);
  cudaFree(ctrnS);
  cudaFree(ctrnC);
  checkCudaError();
}


template <typename Float>
void doCudaFFT(void *cnRes_gv, void *cnRes_vv, void *cnResTmp_gv,void *cnResTmp_vv){
  static cufftHandle      fftPlan;
  static int              init = 0;
  int                     nRank[3]         = {GK_localL[0], GK_localL[1], GK_localL[2]};
  const int               Vol              = GK_localL[0]*GK_localL[1]*GK_localL[2];
  static cudaStream_t     streamCuFFT;
  cudaStreamCreate(&streamCuFFT);

  if(cufftPlanMany(&fftPlan, 3, nRank, nRank, 1, Vol, nRank, 1, Vol, CUFFT_Z2Z, 16*GK_localL[3]) != CUFFT_SUCCESS) errorQuda("Error in the FFT!!!\n");
  cufftSetCompatibilityMode       (fftPlan, CUFFT_COMPATIBILITY_NATIVE);
  cufftSetStream                  (fftPlan, streamCuFFT);
  checkCudaError();
  void* ctrnS;
  if((cudaMalloc(&ctrnS, sizeof(Float)*32*Vol*GK_localL[3])) == cudaErrorMemoryAllocation) errorQuda("Error with memory allocation\n");

  cudaMemcpy(ctrnS, cnRes_vv, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyHostToDevice);
  if(typeid(Float) == typeid(double))if(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  if(typeid(Float) == typeid(float))if(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  cudaMemcpy(cnResTmp_vv, ctrnS, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyDeviceToHost);

  cudaMemcpy(ctrnS, cnRes_gv, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyHostToDevice);
  if(typeid(Float) == typeid(double))if(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  if(typeid(Float) == typeid(float))if(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  cudaMemcpy(cnResTmp_gv, ctrnS, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyDeviceToHost);


  cudaFree(ctrnS);
  cufftDestroy            (fftPlan);
  cudaStreamDestroy       (streamCuFFT);
  checkCudaError();
}

template <typename Float>
void doCudaFFT_v2(void *cnIn, void *cnOut){
  static cufftHandle      fftPlan;
  static int              init = 0;
  int                     nRank[3]         = {GK_localL[0], GK_localL[1], GK_localL[2]};
  const int               Vol              = GK_localL[0]*GK_localL[1]*GK_localL[2];
  static cudaStream_t     streamCuFFT;
  cudaStreamCreate(&streamCuFFT);

  if(cufftPlanMany(&fftPlan, 3, nRank, nRank, 1, Vol, nRank, 1, Vol, CUFFT_Z2Z, 16*GK_localL[3]) != CUFFT_SUCCESS) errorQuda("Error in the FFT!!!\n");
  cufftSetCompatibilityMode       (fftPlan, CUFFT_COMPATIBILITY_NATIVE);
  cufftSetStream                  (fftPlan, streamCuFFT);
  checkCudaError();
  void* ctrnS;
  if((cudaMalloc(&ctrnS, sizeof(Float)*32*Vol*GK_localL[3])) == cudaErrorMemoryAllocation) errorQuda("Error with memory allocation\n");

  cudaMemcpy(ctrnS, cnIn, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyHostToDevice);
  if(typeid(Float) == typeid(double))if(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  if(typeid(Float) == typeid(float))if(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  cudaMemcpy(cnOut, ctrnS, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyDeviceToHost);

  cudaFree(ctrnS);
  cufftDestroy            (fftPlan);
  cudaStreamDestroy       (streamCuFFT);
  checkCudaError();
}

static int** allocateMomMatrix(int Q_sq){
  int **mom;
  if((mom = (int **) malloc(sizeof(int*)*GK_localL[0]*GK_localL[1]*GK_localL[2])) == NULL) errorQuda("Error allocate memory for momenta\n");
  for(int ip=0; ip<GK_localL[0]*GK_localL[1]*GK_localL[2]; ip++)
    if((mom[ip] = (int *) malloc(sizeof(int)*3)) == NULL)errorQuda("Error allocate memory for momenta\n");
  int momIdx       = 0;
  int totMom       = 0;
  
  for(int pz = 0; pz < GK_localL[2]; pz++)
    for(int py = 0; py < GK_localL[1]; py++)
      for(int px = 0; px < GK_localL[0]; px++){
	  if      (px < GK_localL[0]/2)
	    mom[momIdx][0]   = px;
	  else
	    mom[momIdx][0]   = px - GK_localL[0];

	  if      (py < GK_localL[1]/2)
	    mom[momIdx][1]   = py;
	  else
	    mom[momIdx][1]   = py - GK_localL[1];

	  if      (pz < GK_localL[2]/2)
	    mom[momIdx][2]   = pz;
	  else
	    mom[momIdx][2]   = pz - GK_localL[2];

	  if((mom[momIdx][0]*mom[momIdx][0]+mom[momIdx][1]*mom[momIdx][1]+mom[momIdx][2]*mom[momIdx][2])<=Q_sq) totMom++;
	  momIdx++;
	}
  return mom;
}

template<typename Float>
void dumpLoop(void *cnRes_gv, void *cnRes_vv, const char *Pref,int accumLevel, int Q_sq){
  int **mom = allocateMomMatrix(Q_sq);
  FILE *ptr_gv;
  FILE *ptr_vv;
  char file_gv[257];
  char file_vv[257];
  sprintf(file_gv, "%s_dOp.loop.%04d.%d_%d",Pref,accumLevel,comm_size(), comm_rank());
  sprintf(file_vv, "%s_Scalar.loop.%04d.%d_%d",Pref,accumLevel,comm_size(), comm_rank());
  ptr_gv = fopen(file_gv,"w");
  ptr_vv = fopen(file_vv,"w");
  if(ptr_gv == NULL || ptr_vv == NULL) errorQuda("Error open files to write loops\n");
  long int Vol = GK_localL[0]*GK_localL[1]*GK_localL[2];
  for(int ip=0; ip < Vol; ip++)
    for(int lt=0; lt < GK_localL[3]; lt++){
      if ((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= Q_sq){
	int t  = lt+comm_coords(default_topo)[3]*GK_localL[3];
	for(int gm=0; gm<16; gm++){                                                             
	    fprintf (ptr_gv, "%02d %02d %+d %+d %+d %+16.15e %+16.15e\n",t, gm, mom[ip][0], mom[ip][1], mom[ip][2],
		     ((Float*)cnRes_gv)[0+2*ip+2*Vol*lt+2*Vol*GK_localL[3]*gm], ((Float*)cnRes_gv)[1+2*ip+2*Vol*lt+2*Vol*GK_localL[3]*gm]);
	    fprintf (ptr_vv, "%02d %02d %+d %+d %+d %+16.15le %+16.15e\n",t, gm, mom[ip][0], mom[ip][1], mom[ip][2],
		     ((Float*)cnRes_vv)[0+2*ip+2*Vol*lt+2*Vol*GK_localL[3]*gm], ((Float*)cnRes_vv)[1+2*ip+2*Vol*lt+2*Vol*GK_localL[3]*gm]);
	}
      }
    }
  printfQuda("data dumped for accumLevel %d\n",accumLevel);
  fclose(ptr_gv);
  fclose(ptr_vv);
  for(int ip=0; ip<Vol; ip++)
    free(mom[ip]);
  free(mom);
}

template<typename Float>
void dumpLoop_ultraLocal(void *cn, const char *Pref,int accumLevel, int Q_sq, int flag){
  int **mom = allocateMomMatrix(Q_sq);
  FILE *ptr;
  char file_name[257];

  switch(flag){
  case 0:
    sprintf(file_name, "%s_Scalar.loop.%04d.%d_%d",Pref,accumLevel,comm_size(), comm_rank());
    break;
  case 1:
    sprintf(file_name, "%s_dOp.loop.%04d.%d_%d",Pref,accumLevel,comm_size(), comm_rank());
    break;
  }
  ptr = fopen(file_name,"w");

  if(ptr == NULL) errorQuda("Error open files to write loops\n");
  long int Vol = GK_localL[0]*GK_localL[1]*GK_localL[2];
  for(int ip=0; ip < Vol; ip++)
    for(int lt=0; lt < GK_localL[3]; lt++){
      if ((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= Q_sq){
	int t  = lt+comm_coords(default_topo)[3]*GK_localL[3];
	for(int gm=0; gm<16; gm++){                                                             
	    fprintf(ptr, "%02d %02d %+d %+d %+d %+16.15e %+16.15e\n",t, gm, mom[ip][0], mom[ip][1], mom[ip][2],
		     ((Float*)cn)[0+2*ip+2*Vol*lt+2*Vol*GK_localL[3]*gm], ((Float*)cn)[1+2*ip+2*Vol*lt+2*Vol*GK_localL[3]*gm]);
	}
      }
    }
  printfQuda("data dumped for accumLevel %d\n",accumLevel);
  fclose(ptr);
  for(int ip=0; ip<Vol; ip++)
    free(mom[ip]);
  free(mom);
}

template<typename Float>
void dumpLoop_oneD(void *cn, const char *Pref,int accumLevel, int Q_sq, int muDir, int flag){
  int **mom = allocateMomMatrix(Q_sq);
  FILE *ptr;
  char file_name[257];

  switch(flag){
  case 0:
    sprintf(file_name, "%s_Loops.loop.%04d.%d_%d",Pref,accumLevel,comm_size(), comm_rank());
    break;
  case 1:
    sprintf(file_name, "%s_LpsDw.loop.%04d.%d_%d",Pref,accumLevel,comm_size(), comm_rank());
    break;
  case 2:
    sprintf(file_name, "%s_LoopsCv.loop.%04d.%d_%d",Pref,accumLevel,comm_size(), comm_rank());
    break;
  case 3:
    sprintf(file_name, "%s_LpsDwCv.loop.%04d.%d_%d",Pref,accumLevel,comm_size(), comm_rank());
    break;
  }
  if(muDir == 0)
    ptr = fopen(file_name,"w");
  else
    ptr = fopen(file_name,"a");

  if(ptr == NULL) errorQuda("Error open files to write loops\n");
  long int Vol = GK_localL[0]*GK_localL[1]*GK_localL[2];
  for(int ip=0; ip < Vol; ip++)
    for(int lt=0; lt < GK_localL[3]; lt++){
      if ((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= Q_sq){
	int t  = lt+comm_coords(default_topo)[3]*GK_localL[3];
	for(int gm=0; gm<16; gm++){                                                             
	  fprintf(ptr, "%02d %02d %02d %+d %+d %+d %+16.15e %+16.15e\n",t, gm, muDir ,mom[ip][0], mom[ip][1], mom[ip][2],
		     ((Float*)cn)[0+2*ip+2*Vol*lt+2*Vol*GK_localL[3]*gm], ((Float*)cn)[1+2*ip+2*Vol*lt+2*Vol*GK_localL[3]*gm]);
	}
      }
    }

  printfQuda("data dumped for accumLevel %d\n",accumLevel);
  fclose(ptr);
  for(int ip=0; ip<Vol; ip++)
    free(mom[ip]);
  free(mom);
}
