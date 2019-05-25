/* C. Kallidonis: GPU kernel to perform di-quark contractions.
 * This function takes as input two forward propagators, performs
 * the relevant contractions the stores the result in another 
 * propagator object.
 * November 2017
 */

#include <qlua_contract.h>
#include <tune_quda.h>

namespace quda {
  
  //-- Main calculation kernel
#define ContractQQ_macro(cntrIdx, A, B, C, D)				\
  __device__ __host__ inline void performContractQQ ## cntrIdx(ColorSpinor<QUDA_REAL,QUDA_Nc,QUDA_Ns> *out, \
							       ContractQQArg *arg, \
							       int x_cb, \
							       int pty) \
  {									\
									\
    int eps[3][3] = { { 0, 1, 2},					\
		      { 1, 2, 0},					\
		      { 2, 0, 1} };					\
    									\
    const int Ns = QUDA_Ns;						\
    const int Nc = QUDA_Nc;						\
    									\
    Vector In1[QUDA_NVEC_PROP];						\
    Vector In2[QUDA_NVEC_PROP];						\
    									\
    for(int i=0;i<arg->nVec;i++){					\
      In1[i] = arg->pIn1[i](x_cb, pty);					\
      In2[i] = arg->pIn2[i](x_cb, pty);					\
    }									\
    rotateVectorBasis(In1,QLUA_quda2qdp);				\
    rotateVectorBasis(In2,QLUA_quda2qdp);				\
    									\
    									\
    for(int p_a = 0; p_a < Nc; p_a++){					\
      int i1 = eps[p_a][0];						\
      int j1 = eps[p_a][1];						\
      int k1 = eps[p_a][2];						\
      for (int p_b = 0; p_b < Nc; p_b++){				\
	int i2 = eps[p_b][0];						\
    	int j2 = eps[p_b][1];						\
	int k2 = eps[p_b][2];						\
    	for (int a = 0; a < Ns; a++){					\
    	  for (int b = 0; b < Ns; b++){					\
	    complex<QUDA_REAL> accum = 0.0;				\
	    for (int c = 0; c < Ns; c++){				\
	      int idx11 = (B)+Ns*i2;					\
	      int idx12 = (B)+Ns*j2;					\
	      int idx21 = (D)+Ns*i2;					\
	      int idx22 = (D)+Ns*j2;					\
	      								\
   	      accum += In1[idx11]((A),i1) * In2[idx22]((C),j1);		\
	      accum -= In1[idx12]((A),i1) * In2[idx21]((C),j1);		\
	      accum -= In1[idx11]((A),j1) * In2[idx22]((C),i1);		\
	      accum += In1[idx12]((A),j1) * In2[idx21]((C),i1);		\
	    }								\
	    out[b+Ns*k1](a,k2) = accum;					\
	  }								\
	}								\
      }									\
    }									\
  } //-- function closes
  
  ContractQQ_macro(12, c,c,a,b);
  ContractQQ_macro(13, c,a,c,b);
  ContractQQ_macro(14, c,a,b,c);
  ContractQQ_macro(23, a,c,c,b);
  ContractQQ_macro(24, a,c,b,c);
  ContractQQ_macro(34, a,b,c,c);
#undef ContractQQ_macro

  
  __device__ __host__ inline void computeContractQQ(ContractQQArg *arg, int x_cb, int pty){
    
    Vector out[QUDA_NVEC_PROP];
    
    switch(arg->cntrID){
    case cntr12:
      performContractQQ12(out, arg, x_cb, pty);
      break;
    case cntr13:
      performContractQQ13(out, arg, x_cb, pty);
      break;
    case cntr14:
      performContractQQ14(out, arg, x_cb, pty);
      break;
    case cntr23:
      performContractQQ23(out, arg, x_cb, pty);
      break;
    case cntr24:
      performContractQQ24(out, arg, x_cb, pty);
      break;
    case cntr34:
      performContractQQ34(out, arg, x_cb, pty);
      break;
    case cntr_INVALID: // Added it just to avoid the compilation warning, check has already been made
      break;
    }    

    rotateVectorBasis(out, QLUA_qdp2quda);
        
    for(int ivec=0;ivec<arg->nVec;ivec++)
      arg->pOut[ivec](x_cb, pty) = out[ivec];
    
  } //-- function closes
  
  
  //-- CPU kernel for performing the contractions
  void ContractQQ_CPU(ContractQQArg arg){    
    for (int parity= 0; parity < arg.nParity; parity++){
      parity = (arg.nParity == 2) ? parity : arg.parity;
      
      for (int x_cb = 0; x_cb < arg.volumeCB; x_cb++){
	computeContractQQ(&arg,x_cb,parity);
      }
    }
  }

  //-- GPU kernel for performing the contractions
  __global__ void ContractQQ_GPU(ContractQQArg *arg_dev){
    
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    
    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (x_cb >= arg_dev->volumeCB) return;
    if (parity >= arg_dev->nParity) return;
    parity = (arg_dev->nParity == 2) ? parity : arg_dev->parity;
    
    computeContractQQ(arg_dev, x_cb, parity);
  }    
  

  //-- Class definition
  class ContractQQ : public TunableVectorY {

  protected:
    ContractQQArg &arg;
    ContractQQArg *arg_dev;
    const ColorSpinorField &meta;
    
    long long flops() const{
      return QUDA_Nc*QUDA_Nc*QUDA_Ns*QUDA_Ns*QUDA_Ns*4*(3+2)*arg.nParity*(long long)meta.VolumeCB();
    }
    long long bytes() const{
      return meta.Bytes() + QUDA_Nc*QUDA_Ns*2*meta.Bytes();
    }
    
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volumeCB; }
    
  public:
  ContractQQ(ContractQQArg &arg, ContractQQArg *arg_dev, const ColorSpinorField &meta) : TunableVectorY(arg.nParity), arg(arg), arg_dev(arg_dev), meta(meta)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
    }
    virtual ~ContractQQ() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	ContractQQ_CPU(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

	ContractQQ_GPU <<<tp.grid,tp.block,tp.shared_bytes,stream>>> (arg_dev);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  }; //-- Class definition


  
  //-- Top level function, called within interface-qlua
  void cudaContractQQ(ColorSpinorField **propOut, ColorSpinorField **propIn1, ColorSpinorField **propIn2, int parity, int Nc, int Ns, QQParam cQQParam){
    
    if(Nc != QUDA_Nc) errorQuda("cudaContractQQ: Supports only Ncolor = %d. Got Nc = %d\n", QUDA_Nc, Nc);
    if(Ns != QUDA_Ns) errorQuda("cudaContractQQ: Supports only Nspin = %d.  Got Ns = %d\n", QUDA_Ns, Ns);
    
    ContractQQArg arg(propOut, propIn1, propIn2, parity, cQQParam);    
    ContractQQArg *arg_dev;
    cudaMalloc((void**)&(arg_dev), sizeof(ContractQQArg) );
    cudaMemcpy(arg_dev, &arg, sizeof(ContractQQArg), cudaMemcpyHostToDevice);
    
    ContractQQ contract(arg, arg_dev, *propIn1[0]);
    contract.apply(0);

    cudaDeviceSynchronize();
    checkCudaError();
  }

  
} //-- namespace quda
