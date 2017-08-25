#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>

#include <color_spinor_array.h>
#include <blas_quda.h>

#include <string.h>
#include <iostream>
#include <misc_helpers.h>
#include <face_quda.h>
#include <dslash_quda.h>

int zeroCopy = 0;

namespace quda {

  bool cudaColorSpinorArray::initGhostFaceBuffer = false;
  size_t cudaColorSpinorArray::ghostFaceBytes = 0;

  cudaColorSpinorArray::cudaColorSpinorArray(const ColorSpinorParam &param) : 
    ColorSpinorArray(param), alloc(false), init(true),
    ghost_field_tex{nullptr,nullptr}, bufferMessageHandler(0)
  {
    // this must come before create
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      norm = param.norm;
    }

    create(param.create);

    if  (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_COPY_FIELD_CREATE) {
      errorQuda("not implemented");
    }
  }

  ColorSpinorArray& cudaColorSpinorArray::operator=(const ColorSpinorArray &src) {
    if (typeid(src) == typeid(cudaColorSpinorArray)) {
      *this = (dynamic_cast<const cudaColorSpinorArray&>(src));
    } else if (typeid(src) == typeid(cpuColorSpinorArray)) {
      *this = (dynamic_cast<const cpuColorSpinorArray&>(src));
    } else {
      errorQuda("Unknown input ColorSpinorArray %s", typeid(src).name());
    }
    return *this;
  }

  cudaColorSpinorArray& cudaColorSpinorArray::operator=(const cpuColorSpinorArray &src) {
    // keep current attributes unless unset
    if (!ColorSpinorArray::init) { // note this will turn a reference field into a regular field
      destroy();
      ColorSpinorArray::operator=(src);
      create(QUDA_COPY_FIELD_CREATE);
    }
    loadSpinorArray(src);
    return *this;
  }

  cudaColorSpinorArray::~cudaColorSpinorArray() {
    destroyComms();
    destroy();
  }

  void cudaColorSpinorArray::create(const QudaFieldCreate create) {

    if (siteSubset == QUDA_FULL_SITE_SUBSET && siteOrder != QUDA_EVEN_ODD_SITE_ORDER) {
      errorQuda("Subset not implemented");
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      switch(mem_type) {
      case QUDA_MEMORY_DEVICE:
	v = pool_device_malloc(bytes);
	if (precision == QUDA_HALF_PRECISION) norm = pool_device_malloc(norm_bytes);
	break;
      case QUDA_MEMORY_MAPPED:
	v_h = mapped_malloc(bytes);
	cudaHostGetDevicePointer(&v, v_h, 0); // set the matching device pointer
	if (precision == QUDA_HALF_PRECISION) {
	  norm_h = mapped_malloc(norm_bytes);
	  cudaHostGetDevicePointer(&norm, norm_h, 0); // set the matching device pointer
	}
	break;
      default:
	errorQuda("Unsupported memory type %d", mem_type);
      }
      alloc = true;
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (siteSubset != QUDA_FULL_SITE_SUBSET) {
	zeroPad();
      }
      else { //temporary hack for the full spinor field sets, manual zeroPad for each component:
	warningQuda("cudaColorSpinorArray::create : siteSubset == QUDA_FULL_SITE_SUBSET, yet I do nothing.\n"); 
      }
    }

  }

  void cudaColorSpinorArray::destroy() {
    if (alloc) {
      switch(mem_type) {
      case QUDA_MEMORY_DEVICE:
	pool_device_free(v);
	if (precision == QUDA_HALF_PRECISION) pool_device_free(norm);
	break;
      case QUDA_MEMORY_MAPPED:
	host_free(v_h);
	if (precision == QUDA_HALF_PRECISION) host_free(norm_h);
	break;
      default:
	errorQuda("Unsupported memory type %d", mem_type);
      }
    }
  }

  // cuda's floating point format, IEEE-754, represents the floating point
  // zero as 4 zero bytes
  void cudaColorSpinorArray::zero() {
    cudaMemsetAsync(v, 0, bytes, streams[Nstream-1]);
    if (precision == QUDA_HALF_PRECISION) cudaMemsetAsync(norm, 0, norm_bytes, streams[Nstream-1]);
  }


  void cudaColorSpinorArray::allocateGhostBuffer(int nFace, bool spin_project) const {

    if (!comm_partitioned()) {
      for (int i=0; i<4; i++) ghost_face_bytes[i] = 0;
      return;
    }

    createGhostZone(nFace, spin_project);

    // temporary work around until the ghost buffer for fine and
    // coarse grid are merged: this ensures we reset the fine ghost
    // buffer if the coarse grid operator allocates a ghost buffer
    // that is larger than the fine grid operator
    static size_t ghostFaceBytes_ = 0;

    // only allocate if not already allocated or buffer required is bigger than previously
    if ( !initGhostFaceBuffer || ghost_bytes > ghostFaceBytes || ghost_bytes > ghostFaceBytes_) {

      if (initGhostFaceBuffer) {
	if (ghost_bytes) {
	  for (int b=0; b<2; b++) {
	    device_pinned_free(ghost_recv_buffer_d[b]);
	    device_pinned_free(ghost_send_buffer_d[b]);
	    host_free(ghost_pinned_buffer_h[b]);
	  }
	}
      }

      if (ghost_bytes > 0) {
	for (int b=0; b<2; ++b) {
	  // gpu receive buffer (use pinned allocator to avoid this being redirected, e.g., by QDPJIT)
	  ghost_recv_buffer_d[b] = device_pinned_malloc(ghost_bytes);

	  // gpu send buffer (use pinned allocator to avoid this being redirected, e.g., by QDPJIT)
	  ghost_send_buffer_d[b] = device_pinned_malloc(ghost_bytes);

	  // pinned buffer used for sending and receiving
	  ghost_pinned_buffer_h[b] = mapped_malloc(2*ghost_bytes);

	  // set the matching device-mapper pointer
	  cudaHostGetDevicePointer(&ghost_pinned_buffer_hd[b], ghost_pinned_buffer_h[b], 0);
	}

	initGhostFaceBuffer = true;
	ghostFaceBytes = ghost_bytes;
	ghostFaceBytes_ = ghost_bytes;
      }

      LatticeField::ghost_field_reset = true; // this signals that we must reset the IPC comms
    }

  }

  void cudaColorSpinorArray::freeGhostBuffer(void)
  {
    destroyIPCComms();

    if (!initGhostFaceBuffer) return;
  
    for (int b=0; b<2; b++) {
      // free receive buffer
      if (ghost_recv_buffer_d[b]) device_pinned_free(ghost_recv_buffer_d[b]);
      ghost_recv_buffer_d[b] = nullptr;

      // free send buffer
      if (ghost_send_buffer_d[b]) device_pinned_free(ghost_send_buffer_d[b]);
      ghost_send_buffer_d[b] = nullptr;

      // free pinned memory buffers
      if (ghost_pinned_buffer_h[b]) host_free(ghost_pinned_buffer_h[b]);
      ghost_pinned_buffer_h[b] = nullptr;
      ghost_pinned_buffer_hd[b] = nullptr;
    }
    initGhostFaceBuffer = false;
  }

  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorArray::packGhost(const int nFace, const QudaParity parity, 
                                       const int dim, const QudaDirection dir,
				       const int dagger, cudaStream_t *stream, 
				       MemoryLocation location [2*QUDA_MAX_DIM], double a, double b)
  {
#ifdef MULTI_GPU
    int face_num = (dir == QUDA_BACKWARDS) ? 0 : (dir == QUDA_FORWARDS) ? 1 : 2;
    void *packBuffer[2*QUDA_MAX_DIM];

    for (int dim=0; dim<4; dim++) {
      for (int dir=0; dir<2; dir++) {
	switch(location[2*dim+dir]) {
	case Device: // pack to local device buffer
	  packBuffer[2*dim+dir] = my_face_dim_dir_d[bufferIndex][dim][dir]; break;
	case Host:   // pack to zero-copy memory
	  packBuffer[2*dim+dir] = my_face_dim_dir_hd[bufferIndex][dim][dir]; break;
	default: errorQuda("Undefined location %d", location[2*dim+dir]);
	}
      }
    }

    packFace<cudaColorSpinorArray>(packBuffer, *this, location, nFace, dagger, parity, dim, face_num, *stream, a, b);
#else
    errorQuda("packGhost not built on single-GPU build");
#endif
  }
 
  // send the ghost zone to the host
  void cudaColorSpinorArray::sendGhost(void *ghost_spinor, const int nFace, const int dim, 
				       const QudaDirection dir, const int dagger, 
				       cudaStream_t *stream) {

#ifdef MULTI_GPU
    int Nvec = (nSpin == 1 || precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1);  // (spin proj.) degrees of freedom
    int Npad = Nint / Nvec; // number Nvec buffers we have
    
    if (dim !=3 || getKernelPackT()) { // use kernels to pack into contiguous buffers then a single cudaMemcpy

      size_t bytes = nFace*Nint*ghostFace[dim]*precision;

      if (precision == QUDA_HALF_PRECISION) bytes += nFace*ghostFace[dim]*sizeof(float);

      void* gpu_buf = (dir == QUDA_BACKWARDS) ? my_face_dim_dir_d[bufferIndex][dim][0] : my_face_dim_dir_d[bufferIndex][dim][1];

      cudaMemcpyAsync(ghost_spinor, gpu_buf, bytes, cudaMemcpyDeviceToHost, *stream);

    } else if (this->TwistFlavor() != QUDA_TWIST_NONDEG_DOUBLET) { // do multiple cudaMemcpys

      const int x4 = nDim==5 ? x[4] : 1;
      const int Nt_minus1_offset = (volumeCB - nFace*ghostFace[3])/x4; // N_t -1 = Vh-Vsh

      int offset = 0;
      if (nSpin == 1) {
	offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset;
      } else if (nSpin == 4) {
	// !dagger: send lower components backwards, send upper components forwards
	// dagger: send upper components backwards, send lower components forwards
	bool upper = dagger ? true : false; // Fwd is !Back  
	if (dir == QUDA_FORWARDS) upper = !upper;
	int lower_spin_offset = Npad*stride;
	if (upper) offset = (dir == QUDA_BACKWARDS ? 0 : Nt_minus1_offset);
	else offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? 0 : Nt_minus1_offset);
      }
    
      size_t len = nFace*(ghostFace[3]/x4)*Nvec*precision;
      size_t dpitch = x4*len;
      size_t spitch = stride*Nvec*precision;

      // QUDA Memcpy NPad's worth. 
      //  -- Dest will point to the right beginning PAD. 
      //  -- Each Pad has size Nvec*Vsh Floats. 
      //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next
      for (int s=0; s<x4; s++) { // loop over multiple 4-d volumes (if they exist)
	void *dst = (char*)ghost_spinor + s*len;
	void *src = (char*)v + (offset + s*(volumeCB/x4))*Nvec*precision;
	cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);

	if (precision == QUDA_HALF_PRECISION) {
	  size_t len = nFace*(ghostFace[3]/x4)*sizeof(float);
	  int norm_offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset*sizeof(float);
	  void *dst = (char*)ghost_spinor + nFace*Nint*ghostFace[3]*precision + s*len;
	  void *src = (char*)norm + norm_offset + s*(volumeCB/x4)*sizeof(float);
	  cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream);
	}
      }
    }else{
      int flavorVolume = volume / 2;
      int flavorTFace  = ghostFace[3] / 2;
      int flavor1_Nt_minus1_offset = (flavorVolume - flavorTFace);
      int flavor2_Nt_minus1_offset = (volume - flavorTFace);
      int flavor1_offset = 0;
      int flavor2_offset = 0;
      // !dagger: send lower components backwards, send upper components forwards
      // dagger: send upper components backwards, send lower components forwards
      bool upper = dagger ? true : false; // Fwd is !Back
      if (dir == QUDA_FORWARDS) upper = !upper;
      int lower_spin_offset = Npad*stride;//ndeg tm: stride=2*flavor_volume+pad
      if (upper) {
        flavor1_offset = (dir == QUDA_BACKWARDS ? 0 : flavor1_Nt_minus1_offset);
        flavor2_offset = (dir == QUDA_BACKWARDS ? flavorVolume : flavor2_Nt_minus1_offset);
      }else{
        flavor1_offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? 0 : flavor1_Nt_minus1_offset);
        flavor2_offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? flavorVolume : flavor2_Nt_minus1_offset);
      }

      // QUDA Memcpy NPad's worth.
      //  -- Dest will point to the right beginning PAD.
      //  -- Each Pad has size Nvec*Vsh Floats.
      //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next

      void *dst = (char*)ghost_spinor;
      void *src = (char*)v + flavor1_offset*Nvec*precision;
      size_t len = flavorTFace*Nvec*precision;
      size_t spitch = stride*Nvec*precision;//ndeg tm: stride=2*flavor_volume+pad
      size_t dpitch = 2*len;
      cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);
      dst = (char*)ghost_spinor+len;
      src = (char*)v + flavor2_offset*Nvec*precision;
      cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);

      if (precision == QUDA_HALF_PRECISION) {
        int Nt_minus1_offset = (flavorVolume - flavorTFace);
        int norm_offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset*sizeof(float);
	void *dst = (char*)ghost_spinor + Nint*ghostFace[3]*precision;
	void *src = (char*)norm + norm_offset;
        size_t dpitch = flavorTFace*sizeof(float);
        size_t spitch = flavorVolume*sizeof(float);
	cudaMemcpy2DAsync(dst, dpitch, src, spitch, flavorTFace*sizeof(float), 2, cudaMemcpyDeviceToHost, *stream);
      }
    }
#else
    errorQuda("sendGhost not built on single-GPU build");
#endif

  }


  void cudaColorSpinorArray::unpackGhost(const void* ghost_spinor, const int nFace, 
					 const int dim, const QudaDirection dir, 
					 const int dagger, cudaStream_t* stream) 
  {
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1);  // (spin proj.) degrees of freedom

    int len = nFace*ghostFace[dim]*Nint*precision;
    const void *src = ghost_spinor;
  
    int ghost_offset = (dir == QUDA_BACKWARDS) ? ghostOffset[dim][0] : ghostOffset[dim][1];
    void *ghost_dst = (char*)ghost_recv_buffer_d[bufferIndex] + precision*ghost_offset;

    if (precision == QUDA_HALF_PRECISION) len += nFace*ghostFace[dim]*sizeof(float);

    cudaMemcpyAsync(ghost_dst, src, len, cudaMemcpyHostToDevice, *stream);
  }


  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorArray::packGhostExtended(const int nFace, const int R[], const QudaParity parity,
					       const int dim, const QudaDirection dir,
					       const int dagger, cudaStream_t *stream, bool zero_copy)
  {
#ifdef MULTI_GPU
    int face_num = (dir == QUDA_BACKWARDS) ? 0 : (dir == QUDA_FORWARDS) ? 1 : 2;
    void *packBuffer[2*QUDA_MAX_DIM];
    MemoryLocation location[2*QUDA_MAX_DIM];

    if (zero_copy) {
      for (int d=0; d<4; d++) {
	packBuffer[2*d+0] = my_face_dim_dir_hd[bufferIndex][d][0];
	packBuffer[2*d+1] = my_face_dim_dir_hd[bufferIndex][d][1];
	location[2*d+0] = Host;
	location[2*d+1] = Host;
      }
    } else {
      for (int d=0; d<4; d++) {
	packBuffer[2*d+0] = my_face_dim_dir_d[bufferIndex][d][0];
	packBuffer[2*d+1] = my_face_dim_dir_d[bufferIndex][d][1];
	location[2*d+0] = Device;
	location[2*d+1] = Device;
      }
    }

    packFaceExtended<cudaColorSpinorArray>(packBuffer, *this, location, nFace, R, dagger, parity, dim, face_num, *stream);
#else
    errorQuda("packGhostExtended not built on single-GPU build");
#endif

  }


  // copy data from host buffer into boundary region of device field
  void cudaColorSpinorArray::unpackGhostExtended(const void* ghost_spinor, const int nFace, const QudaParity parity,
                                                 const int dim, const QudaDirection dir, 
                                                 const int dagger, cudaStream_t* stream, bool zero_copy)
  {
    // First call the regular unpackGhost routine to copy data into the `usual' ghost-zone region 
    // of the data array 
    unpackGhost(ghost_spinor, nFace, dim, dir, dagger, stream);

    // Next step is to copy data from the ghost zone back to the interior region
    int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1); // (spin proj.) degrees of freedom

    int len = nFace*ghostFace[dim]*Nint;
    int offset = length + ghostOffset[dim][0];
    offset += (dir == QUDA_BACKWARDS) ? 0 : len;

#ifdef MULTI_GPU
    const int face_num = 2;
    const bool unpack = true;
    const int R[4] = {0,0,0,0};
    void *packBuffer[2*QUDA_MAX_DIM];
    MemoryLocation location[2*QUDA_MAX_DIM];

    if (zero_copy) {
      for (int d=0; d<4; d++) {
	packBuffer[2*d+0] = my_face_dim_dir_hd[bufferIndex][d][0];
	packBuffer[2*d+1] = my_face_dim_dir_hd[bufferIndex][d][1];
	location[2*d+0] = Host;
	location[2*d+1] = Host;
      }
    } else {
      for (int d=0; d<4; d++) {
	packBuffer[2*d+0] = my_face_dim_dir_d[bufferIndex][d][0];
	packBuffer[2*d+1] = my_face_dim_dir_d[bufferIndex][d][1];
	location[2*d+0] = Device;
	location[2*d+1] = Device;
      }
    }

    packFaceExtended<cudaColorSpinorArray>(packBuffer, *this, location, nFace, R, dagger, parity, dim, face_num, *stream, unpack);
#else
    errorQuda("unpackGhostExtended not built on single-GPU build");
#endif
  }


  cudaStream_t *stream;

  void cudaColorSpinorArray::createComms(int nFace, bool spin_project) {

    allocateGhostBuffer(nFace,spin_project); // allocate the ghost buffer if not yet allocated

    // ascertain if this instance needs its comms buffers to be updated
    bool comms_reset = ghost_field_reset || // FIXME add send buffer check
      (my_face_h[0] != ghost_pinned_buffer_h[0]) || (my_face_h[1] != ghost_pinned_buffer_h[1]) || // pinned buffers
      (ghost_field_tex[0] != ghost_recv_buffer_d[0]) || (ghost_field_tex[1] != ghost_recv_buffer_d[1]); // receive buffers

    if (!initComms || comms_reset) {

      destroyComms(); // if we are requesting a new number of faces destroy and start over

      int Nint = nColor * nSpin * 2 / (nSpin == 4 && spin_project ? 2 : 1); // number of internal degrees of freedom

      for (int i=0; i<nDimComms; i++) { // compute size of ghost buffers required
	if (!commDimPartitioned(i)) { ghost_face_bytes[i] = 0; continue; }
	ghost_face_bytes[i] = nFace*ghostFace[i]*Nint*precision;
	if (precision == QUDA_HALF_PRECISION) ghost_face_bytes[i] += nFace*ghostFace[i]*sizeof(float);
      }

      // initialize the ghost pinned buffers
      for (int b=0; b<2; b++) {
	my_face_h[b] = ghost_pinned_buffer_h[b];
	my_face_hd[b] = ghost_pinned_buffer_hd[b];
	from_face_h[b] = static_cast<char*>(my_face_h[b]) + ghost_bytes;
	from_face_hd[b] = static_cast<char*>(my_face_hd[b]) + ghost_bytes;
      }

      // initialize the ghost receive pointers
      for (int i=0; i<nDimComms; ++i) {
	if (commDimPartitioned(i)) {
	  for (int b=0; b<2; b++) {
	    ghost[b][i] = static_cast<char*>(ghost_recv_buffer_d[b]) + ghostOffset[i][0]*precision;
	    if (precision == QUDA_HALF_PRECISION)
	      ghostNorm[b][i] = static_cast<char*>(ghost_recv_buffer_d[b]) + ghostNormOffset[i][0]*QUDA_SINGLE_PRECISION;
	  }
	}
      }

      // initialize ghost send pointers
      size_t offset = 0;
      for (int i=0; i<nDimComms; i++) {
	if (!commDimPartitioned(i)) continue;

	for (int b=0; b<2; ++b) {
	  my_face_dim_dir_h[b][i][0] = static_cast<char*>(my_face_h[b]) + offset;
	  from_face_dim_dir_h[b][i][0] = static_cast<char*>(from_face_h[b]) + offset;

	  my_face_dim_dir_hd[b][i][0] = static_cast<char*>(my_face_hd[b]) + offset;
	  from_face_dim_dir_hd[b][i][0] = static_cast<char*>(from_face_hd[b]) + offset;

	  my_face_dim_dir_d[b][i][0] = static_cast<char*>(ghost_send_buffer_d[b]) + offset;
	  from_face_dim_dir_d[b][i][0] = static_cast<char*>(ghost_recv_buffer_d[b]) + ghostOffset[i][0]*precision;
	} // loop over b
	offset += ghost_face_bytes[i];

	for (int b=0; b<2; ++b) {
	  my_face_dim_dir_h[b][i][1] = static_cast<char*>(my_face_h[b]) + offset;
	  from_face_dim_dir_h[b][i][1] = static_cast<char*>(from_face_h[b]) + offset;

	  my_face_dim_dir_hd[b][i][1] = static_cast<char*>(my_face_hd[b]) + offset;
	  from_face_dim_dir_hd[b][i][1] = static_cast<char*>(from_face_hd[b]) + offset;

	  my_face_dim_dir_d[b][i][1] = static_cast<char*>(ghost_send_buffer_d[b]) + offset;
	  from_face_dim_dir_d[b][i][1] = static_cast<char*>(ghost_recv_buffer_d[b]) + ghostOffset[i][1]*precision;
	} // loop over b
	offset += ghost_face_bytes[i];

      } // loop over dimension

      bool gdr = comm_gdr_enabled(); // only allocate rdma buffers if GDR enabled

      // initialize the message handlers
      for (int i=0; i<nDimComms; i++) {
	if (!commDimPartitioned(i)) continue;

	for (int b=0; b<2; ++b) {
	  mh_send_fwd[b][i] = comm_declare_send_relative(my_face_dim_dir_h[b][i][1], i, +1, ghost_face_bytes[i]);
	  mh_send_back[b][i] = comm_declare_send_relative(my_face_dim_dir_h[b][i][0], i, -1, ghost_face_bytes[i]);

	  mh_recv_fwd[b][i] = comm_declare_receive_relative(from_face_dim_dir_h[b][i][1], i, +1, ghost_face_bytes[i]);
	  mh_recv_back[b][i] = comm_declare_receive_relative(from_face_dim_dir_h[b][i][0], i, -1, ghost_face_bytes[i]);

	  mh_send_rdma_fwd[b][i] = gdr ? comm_declare_send_relative(my_face_dim_dir_d[b][i][1], i, +1, ghost_face_bytes[i]) : nullptr;
	  mh_send_rdma_back[b][i] = gdr ? comm_declare_send_relative(my_face_dim_dir_d[b][i][0], i, -1, ghost_face_bytes[i]) : nullptr;

	  mh_recv_rdma_fwd[b][i] = gdr ? comm_declare_receive_relative(from_face_dim_dir_d[b][i][1], i, +1, ghost_face_bytes[i]) : nullptr;
	  mh_recv_rdma_back[b][i] = gdr ? comm_declare_receive_relative(from_face_dim_dir_d[b][i][0], i, -1, ghost_face_bytes[i]) : nullptr;
	} // loop over b

      } // loop over dimension
     
      initComms = true;
      checkCudaError();
    }

    if (LatticeField::ghost_field_reset) destroyIPCComms();
    createIPCComms();
  }

  void cudaColorSpinorArray::destroyComms()
  {
    if (initComms) {

      for (int b=0; b<2; ++b) {
	for (int i=0; i<nDimComms; i++) {
	  if (commDimPartitioned(i)) {
	    if (mh_recv_fwd[b][i]) comm_free(mh_recv_fwd[b][i]);
	    if (mh_recv_back[b][i]) comm_free(mh_recv_back[b][i]);
	    if (mh_send_fwd[b][i]) comm_free(mh_send_fwd[b][i]);
	    if (mh_send_back[b][i]) comm_free(mh_send_back[b][i]);

	    if (mh_recv_rdma_fwd[b][i]) comm_free(mh_recv_rdma_fwd[b][i]);
	    if (mh_recv_rdma_back[b][i]) comm_free(mh_recv_rdma_back[b][i]);
	    if (mh_send_rdma_fwd[b][i]) comm_free(mh_send_rdma_fwd[b][i]);
	    if (mh_send_rdma_back[b][i]) comm_free(mh_send_rdma_back[b][i]);
	  }
	}
      } // loop over b

      initComms = false;
      checkCudaError();
    }

  }

  void cudaColorSpinorArray::streamInit(cudaStream_t *stream_p) {
    stream = stream_p;
  }

  void cudaColorSpinorArray::pack(int nFace, int parity, int dagger, int stream_idx,
				  MemoryLocation location[2*QUDA_MAX_DIM], double a, double b)
  {
    createComms(nFace); // must call this first

    const int dim=-1; // pack all partitioned dimensions
 
    packGhost(nFace, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[stream_idx], location, a, b);
  }

  void cudaColorSpinorArray::packExtended(const int nFace, const int R[], const int parity, 
                                          const int dagger, const int dim,
                                          cudaStream_t *stream_p, const bool zero_copy)
  {
    createComms(nFace); // must call this first

    stream = stream_p;
 
    packGhostExtended(nFace, R, (QudaParity)parity, dim, QUDA_BOTH_DIRS, dagger, &stream[zero_copy ? 0 : (Nstream-1)], zero_copy);
  }

  void cudaColorSpinorArray::gather(int nFace, int dagger, int dir, cudaStream_t* stream_p)
  {
    int dim = dir/2;

    // If stream_p != 0, use pack_stream, else use the stream array
    cudaStream_t *pack_stream = (stream_p) ? stream_p : stream+dir;

    if (dir%2 == 0) {
      // backwards copy to host
      if (comm_peer2peer_enabled(0,dim)) return;

      sendGhost(my_face_dim_dir_h[bufferIndex][dim][0], nFace, dim, QUDA_BACKWARDS, dagger, pack_stream);
    } else {
      // forwards copy to host
      if (comm_peer2peer_enabled(1,dim)) return;

      sendGhost(my_face_dim_dir_h[bufferIndex][dim][1], nFace, dim, QUDA_FORWARDS, dagger, pack_stream);
    }
  }


  void cudaColorSpinorArray::recvStart(int nFace, int dir, int dagger, cudaStream_t* stream_p, bool gdr) {

    int dim = dir/2;
    if (!commDimPartitioned(dim)) return;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but not GDR is not enabled");

    if (dir%2 == 0) { // sending backwards
      if (comm_peer2peer_enabled(1,dim)) {
	// receive from the processor in the +1 direction
	comm_start(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else if (gdr) {
        // Prepost receive
        comm_start(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
        // Prepost receive
        comm_start(mh_recv_fwd[bufferIndex][dim]);
      }
    } else { //sending forwards
      // Prepost receive
      if (comm_peer2peer_enabled(0,dim)) {
	comm_start(mh_recv_p2p_back[bufferIndex][dim]);
      } else if (gdr) {
        comm_start(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
        comm_start(mh_recv_back[bufferIndex][dim]);
      }
    }
  }


  void cudaColorSpinorArray::sendStart(int nFace, int d, int dagger, cudaStream_t* stream_p, bool gdr) {

    int dim = d/2;
    int dir = d%2;
    if (!commDimPartitioned(dim)) return;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but not GDR is not enabled");

    int Nvec = (nSpin == 1 || precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
    int Nint = (nColor * nSpin * 2)/(nSpin == 4 ? 2 : 1); // (spin proj.) degrees of freedom
    int Npad = Nint/Nvec;

    if (!comm_peer2peer_enabled(dir,dim)) {
      if (dir == 0)
	if (gdr) comm_start(mh_send_rdma_back[bufferIndex][dim]);
	else comm_start(mh_send_back[bufferIndex][dim]);
      else
	if (gdr) comm_start(mh_send_rdma_fwd[bufferIndex][dim]);
	else comm_start(mh_send_fwd[bufferIndex][dim]);
    } else { // doing peer-to-peer
      cudaStream_t *copy_stream = (stream_p) ? stream_p : stream + d;

      // all goes here
      void* ghost_dst = static_cast<char*>(ghost_remote_send_buffer_d[bufferIndex][dim][dir])
	+ precision*ghostOffset[dim][(dir+1)%2];
      void *ghost_norm_dst = static_cast<char*>(ghost_remote_send_buffer_d[bufferIndex][dim][dir])
	+ QUDA_SINGLE_PRECISION*ghostNormOffset[dim][(d+1)%2];

      if (dim != 3 || getKernelPackT()) {

	cudaMemcpyAsync(ghost_dst,
			my_face_dim_dir_d[bufferIndex][dim][dir],
			ghost_face_bytes[dim],
			cudaMemcpyDeviceToDevice,
			*copy_stream); // copy to forward processor

      } else if (this->TwistFlavor() != QUDA_TWIST_NONDEG_DOUBLET) {

	const int x4 = nDim==5 ? x[4] : 1;
	const int Nt_minus_offset = (volumeCB - nFace*ghostFace[3])/x4;

	int offset = 0;
	if (nSpin == 1) {
	  offset = (dir == 0) ? 0 : Nt_minus_offset;
	} else if (nSpin == 4) {
	  // !dagger: send lower components backwards, send upper components forwards
	  // dagger: send upper components backwards, send lower components forwards
	  bool upper = dagger ? true : false;
	  if (dir == 1) upper = !upper;
	  int lower_spin_offset = Npad*stride;
	  if (dir == 0) {
	    offset = upper ? 0 : lower_spin_offset;
	  } else {
	    offset = (upper) ? Nt_minus_offset : lower_spin_offset + Nt_minus_offset;
	  }
	}

	size_t len = nFace*(ghostFace[3]/x4)*Nvec*precision;
	size_t dpitch = x4*len;
	size_t spitch = stride*Nvec*precision;

	for (int s=0; s<x4; s++) {
	  void *dst = (char*)ghost_dst + s*len;
	  void *src = (char*)v + (offset + s*(volumeCB/x4))*Nvec*precision;
	  // start the copy
	  cudaMemcpy2DAsync(dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

	  if (precision == QUDA_HALF_PRECISION) {
	    size_t len = nFace*(ghostFace[3]/x4)*sizeof(float);
	    int norm_offset = (dir == 0) ? 0 : Nt_minus_offset*sizeof(float);
	    void *dst = (char*)ghost_norm_dst + s*len;
	    void *src = static_cast<char*>(norm) + norm_offset + s*(volumeCB/x4)*sizeof(float);
	    cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToDevice, *copy_stream);
	  }
	}
      } else { // twisted doublet
	int flavorVolume = volume / 2;
	int flavorTFace  = ghostFace[3] / 2;
	int flavor1_Nt_minus1_offset = (flavorVolume - flavorTFace);
	int flavor2_Nt_minus1_offset = (volume - flavorTFace);
	int flavor1_offset = 0;
	int flavor2_offset = 0;
	// !dagger: send lower components backwards, send upper components forwards
	// dagger: send upper components backwards, send lower components forwards
	bool upper = dagger ? true : false; // Fwd is !Back
	if (dir == 1) upper = !upper;
	int lower_spin_offset = Npad*stride;//ndeg tm: stride=2*flavor_volume+pad
	if (upper) {
	  flavor1_offset = (dir == 0 ? 0 : flavor1_Nt_minus1_offset);
	  flavor2_offset = (dir == 0 ? flavorVolume : flavor2_Nt_minus1_offset);
	}else{
	  flavor1_offset = lower_spin_offset + (dir == 0 ? 0 : flavor1_Nt_minus1_offset);
	  flavor2_offset = lower_spin_offset + (dir == 0 ? flavorVolume : flavor2_Nt_minus1_offset);
	}

	// QUDA Memcpy NPad's worth.
	//  -- Dest will point to the right beginning PAD.
	//  -- Each Pad has size Nvec*Vsh Floats.
	//  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next

	void *src = static_cast<char*>(v) + flavor1_offset*Nvec*precision;
	size_t len = flavorTFace*Nvec*precision;
	size_t spitch = stride*Nvec*precision;//ndeg tm: stride=2*flavor_volume+pad
	size_t dpitch = 2*len;
	cudaMemcpy2DAsync(ghost_dst, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

	src = static_cast<char*>(v) + flavor2_offset*Nvec*precision;
	cudaMemcpy2DAsync(static_cast<char*>(ghost_dst)+len, dpitch, src, spitch, len, Npad, cudaMemcpyDeviceToDevice, *copy_stream);

	if (precision == QUDA_HALF_PRECISION) {
	  int norm_offset = (dir == 0) ? 0 : flavor1_Nt_minus1_offset*sizeof(float);
	  void *src = static_cast<char*>(norm) + norm_offset;
	  size_t dpitch = flavorTFace*sizeof(float);
	  size_t spitch = flavorVolume*sizeof(float);
	  cudaMemcpy2DAsync(ghost_norm_dst, dpitch, src, spitch, flavorTFace*sizeof(float), 2, cudaMemcpyDeviceToDevice, *copy_stream);
	}
      }

      if (dir == 0) {
	// record the event
	cudaEventRecord(ipcCopyEvent[bufferIndex][0][dim], *copy_stream);
	// send to the propcessor in the -1 direction
	comm_start(mh_send_p2p_back[bufferIndex][dim]);
      } else {
	cudaEventRecord(ipcCopyEvent[bufferIndex][1][dim], *copy_stream);
	// send to the processor in the +1 direction
	comm_start(mh_send_p2p_fwd[bufferIndex][dim]);
      }
    }
  }

  void cudaColorSpinorArray::commsStart(int nFace, int dir, int dagger, cudaStream_t* stream_p, bool gdr) {
    recvStart(nFace, dir, dagger, stream_p, gdr);
    sendStart(nFace, dir, dagger, stream_p, gdr);
  }


  static bool complete_recv_fwd[QUDA_MAX_DIM] = { };
  static bool complete_recv_back[QUDA_MAX_DIM] = { };
  static bool complete_send_fwd[QUDA_MAX_DIM] = { };
  static bool complete_send_back[QUDA_MAX_DIM] = { };

  int cudaColorSpinorArray::commsQuery(int nFace, int dir, int dagger, cudaStream_t *stream_p, bool gdr) {

    int dim = dir/2;
    if (!commDimPartitioned(dim)) return 0;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but not GDR is not enabled");

    if (dir%2==0) {

      if (comm_peer2peer_enabled(1,dim)) {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else if (gdr) {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_fwd[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(0,dim)) {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_p2p_back[bufferIndex][dim]);
      } else if (gdr) {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_rdma_back[bufferIndex][dim]);
      } else {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_back[bufferIndex][dim]);
      }

      if (complete_recv_fwd[dim] && complete_send_back[dim]) {
	complete_recv_fwd[dim] = false;
	complete_send_back[dim] = false;
	return 1;
      }

    } else { // dir%2 == 1

      if (comm_peer2peer_enabled(0,dim)) {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_p2p_back[bufferIndex][dim]);
      } else if (gdr) {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_back[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(1,dim)) {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_p2p_fwd[bufferIndex][dim]);
      } else if (gdr) {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_rdma_fwd[bufferIndex][dim]);
      } else {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_fwd[bufferIndex][dim]);
      }

      if (complete_recv_back[dim] && complete_send_fwd[dim]) {
	complete_recv_back[dim] = false;
	complete_send_fwd[dim] = false;
	return 1;
      }

    }

    return 0;
  }

  void cudaColorSpinorArray::commsWait(int nFace, int dir, int dagger, cudaStream_t *stream_p, bool gdr) {
    int dim = dir / 2;
    if (!commDimPartitioned(dim)) return;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but not GDR is not enabled");

    if (dir%2==0) {

      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_recv_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][1][dim]);
      } else if (gdr) {
	comm_wait(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_fwd[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_send_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][0][dim]);
      } else if (gdr) {
	comm_wait(mh_send_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_send_back[bufferIndex][dim]);
      }
    } else {
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_recv_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][0][dim]);
      } else if (gdr) {
	comm_wait(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_back[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_send_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][1][dim]);
      } else if (gdr) {
	comm_wait(mh_send_rdma_fwd[bufferIndex][dim]);
      } else {
	comm_wait(mh_send_fwd[bufferIndex][dim]);
      }
    }

    return;
  }

  void cudaColorSpinorArray::scatter(int nFace, int dagger, int dim_dir, cudaStream_t* stream_p)
  {
    int dim = dim_dir/2;
    int dir = (dim_dir+1)%2; // dir = 1 - receive from forwards, dir == 0 recive from backwards
    if (!commDimPartitioned(dim)) return;

    if (comm_peer2peer_enabled(dir,dim)) return;
    unpackGhost(from_face_dim_dir_h[bufferIndex][dim][dir], nFace, dim, dir == 0 ? QUDA_BACKWARDS : QUDA_FORWARDS, dagger, stream_p);
  }

  void cudaColorSpinorArray::scatter(int nFace, int dagger, int dim_dir)
  {
    int dim = dim_dir/2;
    int dir = (dim_dir+1)%2; // dir = 1 - receive from forwards, dir == 0 receive from backwards
    if (!commDimPartitioned(dim)) return;

    if (comm_peer2peer_enabled(dir,dim)) return;
    unpackGhost(from_face_dim_dir_h[bufferIndex][dim][dir], nFace, dim, dir == 0 ? QUDA_BACKWARDS : QUDA_FORWARDS, dagger, &stream[dim_dir]);
  }

  void cudaColorSpinorArray::scatterExtended(int nFace, int parity, int dagger, int dim_dir)
  {
    bool zero_copy = false;
    int dim = dim_dir/2;
    int dir = (dim_dir+1)%2; // dir = 1 - receive from forwards, dir == 0 receive from backwards
    if (!commDimPartitioned(dim)) return;
    unpackGhostExtended(from_face_dim_dir_h[bufferIndex][dim][dir], nFace, static_cast<QudaParity>(parity), dim, dir == 0 ? QUDA_BACKWARDS : QUDA_FORWARDS, dagger, &stream[2*dim/*+0*/], zero_copy);
  }
 
  void cudaColorSpinorArray::exchangeGhost(QudaParity parity, int nFace, int dagger, const MemoryLocation *pack_destination_,
					   const MemoryLocation *halo_location_, bool gdr_send, bool gdr_recv)  const {
    if ((gdr_send || gdr_recv) && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but not GDR is not enabled");
    const_cast<cudaColorSpinorField&>(*this).createComms(nFace, false);

    // first set default values to device if needed
    MemoryLocation pack_destination[2*QUDA_MAX_DIM], halo_location[2*QUDA_MAX_DIM];
    for (int i=0; i<8; i++) {
      pack_destination[i] = pack_destination_ ? pack_destination_[i] : Device;
      halo_location[i] = halo_location_ ? halo_location_[i] : Device;
    }

    // If this is set to true, then we are assuming that the send
    // buffers are in a single contiguous memory space and we çan
    // aggregate all cudaMemcpys to reduce latency.  This only applies
    // if the memory locations are all "Device".
    bool fused_pack_memcpy = true;

    // If this is set to true, then we are assuming that the send
    // buffers are in a single contiguous memory space and we çan
    // aggregate all cudaMemcpys to reduce latency.  This only applies
    // if the memory locations are all "Device".
    bool fused_halo_memcpy = true;

    // set to true if any of the ghost packing is being done to Host memory
    bool pack_host = false;

    // set to true if the final halos will be left in Host memory
    bool halo_host = false;

    void *send[2*QUDA_MAX_DIM];
    for (int d=0; d<4; d++) {
      send[2*d+0] = pack_destination[2*d+0] == Host ? my_face_dim_dir_hd[bufferIndex][d][0] : my_face_dim_dir_d[bufferIndex][d][0];
      send[2*d+1] = pack_destination[2*d+1] == Host ? my_face_dim_dir_hd[bufferIndex][d][1] : my_face_dim_dir_d[bufferIndex][d][1];
      ghost_buf[2*d+0] = halo_location[2*d+0] == Host ? from_face_dim_dir_hd[bufferIndex][d][0] : from_face_dim_dir_d[bufferIndex][d][0];
      ghost_buf[2*d+1] = halo_location[2*d+1] == Host ? from_face_dim_dir_hd[bufferIndex][d][1] : from_face_dim_dir_d[bufferIndex][d][1];
      if (pack_destination[2*d+0] != Device || pack_destination[2*d+1] != Device) fused_pack_memcpy = false;
      if (halo_location[2*d+0] != Device || halo_location[2*d+1] != Device) fused_halo_memcpy = false;

      if (pack_destination[2*d+0] == Host || pack_destination[2*d+1] == Host) pack_host = true;
      if (halo_location[2*d+0] == Host || halo_location[2*d+1] == Host) halo_host = true;
    }

    //-C.K. CHECK: Might need to pass cudaColorSpinorField as template
    genericPackGhost<ColorSpinorField>(send, *this, parity, nFace, dagger, pack_destination); // FIXME - need support for asymmetric topologies

    size_t total_bytes = 0;
    for (int i=0; i<nDimComms; i++) if (comm_dim_partitioned(i)) total_bytes += 2*ghost_face_bytes[i]; // 2 for fwd/bwd

    if (!gdr_send)  {
      if (!fused_pack_memcpy) {
	for (int i=0; i<nDimComms; i++) {
	  if (comm_dim_partitioned(i)) {
	    if (pack_destination[2*i+0] == Device) qudaMemcpy(my_face_dim_dir_h[bufferIndex][i][0], my_face_dim_dir_d[bufferIndex][i][0],
							      ghost_face_bytes[i], cudaMemcpyDeviceToHost);
	    if (pack_destination[2*i+1] == Device) qudaMemcpy(my_face_dim_dir_h[bufferIndex][i][1], my_face_dim_dir_d[bufferIndex][i][1],
							      ghost_face_bytes[i], cudaMemcpyDeviceToHost);
	  }
	}
      } else if (total_bytes && !pack_host) {
	qudaMemcpy(my_face_h[bufferIndex], ghost_send_buffer_d[bufferIndex], total_bytes, cudaMemcpyDeviceToHost);
      }
    }

    for (int i=0; i<nDimComms; i++) { // prepost receive
      if (comm_dim_partitioned(i)) {
	comm_start(gdr_recv ? mh_recv_rdma_back[bufferIndex][i] : mh_recv_back[bufferIndex][i]);
	comm_start(gdr_recv ? mh_recv_rdma_fwd[bufferIndex][i] : mh_recv_fwd[bufferIndex][i]);
      }
    }

    if (gdr_send || pack_host) cudaDeviceSynchronize(); // need to make sure packing has finished before kicking off MPI

    for (int i=0; i<nDimComms; i++) {
      if (comm_dim_partitioned(i)) {
	comm_start(gdr_send ? mh_send_rdma_fwd[bufferIndex][i] : mh_send_fwd[bufferIndex][i]);
	comm_start(gdr_send ? mh_send_rdma_back[bufferIndex][i] : mh_send_back[bufferIndex][i]);
      }
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_wait(gdr_send ? mh_send_rdma_fwd[bufferIndex][i] : mh_send_fwd[bufferIndex][i]);
      comm_wait(gdr_send ? mh_send_rdma_back[bufferIndex][i] : mh_send_back[bufferIndex][i]);
      comm_wait(gdr_recv ? mh_recv_rdma_back[bufferIndex][i] : mh_recv_back[bufferIndex][i]);
      comm_wait(gdr_recv ? mh_recv_rdma_fwd[bufferIndex][i] : mh_recv_fwd[bufferIndex][i]);
    }

    if (!gdr_recv) {
      if (!fused_halo_memcpy) {
	for (int i=0; i<nDimComms; i++) {
	  if (!comm_dim_partitioned(i)) continue;
	  if (halo_location[2*i+0] == Device) qudaMemcpy(from_face_dim_dir_d[bufferIndex][i][0], from_face_dim_dir_h[bufferIndex][i][0],
							 ghost_face_bytes[i], cudaMemcpyHostToDevice);
	  if (halo_location[2*i+1] == Device) qudaMemcpy(from_face_dim_dir_d[bufferIndex][i][1], from_face_dim_dir_h[bufferIndex][i][1],
							 ghost_face_bytes[i], cudaMemcpyHostToDevice);
	}
      } else if (total_bytes && !halo_host) {
	qudaMemcpy(ghost_recv_buffer_d[bufferIndex], from_face_h[bufferIndex], total_bytes, cudaMemcpyHostToDevice);
      }
    }

  }

  std::ostream& operator<<(std::ostream &out, const cudaColorSpinorArray &a) {
    out << (const ColorSpinorArray&)a;
    out << "v = " << a.v << std::endl;
    out << "norm = " << a.norm << std::endl;
    out << "alloc = " << a.alloc << std::endl;
    out << "init = " << a.init << std::endl;
    return out;
  }

} // namespace quda
