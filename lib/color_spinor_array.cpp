#include <color_spinor_array.h>
#include <string.h>
#include <iostream>
#include <typeinfo>
#include <face_quda.h>

namespace quda {

  ColorSpinorArray::ColorSpinorArray(const ColorSpinorParam &param)
    : LatticeField(param), init(false), v(0), norm(0),
      ghost( ), ghostNorm( ), ghostFace( ), ghostOffset( ), ghostNormOffset( ),
      ghost_length(0), ghost_norm_length(0),
      bytes(0), norm_bytes(0), ghost_bytes(0)
  {
    create(param.nDim, param.x, param.nColor, param.nSpin,
	   param.precision, param.pad, param.siteSubset, param.siteOrder,
	   param.fieldOrder);
  }

  ColorSpinorArray::ColorSpinorArray(const ColorSpinorArray &array)
    : LatticeField(array), init(false), v(0), norm(0),
      ghost( ), ghostNorm( ), ghostFace( ), ghostOffset( ), ghostNormOffset( ),
      ghost_length(0), ghost_norm_length(0),
      bytes(0), norm_bytes(0), ghost_bytes(0)
  {
    create(array.nDim, array.x, array.nColor, array.nSpin,
	   array.precision, array.pad, array.siteSubset, array.siteOrder,
	   array.fieldOrder);
  }

  ColorSpinorArray::~ColorSpinorArray() {
    destroy();
  }

  void ColorSpinorArray::createGhostZone(int nFace, bool spin_project) const {

    if (typeid(*this) == typeid(cpuColorSpinorArray)) {
      ghost_length = 0;
      ghost_norm_length = 0;
      return;
    }

    // For Wilson we half the number of effective faces if the fields are spin projected.
    int num_faces = ((nSpin == 4 && spin_project) ? 1 : 2) * nFace;
    int num_norm_faces = 2*nFace;

    // calculate size of ghost zone required
    int ghostVolume = 0;
    int dims = nDim == 5 ? (nDim - 1) : nDim;
    int x5   = nDim == 5 ? x[4] : 1; ///includes DW  and non-degenerate TM ghosts
    for (int i=0; i<dims; i++) {
      ghostFace[i] = 0;
      if (commDimPartitioned(i)) {
	ghostFace[i] = 1;
	for (int j=0; j<dims; j++) {
	  if (i==j) continue;
	  ghostFace[i] *= x[j];
	}
	ghostFace[i] *= x5; ///temporal hack : extra dimension for DW ghosts
	if (i==0 && siteSubset != QUDA_FULL_SITE_SUBSET) ghostFace[i] /= 2;
	ghostVolume += ghostFace[i];
      }
      if (i==0) {
	ghostOffset[i][0] = 0;
      } else {
        if (precision == QUDA_HALF_PRECISION) {
          ghostOffset[i][0] = (ghostNormOffset[i-1][1] + num_norm_faces*ghostFace[i-1]/2)*sizeof(float)/sizeof(short);
          // Adjust so that the offsets are multiples of 4 shorts
          // This ensures that the dslash kernel can read the ghost field data as an array of short4's
          ghostOffset[i][0] = 4*((ghostOffset[i][0] + 3)/4);
        } else {
	  ghostOffset[i][0] = ghostOffset[i-1][0] + num_faces*ghostFace[i-1]*nSpin*nColor*2;
        }
      }

      if (precision == QUDA_HALF_PRECISION) {
        ghostNormOffset[i][0] = (ghostOffset[i][0] + (num_faces*ghostFace[i]*nSpin*nColor*2/2))*sizeof(short)/sizeof(float);
        ghostOffset[i][1] = (ghostNormOffset[i][0] + num_norm_faces*ghostFace[i]/2)*sizeof(float)/sizeof(short);
        // Adjust so that the offsets are multiples of 4 shorts
        // This ensures that the dslash kernel can read the ghost field data as an array of short4's
        ghostOffset[i][1] = 4*((ghostOffset[i][1] + 3)/4);
        ghostNormOffset[i][1] = (ghostOffset[i][1] + (num_faces*ghostFace[i]*nSpin*nColor*2/2))*sizeof(short)/sizeof(float);
      } else {
        ghostOffset[i][1] = ghostOffset[i][0] + num_faces*ghostFace[i]*nSpin*nColor*2/2;
      }

      if (getVerbosity() == QUDA_DEBUG_VERBOSE)
	printfQuda("face %d = %6d commDimPartitioned = %6d ghostOffset = %6d %6d ghostNormOffset = %6d, %6d\n",
		   i, ghostFace[i], commDimPartitioned(i), ghostOffset[i][0], ghostOffset[i][1], ghostNormOffset[i][0], ghostNormOffset[i][1]);
    } // dim

    int ghostNormVolume = num_norm_faces * ghostVolume;
    ghostVolume *= num_faces;

    ghost_length = ghostVolume*nColor*nSpin*2;
    ghost_norm_length = (precision == QUDA_HALF_PRECISION) ? ghostNormVolume : 0;

    if (getVerbosity() == QUDA_DEBUG_VERBOSE) {
      printfQuda("Allocated ghost volume = %d, ghost norm volume %d\n", ghostVolume, ghostNormVolume);
      printfQuda("ghost length = %lu, ghost norm length = %lu\n", ghost_length, ghost_norm_length);
    }

    ghost_bytes = (size_t)ghost_length*precision;
    if (precision == QUDA_HALF_PRECISION) ghost_bytes += ghost_norm_length*sizeof(float);
    if (isNative()) ghost_bytes = ALIGNMENT_ADJUST(ghost_bytes);

  } // createGhostZone

  void ColorSpinorArray::create(int Ndim, const int *X, int Nc, int Ns,
				QudaPrecision Prec, int Pad, QudaSiteSubset siteSubset,
				QudaSiteOrder siteOrder, QudaFieldOrder fieldOrder) {
    this->siteSubset = siteSubset;
    this->siteOrder = siteOrder;
    this->fieldOrder = fieldOrder;

    if (Ndim > QUDA_MAX_DIM){
      errorQuda("Number of dimensions nDim = %d too great", Ndim);
    }
    nDim = Ndim;
    nColor = Nc;
    nSpin = Ns;

    precision = Prec;
    volume = 1;
    for (int d=0; d<nDim; d++) {
      x[d] = X[d];
      volume *= x[d];
    }
    volumeCB = siteSubset == QUDA_PARITY_SITE_SUBSET ? volume : volume/2;

    pad = Pad;
    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      stride = volume/2 + pad; // padding is based on half volume
      length = 2*stride*nColor*nSpin*2;
    } else {
      stride = volume + pad;
      length = stride*nColor*nSpin*2;
    }

    real_length = volume*nColor*nSpin*2; // physical length

    bytes = (size_t)length * precision; // includes pads and ghost zones
    if (isNative()) bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(bytes/2) : ALIGNMENT_ADJUST(bytes);

    if (precision == QUDA_HALF_PRECISION) {
      norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET ? 2*stride : stride) * sizeof(float);
      if (isNative()) norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(norm_bytes/2) : ALIGNMENT_ADJUST(norm_bytes);
    } else {
      norm_bytes = 0;
    }

    init = true;

    setTuningString();
  }

  void ColorSpinorArray::setTuningString() {
    char vol_tmp[TuneKey::volume_n];
    int check;
    check = snprintf(vol_string, TuneKey::volume_n, "%d", x[0]);
    if (check < 0 || check >= TuneKey::volume_n) errorQuda("Error writing volume string");
    for (int d=1; d<nDim; d++) {
      strcpy(vol_tmp, vol_string);
      check = snprintf(vol_string, TuneKey::volume_n, "%sx%d", vol_tmp, x[d]);
      if (check < 0 || check >= TuneKey::volume_n) errorQuda("Error writing volume string");
    }

    int aux_string_n = TuneKey::aux_n / 2;
    char aux_tmp[aux_string_n];
    check = snprintf(aux_string, aux_string_n, "vol=%d,stride=%d,precision=%d,Ns=%d,Nc=%d",
		     volume, stride, precision, nSpin, nColor);
    if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");
  }

  void ColorSpinorArray::destroy() {
    init = false;
  }

  ColorSpinorArray& ColorSpinorArray::operator=(const ColorSpinorArray &src) {

    if (&src != this) {
      create(src.nDim, src.x, src.nColor, src.nSpin,
	     src.precision, src.pad, src.siteSubset,
	     src.siteOrder, src.fieldOrder);
    }

    return *this;
  }

  void ColorSpinorArray::exchange(void **ghost, void **sendbuf, int nFace) const {

    // FIXME: use LatticeField MsgHandles
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];
    size_t bytes[4];

    const int Ninternal = 2*nColor*nSpin;
    size_t total_bytes = 0;
    for (int i=0; i<nDimComms; i++) {
      bytes[i] = siteSubset*nFace*surfaceCB[i]*Ninternal*precision;
      if (comm_dim_partitioned(i)) total_bytes += 2*bytes[i]; // 2 for fwd/bwd
    }

    void *total_send = nullptr;
    void *total_recv = nullptr;
    void *send_fwd[4];
    void *send_back[4];
    void *recv_fwd[4];
    void *recv_back[4];

    // leave this option in there just in case
    bool no_comms_fill = false;

    // If this is set to false, then we are assuming that the send and
    // ghost buffers are in a single contiguous memory space.  Setting
    // to false means we aggregate all cudaMemcpys which reduces
    // latency.
    bool fine_grained_memcpy = false;

    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      for (int i=0; i<nDimComms; i++) {
	if (comm_dim_partitioned(i)) {
	  send_back[i] = sendbuf[2*i + 0];
	  send_fwd[i]  = sendbuf[2*i + 1];
	  recv_fwd[i]  =   ghost[2*i + 1];
	  recv_back[i] =   ghost[2*i + 0];
	} else if (no_comms_fill) {
	  memcpy(ghost[2*i+1], sendbuf[2*i+0], bytes[i]);
	  memcpy(ghost[2*i+0], sendbuf[2*i+1], bytes[i]);
	}
      }
    } else { // FIXME add GPU_COMMS support
      if (total_bytes) {
	total_send = pool_pinned_malloc(total_bytes);
	total_recv = pool_pinned_malloc(total_bytes);
      }
      size_t offset = 0;
      for (int i=0; i<nDimComms; i++) {
	if (comm_dim_partitioned(i)) {
	  send_back[i] = static_cast<char*>(total_send) + offset;
	  recv_back[i] = static_cast<char*>(total_recv) + offset;
	  offset += bytes[i];
	  send_fwd[i] = static_cast<char*>(total_send) + offset;
	  recv_fwd[i] = static_cast<char*>(total_recv) + offset;
	  offset += bytes[i];
	  if (fine_grained_memcpy) {
	    qudaMemcpy(send_back[i], sendbuf[2*i + 0], bytes[i], cudaMemcpyDeviceToHost);
	    qudaMemcpy(send_fwd[i],  sendbuf[2*i + 1], bytes[i], cudaMemcpyDeviceToHost);
	  }
	} else if (no_comms_fill) {
	  qudaMemcpy(ghost[2*i+1], sendbuf[2*i+0], bytes[i], cudaMemcpyDeviceToDevice);
	  qudaMemcpy(ghost[2*i+0], sendbuf[2*i+1], bytes[i], cudaMemcpyDeviceToDevice);
	}
      }
      if (!fine_grained_memcpy && total_bytes) {
	// find first non-zero pointer
	void *send_ptr = nullptr;
	for (int i=0; i<nDimComms; i++) {
	  if (comm_dim_partitioned(i)) {
	    send_ptr = sendbuf[2*i];
	    break;
	  }
	}
	qudaMemcpy(total_send, send_ptr, total_bytes, cudaMemcpyDeviceToHost);
      }
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      mh_send_fwd[i] = comm_declare_send_relative(send_fwd[i], i, +1, bytes[i]);
      mh_send_back[i] = comm_declare_send_relative(send_back[i], i, -1, bytes[i]);
      mh_from_fwd[i] = comm_declare_receive_relative(recv_fwd[i], i, +1, bytes[i]);
      mh_from_back[i] = comm_declare_receive_relative(recv_back[i], i, -1, bytes[i]);
    }

    for (int i=0; i<nDimComms; i++) {
      if (comm_dim_partitioned(i)) {
	comm_start(mh_from_back[i]);
	comm_start(mh_from_fwd[i]);
	comm_start(mh_send_fwd[i]);
	comm_start(mh_send_back[i]);
      }
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_wait(mh_send_fwd[i]);
      comm_wait(mh_send_back[i]);
      comm_wait(mh_from_back[i]);
      comm_wait(mh_from_fwd[i]);
    }

    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      for (int i=0; i<nDimComms; i++) {
	if (!comm_dim_partitioned(i)) continue;
	if (fine_grained_memcpy) {
	  qudaMemcpy(ghost[2*i+0], recv_back[i], bytes[i], cudaMemcpyHostToDevice);
	  qudaMemcpy(ghost[2*i+1], recv_fwd[i], bytes[i], cudaMemcpyHostToDevice);
	}
      }

      if (!fine_grained_memcpy && total_bytes) {
	// find first non-zero pointer
	void *ghost_ptr = nullptr;
	for (int i=0; i<nDimComms; i++) {
	  if (comm_dim_partitioned(i)) {
	    ghost_ptr = ghost[2*i];
	    break;
	  }
	}
	qudaMemcpy(ghost_ptr, total_recv, total_bytes, cudaMemcpyHostToDevice);
      }

      if (total_bytes) {
	pool_pinned_free(total_send);
	pool_pinned_free(total_recv);
      }
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_free(mh_send_fwd[i]);
      comm_free(mh_send_back[i]);
      comm_free(mh_from_back[i]);
      comm_free(mh_from_fwd[i]);
    }
  }

  bool ColorSpinorArray::isNative() const {
    if (precision == QUDA_DOUBLE_PRECISION) {
      if (fieldOrder  == QUDA_FLOAT2_FIELD_ORDER) return true;
    } else if (precision == QUDA_SINGLE_PRECISION ||
	       precision == QUDA_HALF_PRECISION) {
      if (nSpin == 4) {
	if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) return true;
      } else if (nSpin == 2) {
	if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) return true;
      } else if (nSpin == 1) {
	if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) return true;
      }
    }
    return false;
  }

  void* ColorSpinorArray::Ghost(const int i) {
    if(siteSubset != QUDA_PARITY_SITE_SUBSET) errorQuda("Site Subset %d is not supported",siteSubset);
    return ghost[i];
  }

  const void* ColorSpinorArray::Ghost(const int i) const {
    if(siteSubset != QUDA_PARITY_SITE_SUBSET) errorQuda("Site Subset %d is not supported",siteSubset);
    return ghost[i];
  }


  void* ColorSpinorArray::GhostNorm(const int i){
    if(siteSubset != QUDA_PARITY_SITE_SUBSET) errorQuda("Site Subset %d is not supported",siteSubset);
    return ghostNorm[i];
  }

  const void* ColorSpinorArray::GhostNorm(const int i) const{
    if(siteSubset != QUDA_PARITY_SITE_SUBSET) errorQuda("Site Subset %d is not supported",siteSubset);
    return ghostNorm[i];
  }

  void* const* ColorSpinorArray::Ghost() const {
    return ghost_buf;
  }


  ColorSpinorArray* ColorSpinorArray::Create(const ColorSpinorParam &param) {

    ColorSpinorArray *array = NULL;
    if (param.location == QUDA_CPU_FIELD_LOCATION) {
      array = new cpuColorSpinorArray(param);
    } else if (param.location== QUDA_CUDA_FIELD_LOCATION) {
      array = new cudaColorSpinorArray(param);
    } else {
      errorQuda("Invalid field location %d", param.location);
    }

    return array;
  }

  ColorSpinorArray* ColorSpinorArray::Create(const ColorSpinorArray &src, const ColorSpinorParam &param) {

    ColorSpinorArray *array = NULL;
    if (param.location == QUDA_CPU_FIELD_LOCATION) {
      array = new cpuColorSpinorArray(src, param);
    } else if (param.location== QUDA_CUDA_FIELD_LOCATION) {
      array = new cudaColorSpinorArray(src, param);
    } else {
      errorQuda("Invalid field location %d", param.location);
    }

    return array;
  }

  std::ostream& operator<<(std::ostream &out, const ColorSpinorArray &a) {
    out << "typedid = " << typeid(a).name() << std::endl;
    out << "nColor = " << a.nColor << std::endl;
    out << "nSpin = " << a.nSpin << std::endl;
    out << "nDim = " << a.nDim << std::endl;
    for (int d=0; d<a.nDim; d++) out << "x[" << d << "] = " << a.x[d] << std::endl;
    out << "volume = " << a.volume << std::endl;
    out << "precision = " << a.precision << std::endl;
    out << "pad = " << a.pad << std::endl;
    out << "stride = " << a.stride << std::endl;
    out << "real_length = " << a.real_length << std::endl;
    out << "length = " << a.length << std::endl;
    out << "ghost_length = " << a.ghost_length << std::endl;
    out << "ghost_norm_length = " << a.ghost_norm_length << std::endl;
    out << "bytes = " << a.bytes << std::endl;
    out << "norm_bytes = " << a.norm_bytes << std::endl;
    out << "siteSubset = " << a.siteSubset << std::endl;
    out << "siteOrder = " << a.siteOrder << std::endl;
    out << "fieldOrder = " << a.fieldOrder << std::endl;

    return out;
  }

} // namespace quda
