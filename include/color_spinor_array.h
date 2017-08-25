//-C.K. A "striped-down" color-spinor class for doing the momentum smearing

#ifndef _COLOR_SPINOR_ARRAY_H
#define _COLOR_SPINOR_ARRAY_H

#include <quda_internal.h>
#include <quda.h>

#include <iostream>

#include <lattice_field.h>
#include <random_quda.h>

namespace quda {

  class cpuColorSpinorArray;
  class cudaColorSpinorArray;

  class ColorSpinorArray : public LatticeField {

  private:
    void create(int nDim, const int *x, int Nc, int Ns,
		QudaPrecision precision, int pad, QudaSiteSubset subset,
		QudaSiteOrder siteOrder, QudaFieldOrder fieldOrder);
    void destroy();

  protected:
    bool init;

    int nColor;
    int nSpin;

    int nDim;
    int x[QUDA_MAX_DIM];

    int volume;
    int volumeCB;
    int pad;
    int stride;

    const QudaTwistFlavorType twistFlavor = QUDA_TWIST_INVALID;

    size_t real_length; // physical length only
    size_t length; // length including pads, but not ghost zone - used for BLAS

    void *v; // the field elements
    void *norm; // the normalization field

    void *v_h; // the field elements
    void *norm_h; // the normalization field

    // multi-GPU parameters

    void* ghost[2][QUDA_MAX_DIM]; // pointers to the ghost regions - NULL by default
    void* ghostNorm[2][QUDA_MAX_DIM]; // pointers to ghost norms - NULL by default

    mutable int ghostFace[QUDA_MAX_DIM];// the size of each face
    mutable int ghostOffset[QUDA_MAX_DIM][2]; // offsets to each ghost zone
    mutable int ghostNormOffset[QUDA_MAX_DIM][2]; // offsets to each ghost zone for norm field

    mutable size_t ghost_length; // length of ghost zone
    mutable size_t ghost_norm_length; // length of ghost zone for norm

    mutable void *ghost_buf[2*QUDA_MAX_DIM]; // wrapper that points to current ghost zone

    size_t bytes; // size in bytes of spinor field
    size_t norm_bytes; // size in bytes of norm field
    mutable size_t ghost_bytes; // size in bytes of the ghost field
    mutable size_t ghost_face_bytes[QUDA_MAX_DIM];

    QudaSiteSubset siteSubset;
    QudaSiteOrder siteOrder;
    QudaFieldOrder fieldOrder;

    void createGhostZone(int nFace, bool spin_project=true) const;

    char aux_string[TuneKey::aux_n]; // used as a label in the autotuner
    void setTuningString(); // set the vol_string and aux_string for use in tuning

  public:
    ColorSpinorArray(const ColorSpinorParam &);

    virtual ~ColorSpinorArray();

    virtual ColorSpinorArray& operator=(const ColorSpinorArray &);

    int Ncolor() const { return nColor; }
    int Nspin() const { return nSpin; }
    int Ndim() const { return nDim; }
    const int* X() const { return x; }
    int X(int d) const { return x[d]; }
    size_t RealLength() const { return real_length; }
    size_t Length() const { return length; }
    int Stride() const { return stride; }
    int Volume() const { return volume; }
    int VolumeCB() const { return siteSubset == QUDA_PARITY_SITE_SUBSET ? volume : volume / 2; }
    int Pad() const { return pad; }
    size_t Bytes() const { return bytes; }
    size_t NormBytes() const { return norm_bytes; }
    size_t GhostBytes() const { return ghost_bytes; }
    size_t GhostNormBytes() const { return ghost_bytes; }
    void PrintDims() const { printfQuda("dimensions=%d %d %d %d\n", x[0], x[1], x[2], x[3]); }

    inline const char *AuxString() const { return aux_string; }

    void* V() {return v;}
    const void* V() const {return v;}
    void* Norm(){return norm;}
    const void* Norm() const {return norm;}
    virtual const void* Ghost2() const { return nullptr; }

    /**
       Do the exchange between neighbouring nodes of the data in
       sendbuf storing the result in recvbuf.  The arrays are ordered
       (2*dim + dir).
       @param recvbuf Packed buffer where we store the result
       @param sendbuf Packed buffer from which we're sending
       @param nFace Number of layers we are exchanging
     */
    void exchange(void **ghost, void **sendbuf, int nFace=1) const;

    /**
       This is a unified ghost exchange function for doing a complete
       halo exchange regardless of the type of field.  All dimensions
       are exchanged and no spin projection is done in the case of
       Wilson fermions.
       @param[in] parity Field parity
       @param[in] nFace Depth of halo exchange
       @param[in] dagger Is this for a dagger operator (only relevant for spin projected Wilson)
       @param[in] pack_destination Destination of the packing buffer
       @param[in] halo_location Destination of the halo reading buffer
       @param[in] gdr_send Are we using GDR for sending
       @param[in] gdr_recv Are we using GDR for receiving
     */
    virtual void exchangeGhost(QudaParity parity, int nFace, int dagger, const MemoryLocation *pack_destination=nullptr,
			       const MemoryLocation *halo_location=nullptr, bool gdr_send=false, bool gdr_recv=false) const = 0;

    /**
      This function returns true if the field is stored in an internal
      field order, given the precision and the length of the spin
      dimension.
      */
    bool isNative() const;

    QudaSiteSubset SiteSubset() const { return siteSubset; }
    QudaSiteOrder SiteOrder() const { return siteOrder; }
    QudaFieldOrder FieldOrder() const { return fieldOrder; }

    size_t GhostLength() const { return ghost_length; }
    const int *GhostFace() const { return ghostFace; }
    int GhostOffset(const int i) const { return ghostOffset[i][0]; }
    int GhostOffset(const int i, const int j) const { return ghostOffset[i][j]; }
    int GhostNormOffset(const int i ) const { return ghostNormOffset[i][0]; }
    int GhostNormOffset(const int i, const int j) const { return ghostNormOffset[i][j]; }

    void* Ghost(const int i);
    const void* Ghost(const int i) const;
    void* GhostNorm(const int i);
    const void* GhostNorm(const int i) const;

    /**
       Return array of pointers to the ghost zones (ordering dim*2+dir)
     */
    void* const* Ghost() const;

    virtual void Source(const QudaSourceType sourceType, const int st=0, const int s=0, const int c=0) = 0;

    static ColorSpinorArray* Create(const ColorSpinorParam &param);
    static ColorSpinorArray* Create(const ColorSpinorArray &src, const ColorSpinorParam &param);

    friend std::ostream& operator<<(std::ostream &out, const ColorSpinorArray &);
    friend class ColorSpinorParam;
  };

  //--------------------------------------------------------------------------------------------------------
  //--------------------------------------------------------------------------------------------------------

  // CUDA implementation
  class cudaColorSpinorArray : public ColorSpinorArray {

    friend class cpuColorSpinorArray;

  private:
    bool alloc; // whether we allocated memory
    bool init;

    bool reference; // whether the field is a reference or not

    static size_t ghostFaceBytes;
    static bool initGhostFaceBuffer;

    mutable void *ghost_field_tex[4]; // instance pointer to GPU halo buffer (used to check if static allocation has changed)

    void create(const QudaFieldCreate);
    void destroy();

    /** Keep track of which pinned-memory buffer we used for creating message handlers */
    size_t bufferMessageHandler;

  public:
    cudaColorSpinorArray(const ColorSpinorParam&);
    virtual ~cudaColorSpinorArray();

    ColorSpinorArray& operator=(const ColorSpinorArray &);
    cudaColorSpinorArray& operator=(const cpuColorSpinorArray&);

    void switchBufferPinned();

    /**
       @brief Create the communication handlers and buffers
       @param[in] nFace Depth of each halo
       @param[in] spin_project Whether the halos are spin projected (Wilson-type fermions only)
    */
    void createComms(int nFace, bool spin_project=true);

    /**
       @brief Destroy the communication handlers and buffers
    */
    void destroyComms();

    /**
       @brief Allocate the ghost buffers
       @param[in] nFace Depth of each halo
       @param[in] spin_project Whether the halos are spin projected (Wilson-type fermions only)
    */
    void allocateGhostBuffer(int nFace, bool spin_project=true) const;

    /**
       @brief Free statically allocated ghost buffers
    */
    static void freeGhostBuffer(void);

    /**
       @brief Packs the cudaColorSpinorField's ghost zone
       @param[in] nFace How many faces to pack (depth)
       @param[in] parity Parity of the field
       @param[in] dim Labels space-time dimensions
       @param[in] dir Pack data to send in forward of backward directions, or both
       @param[in] dagger Whether the operator is the Hermitian conjugate or not
       @param[in] stream Which stream to use for the kernel
       @param[out] buffer Optional parameter where the ghost should be
       stored (default is to use cudaColorSpinorField::ghostFaceBuffer)
       @param[in] location Are we packing directly into local device memory, zero-copy memory or remote memory
       @param[in] a Twisted mass parameter (default=0)
       @param[in] b Twisted mass parameter (default=0)
    */
    void packGhost(const int nFace, const QudaParity parity, const int dim, const QudaDirection dir, const int dagger,
                   cudaStream_t* stream, MemoryLocation location[2*QUDA_MAX_DIM], double a=0, double b=0);


    void packGhostExtended(const int nFace, const int R[], const QudaParity parity, const int dim, const QudaDirection dir,
                           const int dagger,cudaStream_t* stream, bool zero_copy=false);


    void packGhost(FullClover &clov, FullClover &clovInv, const int nFace, const QudaParity parity, const int dim,
                   const QudaDirection dir, const int dagger, cudaStream_t* stream, void *buffer=0, double a=0);

    /**
      Initiate the gpu to cpu send of the ghost zone (halo)
      @param ghost_spinor Where to send the ghost zone
      @param nFace Number of face to send
      @param dim The lattice dimension we are sending
      @param dir The direction (QUDA_BACKWARDS or QUDA_FORWARDS)
      @param dagger Whether the operator is daggerer or not
      @param stream The array of streams to use
    */
    void sendGhost(void *ghost_spinor, const int nFace, const int dim, const QudaDirection dir,
		   const int dagger, cudaStream_t *stream);

    /**
      Initiate the cpu to gpu send of the ghost zone (halo)
      @param ghost_spinor Source of the ghost zone
      @param nFace Number of face to send
      @param dim The lattice dimension we are sending
      @param dir The direction (QUDA_BACKWARDS or QUDA_FORWARDS)
      @param dagger Whether the operator is daggerer or not
      @param stream The array of streams to use
    */
    void unpackGhost(const void* ghost_spinor, const int nFace, const int dim,
		     const QudaDirection dir, const int dagger, cudaStream_t* stream);

    /**
      Initiate the cpu to gpu copy of the extended border region
      @param ghost_spinor Source of the ghost zone
      @param parity Parity of the field
      @param nFace Number of face to send
      @param dim The lattice dimension we are sending
      @param dir The direction (QUDA_BACKWARDS or QUDA_FORWARDS)
      @param dagger Whether the operator is daggered or not
      @param stream The array of streams to use
      @param zero_copy Whether we are unpacking from zero_copy memory
    */
    void unpackGhostExtended(const void* ghost_spinor, const int nFace, const QudaParity parity,
                             const int dim, const QudaDirection dir, const int dagger, cudaStream_t* stream, bool zero_copy);


    void streamInit(cudaStream_t *stream_p);

    void pack(int nFace, int parity, int dagger, int stream_idx,
              MemoryLocation location[], double a=0, double b=0);

    void packExtended(const int nFace, const int R[], const int parity, const int dagger,
		      const int dim,  cudaStream_t *stream_p, const bool zeroCopyPack=false);

    void gather(int nFace, int dagger, int dir, cudaStream_t *stream_p=NULL);

    void recvStart(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL, bool gdr=false);
    void sendStart(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL, bool gdr=false);
    void commsStart(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL, bool gdr=false);
    int commsQuery(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL, bool gdr=false);
    void commsWait(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL, bool gdr=false);

    void scatter(int nFace, int dagger, int dir, cudaStream_t *stream_p);
    void scatter(int nFace, int dagger, int dir);

    void scatterExtended(int nFace, int parity, int dagger, int dir);

    const void* Ghost2() const { return ghost_field_tex[bufferIndex]; }

    /**
       @brief This is a unified ghost exchange function for doing a complete
       halo exchange regardless of the type of field.  All dimensions
       are exchanged and no spin projection is done in the case of
       Wilson fermions.
       @param[in] parity Field parity
       @param[in] nFace Depth of halo exchange
       @param[in] dagger Is this for a dagger operator (only relevant for spin projected Wilson)
       @param[in] pack_destination Destination of the packing buffer
       @param[in] halo_location Destination of the halo reading buffer
       @param[in] gdr_send Are we using GDR for sending
       @param[in] gdr_recv Are we using GDR for receiving
    */
    void exchangeGhost(QudaParity parity, int nFace, int dagger, const MemoryLocation *pack_destination=nullptr,
                       const MemoryLocation *halo_location=nullptr, bool gdr_send=false, bool gdr_recv=false) const;

    void zero();

    friend std::ostream& operator<<(std::ostream &out, const cudaColorSpinorField &);
  };

  //--------------------------------------------------------------------------------------------------------
  //--------------------------------------------------------------------------------------------------------

  // CPU implementation
  class cpuColorSpinorArray : public ColorSpinorArray {

    friend class cudaColorSpinorArray;

  public:
    static void* fwdGhostFaceBuffer[QUDA_MAX_DIM]; //cpu memory
    static void* backGhostFaceBuffer[QUDA_MAX_DIM]; //cpu memory
    static void* fwdGhostFaceSendBuffer[QUDA_MAX_DIM]; //cpu memory
    static void* backGhostFaceSendBuffer[QUDA_MAX_DIM]; //cpu memory
    static int initGhostFaceBuffer;
    static size_t ghostFaceBytes[QUDA_MAX_DIM];

  private:
    bool init;
    bool reference; // whether the field is a reference or not

    void create(const QudaFieldCreate);
    void destroy();

  public:
    cpuColorSpinorArray(const cpuColorSpinorArray&);
    cpuColorSpinorArray(const ColorSpinorArray&);
    cpuColorSpinorArray(const ColorSpinorArray&, const ColorSpinorParam&);
    cpuColorSpinorArray(const ColorSpinorParam&);
    virtual ~cpuColorSpinorArray();

    ColorSpinorArray& operator=(const ColorSpinorArray &);
    cpuColorSpinorArray& operator=(const cpuColorSpinorArray&);
    cpuColorSpinorArray& operator=(const cudaColorSpinorArray&);

    /**
       @brief Allocate the ghost buffers
       @param[in] nFace Depth of each halo
    */
    void allocateGhostBuffer(int nFace) const;
    static void freeGhostBuffer(void);

    void packGhost(void **ghost, const QudaParity parity, const int nFace, const int dagger) const;
    void unpackGhost(void* ghost_spinor, const int dim,
                     const QudaDirection dir, const int dagger);

    void zero();

    /**
       @brieff This is a unified ghost exchange function for doing a complete
       halo exchange regardless of the type of field.  All dimensions
       are exchanged and no spin projection is done in the case of
       Wilson fermions.
       @param[in] parity Field parity
       @param[in] nFace Depth of halo exchange
       @param[in] dagger Is this for a dagger operator (only relevant for spin projected Wilson)
       @param[in] pack_destination Destination of the packing buffer
       @param[in] halo_location Destination of the halo reading buffer
       @param[in] gdr_send Dummy for CPU
       @param[in] gdr_recv Dummy for GPU
    */
    void exchangeGhost(QudaParity parity, int nFace, int dagger, const MemoryLocation *pack_destination=nullptr,
                       const MemoryLocation *halo_location=nullptr, bool gdr_send=false, bool gdr_recv=false) const;
  };

  //--------------------------------------------------------------------------------------------------------
  //--------------------------------------------------------------------------------------------------------

  /**
     @brief Generic ghost packing routine

     @param[out] ghost Array of packed ghosts with array ordering [2*dim+dir]
     @param[in] a Input field that is being packed
     @param[in] parity Which parity are we packing
     @param[in] dagger Is for a dagger operator (presently ignored)
     @param[in[ location Array specifiying the memory location of each resulting ghost [2*dim+dir]
  */

  //-C.K. Overloaded: Added a template to accommodate both ColorSpinorField and ColorSpinorArray when calling the function
  template <typename TColSpin>
  void genericPackGhost(void **ghost, const TColSpin &a, QudaParity parity,
                        int nFace, int dagger, MemoryLocation *destination=nullptr);


} // namespace quda

#endif // _COLOR_SPINOR_ARRAY_H
