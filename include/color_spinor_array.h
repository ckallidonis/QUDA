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
