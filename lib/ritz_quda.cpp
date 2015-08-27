#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <lanczos_quda.h>
#include <ritz_quda.h>
#include <face_quda.h>
#include <iostream>
#include <complex>

namespace quda {

  void RitzMat::operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in) const
  {
    const double alpha = pow(cheby_param[0], 2);
    const double beta  = pow(cheby_param[1]+fabs(shift), 2);

    const double c1 = 2.0*(alpha+beta)/(alpha-beta); 
    const double c0 = 2.0/(alpha+beta); 

    bool reset1 = newTmp( &tmp1, in);
    bool reset2 = newTmp( &tmp2, in);

    *(tmp2) = in;
    dirac_mat( *(tmp1), in);

    axpbyCuda(-0.5*c1, const_cast<cudaColorSpinorField&>(in), 0.5*c0*c1, *(tmp1));
    for(int i=2; i < N_Poly+1; ++i)
    {
      dirac_mat(out,*(tmp1));
      axpbyCuda(-c1,*(tmp1),c0*c1,out);
      axpyCuda(-1.0,*(tmp2),out);
      //printfQuda("ritzMat: Ritz mat loop %d\n",i);

      if(i != N_Poly)
      {
        // tmp2 = tmp
        // tmp = out
        cudaColorSpinorField *swap_Tmp = tmp2;
        tmp2 = tmp1;
        tmp1 = swap_Tmp;
        *(tmp1) = out;
      }
    }
    deleteTmp(&(tmp1), reset1);
    deleteTmp(&(tmp2), reset2);

  }

  void RitzMat::operator()(cudaColorSpinorField &out, const cudaColorSpinorField &in, double amin, double amax) const
  {



   double delta,theta;
   double sigma,sigma1,sigma_old;
   double d1,d2,d3;


   double a = amin;
   double b = amax;

   delta = (b-a)/2.0;
   theta = (b+a)/2.0;

   sigma1 = -delta/theta;

   copyCuda(out,in);
   if(N_Poly == 0){
      return;
   }

   //T_1(Q) = [2/(b-a)*Q - (b+a)/(b-a)]*sigma_1 
   d1 =  sigma1/delta;
   d2 =  1.0;

   dirac_mat( out, in);
   //   axCuda(1./(2.*shift),out);
   axpbyCuda(d2, const_cast<cudaColorSpinorField&>(in), d1, out);

   if(N_Poly==1){
     return;
   }

   cudaColorSpinorField *tm1 = new cudaColorSpinorField(in);
   cudaColorSpinorField *tm2 = new cudaColorSpinorField(in);
   //degree >=2
   //==========

   //T_0 = S
   //T_1 = R

   copyCuda(*tm1,in);
   copyCuda(*tm2,out);

   sigma_old = sigma1;

   for(int i=2; i <= N_Poly; i++)
   {
      sigma = 1.0/(2.0/sigma1-sigma_old);
      
      d1 = 2.0*sigma/delta;
      d2 = -d1*theta;
      d3 = -sigma*sigma_old;

      dirac_mat( out, *tm2);
      // axCuda(1./(2.*shift),out);
      axCuda(d3,*tm1);
      std::complex<double> d1c(d1,0);
      std::complex<double> d2c(d2,0);
      cxpaypbzCuda(*tm1,d2c,*tm2,d1c,out); // out=tmp1 + d2*tmp2 + d1*out

      copyCuda(*tm1,*tm2);
      copyCuda(*tm2,out);

      sigma_old  = sigma;
   }

   delete tm1;
   delete tm2;


   return;
  }


  RitzMat::~RitzMat() {;}
  bool RitzMat::newTmp(cudaColorSpinorField **tmp, const cudaColorSpinorField &a) const{
    if (*tmp) return false;
    ColorSpinorParam param(a);
    param.create = QUDA_ZERO_FIELD_CREATE;
    *tmp = new cudaColorSpinorField(a, param);
    return true;
  }

  void RitzMat::deleteTmp(cudaColorSpinorField **a, const bool &reset) const{
    if (reset) {
      delete *a;
      *a = NULL;
    }
  }
}
