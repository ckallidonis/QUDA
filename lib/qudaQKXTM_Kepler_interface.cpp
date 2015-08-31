#include <qudaQKXTM_Kepler.cpp>
#include <sys/stat.h>
#define TIMING_REPORT
template  class QKXTM_Field_Kepler<double>;
template  class QKXTM_Gauge_Kepler<double>;
template  class QKXTM_Vector_Kepler<double>;
template  class QKXTM_Propagator_Kepler<double>;
template  class QKXTM_Propagator3D_Kepler<double>;
template  class QKXTM_Vector3D_Kepler<double>;

template  class QKXTM_Field_Kepler<float>;
template  class QKXTM_Gauge_Kepler<float>;
template  class QKXTM_Vector_Kepler<float>;
template  class QKXTM_Propagator_Kepler<float>;
template  class QKXTM_Propagator3D_Kepler<float>;
template  class QKXTM_Vector3D_Kepler<float>;

static bool exists_file (const char* name) {
  return ( access( name, F_OK ) != -1 );
}


void testPlaquette(void **gauge){
  QKXTM_Gauge_Kepler<float> *gauge_object = new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);
  gauge_object->printInfo();
  gauge_object->packGauge(gauge);
  gauge_object->loadGauge();
  gauge_object->calculatePlaq();
  delete gauge_object;

  QKXTM_Gauge_Kepler<double> *gauge_object_2 = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  gauge_object_2->printInfo();
  gauge_object_2->packGauge(gauge);
  gauge_object_2->loadGauge();
  gauge_object_2->calculatePlaq();
  delete gauge_object_2;
}

void testGaussSmearing(void **gauge){

  QKXTM_Gauge_Kepler<double> *gauge_object = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  gauge_object->printInfo();
  gauge_object->packGauge(gauge);
  gauge_object->loadGauge();
  gauge_object->calculatePlaq();

  QKXTM_Vector_Kepler<double> *vecIn = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *vecOut = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  void *input_vector = malloc(GK_localVolume*4*3*2*sizeof(double));
  *((double*) input_vector) = 1.;
  vecIn->packVector((double*) input_vector);
  vecIn->loadVector();
  vecOut->gaussianSmearing(*vecIn,*gauge_object);
  vecOut->download();
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      printf("%+e %+e\n",vecOut->H_elem()[mu*3*2+c1*2+0],vecOut->H_elem()[mu*3*2+c1*2+1]);

  delete vecOut;
  delete gauge_object;
}

void invertWritePropsNoApe_SL_v2_Kepler(void **gauge, void **gaugeAPE ,QudaInvertParam *param ,QudaGaugeParam *gauge_param,quda::qudaQKXTMinfo_Kepler info, char *prop_path){
  profileInvert.Start(QUDA_PROFILE_TOTAL);
  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);
  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gaugeAPE = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gaugeAPE->packGauge(gaugeAPE);
  K_gaugeAPE->loadGauge();
  K_gaugeAPE->calculatePlaq();

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                     
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.Start(QUDA_PROFILE_H2D);

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  ColorSpinorParam cpuParam(input_vector,*param,X,pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = output_vector;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);


  profileInvert.Stop(QUDA_PROFILE_H2D);
  setTuning(param->tune);

  if (pc_solution && !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }
  if (!mat_solution && !pc_solution && pc_solve) {
    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
  }
  if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
  if( param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  QKXTM_Vector_Kepler<double> *K_vectorTmp = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR); 
  QKXTM_Vector_Kepler<double> *K_vectorGauss = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR); 

  char tempFilename[257];

   for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = info.sourcePosition[0][i] - comm_coords(default_topo)[i] * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   K_vectorTmp->packVector((double*) input_vector);
   K_vectorTmp->loadVector();
   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   K_vectorGauss->uploadToCuda(b);

   //   K_vectorGauss->download();
   // K_vectorGauss->norm2Host();
   // exit(-1);

   //   K_vectorTmp->norm2Host();
   // K_vectorTmp->uploadToCuda(b);
   // double nb = norm2(*b);
   //  if(nb==0.0)errorQuda("Source has zero norm");

   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, param->solution_type); // prepares the source vector 
   checkCudaError();
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);


   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   
   (*solve)(*out, *in);      
   dirac.reconstruct(*x, *b, param->solution_type);

   K_vectorTmp->downloadFromCuda(x);
   if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     K_vectorTmp->scaleVector(2*param->kappa);
   }

   //   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   //  K_vectorGauss->download();
   K_vectorTmp->download();

   sprintf(tempFilename,"%s_up.%04d",prop_path,ip);
   K_vectorTmp->write(tempFilename);
   delete tmp_up;
   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   // down
   K_vectorTmp->packVector((double*) input_vector);
   K_vectorTmp->loadVector();
   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);


   K_vectorGauss->uploadToCuda(b);

   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);


   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, param->solution_type);
   K_vectorTmp->downloadFromCuda(x);
   if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     K_vectorTmp->scaleVector(2*param->kappa);
   }

   //   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   //  K_vectorGauss->download();
   K_vectorTmp->download();
   sprintf(tempFilename,"%s_down.%04d",prop_path,ip);
   K_vectorTmp->write(tempFilename);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   

   }



 free(input_vector);
 free(output_vector);

 delete K_vectorTmp;
 delete K_vectorGauss;
 delete K_gaugeAPE;
 delete solve;
 delete h_b;
 delete h_x;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;

 popVerbosity();
 saveTuneCache(getVerbosity());
 profileInvert.Stop(QUDA_PROFILE_TOTAL);


}


void invertWritePropsNoApe_SL_v2_Kepler_single(void **gauge, void **gaugeAPE ,QudaInvertParam *param ,QudaGaugeParam *gauge_param,quda::qudaQKXTMinfo_Kepler info, char *prop_path){
  profileInvert.Start(QUDA_PROFILE_TOTAL);
  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);
  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<float> *K_gaugeAPE = new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);
  K_gaugeAPE->packGauge(gaugeAPE);
  K_gaugeAPE->loadGauge();
  K_gaugeAPE->calculatePlaq();

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                     
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.Start(QUDA_PROFILE_H2D);

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(float));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(float));

  ColorSpinorParam cpuParam(input_vector,*param,X,pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = output_vector;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);


  profileInvert.Stop(QUDA_PROFILE_H2D);
  setTuning(param->tune);

  if (pc_solution && !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }
  if (!mat_solution && !pc_solution && pc_solve) {
    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
  }
  if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
  if( param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  QKXTM_Vector_Kepler<float> *K_vectorTmp = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR); 
  QKXTM_Vector_Kepler<float> *K_vectorGauss = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR); 

  char tempFilename[257];

   for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(float));

   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = info.sourcePosition[0][i] - comm_coords(default_topo)[i] * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (float*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   K_vectorTmp->packVector((float*) input_vector);
   K_vectorTmp->loadVector();
   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   K_vectorGauss->uploadToCuda(b);

   //   K_vectorGauss->download();
   // K_vectorGauss->norm2Host();
   // exit(-1);

   //   K_vectorTmp->norm2Host();
   // K_vectorTmp->uploadToCuda(b);
   // double nb = norm2(*b);
   //  if(nb==0.0)errorQuda("Source has zero norm");

   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, param->solution_type); // prepares the source vector 
   checkCudaError();
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);


   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   
   (*solve)(*out, *in);      
   dirac.reconstruct(*x, *b, param->solution_type);

   K_vectorTmp->downloadFromCuda(x);
   if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     K_vectorTmp->scaleVector(2*param->kappa);
   }

   //   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   //  K_vectorGauss->download();
   K_vectorTmp->download();

   sprintf(tempFilename,"%s_up.%04d",prop_path,ip);
   K_vectorTmp->write(tempFilename);
   delete tmp_up;
   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   // down
   K_vectorTmp->packVector((float*) input_vector);
   K_vectorTmp->loadVector();
   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);


   K_vectorGauss->uploadToCuda(b);

   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);


   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, param->solution_type);
   K_vectorTmp->downloadFromCuda(x);
   if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     K_vectorTmp->scaleVector(2*param->kappa);
   }

   //   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   //  K_vectorGauss->download();
   K_vectorTmp->download();
   sprintf(tempFilename,"%s_down.%04d",prop_path,ip);
   K_vectorTmp->write(tempFilename);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   

   }



 free(input_vector);
 free(output_vector);

 delete K_vectorTmp;
 delete K_vectorGauss;
 delete K_gaugeAPE;
 delete solve;
 delete h_b;
 delete h_x;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;

 popVerbosity();
 saveTuneCache(getVerbosity());
 profileInvert.Stop(QUDA_PROFILE_TOTAL);


}

void checkReadingEigenVectors(int N_eigenVectors, char* pathIn, char *pathOut, char* pathEigenValues){
  QKXTM_Deflation_Kepler<float> *deflation = new QKXTM_Deflation_Kepler<float>(N_eigenVectors,false);
  deflation->printInfo();
  deflation->readEigenVectors(pathIn);
  deflation->writeEigenVectors_ASCI(pathOut);
  deflation->readEigenValues(pathEigenValues);
  for(int i = 0 ; i < N_eigenVectors; i++)
    printf("%e\n",deflation->EigenValues()[i]);
  delete deflation;
}

void checkDeflateVectorQuda(void **gauge,QudaInvertParam *param ,QudaGaugeParam *gauge_param,char *filename_eigenValues, char *filename_eigenVectors, char *filename_out,int NeV){
  bool flag_eo;
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<float> *deflation = new QKXTM_Deflation_Kepler<float>(NeV,flag_eo);

  deflation->readEigenValues(filename_eigenValues);
  deflation->readEigenVectors(filename_eigenVectors);
  deflation->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation->multiply_by_phase();
  QKXTM_Vector_Kepler<float> *vecIn = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<float> *vecOut = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<float> *vecTest = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);

  // just for test set vec to all elements to be 1
  vecIn->zero_host();
  vecOut->zero_host();

  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < 4 ; mu++)
      for(int c = 0 ; c < 3 ; c++)
	vecIn->H_elem()[iv*4*3*2+mu*c*2+c*2+0] = 1.;
  
  deflation->deflateGuessVector(*vecOut,*vecIn);
  vecOut->download();
  vecTest->zero_host();

  std::complex<float> *cmplx_vecTest = NULL;
  std::complex<float> *cmplx_U = NULL;
  std::complex<float> *cmplx_b = NULL;


  for(int ie = 0 ; ie < NeV ; ie++){
    cmplx_vecTest = (std::complex<float>*) vecTest->H_elem();
    cmplx_U = (std::complex<float>*) deflation->H_elem()[ie];
    cmplx_b = (std::complex<float>*) vecIn->H_elem();
    for(int alpha = 0 ; alpha < (GK_localVolume/2)*4*3 ; alpha++)
      for(int beta = 0 ; beta < (GK_localVolume/2)*4*3 ; beta++)
	cmplx_vecTest[alpha] = cmplx_vecTest[alpha] + cmplx_U[ alpha] * (1./deflation->EigenValues()[ie]) * conj(cmplx_U[beta]) * cmplx_b[beta]; 
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out=fopen(filename_out,"w"); 
    if(ptr_out == NULL)errorQuda("Error open file for writing\n");
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int mu = 0 ; mu < 4 ; mu++)
	for(int c = 0 ; c < 3 ; c++)
	  fprintf(ptr_out,"%+e %+e \t %+e %+e \t %e %e\n",vecIn->H_elem()[iv*4*3*2+mu*c*2+c*2+0],vecIn->H_elem()[iv*4*3*2+mu*c*2+c*2+1],vecOut->H_elem()[iv*4*3*2+mu*c*2+c*2+0],vecOut->H_elem()[iv*4*3*2+mu*c*2+c*2+1],vecTest->H_elem()[iv*4*3*2+mu*c*2+c*2+0],vecTest->H_elem()[iv*4*3*2+mu*c*2+c*2+1]);
  }

  /*

  std::complex<float> *temp = (std::complex<float>*)malloc(NeV*2*sizeof(float));
  memset(temp,0,NeV*2*sizeof(float));
  for(int ie = 0 ; ie < NeV ; ie++){
    std::complex<float> *pointer = (std::complex<float>*) deflation->H_elem()[ie];
    std::complex<float> *cmplx_b = (std::complex<float>*) vecIn->H_elem();
    for(int iv = 0 ; iv < (GK_localVolume/2)*4*3 ; iv++)
      temp[ie] = temp[ie] + conj(pointer[iv]) * cmplx_b[iv];
    printfQuda("%+e %+e\n",temp[ie].real(),temp[ie].imag());
  }
  */
  delete vecTest;
  delete deflation;
  delete vecIn;
  delete vecOut;
}

void checkEigenVectorQuda(void **gauge,QudaInvertParam *param ,QudaGaugeParam *gauge_param,char *filename_eigenValues, char *filename_eigenVectors, char *filename_out,int NeV){
  bool flag_eo;
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);

  deflation->readEigenValues(filename_eigenValues);
  deflation->readEigenVectors(filename_eigenVectors);
  deflation->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation->multiply_by_phase();

  if (!initialized) errorQuda("QUDA not initialized");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  if (cloverPrecise == NULL && ((param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)))
    errorQuda("Clover field not allocated");
  if (cloverInvPrecise == NULL && param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  QKXTM_Gauge_Kepler<double> *qkxTM_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  qkxTM_gauge->packGauge(gauge);
  qkxTM_gauge->loadGauge();
  qkxTM_gauge->calculatePlaq();


  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  //  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION || inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);


  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  double *input_vector =(double*) malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));  
  double *output_vector =(double*) malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));  

  ColorSpinorParam cpuParam((void*)input_vector,*param,X,pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = output_vector;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);
  delete h_b;
  delete h_x;

  setTuning(param->tune);
  QKXTM_Vector_Kepler<double> *vec = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  DiracParam diracParam;
  setDiracParam(diracParam, param, true);
  Dirac *dirac = Dirac::create(diracParam);


  for(int i = 0 ; i < deflation->NeVs() ; i++){
    zeroCuda(*b);
    zeroCuda(*x);
    deflation->copyEigenVectorToQKXTM_Vector_Kepler(i,input_vector);
    vec->packVector(input_vector);
    vec->loadVector();
    vec->uploadToCuda(b,flag_eo);
    dirac->MdagM(*x,*b);
    vec->downloadFromCuda(x,flag_eo);
    vec->download();
    deflation->copyEigenVectorFromQKXTM_Vector_Kepler(i,vec->H_elem());
  }

  free(input_vector);
  free(output_vector);
  deflation->writeEigenVectors_ASCI(filename_out);

   delete dirac;  
   delete b;
   delete x;
   delete vec;
   delete qkxTM_gauge;
   delete deflation;

}

void checkDeflateAndInvert(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues, char *filename_eigenVectors, char *filename_out,int NeV ){
  bool flag_eo;


  double t1,t2;


  profileInvert.Start(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  deflation->printInfo();
  t1 = MPI_Wtime();
  deflation->readEigenValues(filename_eigenValues);
  deflation->readEigenVectors(filename_eigenVectors);
  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");
  deflation->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation->multiply_by_phase();

  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                                                         
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.Start(QUDA_PROFILE_H2D);


  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  ColorSpinorParam cpuParam(input_vector,*param,X,pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = output_vector;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));


  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField( cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.Stop(QUDA_PROFILE_H2D);
  setTuning(param->tune);


  zeroCuda(*x);
  zeroCuda(*b);

  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  t1 = MPI_Wtime();
  if(comm_rank() == 0)  *((double*) input_vector) = 1.;
  K_vector->packVector((double*) input_vector);
  K_vector->loadVector();
  K_vector->uploadToCuda(b,flag_eo);
  dirac.prepare(in,out,*x,*b,param->solution_type);

  // in is reference to the b but for a parity sinlet
  // out is reference to the x but for a parity sinlet
  cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
  dirac.Mdag(*in, *tmp);
  delete tmp;
  // now the the source vector b is ready to perform deflation and find the initial guess
  K_vector->downloadFromCuda(in,flag_eo);
  K_vector->download();
  deflation->deflateGuessVector(*K_guess,*K_vector);
  K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
  t2 = MPI_Wtime();
#ifdef TIMING_REPORT
  printfQuda("Timing report for deflation procudure is %f sec\n",t2-t1);
#endif

  //  zeroCuda(*out); // remove it later , just for test

  fflush(stdout);

  t1 = MPI_Wtime();
  (*solve)(*out,*in);
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report inversion is %f sec\n",t2-t1);
#endif


  dirac.reconstruct(*x,*b,param->solution_type);
  K_vector->downloadFromCuda(x,flag_eo);
  if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
    K_vector->scaleVector(2*param->kappa);
  }
  K_vector->download();

  /*
  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out=fopen(filename_out,"w"); 
    if(ptr_out == NULL)errorQuda("Error open file for writing\n");
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int mu = 0 ; mu < 4 ; mu++)
	for(int c = 0 ; c < 3 ; c++)
	  fprintf(ptr_out,"%+e %+e\n",K_vector->H_elem()[iv*4*3*2+mu*c*2+c*2+0],K_vector->H_elem()[iv*4*3*2+mu*c*2+c*2+1]);
  }
  */

  /*
  K_guess->download();
  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out=fopen(filename_out,"w"); 
    if(ptr_out == NULL)errorQuda("Error open file for writing\n");
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int mu = 0 ; mu < 4 ; mu++)
	for(int c = 0 ; c < 3 ; c++)
	  fprintf(ptr_out,"%+e %+e\n",K_guess->H_elem()[iv*4*3*2+mu*c*2+c*2+0],K_guess->H_elem()[iv*4*3*2+mu*c*2+c*2+1]);
  }
  */

  free(input_vector);
  free(output_vector);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation;
  delete h_x;
  delete h_b;
  delete x;
  delete b;

  popVerbosity();
  saveTuneCache(getVerbosity());
  profileInvert.Stop(QUDA_PROFILE_TOTAL);

}


void DeflateAndInvert_twop(void **gaugeSmeared, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_up, char *filename_eigenVectors_up, char *filename_eigenValues_down, char *filename_eigenVectors_down, char *filename_out,int NeV, qudaQKXTMinfo_Kepler info ){
  bool flag_eo;
  double t1,t2;

  profileInvert.Start(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_up = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  void *input_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<float> *K_temp = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);

  QKXTM_Propagator_Kepler<float> *K_prop_up = new QKXTM_Propagator_Kepler<float>(DEVICE,PROPAGATOR);
  QKXTM_Propagator_Kepler<float> *K_prop_down = new QKXTM_Propagator_Kepler<float>(DEVICE,PROPAGATOR);  
  QKXTM_Contraction_Kepler<float> *K_contract = new QKXTM_Contraction_Kepler<float>();
  printfQuda("Memory allocation was successfull\n");

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  deflation_up->printInfo();
  t1 = MPI_Wtime();
  deflation_up->readEigenValues(filename_eigenValues_up);
  deflation_up->readEigenVectors(filename_eigenVectors_up);
  deflation_up->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_up->multiply_by_phase();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);


  K_gauge->packGauge(gaugeSmeared);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.Start(QUDA_PROFILE_H2D);


  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();


  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  ColorSpinorParam cpuParam(input_vector,*param,X,pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = output_vector;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));


  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField( cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.Stop(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  zeroCuda(*x);
  zeroCuda(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  int my_src[4];
  char filename_mesons[257];
  char filename_baryons[257];

  for(int isource = 0 ; isource < info.Nsources ; isource++){
    sprintf(filename_mesons,"%s.mesons.SS.%02d.%02d.%02d.%02d.dat",filename_out,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
    sprintf(filename_baryons,"%s.baryons.SS.%02d.%02d.%02d.%02d.dat",filename_out,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
    bool checkMesons, checkBaryons;
    checkMesons = exists_file(filename_mesons);
    checkBaryons = exists_file(filename_baryons);
    if( (checkMesons == true) && (checkBaryons == true) ) continue;
    for(int isc = 0 ; isc < 12 ; isc++){
      t1 = MPI_Wtime();
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_PLUS);
      x->changeTwist(QUDA_TWIST_PLUS);
      b->Even().changeTwist(QUDA_TWIST_PLUS);
      b->Odd().changeTwist(QUDA_TWIST_PLUS);
      x->Even().changeTwist(QUDA_TWIST_PLUS);
      x->Odd().changeTwist(QUDA_TWIST_PLUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gauge);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_up);
      delete tmp_up;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_up->deflateGuessVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
      //  zeroCuda(*out); // remove it later , just for test
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }
      K_guess->gaussianSmearing(*K_vector,*K_gauge);
      K_temp->castDoubleToFloat(*K_guess);
      K_prop_up->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      t2 = MPI_Wtime();
      printfQuda("Inversion up = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
      //////////////////////////////////////////////////////////
      //////////////////////////////////////////////////////////
      /////////////////////////////////////////////////////////
      t1 = MPI_Wtime();
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_MINUS);
      x->changeTwist(QUDA_TWIST_MINUS);
      b->Even().changeTwist(QUDA_TWIST_MINUS);
      b->Odd().changeTwist(QUDA_TWIST_MINUS);
      x->Even().changeTwist(QUDA_TWIST_MINUS);
      x->Odd().changeTwist(QUDA_TWIST_MINUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gauge);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_down);
      delete tmp_down;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_down->deflateGuessVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
      //  zeroCuda(*out); // remove it later , just for test
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }
      K_guess->gaussianSmearing(*K_vector,*K_gauge);
      K_temp->castDoubleToFloat(*K_guess);
      K_prop_down->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      t2 = MPI_Wtime();
      printfQuda("Inversion down = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
    } // close loop over 12 spin-color

    K_prop_up->rotateToPhysicalBase_device(+1);
    K_prop_down->rotateToPhysicalBase_device(-1);
    t1 = MPI_Wtime();
    K_contract->contractMesons(*K_prop_up,*K_prop_down,filename_mesons,isource);
    K_contract->contractBaryons(*K_prop_up,*K_prop_down,filename_baryons,isource);
    t2 = MPI_Wtime();
    printfQuda("Contractions for source = %d finished in time %f sec\n",isource,t2-t1);
  } // close loop over source positions


  free(input_vector);
  free(output_vector);
  delete K_temp;
  delete K_contract;
  delete K_prop_down;
  delete K_prop_up;
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation_up;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;

  popVerbosity();
  saveTuneCache(getVerbosity());
  profileInvert.Stop(QUDA_PROFILE_TOTAL);

}

template <typename Float>
void getStochasticRandomSource(void *spinorIn, gsl_rng *rNum){
  memset(spinorIn,0,GK_localVolume*12*2*sizeof(Float));
  for(int i = 0; i<GK_localVolume*12; i++){
    int randomNumber = gsl_rng_uniform_int(rNum, 4);

    //    ((Float*) spinorIn)[i*2] = 1.0;
    //    ((Float*) spinorIn)[i*2+1] = 0.0;

    switch  (randomNumber){
      case 0:
	((Float*) spinorIn)[i*2] = 1.;
	break;
      case 1:
	((Float*) spinorIn)[i*2] = -1.;
	break;
      case 2:
	((Float*) spinorIn)[i*2+1] = 1.;
	break;
      case 3:
	((Float*) spinorIn)[i*2+1] = -1.;
	break;
    }

  }

}


void DeflateAndInvert_loop(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_down, char *filename_eigenVectors_down,char *filename_out , int NeV , int Nstoch, int seed ,int NdumpStep, qudaQKXTMinfo_Kepler info){
  bool flag_eo;
  double t1,t2;

  profileInvert.Start(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  deflation_down->printInfo();

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  t1 = MPI_Wtime();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.Start(QUDA_PROFILE_H2D);


  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;
  cudaColorSpinorField *tmp3 = NULL;
  cudaColorSpinorField *tmp4 = NULL;


  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  ColorSpinorParam cpuParam(input_vector,*param,X,pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = output_vector;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));


  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField( cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  tmp3 = new cudaColorSpinorField(cudaParam);
  tmp4 = new cudaColorSpinorField(cudaParam);

  profileInvert.Stop(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  zeroCuda(*x);
  zeroCuda(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  void    *cnRes_vv;
  void    *cnRes_gv;

  void    *cnResTmp_vv;
  void    *cnResTmp_gv;

  if((cudaHostAlloc(&cnRes_vv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_vv\n");
  if((cudaHostAlloc(&cnRes_gv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_gv\n");

  cudaMemset      (cnRes_vv, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset      (cnRes_gv, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cnResTmp_vv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_vv\n");
  if((cudaHostAlloc(&cnResTmp_gv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_gv\n");

  cudaMemset      (cnResTmp_vv, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset      (cnResTmp_gv, 0, sizeof(double)*2*16*GK_localVolume);

  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  for(int is = 0 ; is < Nstoch ; is++){
    t1 = MPI_Wtime();
    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum);
    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_up);
    delete tmp_up;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation_down->deflateGuessVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  zeroCuda(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);
    oneEndTrick<double>(*x,*tmp3,*tmp4,param,cnRes_gv,cnRes_vv);
    t2 = MPI_Wtime();
    printfQuda("Stoch %d finished in %f sec\n",is,t2-t1);
    if( (is+1)%NdumpStep == 0){
      doCudaFFT<double>(cnRes_gv,cnRes_vv,cnResTmp_gv,cnResTmp_vv);
      dumpLoop<double>(cnResTmp_gv,cnResTmp_vv,filename_out,is+1,info.Q_sq);
    }
  } // close loop over source positions

  cudaFreeHost(cnRes_gv);
  cudaFreeHost(cnRes_vv);

  cudaFreeHost(cnResTmp_gv);
  cudaFreeHost(cnResTmp_vv);

  free(input_vector);
  free(output_vector);
  gsl_rng_free(rNum);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete tmp3;
  delete tmp4;
  popVerbosity();
  saveTuneCache(getVerbosity());
  profileInvert.Stop(QUDA_PROFILE_TOTAL);

}

void DeflateAndInvert_loop_w_One_Der(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_down, char *filename_eigenVectors_down,char *filename_out , int NeV , int Nstoch, int seed ,int NdumpStep, qudaQKXTMinfo_Kepler info){
  bool flag_eo;
  double t1,t2;

  profileInvert.Start(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  deflation_down->printInfo();

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  t1 = MPI_Wtime();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.Start(QUDA_PROFILE_H2D);


  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;
  cudaColorSpinorField *tmp3 = NULL;
  cudaColorSpinorField *tmp4 = NULL;


  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  ColorSpinorParam cpuParam(input_vector,*param,X,pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = output_vector;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));


  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField( cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  tmp3 = new cudaColorSpinorField(cudaParam);
  tmp4 = new cudaColorSpinorField(cudaParam);

  profileInvert.Stop(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  zeroCuda(*x);
  zeroCuda(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  ////////////////////////// Allocate memory for local
  void    *cnRes_vv;
  void    *cnRes_gv;

  void    *cnTmp;

  if((cudaHostAlloc(&cnRes_vv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_vv\n");
  if((cudaHostAlloc(&cnRes_gv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_gv\n");

  cudaMemset      (cnRes_vv, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset      (cnRes_gv, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cnTmp, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnTmp\n");

  cudaMemset      (cnTmp, 0, sizeof(double)*2*16*GK_localVolume);
  ///////////////////////////////////////////////////
  //////////// Allocate memory for one-Der and conserved current
  void    **cnD_vv;
  void    **cnD_gv;
  void    **cnC_vv;
  void    **cnC_gv;

  cnD_vv   = (void**) malloc(sizeof(double*)*2*4);
  cnD_gv   = (void**) malloc(sizeof(double*)*2*4);
  cnC_vv   = (void**) malloc(sizeof(double*)*2*4);
  cnC_gv   = (void**) malloc(sizeof(double*)*2*4);

  if(cnD_gv == NULL)errorQuda("Error allocating memory cnD_gv higher level\n");
  if(cnD_vv == NULL)errorQuda("Error allocating memory cnD_vv higher level\n");
  if(cnC_gv == NULL)errorQuda("Error allocating memory cnC_gv higher level\n");
  if(cnC_vv == NULL)errorQuda("Error allocating memory cnC_vv higher level\n");
  cudaDeviceSynchronize();

  for(int mu = 0; mu < 4 ; mu++){
    if((cudaHostAlloc(&(cnD_vv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD_vv\n");
    if((cudaHostAlloc(&(cnD_gv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD_gv\n");
    if((cudaHostAlloc(&(cnC_vv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC_vv\n");
    if((cudaHostAlloc(&(cnC_gv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC_gv\n");
  }
  cudaDeviceSynchronize();
  ///////////////////////////////////////////////////
  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  for(int is = 0 ; is < Nstoch ; is++){
    t1 = MPI_Wtime();
    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum);
    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_up);
    delete tmp_up;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation_down->deflateGuessVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  zeroCuda(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);
    oneEndTrick_w_One_Der<double>(*x,*tmp3,*tmp4,param,cnRes_gv,cnRes_vv,cnD_gv,cnD_vv,cnC_gv,cnC_vv);
    t2 = MPI_Wtime();
    printfQuda("Stoch %d finished in %f sec\n",is,t2-t1);
    if( (is+1)%NdumpStep == 0){
      doCudaFFT_v2<double>(cnRes_vv,cnTmp);
      dumpLoop_ultraLocal<double>(cnTmp,filename_out,is+1,info.Q_sq,0); // Scalar
      doCudaFFT_v2<double>(cnRes_gv,cnTmp);
      dumpLoop_ultraLocal<double>(cnTmp,filename_out,is+1,info.Q_sq,1); // dOp
      for(int mu = 0 ; mu < 4 ; mu++){
	doCudaFFT_v2<double>(cnD_vv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,0); // Loops
	doCudaFFT_v2<double>(cnD_gv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,1); // LpsDw

	doCudaFFT_v2<double>(cnC_vv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,2); // LpsDw noether
	doCudaFFT_v2<double>(cnC_gv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,3); // LpsDw noether
      }
    } // close loop for dump loops

  } // close loop over source positions

  cudaFreeHost(cnRes_gv);
  cudaFreeHost(cnRes_vv);

  cudaFreeHost(cnTmp);

  for(int mu = 0 ; mu < 4 ; mu++){
    cudaFreeHost(cnD_vv[mu]);
    cudaFreeHost(cnD_gv[mu]);
    cudaFreeHost(cnC_vv[mu]);
    cudaFreeHost(cnC_gv[mu]);
  }
  
  free(cnD_vv);
  free(cnD_gv);
  free(cnC_vv);
  free(cnC_gv);

  free(input_vector);
  free(output_vector);
  gsl_rng_free(rNum);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete tmp3;
  delete tmp4;
  popVerbosity();
  saveTuneCache(getVerbosity());
  profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void DeflateAndInvert_loop_w_One_Der_volumeSource(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_up, char *filename_eigenVectors_up, char *filename_eigenValues_down, char *filename_eigenVectors_down,char *filename_out , int NeV , int Nstoch, int seed ,int NdumpStep, qudaQKXTMinfo_Kepler info){
  bool flag_eo;
  double t1,t2;

  profileInvert.Start(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_up = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  deflation_down->printInfo();

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  t1 = MPI_Wtime();

  deflation_up->readEigenValues(filename_eigenValues_up);
  deflation_up->readEigenVectors(filename_eigenVectors_up);
  deflation_up->rotateFromChiralToUKQCD();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.Start(QUDA_PROFILE_H2D);


  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;
  cudaColorSpinorField *tmp = NULL;


  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  ColorSpinorParam cpuParam(input_vector,*param,X,pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = output_vector;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));


  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField( cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  tmp = new cudaColorSpinorField(cudaParam);

  profileInvert.Stop(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  zeroCuda(*x);
  zeroCuda(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  ////////////////////////// Allocate memory for local
  void    *cn_local_up;
  void    *cn_local_down;
  void    *cnTmp;

  if((cudaHostAlloc(&cn_local_up, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cn_local_up\n");
  cudaMemset      (cn_local_up, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cn_local_down, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cn_local_down\n");
  cudaMemset      (cn_local_down, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cnTmp, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnTmp\n");
  cudaMemset      (cnTmp, 0, sizeof(double)*2*16*GK_localVolume);
  ///////////////////////////////////////////////////
  //////////// Allocate memory for one-Der and conserved current
  void    **cnD_up;
  void    **cnC_up;
  void    **cnD_down;
  void    **cnC_down;

  cnD_up   = (void**) malloc(sizeof(double*)*2*4);
  cnC_up   = (void**) malloc(sizeof(double*)*2*4);
  cnD_down   = (void**) malloc(sizeof(double*)*2*4);
  cnC_down   = (void**) malloc(sizeof(double*)*2*4);

  if(cnD_up == NULL)errorQuda("Error allocating memory cnD higher level\n");
  if(cnC_up == NULL)errorQuda("Error allocating memory cnC higher level\n");

  if(cnD_down == NULL)errorQuda("Error allocating memory cnD higher level\n");
  if(cnC_down == NULL)errorQuda("Error allocating memory cnC higher level\n");

  cudaDeviceSynchronize();

  for(int mu = 0; mu < 4 ; mu++){
    if((cudaHostAlloc(&(cnD_up[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD\n");
    if((cudaHostAlloc(&(cnC_up[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC\n");

    if((cudaHostAlloc(&(cnD_down[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD\n");
    if((cudaHostAlloc(&(cnC_down[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC\n");

  }
  cudaDeviceSynchronize();
  ///////////////////////////////////////////////////
  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  for(int is = 0 ; is < Nstoch ; is++){

    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum);
    t1 = MPI_Wtime();

#define CROSSCHECK
#ifdef CROSSCHECK
    FILE *ptr_xi;
    ptr_xi=fopen("/users/krikitos/run/test_loop/volumeSource.In","w");
    for(int ii = 0 ; ii < X[0]*X[1]*X[2]*X[3]*spinorSiteSize/2 ; ii++)
      fprintf(ptr_xi,"%+e %+e\n",((double*) input_vector)[ii*2+0], ((double*) input_vector)[ii*2+1]);
#endif

    // for up/////////////////////////////////////
    b->changeTwist(QUDA_TWIST_PLUS);
    x->changeTwist(QUDA_TWIST_PLUS);
    b->Even().changeTwist(QUDA_TWIST_PLUS);
    b->Odd().changeTwist(QUDA_TWIST_PLUS);
    x->Even().changeTwist(QUDA_TWIST_PLUS);
    x->Odd().changeTwist(QUDA_TWIST_PLUS);

    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_up);
    delete tmp_up;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation_up->deflateGuessVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  zeroCuda(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);


#ifdef CROSSCHECK
    K_guess->downloadFromCuda(x,flag_eo);
    K_guess->download();
    FILE *ptr_phi_up;
    ptr_phi_up=fopen("/users/krikitos/run/test_loop/volumeSource_up.Out","w");
    for(int ii = 0 ; ii < X[0]*X[1]*X[2]*X[3]*spinorSiteSize/2 ; ii++)
      fprintf(ptr_phi_up,"%+e %+e\n",K_guess->H_elem()[ii*2+0], K_guess->H_elem()[ii*2+1]);
#endif

    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);

    volumeSource_w_One_Der<double>(*x,*b,*tmp,param,cn_local_up,cnD_up,cnC_up);
    t2 = MPI_Wtime();
    printfQuda("Stoch %d for up finished in %f sec\n",is,t2-t1);
    if( (is+1)%NdumpStep == 0){
      doCudaFFT_v2<double>(cn_local_up,cnTmp);
      dumpLoop_ultraLocal_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,"ultralocal_up"); 
      for(int mu = 0 ; mu < 4 ; mu++){
	doCudaFFT_v2<double>(cnD_up[mu],cnTmp);
	dumpLoop_oneD_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,"oneD_up"); 

	doCudaFFT_v2<double>(cnC_up[mu],cnTmp);
	dumpLoop_oneD_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,"noe_up"); 
      }
    } // close loop for dump loops

    //////////////////// for down
    t1 = MPI_Wtime();

    b->changeTwist(QUDA_TWIST_MINUS);
    x->changeTwist(QUDA_TWIST_MINUS);
    b->Even().changeTwist(QUDA_TWIST_MINUS);
    b->Odd().changeTwist(QUDA_TWIST_MINUS);
    x->Even().changeTwist(QUDA_TWIST_MINUS);
    x->Odd().changeTwist(QUDA_TWIST_MINUS);

    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_down);
    delete tmp_down;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation_down->deflateGuessVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  zeroCuda(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);

#ifdef CROSSCHECK
    K_guess->downloadFromCuda(x,flag_eo);
    K_guess->download();
    FILE *ptr_phi_down;
    ptr_phi_down=fopen("/users/krikitos/run/test_loop/volumeSource_down.Out","w");
    for(int ii = 0 ; ii < X[0]*X[1]*X[2]*X[3]*spinorSiteSize/2 ; ii++)
      fprintf(ptr_phi_down,"%+e %+e\n",K_guess->H_elem()[ii*2+0], K_guess->H_elem()[ii*2+1]);
#endif


    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);

    volumeSource_w_One_Der<double>(*x,*b,*tmp,param,cn_local_down,cnD_down,cnC_down);
    t2 = MPI_Wtime();
    printfQuda("Stoch %d for down finished in %f sec\n",is,t2-t1);
    if( (is+1)%NdumpStep == 0){
      doCudaFFT_v2<double>(cn_local_down,cnTmp);
      dumpLoop_ultraLocal_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,"ultralocal_down"); // Scalar
      for(int mu = 0 ; mu < 4 ; mu++){
	doCudaFFT_v2<double>(cnD_down[mu],cnTmp);
	dumpLoop_oneD_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,"oneD_down"); // Loops

	doCudaFFT_v2<double>(cnC_down[mu],cnTmp);
	dumpLoop_oneD_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,"noe_down"); // Loops noether
      }
    } // close loop for dump loops

  } // close loop over source positions

  cudaFreeHost(cn_local_up);
  cudaFreeHost(cn_local_down);
  cudaFreeHost(cnTmp);

  for(int mu = 0 ; mu < 4 ; mu++){
    cudaFreeHost(cnD_up[mu]);
    cudaFreeHost(cnD_down[mu]);
    cudaFreeHost(cnC_up[mu]);
    cudaFreeHost(cnC_down[mu]);
  }
  
  free(cnD_up);
  free(cnD_down);
  free(cnC_up);
  free(cnC_down);

  free(input_vector);
  free(output_vector);
  gsl_rng_free(rNum);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete tmp;
  popVerbosity();
  saveTuneCache(getVerbosity());
  profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

/*
#ifdef TEST
    FILE *ptr_file;
    ptr_file = fopen("/users/krikitos/run/test_loop/source.dat","w");
    if(ptr_file == NULL)errorQuda("cannot open file for writting\n");
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int mu = 0 ; mu < 4 ; mu++)
	for(int c1 = 0 ; c1 < 3 ; c1++){
	  fprintf(ptr_file,"%f %f\n",((double*)input_vector)[iv*4*3*2+mu*3*2+c1*2+0],((double*)input_vector)[iv*4*3*2+mu*3*2+c1*2+1]);
	}
#endif

 */

void DeflateAndInvert_threepTwop(void **gaugeSmeared, void **gauge, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_up, char *filename_eigenVectors_up, char *filename_eigenValues_down, char *filename_eigenVectors_down, char *filename_twop, char *filename_threep,int NeV, qudaQKXTMinfo_Kepler info, WHICHPARTICLE NUCLEON, WHICHPROJECTOR PID ){
  bool flag_eo;
  double t1,t2;

  profileInvert.Start(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_up = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  void *input_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  QKXTM_Gauge_Kepler<double> *K_gaugeSmeared = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  QKXTM_Gauge_Kepler<float> *K_gaugeContractions = new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);

  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<float> *K_temp = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);


  QKXTM_Propagator_Kepler<float> *K_prop_up = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);
  QKXTM_Propagator_Kepler<float> *K_prop_down = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);  
  QKXTM_Propagator_Kepler<float> *K_seqProp = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);

  QKXTM_Propagator3D_Kepler<float> *K_prop3D_up = new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);
  QKXTM_Propagator3D_Kepler<float> *K_prop3D_down = new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);

  QKXTM_Contraction_Kepler<float> *K_contract = new QKXTM_Contraction_Kepler<float>();
  printfQuda("Memory allocation was successfull\n");

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  deflation_up->printInfo();
  t1 = MPI_Wtime();
  deflation_up->readEigenValues(filename_eigenValues_up);
  deflation_up->readEigenVectors(filename_eigenVectors_up);
  deflation_up->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_up->multiply_by_phase();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);


  K_gaugeContractions->packGauge(gauge);
  K_gaugeContractions->loadGauge();
  //  K_gaugeContractions->calculate(); do not do it because I changed the sign due to antiperiodic boundary conditions


  K_gaugeSmeared->packGauge(gaugeSmeared);
  K_gaugeSmeared->loadGauge();
  K_gaugeSmeared->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.Start(QUDA_PROFILE_H2D);


  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();


  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  ColorSpinorParam cpuParam(input_vector,*param,X,pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = output_vector;
  ColorSpinorField *h_x = (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));


  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField( cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.Stop(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  zeroCuda(*x);
  zeroCuda(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  int my_src[4];
  char filename_mesons[257];
  char filename_baryons[257];


  for(int isource = 0 ; isource < info.Nsources ; isource++){

    if(info.run2pt_src[isource] && info.run3pt_src[isource]){
      for(int isc = 0 ; isc < 12 ; isc++){
	///////////////////////////////////////////////////////////////////////////////// forward prop for up quark ///////////////////////////
	t1 = MPI_Wtime();
	memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
	b->changeTwist(QUDA_TWIST_PLUS);
	x->changeTwist(QUDA_TWIST_PLUS);
	b->Even().changeTwist(QUDA_TWIST_PLUS);
	b->Odd().changeTwist(QUDA_TWIST_PLUS);
	x->Even().changeTwist(QUDA_TWIST_PLUS);
	x->Odd().changeTwist(QUDA_TWIST_PLUS);
	for(int i = 0 ; i < 4 ; i++)
	  my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];
	
	if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	  *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;
	
	K_vector->packVector((double*) input_vector);
	K_vector->loadVector();
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_guess->uploadToCuda(b,flag_eo);
	dirac.prepare(in,out,*x,*b,param->solution_type);
	// in is reference to the b but for a parity sinlet
	// out is reference to the x but for a parity sinlet
	cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
	dirac.Mdag(*in, *tmp_up);
	delete tmp_up;
	// now the the source vector b is ready to perform deflation and find the initial guess
	K_vector->downloadFromCuda(in,flag_eo);
	K_vector->download();
	deflation_up->deflateGuessVector(*K_guess,*K_vector);
	K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
	//  zeroCuda(*out); // remove it later , just for test
	(*solve)(*out,*in);
	dirac.reconstruct(*x,*b,param->solution_type);
	K_vector->downloadFromCuda(x,flag_eo);
	if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  K_vector->scaleVector(2*param->kappa);
	}
	
	K_temp->castDoubleToFloat(*K_vector);
	K_prop_up->absorbVectorToDevice(*K_temp,isc/3,isc%3);
	t2 = MPI_Wtime();
	printfQuda("Inversion up = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
	//////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////// Forward prop for down quark ///////////////////////////////////
	/////////////////////////////////////////////////////////
	t1 = MPI_Wtime();
	memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
	b->changeTwist(QUDA_TWIST_MINUS);
	x->changeTwist(QUDA_TWIST_MINUS);
	b->Even().changeTwist(QUDA_TWIST_MINUS);
	b->Odd().changeTwist(QUDA_TWIST_MINUS);
	x->Even().changeTwist(QUDA_TWIST_MINUS);
	x->Odd().changeTwist(QUDA_TWIST_MINUS);
	for(int i = 0 ; i < 4 ; i++)
	  my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];
	
	if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	  *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;
	
	K_vector->packVector((double*) input_vector);
	K_vector->loadVector();
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_guess->uploadToCuda(b,flag_eo);
	dirac.prepare(in,out,*x,*b,param->solution_type);
	// in is reference to the b but for a parity sinlet
	// out is reference to the x but for a parity sinlet
	cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
	dirac.Mdag(*in, *tmp_down);
	delete tmp_down;
	// now the the source vector b is ready to perform deflation and find the initial guess
	K_vector->downloadFromCuda(in,flag_eo);
	K_vector->download();
	deflation_down->deflateGuessVector(*K_guess,*K_vector);
	K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
	//  zeroCuda(*out); // remove it later , just for test
	(*solve)(*out,*in);
	dirac.reconstruct(*x,*b,param->solution_type);
	K_vector->downloadFromCuda(x,flag_eo);
	if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  K_vector->scaleVector(2*param->kappa);
	}

	K_temp->castDoubleToFloat(*K_vector);
	K_prop_down->absorbVectorToDevice(*K_temp,isc/3,isc%3);
	t2 = MPI_Wtime();
	printfQuda("Inversion down = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
      } // close loop over 12 spin-color

    }//-check if running for 2pt and/or 3pt

    if(info.run2pt_src[isource]){
      sprintf(filename_mesons,"%s.mesons.SS.%02d.%02d.%02d.%02d.dat",filename_twop,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
      sprintf(filename_baryons,"%s.baryons.SS.%02d.%02d.%02d.%02d.dat",filename_twop,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
      bool checkMesons, checkBaryons;
      checkMesons = exists_file(filename_mesons);
      checkBaryons = exists_file(filename_baryons);
      if( (checkMesons != true) && (checkBaryons != true) ){ // continue; // because threep are written before twop if I checked twop I know that threep are fine
	// smear the forward propagators and perform the 2pt-correlation function
	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    K_temp->copyPropagator(*K_prop_up,nu,c2);
	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    K_prop_up->absorbVectorToDevice(*K_temp,nu,c2);
	    
	    K_temp->copyPropagator(*K_prop_down,nu,c2);
	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    K_prop_down->absorbVectorToDevice(*K_temp,nu,c2);
	  }
	/////
	K_prop_up->rotateToPhysicalBase_device(+1);
	K_prop_down->rotateToPhysicalBase_device(-1);
	t1 = MPI_Wtime();
	K_contract->contractMesons(*K_prop_up,*K_prop_down,filename_mesons,isource);
	K_contract->contractBaryons(*K_prop_up,*K_prop_down,filename_baryons,isource);
	t2 = MPI_Wtime();
	printfQuda("2pt contractions for source = %d finished in time %f sec\n",isource,t2-t1);
      }
    }

    // Perform the sequential inversion and 3pt-correlation function
    //-C.Kallidonis: Loop over the number of sink-source separations
    int my_fixSinkTime;
    char filename_threep_tsink[257];
    for(int its=0;its<info.Ntsink;its++){
      my_fixSinkTime = (info.tsinkSource[its] + info.sourcePosition[isource][3])%GK_totalL[3] - comm_coords(default_topo)[3] * X[3];
      sprintf(filename_threep_tsink,"%s_tsink%d",filename_threep,info.tsinkSource[its]);
      printfQuda("The three-point function base name is: %s\n",filename_threep_tsink);
      

      /////////////////////////////////// Smearing on the 3D propagators
      t1 = MPI_Wtime();
      K_temp->zero_device();
      checkCudaError();
      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
	K_prop3D_up->absorbTimeSlice(*K_prop_up,my_fixSinkTime);
	K_prop3D_down->absorbTimeSlice(*K_prop_down,my_fixSinkTime);
      }
      comm_barrier();

      for(int nu = 0 ; nu < 4 ; nu++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  // up //
	  K_temp->zero_device();
	  if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_temp->copyPropagator3D(*K_prop3D_up,my_fixSinkTime,nu,c2);
	  comm_barrier();
	  K_vector->castFloatToDouble(*K_temp);
	  K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	  K_temp->castDoubleToFloat(*K_guess);
	  if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_prop3D_up->absorbVectorTimeSlice(*K_temp,my_fixSinkTime,nu,c2);
	  comm_barrier();
	  K_temp->zero_device();

	// down //
	  if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_temp->copyPropagator3D(*K_prop3D_down,my_fixSinkTime,nu,c2);
	  comm_barrier();
	  K_vector->castFloatToDouble(*K_temp);
	  K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	  K_temp->castDoubleToFloat(*K_guess);
	  if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_prop3D_down->absorbVectorTimeSlice(*K_temp,my_fixSinkTime,nu,c2);
	  comm_barrier();
	  K_temp->zero_device();	
	}
      t2 = MPI_Wtime();
      printfQuda("Time needed to prepare the 3D props for sink-source[%d]=%d is %f sec\n",its,info.tsinkSource[its],t2-t1);


      /////////////////////////////////////////sequential propagator for the part 1
      for(int nu = 0 ; nu < 4 ; nu++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  t1 = MPI_Wtime();
	  K_temp->zero_device();
	  if(NUCLEON == PROTON){
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart1(*K_temp,*K_prop3D_up, *K_prop3D_down, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	  else if(NUCLEON == NEUTRON){
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart1(*K_temp,*K_prop3D_down, *K_prop3D_up, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	  comm_barrier();
	  K_temp->conjugate();
	  K_temp->apply_gamma5();
	  K_vector->castFloatToDouble(*K_temp);
	  //
	  K_vector->scaleVector(1e+10);
	  //
	  K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	  if(NUCLEON == PROTON){
	    b->changeTwist(QUDA_TWIST_MINUS); x->changeTwist(QUDA_TWIST_MINUS); b->Even().changeTwist(QUDA_TWIST_MINUS);
	    b->Odd().changeTwist(QUDA_TWIST_MINUS); x->Even().changeTwist(QUDA_TWIST_MINUS); x->Odd().changeTwist(QUDA_TWIST_MINUS);
	  }
	  else{
	    b->changeTwist(QUDA_TWIST_PLUS); x->changeTwist(QUDA_TWIST_PLUS); b->Even().changeTwist(QUDA_TWIST_PLUS);
	    b->Odd().changeTwist(QUDA_TWIST_PLUS); x->Even().changeTwist(QUDA_TWIST_PLUS); x->Odd().changeTwist(QUDA_TWIST_PLUS);
	  }
	  K_guess->uploadToCuda(b,flag_eo);
	  dirac.prepare(in,out,*x,*b,param->solution_type);
	  
	  cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
	  dirac.Mdag(*in, *tmp);
	  delete tmp;
	  K_vector->downloadFromCuda(in,flag_eo);
	  K_vector->download();
	  if(NUCLEON == PROTON)
	    deflation_down->deflateGuessVector(*K_guess,*K_vector);
	  else if(NUCLEON == NEUTRON)
	    deflation_up->deflateGuessVector(*K_guess,*K_vector);
	  K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
	  (*solve)(*out,*in);
	  dirac.reconstruct(*x,*b,param->solution_type);
	  K_vector->downloadFromCuda(x,flag_eo);
	  if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	    K_vector->scaleVector(2*param->kappa);
	  }
	  //
	  K_vector->scaleVector(1e-10);
	  //
	  K_temp->castDoubleToFloat(*K_vector);
	  K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	  t2 = MPI_Wtime();
	  printfQuda("Inversion for seq prop part 1 = %d,  for source = %d and sink-source = %d finished in time %f sec\n",nu*3+c2,isource,info.tsinkSource[its],t2-t1);
	}

      ////////////////// Contractions for part 1 ////////////////
      t1 = MPI_Wtime();
      if(NUCLEON == PROTON){
	K_contract->contractFixSink(*K_seqProp, *K_prop_up, *K_gaugeContractions, PID, NUCLEON, 1, filename_threep_tsink, isource, info.tsinkSource[its]);
      }
      if(NUCLEON == NEUTRON){
	K_contract->contractFixSink(*K_seqProp, *K_prop_down, *K_gaugeContractions, PID, NUCLEON, 1, filename_threep_tsink, isource, info.tsinkSource[its]);
      }                
      t2 = MPI_Wtime();
      printfQuda("Time for fix sink contractions for part 1 at sink-source = %d is %f sec\n",info.tsinkSource[its],t2-t1);
      /////////////////////////////////////////sequential propagator for the part 2
      for(int nu = 0 ; nu < 4 ; nu++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  t1 = MPI_Wtime();
	  K_temp->zero_device();
	  if(NUCLEON == PROTON){
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart2(*K_temp,*K_prop3D_up, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	  else if(NUCLEON == NEUTRON){
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart2(*K_temp,*K_prop3D_down, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	  comm_barrier();
	  K_temp->conjugate();
	  K_temp->apply_gamma5();
	  K_vector->castFloatToDouble(*K_temp);
	  //
	  K_vector->scaleVector(1e+10);
	  //
	  K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	  if(NUCLEON == PROTON){
	    b->changeTwist(QUDA_TWIST_PLUS); x->changeTwist(QUDA_TWIST_PLUS); b->Even().changeTwist(QUDA_TWIST_PLUS);
	    b->Odd().changeTwist(QUDA_TWIST_PLUS); x->Even().changeTwist(QUDA_TWIST_PLUS); x->Odd().changeTwist(QUDA_TWIST_PLUS);
	  }
	  else{
	    b->changeTwist(QUDA_TWIST_MINUS); x->changeTwist(QUDA_TWIST_MINUS); b->Even().changeTwist(QUDA_TWIST_MINUS);
	    b->Odd().changeTwist(QUDA_TWIST_MINUS); x->Even().changeTwist(QUDA_TWIST_MINUS); x->Odd().changeTwist(QUDA_TWIST_MINUS);
	  }
	  K_guess->uploadToCuda(b,flag_eo);
	  dirac.prepare(in,out,*x,*b,param->solution_type);
	  
	  cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
	  dirac.Mdag(*in, *tmp);
	  delete tmp;
	  K_vector->downloadFromCuda(in,flag_eo);
	  K_vector->download();
	  if(NUCLEON == PROTON)
	    deflation_up->deflateGuessVector(*K_guess,*K_vector);
	  else if(NUCLEON == NEUTRON)
	    deflation_down->deflateGuessVector(*K_guess,*K_vector);
	  K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
	  (*solve)(*out,*in);
	  dirac.reconstruct(*x,*b,param->solution_type);
	  K_vector->downloadFromCuda(x,flag_eo);
	  if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	    K_vector->scaleVector(2*param->kappa);
	  }
	  //
	  K_vector->scaleVector(1e-10);
	  //
	  K_temp->castDoubleToFloat(*K_vector);
	  K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	  t2 = MPI_Wtime();
	  printfQuda("Inversion for seq prop part 2 = %d,  for source = %d and sink-source = %d finished in time %f sec\n",nu*3+c2,isource,info.tsinkSource[its],t2-t1);
	}

      ////////////////// Contractions for part 2 ////////////////
      t1 = MPI_Wtime();
      if(NUCLEON == PROTON)
	K_contract->contractFixSink(*K_seqProp, *K_prop_down, *K_gaugeContractions, PID, NUCLEON, 2, filename_threep_tsink, isource, info.tsinkSource[its]);
      if(NUCLEON == NEUTRON)
	K_contract->contractFixSink(*K_seqProp, *K_prop_up, *K_gaugeContractions, PID, NUCLEON, 2, filename_threep_tsink, isource, info.tsinkSource[its]);
      t2 = MPI_Wtime();

      printfQuda("Time for fix sink contractions for part 2 at sink-source = %d is %f sec\n",info.tsinkSource[its],t2-t1);

    }//-loop over sink-source separations      



  } // close loop over source positions


  free(input_vector);
  free(output_vector);
  delete K_temp;
  delete K_contract;
  delete K_prop_down;
  delete K_prop_up;
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gaugeSmeared;
  delete deflation_up;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete K_gaugeContractions;
  delete K_seqProp;
  delete K_prop3D_up;
  delete K_prop3D_down;

  popVerbosity();
  saveTuneCache(getVerbosity());
  profileInvert.Stop(QUDA_PROFILE_TOTAL);

}
