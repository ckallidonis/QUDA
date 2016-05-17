#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include "face_quda.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <gauge_qio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <contractQuda.h>
#include <qudaQKXTM_Kepler.h>

// Wilson, clover-improved Wilson, twisted mass, and domain wall are supported.
extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;
extern QudaInverterType  inv_type;
extern QudaInverterType  precon_type;
extern int multishift; // whether to test multi-shift or standard solver
extern double mass; // mass of Dirac operator

extern char latfile[];
extern char latfile_smeared[];
extern char verbosity_level[];
extern int traj;

extern void usage(char** );

extern int Q_sq;
extern double muValue;
extern double kappa;
extern char prop_path[];
extern double csw;

//-C.K. Loop parameters
extern int Nstoch;
extern unsigned long int seed;
extern char loop_fname[];
extern char loop_file_format[];
extern int Ndump;
extern int smethod;
extern char filename_dSteps[];
extern bool useTSM;
extern int TSM_NHP;
extern int TSM_NLP;
extern int TSM_NdumpHP;
extern int TSM_NdumpLP;
extern int TSM_maxiter;
extern double TSM_tol;

//-C.K. ARPACK Parameters
extern int PolyDeg;
extern int nEv;
extern int nKv;
extern char *spectrumPart;
extern bool isACC;
extern double tolArpack;
extern int maxIterArpack;
extern char arpack_logfile[];
extern double amin;
extern double amax;
extern bool isEven;
extern bool isFullOp;

extern char source_type[];

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;
  
}

int main(int argc, char **argv)
{
  using namespace quda;

  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION){
    prec_sloppy = prec;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    link_recon_sloppy = link_recon;
  }

  // initialize QMP or MPI
#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  if ( dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_TWISTED_CLOVER_DSLASH 
       && dslash_type != QUDA_CLOVER_WILSON_DSLASH){
    printfQuda("This test is only for twisted mass or twisted clover operator\n");
    exit(-1);
  }


  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = prec;
  QudaPrecision cuda_prec_sloppy = prec_sloppy;
  QudaPrecision cuda_prec_precondition = QUDA_HALF_PRECISION;


  //-C.K. Pass ARPACK parameters to arpackInfo
  qudaQKXTM_arpackInfo arpackInfo;

  arpackInfo.PolyDeg = PolyDeg;
  arpackInfo.nEv = nEv;
  arpackInfo.nKv = nKv;
  arpackInfo.isACC = isACC;
  arpackInfo.tolArpack = tolArpack;
  arpackInfo.maxIterArpack = maxIterArpack;
  strcpy(arpackInfo.arpack_logfile,arpack_logfile);
  arpackInfo.amin = amin;
  arpackInfo.amax = amax;
  arpackInfo.isEven = isEven;
  arpackInfo.isFullOp = isFullOp;

  if(strcmp(spectrumPart,"SR")==0) arpackInfo.spectrumPart = SR;
  else if(strcmp(spectrumPart,"LR")==0) arpackInfo.spectrumPart = LR;
  else if(strcmp(spectrumPart,"SM")==0) arpackInfo.spectrumPart = SM;
  else if(strcmp(spectrumPart,"LM")==0) arpackInfo.spectrumPart = LM;
  else if(strcmp(spectrumPart,"SI")==0) arpackInfo.spectrumPart = SI;
  else if(strcmp(spectrumPart,"LI")==0) arpackInfo.spectrumPart = LI;
  else{
    printf("Error: Your spectrumPart option is suspicious\n");
    exit(-1);
  }
  //-----------------------------------------------------------------------------------------

  QudaGaugeParam gauge_param = newQudaGaugeParam();

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = 1.0;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  //  gauge_param.t_boundary = QUDA_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_sloppy;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.ga_pad = 0; // 24*24*24/2;
  //-----------------------------------------------------------------------------------------

  QudaInvertParam inv_param = newQudaInvertParam();

  inv_param.Ls = 1;

  inv_param.dslash_type = dslash_type;
  inv_param.kappa = kappa;
  inv_param.mu=muValue;
  inv_param.epsilon = 0.;
  inv_param.twist_flavor = QUDA_TWIST_MINUS;

  double kappa5 = 1.;

  // offsets used only by multi-shift solver
  inv_param.num_offset = 4;
  double offset[4] = {0.01, 0.02, 0.03, 0.04};
  for (int i=0; i<inv_param.num_offset; i++) inv_param.offset[i] = offset[i];

  inv_param.inv_type = inv_type;

  if(isEven) inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  else inv_param.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;

  inv_param.solution_type = QUDA_MAT_SOLUTION;
  if(isFullOp){
    inv_param.solve_type = QUDA_NORMOP_SOLVE;
  }
  else{
    inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

  inv_param.pipeline = 0;

  double precision = 1e-9;

  inv_param.Nsteps = 2;
  inv_param.gcrNkrylov = 10;
  inv_param.tol = precision;
  inv_param.tol_restart = 1e-3; //now theoretical background for this parameter...

#if __COMPUTE_CAPABILITY__ >= 200
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL | QUDA_HEAVY_QUARK_RESIDUAL);
  inv_param.tol_hq = precision; // specify a tolerance for the residual for heavy quark residual
#else
  // Pre Fermi architecture only supports L2 relative residual norm
  inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
#endif
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = 50000;
  inv_param.reliable_delta = 1e-2;
  inv_param.use_sloppy_partial_accumulator = 0;
  inv_param.max_res_increase = 1;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = precon_type;
    
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_SILENT;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  gauge_param.ga_pad = 0; // 24*24*24/2;
  inv_param.sp_pad = 0; // 24*24*24/2;
  inv_param.cl_pad = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = csw*inv_param.kappa;
  }

  if(strcmp(verbosity_level,"verbose")==0)   inv_param.verbosity = QUDA_VERBOSE;
  else if(strcmp(verbosity_level,"summarize")==0) inv_param.verbosity = QUDA_SUMMARIZE;
  else if(strcmp(verbosity_level,"silent")==0)    inv_param.verbosity = QUDA_SILENT;
  else{
    warningQuda("Unknown verbosity level %s. Proceeding with QUDA_SUMMARIZE verbosity level\n",verbosity_level);
    inv_param.verbosity = QUDA_SUMMARIZE;
  }

  // declare the dimensions of the communication grid
  initCommsGridQuda(4, gridsize_from_cmdline, NULL, NULL);

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded

  setDims(gauge_param.X);
  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover_inv=0, *clover=0;
  void *gauge_Plaq[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
    gauge_Plaq[dir] = malloc(V*gaugeSiteSize*gSize);
  }


  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    sprintf(latfile,"%s.%04d",latfile,traj);    
    readLimeGauge(gauge, latfile, &gauge_param, &inv_param, gridsize_from_cmdline);
    for(int mu = 0 ; mu < 4 ; mu++)memcpy(gauge_Plaq[mu],gauge[mu],V*9*2*sizeof(double));
    mapEvenOddToNormalGauge(gauge_Plaq,gauge_param,xdim,ydim,zdim,tdim);
    applyBoundaryCondition(gauge, V/2 ,&gauge_param);
  } else { // else generate a random SU(3) field
    construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
  }

  //  if (strcmp(latfile_smeared,"")) {
    //readLimeGaugeSmeared(gauge_APE, latfile_smeared, &gauge_param, &inv_param, gridsize_from_cmdline);        // first read gauge field without apply BC
    //mapEvenOddToNormalGauge(gauge_APE,gauge_param,xdim,ydim,zdim,tdim);
  //} else { // else generate a random SU(3) field
    // construct_gauge_field(gauge_APE, 1, gauge_param.cpu_prec, &gauge_param);
  //}


  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = (inv_param.clover_cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    construct_clover_field(clover_inv, norm, diag, inv_param.clover_cpu_prec);


    // The uninverted clover term is only needed when solving the unpreconditioned
    // system or when using "asymmetric" even/odd preconditioning.
    int preconditioned = (inv_param.solve_type == QUDA_DIRECT_PC_SOLVE ||
			  inv_param.solve_type == QUDA_NORMOP_PC_SOLVE);
    int asymmetric = preconditioned &&
                         (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
                          inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);
    if (!preconditioned) {
      clover = clover_inv;
      clover_inv = NULL;
    } else if (asymmetric) { // fake it by using the same random matrix
      clover = clover_inv;   // for both clover and clover_inv
    } else {
      clover = NULL;
    }
  }


  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library

  qudaQKXTMinfo_Kepler info;

  info.lL[0] = xdim;
  info.lL[1] = ydim;
  info.lL[2] = zdim;
  info.lL[3] = tdim;
  info.Q_sq = Q_sq;
  if( strcmp(source_type,"random")==0 ) info.source_type = RANDOM;
  else if( strcmp(source_type,"unity")==0 ) info.source_type = UNITY;
  else{
    printf("Wrong type for stochastic source type. Must be either random/unity. Exiting.\n");
    exit(1);
  }

  initQuda(device);
  init_qudaQKXTM_Kepler(&info);
  printf_qudaQKXTM_Kepler();



  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  for(int i = 0 ; i < 4 ; i++){
    free(gauge[i]);
  } 

  // load the clover term, if desired
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);

  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(NULL, NULL, &inv_param);
  //if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(clover,clover_inv, &inv_param);


  //-C.K. Pass loop parameters to loopInfo
  qudaQKXTM_loopInfo loopInfo;

  loopInfo.Nstoch = Nstoch;
  loopInfo.seed = seed;
  loopInfo.Ndump = Ndump;
  loopInfo.traj = traj;
  loopInfo.Qsq = Q_sq;
  loopInfo.smethod = smethod;
  strcpy(loopInfo.file_format,loop_file_format);
  strcpy(loopInfo.loop_fname,loop_fname);

  if(loopInfo.Nstoch%loopInfo.Ndump==0) loopInfo.Nprint = loopInfo.Nstoch/loopInfo.Ndump;
  else errorQuda("NdumpStep MUST divide Nstoch exactly! Exiting.\n");

  if(strcmp(filename_dSteps,"none")==0){
    loopInfo.nSteps_defl = 1;
    loopInfo.deflStep[0] = nEv;
  }
  else{
    FILE *ptr_dstep;    
    if( (ptr_dstep = fopen(filename_dSteps,"r"))==NULL ){
      fprintf(stderr,"Cannot open %s for reading. Exiting\n",filename_dSteps);
      exit(-1);
    }
    fscanf(ptr_dstep,"%d\n",&loopInfo.nSteps_defl);
    fscanf(ptr_dstep,"%d\n",&loopInfo.deflStep[0]);
    if(loopInfo.deflStep[0]>nEv){
      printf("ERROR: Supplied deflation step is larger than eigenvalues requested. Exiting.\n");
      exit(-1);
    }
    for(int s=1;s<loopInfo.nSteps_defl;s++){
      fscanf(ptr_dstep,"%d\n",&loopInfo.deflStep[s]);
      if(loopInfo.deflStep[s]<loopInfo.deflStep[s-1]){
	printf("ERROR: Deflation steps MUST be in ascending order. Exiting.\n");
	exit(-1);
      }
      if(loopInfo.deflStep[s]>nEv){
	printf("WARNING: Supplied deflation step %d is larger than eigenvalues requested. Discarding this step.\n",s);
	s--;
	loopInfo.nSteps_defl--;
      }
    }
    fclose(ptr_dstep);

    //- This is to always make sure that the total number of eigenvalues is included
    if(loopInfo.deflStep[loopInfo.nSteps_defl-1] != nEv){
      loopInfo.nSteps_defl++;
      loopInfo.deflStep[loopInfo.nSteps_defl-1] = nEv;
    }
  }

  //- TSM parameters
  loopInfo.useTSM = useTSM;
  if(useTSM){
    loopInfo.TSM_NHP = TSM_NHP;
    loopInfo.TSM_NLP = TSM_NLP;
    loopInfo.TSM_NdumpHP = TSM_NdumpHP;
    loopInfo.TSM_NdumpLP = TSM_NdumpLP;

    if(loopInfo.TSM_NHP%loopInfo.TSM_NdumpHP==0) loopInfo.TSM_NprintHP = loopInfo.TSM_NHP/loopInfo.TSM_NdumpHP;
    else errorQuda("TSM_NdumpHP MUST divide TSM_NHP exactly! Exiting.\n");
    if(loopInfo.TSM_NLP%loopInfo.TSM_NdumpLP==0) loopInfo.TSM_NprintLP = loopInfo.TSM_NLP/loopInfo.TSM_NdumpLP;
    else errorQuda("TSM_NdumpLP MUST divide TSM_NLP exactly! Exiting.\n");

    loopInfo.TSM_tol = TSM_tol;
    loopInfo.TSM_maxiter = TSM_maxiter;
    if( (TSM_maxiter==0) && (TSM_tol==0) ) errorQuda("Criterion for low-precision sources not set!\n");
    if(TSM_tol!=0) errorQuda("Setting the tolerance as low-precision criterion for Truncated Solver method not supported! Re-run using --TSM_maxiter <iter> as criterion.\n");
  }
  //-----------------------------------------------------------------------------------------


  if(isFullOp) calcEigenVectors_loop_wOneD_FullOp(gauge_Plaq, &inv_param, &gauge_param, arpackInfo, loopInfo, info);
  else         calcEigenVectors_loop_wOneD_EvenOdd(gauge_Plaq, &inv_param, &gauge_param, arpackInfo, loopInfo, info);
  
  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  for(int i = 0 ; i < 4 ; i++){
    free(gauge_Plaq[i]);
  }
  
  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif

  return 0;
}
