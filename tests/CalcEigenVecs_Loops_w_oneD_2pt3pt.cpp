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

extern void usage(char** );

extern int src[];
extern int Ntsink;
extern char pathList_tsink[];
extern int Q_sq;
extern int nsmearAPE;
extern int nsmearGauss;
extern double alphaAPE;
extern double alphaGauss;
extern char twop_filename[];
extern char threep_filename[];
extern double muValue;
extern double kappa;
extern char prop_path[];
extern double csw;

extern int numSourcePositions;
extern char pathListSourcePositions[];
extern char pathListRun3pt[];
extern char run3pt[];

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

//-C.K. Loop parameters
extern int Nstoch;
extern unsigned long int seed;
extern char loop_fname[];
extern int Ndump;
extern int smethod;


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
  else errorQuda("Error: Your spectrumPart option is suspicious\n");
  //-----------------------------------------------------------------------------------------


  //C.K. Pass loop parameters to loopInfo
  qudaQKXTM_loopInfo loopInfo;

  loopInfo.Nstoch = Nstoch;
  loopInfo.seed = seed;
  loopInfo.Ndump = Ndump;
  loopInfo.smethod = smethod;
  strcpy(loopInfo.loop_fname,loop_fname);
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
  inv_param.twist_flavor = QUDA_TWIST_PLUS;

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

  inv_param.verbosity = QUDA_SILENT;

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
  void *gauge_APE[4];
  void *gaugeContract[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
    gauge_APE[dir] = malloc(V*gaugeSiteSize*gSize);
    gaugeContract[dir] = malloc(V*gaugeSiteSize*gSize);
    if( gauge[dir] == NULL || gauge_APE[dir] == NULL || gaugeContract[dir] == NULL ) errorQuda("error allocate memory host gauge field\n"); 
  }


  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    readLimeGauge(gauge, latfile, &gauge_param, &inv_param, gridsize_from_cmdline);
    applyBoundaryCondition(gauge, V/2 ,&gauge_param);
    for(int mu = 0 ; mu < 4 ; mu++) memcpy(gaugeContract[mu],gauge[mu],V*9*2*sizeof(double));
    mapEvenOddToNormalGauge(gaugeContract,gauge_param,xdim,ydim,zdim,tdim);
  } else { // else generate a random SU(3) field
    construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
  }

  if (strcmp(latfile_smeared,"")) {
    readLimeGaugeSmeared(gauge_APE, latfile_smeared, &gauge_param, &inv_param, gridsize_from_cmdline);        // first read gauge field without apply BC
    mapEvenOddToNormalGauge(gauge_APE,gauge_param,xdim,ydim,zdim,tdim);
  } else { // else generate a random SU(3) field
    construct_gauge_field(gauge_APE, 1, gauge_param.cpu_prec, &gauge_param);
  }



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

  info.nsmearAPE = nsmearAPE;
  info.nsmearGauss = nsmearGauss;
  info.alphaAPE = alphaAPE;
  info.alphaGauss = alphaGauss;
  info.lL[0] = xdim;
  info.lL[1] = ydim;
  info.lL[2] = zdim;
  info.lL[3] = tdim;
  info.Nsources = numSourcePositions;
  info.Q_sq = Q_sq;
  //  info.tsinkSource=t_sinkSource;
  info.Ntsink = Ntsink;


  if(strcmp(run3pt,"all")==0 || strcmp(run3pt,"ALL")==0){
    printfQuda("Will run for all %d source-positions for 2pt- and 3pt- functions\n",numSourcePositions);
    for(int is = 0; is < numSourcePositions; is++){
      info.run3pt_src[is] = 1;
    }
  }
  else if(strcmp(run3pt,"file")==0 || strcmp(run3pt,"FILE")==0){
    printfQuda("Will read from file %s for which source-positions for 3pt- functions to run\n",pathListRun3pt);  
    FILE *ptr_run3pt;
    ptr_run3pt = fopen(pathListRun3pt,"r");
    if(ptr_run3pt == NULL){
      fprintf(stderr,"Error opening file %s \n",pathListRun3pt);
      exit(-1);
    }

    int nRun3pt = 0;
    for(int is = 0; is < numSourcePositions; is++){
      fscanf(ptr_run3pt,"%d\n",&(info.run3pt_src[is]));
      nRun3pt += info.run3pt_src[is];
    }
    printfQuda("Will run for %d source-positions for 3pt-functions\n",nRun3pt);  

    fclose(ptr_run3pt);
  }
  else{
    printfQuda("Option --run3pt only accepts all/ALL and file/FILE parameters, or, if running for all source-positions, just disregard it.\n");
    exit(-1);
  }


  FILE *ptr_sources;
  ptr_sources = fopen(pathListSourcePositions,"r");
  if(ptr_sources == NULL){
    fprintf(stderr,"Error open file to read the source positions\n");
    exit(-1);
  }
  for(int is = 0 ; is < numSourcePositions ; is++)
    fscanf(ptr_sources,"%d %d %d %d",&(info.sourcePosition[is][0]),&(info.sourcePosition[is][1]), &(info.sourcePosition[is][2]), &(info.sourcePosition[is][3]));

  fclose(ptr_sources);


  //-C.Kallidonis: Read in the sink-source separations
  FILE *ptr_tsink;
  ptr_tsink = fopen(pathList_tsink,"r");
  if(ptr_sources == NULL){
    fprintf(stderr,"Error opening file for sink-source separations\n");
    exit(-1);
  }
  for(int it = 0 ; it < Ntsink ; it++){
    fscanf(ptr_tsink,"%d\n",&(info.tsinkSource[it]));
    printfQuda("Got source sink time separation %d: %d\n",it,info.tsinkSource[it]);
  }

  fclose(ptr_tsink);

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


  //  if(isFullOp) calcEigenVectors_loop_wOneD_2pt3pt_FullOp();
  //  else 

  calcEigenVectors_loop_wOneD_2pt3pt_EvenOdd(gauge_APE, gaugeContract, &gauge_param, &inv_param, gaugeContract, &gauge_param, &inv_param,
					     arpackInfo, loopInfo, info, twop_filename, threep_filename, NEUTRON, G4);

  //  if(isFullOp) calcEigenVectors_threepTwop_FullOp(gauge_APE, gaugeContract, &gauge_param, &inv_param, arpackInfo, info, twop_filename, threep_filename, NEUTRON, G4); 
  //  else        calcEigenVectors_threepTwop_EvenOdd(gauge_APE, gaugeContract, &gauge_param, &inv_param, arpackInfo, info, twop_filename, threep_filename, NEUTRON, G4);


  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  for(int i = 0 ; i < 4 ; i++){
    free(gauge_APE[i]);
    free(gaugeContract[i]);
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
