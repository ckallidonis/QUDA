
//#include <core_def.h>

// shared memory for thread reduction
__shared__ double shared_cache[THREADS_PER_BLOCK];

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

// take indices on 4d lattice
int cacheIndex = threadIdx.x;

int x_id, y_id, z_id, t_id;
int r1,r2;

r1 = sid/(c_localL[0]);
r2 = r1/(c_localL[1]);
x_id = sid - r1*(c_localL[0]);
y_id = r1 - r2*(c_localL[1]);
t_id = r2/(c_localL[2]);
z_id = r2 - t_id*(c_localL[2]);


//if(blockIdx.x == 0) cuPrintf("%d %d %d %d\n",x_id,y_id,z_id,t_id);

// take forward and backward points index
int pointPlus[4];
//int pointMinus[4];

pointPlus[0] = LEXIC(t_id,z_id,y_id,(x_id+1)%c_localL[0],c_localL); 
pointPlus[1] = LEXIC(t_id,z_id,(y_id+1)%c_localL[1],x_id,c_localL);
pointPlus[2] = LEXIC(t_id,(z_id+1)%c_localL[2],y_id,x_id,c_localL);
pointPlus[3] = LEXIC((t_id+1)%c_localL[3],z_id,y_id,x_id,c_localL);
//pointMinus[0] = LEXIC(t_id,z_id,y_id,(x_id-1+c_localL[0])%c_localL[0],c_localL);
//pointMinus[1] = LEXIC(t_id,z_id,(y_id-1+c_localL[1])%c_localL[1],x_id,c_localL);
//pointMinus[2] = LEXIC(t_id,(z_id-1+c_localL[2])%c_localL[2],y_id,x_id,c_localL);
//pointMinus[3] = LEXIC((t_id-1+c_localL[3])%c_localL[3],z_id,y_id,x_id,c_localL);



// x direction
if(c_dimBreak[0] == true){

  if(x_id == c_localL[0] -1)
    pointPlus[0] = c_plusGhost[0]*c_nDim*c_nColor*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
  //if(x_id == 0)
  // pointMinus[0] = c_minusGhost[0] + LEXIC_TZY(t_id,z_id,y_id,c_localL);

 }

// y direction
if(c_dimBreak[1] == true){

  if(y_id == c_localL[1] -1)
    pointPlus[1] = c_plusGhost[1]*c_nDim*c_nColor*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
  //if(y_id == 0)
  //  pointMinus[1] = c_minusGhost[1] + LEXIC_TZX(t_id,z_id,x_id,c_localL);

 }

// z direction 
if(c_dimBreak[2] == true){

  if(z_id == c_localL[2] -1)
    pointPlus[2] = c_plusGhost[2]*c_nDim*c_nColor*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);
  //if(z_id == 0)
  //  pointMinus[2] = c_minusGhost[2] + LEXIC_TYX(t_id,y_id,x_id,c_localL);


 }

// t direction
if(c_dimBreak[3] == true){

  if(t_id == c_localL[3] -1)
    pointPlus[3] = c_plusGhost[3]*c_nDim*c_nColor*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);
  //if(t_id == 0)
  // pointMinus[3] = c_minusGhost[3] + LEXIC_ZYX(z_id,y_id,x_id,c_localL);


 }



double2 G1_0 , G1_1 , G1_2 , G1_3 , G1_4 , G1_5 , G1_6 , G1_7 , G1_8;
double2 G2_0 , G2_1 , G2_2 , G2_3 , G2_4 , G2_5 , G2_6 , G2_7 , G2_8;
double2 G3_0 , G3_1 , G3_2 , G3_3 , G3_4 , G3_5 , G3_6 , G3_7 , G3_8;
double2 G4_0 , G4_1 , G4_2 , G4_3 , G4_4 , G4_5 , G4_6 , G4_7 , G4_8;
double trace = 0.;

// term1 trace[U^0_x * U^1_{x+0} * U^{0+}_{x+1} * U^{1+}_x]
READGAUGE(G1,gaugeTexPlaq,0,sid,c_stride);

/*
if(sid == gridDim.x*blockDim.x -1){
  cuPrintf("%e %e\n",G1_0.x,G1_0.y);
  cuPrintf("%e %e\n",G1_1.x,G1_1.y);
  cuPrintf("%e %e\n",G1_2.x,G1_2.y);
  cuPrintf("%e %e\n",G1_3.x,G1_3.y);
  cuPrintf("%e %e\n",G1_4.x,G1_4.y);
  cuPrintf("%e %e\n",G1_5.x,G1_5.y);
  cuPrintf("%e %e\n",G1_6.x,G1_6.y);
  cuPrintf("%e %e\n",G1_7.x,G1_7.y);
  cuPrintf("%e %e\n",G1_8.x,G1_8.y);
  }
*/

if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G2,gaugeTexPlaq,1,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G2,gaugeTexPlaq,1,pointPlus[0],c_stride);}

MUL3X3(g3_,g1_,g2_);


if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G1,gaugeTexPlaq,0,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G1,gaugeTexPlaq,0,pointPlus[1],c_stride);}
READGAUGE(G2,gaugeTexPlaq,1,sid,c_stride);

MUL3X3(g4_,g1_T,g2_T);

trace += TRACEMUL3X3(g3_,g4_);

//if(sid == 0)cuPrintf("%e\n",trace);

// term2 trace[U^0_x * U^2_{x+0} * U^{0+}_{x+2} * U^{2+}_x]
READGAUGE(G1,gaugeTexPlaq,0,sid,c_stride);

if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G2,gaugeTexPlaq,2,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G2,gaugeTexPlaq,2,pointPlus[0],c_stride);}

MUL3X3(g3_,g1_,g2_);

if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G1,gaugeTexPlaq,0,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G1,gaugeTexPlaq,0,pointPlus[2],c_stride);}
READGAUGE(G2,gaugeTexPlaq,2,sid,c_stride);

MUL3X3(g4_,g1_T,g2_T);

trace += TRACEMUL3X3(g3_,g4_);
//if(sid == 0)cuPrintf("%e\n",trace);
// term3 trace[U^0_x * U^3_{x+0} * U^{0+}_{x+3} * U^{3+}_x]
READGAUGE(G1,gaugeTexPlaq,0,sid,c_stride);
if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G2,gaugeTexPlaq,3,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G2,gaugeTexPlaq,3,pointPlus[0],c_stride);}

MUL3X3(g3_,g1_,g2_);

if(c_dimBreak[3] == true &&(t_id == c_localL[3]-1)){
  READGAUGE(G1,gaugeTexPlaq,0,pointPlus[3],c_surface[3]);}
 else{
   READGAUGE(G1,gaugeTexPlaq,0,pointPlus[3],c_stride);}
READGAUGE(G2,gaugeTexPlaq,3,sid,c_stride);

MUL3X3(g4_,g1_T,g2_T);

trace += TRACEMUL3X3(g3_,g4_);
//if(sid == 0)cuPrintf("%e\n",trace);
// term4 trace[U^1_x * U^2_{x+1} * U^{1+}_{x+2} * U^{2+}_x]
READGAUGE(G1,gaugeTexPlaq,1,sid,c_stride);
if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G2,gaugeTexPlaq,2,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G2,gaugeTexPlaq,2,pointPlus[1],c_stride);}

MUL3X3(g3_,g1_,g2_);

if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G1,gaugeTexPlaq,1,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G1,gaugeTexPlaq,1,pointPlus[2],c_stride);}
READGAUGE(G2,gaugeTexPlaq,2,sid,c_stride);

MUL3X3(g4_,g1_T,g2_T);

trace += TRACEMUL3X3(g3_,g4_);
//if(sid == 0)cuPrintf("%e\n",trace);
// term5 trace[U^1_x * U^3_{x+1} * U^{1+}_{x+3} * U^{3+}_x]
READGAUGE(G1,gaugeTexPlaq,1,sid,c_stride);
if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G2,gaugeTexPlaq,3,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G2,gaugeTexPlaq,3,pointPlus[1],c_stride);}

MUL3X3(g3_,g1_,g2_);

if(c_dimBreak[3] == true && (t_id == c_localL[3]-1)){
  READGAUGE(G1,gaugeTexPlaq,1,pointPlus[3],c_surface[3]);}
 else{
   READGAUGE(G1,gaugeTexPlaq,1,pointPlus[3],c_stride);}
READGAUGE(G2,gaugeTexPlaq,3,sid,c_stride);

MUL3X3(g4_,g1_T,g2_T);

trace += TRACEMUL3X3(g3_,g4_);
//if(sid == 0)cuPrintf("%e\n",trace);
// term6 trace[U^2_x * U^3_{x+2} * U^{2+}_{x+3} * U^{3+}_x]
READGAUGE(G1,gaugeTexPlaq,2,sid,c_stride);
if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G2,gaugeTexPlaq,3,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G2,gaugeTexPlaq,3,pointPlus[2],c_stride);}

MUL3X3(g3_,g1_,g2_);

if(c_dimBreak[3] == true && (t_id == c_localL[3]-1)){
  READGAUGE(G1,gaugeTexPlaq,2,pointPlus[3],c_surface[3]);}
 else{
   READGAUGE(G1,gaugeTexPlaq,2,pointPlus[3],c_stride);}
READGAUGE(G2,gaugeTexPlaq,3,sid,c_stride);

MUL3X3(g4_,g1_T,g2_T);

trace += TRACEMUL3X3(g3_,g4_);
//if(sid == 0)cuPrintf("%e\n",trace);

//if(blockIdx.x == 0)cuPrintf("%d %d %d %d %d \t %e\n",sid,x_id,y_id,z_id,t_id,trace);
//if(blockIdx.x == 0) cuPrintf("%d %d %d %d\n",x_id,y_id,z_id,t_id);

// now each thread write the result in share memory
shared_cache[cacheIndex] = trace;



__syncthreads(); // synchronize threads to be sure that all have written their register trace to share memory

// for reduction threads per block must be power of 2 ( this is always my case)

int i = blockDim.x/2;

while (i != 0){

  if(cacheIndex < i)
    shared_cache[cacheIndex] += shared_cache[cacheIndex + i];

  __syncthreads();
  i /= 2;

 }

// now on the first element of the shared memory we have the reduction of block threads
if(cacheIndex == 0)
  partial_plaq[blockIdx.x] = shared_cache[0];   // write result back to global memory
