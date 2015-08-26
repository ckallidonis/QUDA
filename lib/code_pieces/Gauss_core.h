

//#include <core_def.h>

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;


int x_id, y_id, z_id, t_id;
int r1,r2;

r1 = sid/(c_localL[0]);
r2 = r1/(c_localL[1]);
x_id = sid - r1*(c_localL[0]);
y_id = r1 - r2*(c_localL[1]);
t_id = r2/(c_localL[2]);
z_id = r2 - t_id*(c_localL[2]);

// take forward and backward points index
int pointPlus[4];
int pointMinus[4];
int pointMinusG[4];

pointPlus[0] = LEXIC(t_id,z_id,y_id,(x_id+1)%c_localL[0],c_localL); 
pointPlus[1] = LEXIC(t_id,z_id,(y_id+1)%c_localL[1],x_id,c_localL);
pointPlus[2] = LEXIC(t_id,(z_id+1)%c_localL[2],y_id,x_id,c_localL);
pointPlus[3] = LEXIC((t_id+1)%c_localL[3],z_id,y_id,x_id,c_localL);
pointMinus[0] = LEXIC(t_id,z_id,y_id,(x_id-1+c_localL[0])%c_localL[0],c_localL);
pointMinus[1] = LEXIC(t_id,z_id,(y_id-1+c_localL[1])%c_localL[1],x_id,c_localL);
pointMinus[2] = LEXIC(t_id,(z_id-1+c_localL[2])%c_localL[2],y_id,x_id,c_localL);
pointMinus[3] = LEXIC((t_id-1+c_localL[3])%c_localL[3],z_id,y_id,x_id,c_localL);

pointMinusG[0] = pointMinus[0];
pointMinusG[1] = pointMinus[1];
pointMinusG[2] = pointMinus[2];
pointMinusG[3] = pointMinus[3];

// x direction
if(c_dimBreak[0] == true){
  if(x_id == c_localL[0] -1)
    pointPlus[0] = c_plusGhost[0]*c_nSpin*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
  if(x_id == 0){
    pointMinus[0] = c_minusGhost[0]*c_nSpin*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
    pointMinusG[0] = c_minusGhost[0]*c_nDim*c_nColor*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
  }

 }
// y direction
if(c_dimBreak[1] == true){
  if(y_id == c_localL[1] -1)
    pointPlus[1] = c_plusGhost[1]*c_nSpin*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
  if(y_id == 0){
    pointMinus[1] = c_minusGhost[1]*c_nSpin*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
    pointMinusG[1] = c_minusGhost[1]*c_nDim*c_nColor*c_nColor +  LEXIC_TZX(t_id,z_id,x_id,c_localL);
  }

 }
// z direction 
if(c_dimBreak[2] == true){
  if(z_id == c_localL[2] -1)
    pointPlus[2] = c_plusGhost[2]*c_nSpin*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);
  if(z_id == 0){
    pointMinus[2] = c_minusGhost[2]*c_nSpin*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);
    pointMinusG[2] = c_minusGhost[2]*c_nDim*c_nColor*c_nColor +  LEXIC_TYX(t_id,y_id,x_id,c_localL);
  }

 }
// t direction
if(c_dimBreak[3] == true){
  if(t_id == c_localL[3] -1)
    pointPlus[3] = c_plusGhost[3]*c_nSpin*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);
  if(t_id == 0){
    pointMinus[3] = c_minusGhost[3]*c_nSpin*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);
    pointMinusG[3] = c_minusGhost[3]*c_nDim*c_nColor*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);
  }

 }

double2 G_0 , G_1 , G_2 , G_3 , G_4 , G_5 , G_6 , G_7 , G_8;
double2 S_0 , S_1 , S_2 , S_3 , S_4 , S_5 , S_6 , S_7 , S_8 , S_9 , S_10 , S_11;
double2 P1_0 , P1_1 , P1_2 , P1_3 , P1_4 , P1_5 , P1_6 , P1_7 , P1_8 , P1_9 , P1_10 , P1_11;
double2 P2_0 , P2_1 , P2_2 , P2_3 , P2_4 , P2_5 , P2_6 , P2_7 , P2_8 , P2_9 , P2_10 , P2_11;

/////////////////////////////////// mu = 0 

READGAUGE(G,gaugeTexAPE,0,sid,c_stride);
/*
if(sid == 0){
  
  cuPrintf("%e %e\n",G_0.x,G_0.y);
  cuPrintf("%e %e\n",G_1.x,G_1.y);
  cuPrintf("%e %e\n",G_2.x,G_2.y);
  cuPrintf("%e %e\n",G_3.x,G_3.y);
  cuPrintf("%e %e\n",G_4.x,G_4.y);
  cuPrintf("%e %e\n",G_5.x,G_5.y);
  cuPrintf("%e %e\n",G_6.x,G_6.y);
  cuPrintf("%e %e\n",G_7.x,G_7.y);
  cuPrintf("%e %e\n",G_8.x,G_8.y);
}
*/

if(c_dimBreak[0] == true && (x_id == c_localL[0] - 1)){
  READVECTOR(S,vectorTexGauss,pointPlus[0],c_surface[0]);}
 else{
   READVECTOR(S,vectorTexGauss,pointPlus[0],c_stride);}

apply_U_on_S(P1_,G_,S_);

if(c_dimBreak[0] == true && (x_id == 0)){
  READGAUGE(G,gaugeTexAPE,0,pointMinusG[0],c_surface[0]);}
 else{
   READGAUGE(G,gaugeTexAPE,0,pointMinusG[0],c_stride);}

if(c_dimBreak[0] == true && (x_id == 0)){
  READVECTOR(S,vectorTexGauss,pointMinus[0],c_surface[0]);}
 else{
   READVECTOR(S,vectorTexGauss,pointMinus[0],c_stride);}

apply_U_DAG_on_S(P2_,G_,S_);
/*
if(sid == 0){
  
  cuPrintf("%e %e\n",G_0.x,G_0.y);
  cuPrintf("%e %e\n",G_1.x,G_1.y);
  cuPrintf("%e %e\n",G_2.x,G_2.y);
  cuPrintf("%e %e\n",G_3.x,G_3.y);
  cuPrintf("%e %e\n",G_4.x,G_4.y);
  cuPrintf("%e %e\n",G_5.x,G_5.y);
  cuPrintf("%e %e\n",G_6.x,G_6.y);
  cuPrintf("%e %e\n",G_7.x,G_7.y);
  cuPrintf("%e %e\n",G_8.x,G_8.y);

  
  cuPrintf("%e %e\n",S_0.x,S_0.y);
  cuPrintf("%e %e\n",S_1.x,S_1.y);
  cuPrintf("%e %e\n",S_2.x,S_2.y);
  cuPrintf("%e %e\n",S_3.x,S_3.y);
  cuPrintf("%e %e\n",S_4.x,S_4.y);
  cuPrintf("%e %e\n",S_5.x,S_5.y);
  cuPrintf("%e %e\n",S_6.x,S_6.y);
  cuPrintf("%e %e\n",S_7.x,S_7.y);
  cuPrintf("%e %e\n",S_8.x,S_8.y);
  cuPrintf("%e %e\n",S_9.x,S_9.y);
  cuPrintf("%e %e\n",S_10.x,S_10.y);
  cuPrintf("%e %e\n",S_11.x,S_11.y);
    
 }
*/  



  out[0*c_nColor*c_stride + 0*c_stride + sid] = P1_0 + P2_0;
  out[0*c_nColor*c_stride + 1*c_stride + sid] = P1_1 + P2_1;
  out[0*c_nColor*c_stride + 2*c_stride + sid] = P1_2 + P2_2;
  out[1*c_nColor*c_stride + 0*c_stride + sid] = P1_3 + P2_3;
  out[1*c_nColor*c_stride + 1*c_stride + sid] = P1_4 + P2_4;
  out[1*c_nColor*c_stride + 2*c_stride + sid] = P1_5 + P2_5;
  out[2*c_nColor*c_stride + 0*c_stride + sid] = P1_6 + P2_6;
  out[2*c_nColor*c_stride + 1*c_stride + sid] = P1_7 + P2_7;
  out[2*c_nColor*c_stride + 2*c_stride + sid] = P1_8 + P2_8;
  out[3*c_nColor*c_stride + 0*c_stride + sid] = P1_9 + P2_9;
  out[3*c_nColor*c_stride + 1*c_stride + sid] = P1_10 + P2_10;
  out[3*c_nColor*c_stride + 2*c_stride + sid] = P1_11 + P2_11;

/////////////////////////////////// mu = 1

READGAUGE(G,gaugeTexAPE,1,sid,c_stride);

if(c_dimBreak[1] == true && (y_id == c_localL[1] - 1)){
  READVECTOR(S,vectorTexGauss,pointPlus[1],c_surface[1]);}
 else{
   READVECTOR(S,vectorTexGauss,pointPlus[1],c_stride);}

  apply_U_on_S(P1_,G_,S_);

if(c_dimBreak[1] == true && (y_id == 0)){
  READGAUGE(G,gaugeTexAPE,1,pointMinusG[1],c_surface[1]);}
 else{
   READGAUGE(G,gaugeTexAPE,1,pointMinusG[1],c_stride);}

if(c_dimBreak[1] == true && (y_id == 0)){
  READVECTOR(S,vectorTexGauss,pointMinus[1],c_surface[1]);}
 else{
   READVECTOR(S,vectorTexGauss,pointMinus[1],c_stride);}

  apply_U_DAG_on_S(P2_,G_,S_);

  out[0*c_nColor*c_stride + 0*c_stride + sid].x += P1_0.x + P2_0.x;
  out[0*c_nColor*c_stride + 1*c_stride + sid].x += P1_1.x + P2_1.x;
  out[0*c_nColor*c_stride + 2*c_stride + sid].x += P1_2.x + P2_2.x;
  out[1*c_nColor*c_stride + 0*c_stride + sid].x += P1_3.x + P2_3.x;
  out[1*c_nColor*c_stride + 1*c_stride + sid].x += P1_4.x + P2_4.x;
  out[1*c_nColor*c_stride + 2*c_stride + sid].x += P1_5.x + P2_5.x;
  out[2*c_nColor*c_stride + 0*c_stride + sid].x += P1_6.x + P2_6.x;
  out[2*c_nColor*c_stride + 1*c_stride + sid].x += P1_7.x + P2_7.x;
  out[2*c_nColor*c_stride + 2*c_stride + sid].x += P1_8.x + P2_8.x;
  out[3*c_nColor*c_stride + 0*c_stride + sid].x += P1_9.x + P2_9.x;
  out[3*c_nColor*c_stride + 1*c_stride + sid].x += P1_10.x + P2_10.x;
  out[3*c_nColor*c_stride + 2*c_stride + sid].x += P1_11.x + P2_11.x;

  out[0*c_nColor*c_stride + 0*c_stride + sid].y += P1_0.y + P2_0.y;
  out[0*c_nColor*c_stride + 1*c_stride + sid].y += P1_1.y + P2_1.y;
  out[0*c_nColor*c_stride + 2*c_stride + sid].y += P1_2.y + P2_2.y;
  out[1*c_nColor*c_stride + 0*c_stride + sid].y += P1_3.y + P2_3.y;
  out[1*c_nColor*c_stride + 1*c_stride + sid].y += P1_4.y + P2_4.y;
  out[1*c_nColor*c_stride + 2*c_stride + sid].y += P1_5.y + P2_5.y;
  out[2*c_nColor*c_stride + 0*c_stride + sid].y += P1_6.y + P2_6.y;
  out[2*c_nColor*c_stride + 1*c_stride + sid].y += P1_7.y + P2_7.y;
  out[2*c_nColor*c_stride + 2*c_stride + sid].y += P1_8.y + P2_8.y;
  out[3*c_nColor*c_stride + 0*c_stride + sid].y += P1_9.y + P2_9.y;
  out[3*c_nColor*c_stride + 1*c_stride + sid].y += P1_10.y + P2_10.y;
  out[3*c_nColor*c_stride + 2*c_stride + sid].y += P1_11.y + P2_11.y;


///////////////////////////////// mu = 2

READGAUGE(G,gaugeTexAPE,2,sid,c_stride);

if(c_dimBreak[2] == true && (z_id == c_localL[2] - 1)){
  READVECTOR(S,vectorTexGauss,pointPlus[2],c_surface[2]);}
 else{
   READVECTOR(S,vectorTexGauss,pointPlus[2],c_stride);}

  apply_U_on_S(P1_,G_,S_);

if(c_dimBreak[2] == true && (z_id == 0)){
  READGAUGE(G,gaugeTexAPE,2,pointMinusG[2],c_surface[2]);}
 else{
   READGAUGE(G,gaugeTexAPE,2,pointMinusG[2],c_stride);}

if(c_dimBreak[2] == true && (z_id == 0)){
  READVECTOR(S,vectorTexGauss,pointMinus[2],c_surface[2]);}
 else{
   READVECTOR(S,vectorTexGauss,pointMinus[2],c_stride);}

  apply_U_DAG_on_S(P2_,G_,S_);

  out[0*c_nColor*c_stride + 0*c_stride + sid].x += P1_0.x + P2_0.x;
  out[0*c_nColor*c_stride + 1*c_stride + sid].x += P1_1.x + P2_1.x;
  out[0*c_nColor*c_stride + 2*c_stride + sid].x += P1_2.x + P2_2.x;
  out[1*c_nColor*c_stride + 0*c_stride + sid].x += P1_3.x + P2_3.x;
  out[1*c_nColor*c_stride + 1*c_stride + sid].x += P1_4.x + P2_4.x;
  out[1*c_nColor*c_stride + 2*c_stride + sid].x += P1_5.x + P2_5.x;
  out[2*c_nColor*c_stride + 0*c_stride + sid].x += P1_6.x + P2_6.x;
  out[2*c_nColor*c_stride + 1*c_stride + sid].x += P1_7.x + P2_7.x;
  out[2*c_nColor*c_stride + 2*c_stride + sid].x += P1_8.x + P2_8.x;
  out[3*c_nColor*c_stride + 0*c_stride + sid].x += P1_9.x + P2_9.x;
  out[3*c_nColor*c_stride + 1*c_stride + sid].x += P1_10.x + P2_10.x;
  out[3*c_nColor*c_stride + 2*c_stride + sid].x += P1_11.x + P2_11.x;

  out[0*c_nColor*c_stride + 0*c_stride + sid].y += P1_0.y + P2_0.y;
  out[0*c_nColor*c_stride + 1*c_stride + sid].y += P1_1.y + P2_1.y;
  out[0*c_nColor*c_stride + 2*c_stride + sid].y += P1_2.y + P2_2.y;
  out[1*c_nColor*c_stride + 0*c_stride + sid].y += P1_3.y + P2_3.y;
  out[1*c_nColor*c_stride + 1*c_stride + sid].y += P1_4.y + P2_4.y;
  out[1*c_nColor*c_stride + 2*c_stride + sid].y += P1_5.y + P2_5.y;
  out[2*c_nColor*c_stride + 0*c_stride + sid].y += P1_6.y + P2_6.y;
  out[2*c_nColor*c_stride + 1*c_stride + sid].y += P1_7.y + P2_7.y;
  out[2*c_nColor*c_stride + 2*c_stride + sid].y += P1_8.y + P2_8.y;
  out[3*c_nColor*c_stride + 0*c_stride + sid].y += P1_9.y + P2_9.y;
  out[3*c_nColor*c_stride + 1*c_stride + sid].y += P1_10.y + P2_10.y;
  out[3*c_nColor*c_stride + 2*c_stride + sid].y += P1_11.y + P2_11.y;

  ////////////////

  READVECTOR(S,vectorTexGauss,sid,c_stride);

  double normalize;
  normalize = 1./(1. + 6. * c_alphaGauss);

  out[0*c_nColor*c_stride + 0*c_stride + sid] = normalize * (S_0 + c_alphaGauss * out[0*c_nColor*c_stride + 0*c_stride + sid]); 
  out[0*c_nColor*c_stride + 1*c_stride + sid] = normalize * (S_1 + c_alphaGauss * out[0*c_nColor*c_stride + 1*c_stride + sid]); 
  out[0*c_nColor*c_stride + 2*c_stride + sid] = normalize * (S_2 + c_alphaGauss * out[0*c_nColor*c_stride + 2*c_stride + sid]); 
  out[1*c_nColor*c_stride + 0*c_stride + sid] = normalize * (S_3 + c_alphaGauss * out[1*c_nColor*c_stride + 0*c_stride + sid]); 
  out[1*c_nColor*c_stride + 1*c_stride + sid] = normalize * (S_4 + c_alphaGauss * out[1*c_nColor*c_stride + 1*c_stride + sid]); 
  out[1*c_nColor*c_stride + 2*c_stride + sid] = normalize * (S_5 + c_alphaGauss * out[1*c_nColor*c_stride + 2*c_stride + sid]); 
  out[2*c_nColor*c_stride + 0*c_stride + sid] = normalize * (S_6 + c_alphaGauss * out[2*c_nColor*c_stride + 0*c_stride + sid]); 
  out[2*c_nColor*c_stride + 1*c_stride + sid] = normalize * (S_7 + c_alphaGauss * out[2*c_nColor*c_stride + 1*c_stride + sid]); 
  out[2*c_nColor*c_stride + 2*c_stride + sid] = normalize * (S_8 + c_alphaGauss * out[2*c_nColor*c_stride + 2*c_stride + sid]); 
  out[3*c_nColor*c_stride + 0*c_stride + sid] = normalize * (S_9 + c_alphaGauss * out[3*c_nColor*c_stride + 0*c_stride + sid]); 
  out[3*c_nColor*c_stride + 1*c_stride + sid] = normalize * (S_10 + c_alphaGauss * out[3*c_nColor*c_stride + 1*c_stride + sid]); 
  out[3*c_nColor*c_stride + 2*c_stride + sid] = normalize * (S_11 + c_alphaGauss * out[3*c_nColor*c_stride + 2*c_stride + sid]); 
 



