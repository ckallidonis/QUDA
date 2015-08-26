# 1 "Gauss_core.h"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "Gauss_core.h"


# 1 "./core_def.h" 1
# 4 "Gauss_core.h" 2

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


if(c_dimBreak[0] == true){
  if(x_id == c_localL[0] -1)
    pointPlus[0] = c_plusGhost[0]*c_nSpin*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
  if(x_id == 0){
    pointMinus[0] = c_minusGhost[0]*c_nSpin*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
    pointMinusG[0] = c_minusGhost[0]*c_nDim*c_nColor*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
  }

 }

if(c_dimBreak[1] == true){
  if(y_id == c_localL[1] -1)
    pointPlus[1] = c_plusGhost[1]*c_nSpin*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
  if(y_id == 0){
    pointMinus[1] = c_minusGhost[1]*c_nSpin*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
    pointMinusG[1] = c_minusGhost[1]*c_nDim*c_nColor*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
  }

 }

if(c_dimBreak[2] == true){
  if(z_id == c_localL[2] -1)
    pointPlus[2] = c_plusGhost[2]*c_nSpin*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);
  if(z_id == 0){
    pointMinus[2] = c_minusGhost[2]*c_nSpin*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);
    pointMinusG[2] = c_minusGhost[2]*c_nDim*c_nColor*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
  }

 }

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



G_0 = fetch_double2(gaugeTexAPE, sid + (0*9 + 0 ) * c_stride); G_1 = fetch_double2(gaugeTexAPE, sid + (0*9 + 1 ) * c_stride); G_2 = fetch_double2(gaugeTexAPE, sid + (0*9 + 2 ) * c_stride); G_3 = fetch_double2(gaugeTexAPE, sid + (0*9 + 3 ) * c_stride); G_4 = fetch_double2(gaugeTexAPE, sid + (0*9 + 4 ) * c_stride); G_5 = fetch_double2(gaugeTexAPE, sid + (0*9 + 5 ) * c_stride); G_6 = fetch_double2(gaugeTexAPE, sid + (0*9 + 6 ) * c_stride); G_7 = fetch_double2(gaugeTexAPE, sid + (0*9 + 7 ) * c_stride); G_8 = fetch_double2(gaugeTexAPE, sid + (0*9 + 8 ) * c_stride);;

if(c_dimBreak[0] == true && (x_id == c_localL[0] - 1)){
  S_0 = fetch_double2(vectorTexGauss, pointPlus[0] + 0 * c_surface[0]); S_1 = fetch_double2(vectorTexGauss, pointPlus[0] + 1 * c_surface[0]); S_2 = fetch_double2(vectorTexGauss, pointPlus[0] + 2 * c_surface[0]); S_3 = fetch_double2(vectorTexGauss, pointPlus[0] + 3 * c_surface[0]); S_4 = fetch_double2(vectorTexGauss, pointPlus[0] + 4 * c_surface[0]); S_5 = fetch_double2(vectorTexGauss, pointPlus[0] + 5 * c_surface[0]); S_6 = fetch_double2(vectorTexGauss, pointPlus[0] + 6 * c_surface[0]); S_7 = fetch_double2(vectorTexGauss, pointPlus[0] + 7 * c_surface[0]); S_8 = fetch_double2(vectorTexGauss, pointPlus[0] + 8 * c_surface[0]); S_9 = fetch_double2(vectorTexGauss, pointPlus[0] + 9 * c_surface[0]); S_10 = fetch_double2(vectorTexGauss, pointPlus[0] + 10 * c_surface[0]); S_11 = fetch_double2(vectorTexGauss, pointPlus[0] + 11 * c_surface[0]);;}
 else{
   S_0 = fetch_double2(vectorTexGauss, pointPlus[0] + 0 * c_stride); S_1 = fetch_double2(vectorTexGauss, pointPlus[0] + 1 * c_stride); S_2 = fetch_double2(vectorTexGauss, pointPlus[0] + 2 * c_stride); S_3 = fetch_double2(vectorTexGauss, pointPlus[0] + 3 * c_stride); S_4 = fetch_double2(vectorTexGauss, pointPlus[0] + 4 * c_stride); S_5 = fetch_double2(vectorTexGauss, pointPlus[0] + 5 * c_stride); S_6 = fetch_double2(vectorTexGauss, pointPlus[0] + 6 * c_stride); S_7 = fetch_double2(vectorTexGauss, pointPlus[0] + 7 * c_stride); S_8 = fetch_double2(vectorTexGauss, pointPlus[0] + 8 * c_stride); S_9 = fetch_double2(vectorTexGauss, pointPlus[0] + 9 * c_stride); S_10 = fetch_double2(vectorTexGauss, pointPlus[0] + 10 * c_stride); S_11 = fetch_double2(vectorTexGauss, pointPlus[0] + 11 * c_stride);;}

P1_0 = G_0 * S_0 + G_1 * S_1 + G_2 * S_2; P1_1 = G_3 * S_0 + G_4 * S_1 + G_5 * S_2; P1_2 = G_6 * S_0 + G_7 * S_1 + G_8 * S_2; P1_3 = G_0 * S_3 + G_1 * S_4 + G_2 * S_5; P1_4 = G_3 * S_3 + G_4 * S_4 + G_5 * S_5; P1_5 = G_6 * S_3 + G_7 * S_4 + G_8 * S_5; P1_6 = G_0 * S_6 + G_1 * S_7 + G_2 * S_8; P1_7 = G_3 * S_6 + G_4 * S_7 + G_5 * S_8; P1_8 = G_6 * S_6 + G_7 * S_7 + G_8 * S_8; P1_9 = G_0 * S_9 + G_1 * S_10 + G_2 * S_11; P1_10 = G_3 * S_9 + G_4 * S_10 + G_5 * S_11; P1_11 = G_6 * S_9 + G_7 * S_10 + G_8 * S_11;;

if(c_dimBreak[0] == true && (x_id == 0)){
  G_0 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 0 ) * c_surface[0]); G_1 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 1 ) * c_surface[0]); G_2 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 2 ) * c_surface[0]); G_3 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 3 ) * c_surface[0]); G_4 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 4 ) * c_surface[0]); G_5 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 5 ) * c_surface[0]); G_6 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 6 ) * c_surface[0]); G_7 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 7 ) * c_surface[0]); G_8 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 8 ) * c_surface[0]);;}
 else{
   G_0 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 0 ) * c_stride); G_1 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 1 ) * c_stride); G_2 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 2 ) * c_stride); G_3 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 3 ) * c_stride); G_4 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 4 ) * c_stride); G_5 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 5 ) * c_stride); G_6 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 6 ) * c_stride); G_7 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 7 ) * c_stride); G_8 = fetch_double2(gaugeTexPlaq, pointMinusG[0] + (0*9 + 8 ) * c_stride);;}

if(c_dimBreak[0] == true && (x_id == 0)){
  S_0 = fetch_double2(vectorTexGauss, pointMinus[0] + 0 * c_surface[0]); S_1 = fetch_double2(vectorTexGauss, pointMinus[0] + 1 * c_surface[0]); S_2 = fetch_double2(vectorTexGauss, pointMinus[0] + 2 * c_surface[0]); S_3 = fetch_double2(vectorTexGauss, pointMinus[0] + 3 * c_surface[0]); S_4 = fetch_double2(vectorTexGauss, pointMinus[0] + 4 * c_surface[0]); S_5 = fetch_double2(vectorTexGauss, pointMinus[0] + 5 * c_surface[0]); S_6 = fetch_double2(vectorTexGauss, pointMinus[0] + 6 * c_surface[0]); S_7 = fetch_double2(vectorTexGauss, pointMinus[0] + 7 * c_surface[0]); S_8 = fetch_double2(vectorTexGauss, pointMinus[0] + 8 * c_surface[0]); S_9 = fetch_double2(vectorTexGauss, pointMinus[0] + 9 * c_surface[0]); S_10 = fetch_double2(vectorTexGauss, pointMinus[0] + 10 * c_surface[0]); S_11 = fetch_double2(vectorTexGauss, pointMinus[0] + 11 * c_surface[0]);;}
 else{
   S_0 = fetch_double2(vectorTexGauss, pointMinus[0] + 0 * c_stride); S_1 = fetch_double2(vectorTexGauss, pointMinus[0] + 1 * c_stride); S_2 = fetch_double2(vectorTexGauss, pointMinus[0] + 2 * c_stride); S_3 = fetch_double2(vectorTexGauss, pointMinus[0] + 3 * c_stride); S_4 = fetch_double2(vectorTexGauss, pointMinus[0] + 4 * c_stride); S_5 = fetch_double2(vectorTexGauss, pointMinus[0] + 5 * c_stride); S_6 = fetch_double2(vectorTexGauss, pointMinus[0] + 6 * c_stride); S_7 = fetch_double2(vectorTexGauss, pointMinus[0] + 7 * c_stride); S_8 = fetch_double2(vectorTexGauss, pointMinus[0] + 8 * c_stride); S_9 = fetch_double2(vectorTexGauss, pointMinus[0] + 9 * c_stride); S_10 = fetch_double2(vectorTexGauss, pointMinus[0] + 10 * c_stride); S_11 = fetch_double2(vectorTexGauss, pointMinus[0] + 11 * c_stride);;}

P2_0 = conj(G_0) * S_0 + conj(G_3) * S_1 + conj(G_6) * S_2; P2_1 = conj(G_1) * S_0 + conj(G_4) * S_1 + conj(G_7) * S_2; P2_2 = conj(G_2) * S_0 + conj(G_5) * S_1 + conj(G_8) * S_2; P2_3 = conj(G_0) * S_3 + conj(G_3) * S_4 + conj(G_6) * S_5; P2_4 = conj(G_1) * S_3 + conj(G_4) * S_4 + conj(G_7) * S_5; P2_5 = conj(G_2) * S_3 + conj(G_5) * S_4 + conj(G_8) * S_5; P2_6 = conj(G_0) * S_6 + conj(G_3) * S_7 + conj(G_6) * S_8; P2_7 = conj(G_1) * S_6 + conj(G_4) * S_7 + conj(G_7) * S_8; P2_8 = conj(G_2) * S_6 + conj(G_5) * S_7 + conj(G_8) * S_8; P2_9 = conj(G_0) * S_9 + conj(G_3) * S_10 + conj(G_6) * S_11; P2_10 = conj(G_1) * S_9 + conj(G_4) * S_10 + conj(G_7) * S_11; P2_11 = conj(G_2) * S_9 + conj(G_4) * S_10 + conj(G_8) * S_11;;

if(sid == 0){
  cuPrintf("%e %e\n",S_0.x,S_0.y);
  cuPrintf("%e %e\n",P1_0.x,P1_0.y);
  cuPrintf("%e %e\n",P2_0.x,P2_0.y);
 }
