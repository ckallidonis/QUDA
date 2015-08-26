
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




// take forward points
int pointPlus[4];
int pointMinus[4];

pointPlus[0] = LEXIC(t_id,z_id,y_id,(x_id+1)%c_localL[0],c_localL); 
pointPlus[1] = LEXIC(t_id,z_id,(y_id+1)%c_localL[1],x_id,c_localL);
pointPlus[2] = LEXIC(t_id,(z_id+1)%c_localL[2],y_id,x_id,c_localL);
pointPlus[3] = LEXIC((t_id+1)%c_localL[3],z_id,y_id,x_id,c_localL);

pointMinus[0] = LEXIC(t_id,z_id,y_id,(x_id-1+c_localL[0])%c_localL[0],c_localL); 
pointMinus[1] = LEXIC(t_id,z_id,(y_id-1+c_localL[1])%c_localL[1],x_id,c_localL);
pointMinus[2] = LEXIC(t_id,(z_id-1+c_localL[2])%c_localL[2],y_id,x_id,c_localL);
pointMinus[3] = LEXIC((t_id-1+c_localL[3])%c_localL[3],z_id,y_id,x_id,c_localL);


// x direction
if(c_dimBreak[0] == true)
  if(x_id == c_localL[0] -1){
    pointPlus[0] = c_plusGhost[0]*c_nDim*c_nColor*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
  }

// y direction
if(c_dimBreak[1] == true)
  if(y_id == c_localL[1] -1){
    pointPlus[1] = c_plusGhost[1]*c_nDim*c_nColor*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
  }
// z direction 
if(c_dimBreak[2] == true)
  if(z_id == c_localL[2] -1){
    pointPlus[2] = c_plusGhost[2]*c_nDim*c_nColor*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);
  }
// t direction
if(c_dimBreak[3] == true)
  if(t_id == c_localL[3] -1){
    pointPlus[3] = c_plusGhost[3]*c_nDim*c_nColor*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);
  }

// x direction
if(c_dimBreak[0] == true)
  if(x_id == 0)
    pointMinus[0] = c_minusGhost[0]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);

// y direction
if(c_dimBreak[1] == true) 
  if(y_id == 0)
    pointMinus[1] = c_minusGhost[1]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);

// z direction 
if(c_dimBreak[2] == true)
  if(z_id == 0)
    pointMinus[2] = c_minusGhost[2]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);

// t direction
if(c_dimBreak[3] == true)
  if(t_id == 0)
    pointMinus[3] = c_minusGhost[3]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);

// because now we calculate staples we need 3 inputs
double2 G1_0 , G1_1 , G1_2 , G1_3 , G1_4 , G1_5 , G1_6 , G1_7 , G1_8;
double2 G2_0 , G2_1 , G2_2 , G2_3 , G2_4 , G2_5 , G2_6 , G2_7 , G2_8;
double2 G3_0 , G3_1 , G3_2 , G3_3 , G3_4 , G3_5 , G3_6 , G3_7 , G3_8;
double2 GP_0 , GP_1 , GP_2 , GP_3 , GP_4 , GP_5 , GP_6 , GP_7 , GP_8;

int prp_position; // propagator position for mu and nu only not colors !!!!!!!!!!

/////////////  S20 //////////////////////////
prp_position = 2*c_nSpin*c_nColor*c_nColor*c_stride + 0*c_nColor*c_nColor*c_stride;

if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G1,gaugeTexHYP,2,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G1,gaugeTexHYP,2,pointPlus[1],c_stride);}

if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G2,gaugeTexHYP,1,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G2,gaugeTexHYP,1,pointPlus[2],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexHYP,1,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

if(c_dimBreak[1] == true && (y_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,2,0,pointMinus[1],c_surface[1]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,2,0,pointMinus[1],c_stride);}


prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0 + GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1 + GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2 + GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3 + GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4 + GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5 + GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6 + GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7 + GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8 + GP_8;


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////// s21 //////////////////////////////////////////////////////////

prp_position = 2*c_nSpin*c_nColor*c_nColor*c_stride + 1*c_nColor*c_nColor*c_stride;

if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G1,gaugeTexHYP,2,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G1,gaugeTexHYP,2,pointPlus[0],c_stride);}

if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G2,gaugeTexHYP,0,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G2,gaugeTexHYP,0,pointPlus[2],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexHYP,0,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

if(c_dimBreak[0] == true && (x_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,2,1,pointMinus[0],c_surface[0]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,2,1,pointMinus[0],c_stride);}



prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0 + GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1 + GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2 + GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3 + GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4 + GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5 + GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6 + GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7 + GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8 + GP_8;

/////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////// s12 //////////////////////////////////////////////////////////

prp_position = 1*c_nSpin*c_nColor*c_nColor*c_stride + 2*c_nColor*c_nColor*c_stride;

if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G1,gaugeTexHYP,1,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G1,gaugeTexHYP,1,pointPlus[0],c_stride);}

if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G2,gaugeTexHYP,0,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G2,gaugeTexHYP,0,pointPlus[1],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexHYP,0,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

if(c_dimBreak[0] == true && (x_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,1,2,pointMinus[0],c_surface[0]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,1,2,pointMinus[0],c_stride);}



prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0 + GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1 + GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2 + GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3 + GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4 + GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5 + GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6 + GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7 + GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8 + GP_8;

/////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////// s10 //////////////////////////////////////////////////////////

prp_position = 1*c_nSpin*c_nColor*c_nColor*c_stride + 0*c_nColor*c_nColor*c_stride;

if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G1,gaugeTexHYP,1,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G1,gaugeTexHYP,1,pointPlus[2],c_stride);}

if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G2,gaugeTexHYP,2,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G2,gaugeTexHYP,2,pointPlus[1],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexHYP,2,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

if(c_dimBreak[2] == true && (z_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,1,0,pointMinus[2],c_surface[2]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,1,0,pointMinus[2],c_stride);}


prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0 + GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1 + GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2 + GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3 + GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4 + GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5 + GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6 + GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7 + GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8 + GP_8;

/////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////// s02 //////////////////////////////////////////////////////////

prp_position = 0*c_nSpin*c_nColor*c_nColor*c_stride + 2*c_nColor*c_nColor*c_stride;

if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G1,gaugeTexHYP,0,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G1,gaugeTexHYP,0,pointPlus[1],c_stride);}

if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G2,gaugeTexHYP,1,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G2,gaugeTexHYP,1,pointPlus[0],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexHYP,1,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

if(c_dimBreak[1] == true && (y_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,0,2,pointMinus[1],c_surface[1]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,0,2,pointMinus[1],c_stride);}


prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0 + GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1 + GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2 + GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3 + GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4 + GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5 + GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6 + GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7 + GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8 + GP_8;

/////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////// s01 //////////////////////////////////////////////////////////

prp_position = 0*c_nSpin*c_nColor*c_nColor*c_stride + 1*c_nColor*c_nColor*c_stride;

if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G1,gaugeTexHYP,0,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G1,gaugeTexHYP,0,pointPlus[2],c_stride);}

if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G2,gaugeTexHYP,2,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G2,gaugeTexHYP,2,pointPlus[0],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexHYP,2,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

if(c_dimBreak[2] == true && (z_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,0,1,pointMinus[2],c_surface[2]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,0,1,pointMinus[2],c_stride);}

prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0 + GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1 + GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2 + GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3 + GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4 + GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5 + GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6 + GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7 + GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8 + GP_8;

/////////////////////////////////////////////////////////////////////////////////////////////////////
