
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

pointPlus[0] = LEXIC(t_id,z_id,y_id,(x_id+1)%c_localL[0],c_localL); 
pointPlus[1] = LEXIC(t_id,z_id,(y_id+1)%c_localL[1],x_id,c_localL);
pointPlus[2] = LEXIC(t_id,(z_id+1)%c_localL[2],y_id,x_id,c_localL);
pointPlus[3] = LEXIC((t_id+1)%c_localL[3],z_id,y_id,x_id,c_localL);


// x direction
if(c_dimBreak[0] == true)
  if(x_id == c_localL[0] -1)
    pointPlus[0] = c_plusGhost[0]*c_nDim*c_nColor*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);


// y direction
if(c_dimBreak[1] == true)
  if(y_id == c_localL[1] -1)
    pointPlus[1] = c_plusGhost[1]*c_nDim*c_nColor*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);

// z direction 
if(c_dimBreak[2] == true)
  if(z_id == c_localL[2] -1)
    pointPlus[2] = c_plusGhost[2]*c_nDim*c_nColor*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);

// t direction
if(c_dimBreak[3] == true)
  if(t_id == c_localL[3] -1)
    pointPlus[3] = c_plusGhost[3]*c_nDim*c_nColor*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);


// because now we calculate staples we need 3 inputs
double2 G1_0 , G1_1 , G1_2 , G1_3 , G1_4 , G1_5 , G1_6 , G1_7 , G1_8;
double2 G2_0 , G2_1 , G2_2 , G2_3 , G2_4 , G2_5 , G2_6 , G2_7 , G2_8;
double2 G3_0 , G3_1 , G3_2 , G3_3 , G3_4 , G3_5 , G3_6 , G3_7 , G3_8;

int prp_position; // propagator position for mu and nu only not colors !!!!!!!!!!

/////////////  mu = 0 , nu = 1 //////////////////////////
prp_position = 0*c_nSpin*c_nColor*c_nColor*c_stride + 1*c_nColor*c_nColor*c_stride;

READGAUGE(G1,gaugeTexAPE,0,sid,c_stride);
if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G2,gaugeTexAPE,1,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G2,gaugeTexAPE,1,pointPlus[0],c_stride);}

MUL3X3(g3_,g1_,g2_);

READGAUGE(G2,gaugeTexAPE,1,sid,c_stride);

MUL3X3(g1_,g2_T,g3_);

prp[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0;
prp[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1; 
prp[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2;
prp[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3; 
prp[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4; 
prp[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5; 
prp[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6; 
prp[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7; 
prp[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8; 



if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G1,gaugeTexAPE,0,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G1,gaugeTexAPE,0,pointPlus[1],c_stride);}

if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G2,gaugeTexAPE,1,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G2,gaugeTexAPE,1,pointPlus[0],c_stride);}
MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexAPE,1,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////// mu = 0 , nu = 2 //////////////////////////////////////////////////////////

prp_position = 0*c_nSpin*c_nColor*c_nColor*c_stride + 2*c_nColor*c_nColor*c_stride;

READGAUGE(G1,gaugeTexAPE,0,sid,c_stride);
if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G2,gaugeTexAPE,2,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G2,gaugeTexAPE,2,pointPlus[0],c_stride);}

MUL3X3(g3_,g1_,g2_);

READGAUGE(G2,gaugeTexAPE,2,sid,c_stride);

MUL3X3(g1_,g2_T,g3_);

prp[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0;
prp[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1; 
prp[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2;
prp[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3; 
prp[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4; 
prp[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5; 
prp[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6; 
prp[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7; 
prp[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8; 

if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G1,gaugeTexAPE,0,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G1,gaugeTexAPE,0,pointPlus[2],c_stride);}

if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G2,gaugeTexAPE,2,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G2,gaugeTexAPE,2,pointPlus[0],c_stride);}

MUL3X3(g3_,g1_,g2_T);
READGAUGE(G2,gaugeTexAPE,2,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].x += G1_0.x;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].x += G1_1.x;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].x += G1_2.x;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].x += G1_3.x;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].x += G1_4.x;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].x += G1_5.x;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].x += G1_6.x;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].x += G1_7.x;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].x += G1_8.x;

out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].y += G1_0.y;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].y += G1_1.y;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].y += G1_2.y;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].y += G1_3.y;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].y += G1_4.y;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].y += G1_5.y;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].y += G1_6.y;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].y += G1_7.y;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].y += G1_8.y;



/////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////// mu =1 , nu = 0 ///////////////////////////////////////////////////////

prp_position = 1*c_nSpin*c_nColor*c_nColor*c_stride + 0*c_nColor*c_nColor*c_stride;

READGAUGE(G1,gaugeTexAPE,1,sid,c_stride);
if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G2,gaugeTexAPE,0,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G2,gaugeTexAPE,0,pointPlus[1],c_stride);}

MUL3X3(g3_,g1_,g2_);

READGAUGE(G2,gaugeTexAPE,0,sid,c_stride);

MUL3X3(g1_,g2_T,g3_);

prp[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0;
prp[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1; 
prp[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2;
prp[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3; 
prp[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4; 
prp[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5; 
prp[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6; 
prp[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7; 
prp[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8; 



if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G1,gaugeTexAPE,1,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G1,gaugeTexAPE,1,pointPlus[0],c_stride);}

if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G2,gaugeTexAPE,0,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G2,gaugeTexAPE,0,pointPlus[1],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexAPE,0,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8;



/////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////  mu = 1 , nu =2 /////////////////////////////////////////////////////

prp_position = 1*c_nSpin*c_nColor*c_nColor*c_stride + 2*c_nColor*c_nColor*c_stride;

READGAUGE(G1,gaugeTexAPE,1,sid,c_stride);
if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G2,gaugeTexAPE,2,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G2,gaugeTexAPE,2,pointPlus[1],c_stride);}

MUL3X3(g3_,g1_,g2_);

READGAUGE(G2,gaugeTexAPE,2,sid,c_stride);

MUL3X3(g1_,g2_T,g3_);

prp[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0;
prp[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1; 
prp[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2;
prp[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3; 
prp[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4; 
prp[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5; 
prp[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6; 
prp[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7; 
prp[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8; 


if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G1,gaugeTexAPE,1,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G1,gaugeTexAPE,1,pointPlus[2],c_stride);}

if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G2,gaugeTexAPE,2,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G2,gaugeTexAPE,2,pointPlus[1],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexAPE,2,sid,c_stride);

MUL3X3(g1_,g2_,g3_);


out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].x += G1_0.x;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].x += G1_1.x;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].x += G1_2.x;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].x += G1_3.x;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].x += G1_4.x;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].x += G1_5.x;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].x += G1_6.x;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].x += G1_7.x;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].x += G1_8.x;

out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].y += G1_0.y;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].y += G1_1.y;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].y += G1_2.y;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].y += G1_3.y;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].y += G1_4.y;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].y += G1_5.y;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].y += G1_6.y;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].y += G1_7.y;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].y += G1_8.y;

/*
if(sid == 0){
  for(int c1 = 0 ; c1 < 3 ; c1++)
    for(int c2 = 0 ; c2 < 3 ; c2++)
      cuPrintf("%e %e\n",out[1*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride].x , out[1*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride].y);
 }
*/
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////// mu = 2 , nu = 0  /////////////////////////////////////
prp_position = 2*c_nSpin*c_nColor*c_nColor*c_stride + 0*c_nColor*c_nColor*c_stride;

READGAUGE(G1,gaugeTexAPE,2,sid,c_stride);
if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G2,gaugeTexAPE,0,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G2,gaugeTexAPE,0,pointPlus[2],c_stride);}

MUL3X3(g3_,g1_,g2_);

READGAUGE(G2,gaugeTexAPE,0,sid,c_stride);

MUL3X3(g1_,g2_T,g3_);

prp[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0;
prp[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1; 
prp[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2;
prp[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3; 
prp[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4; 
prp[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5; 
prp[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6; 
prp[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7; 
prp[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8; 



if(c_dimBreak[0] == true && (x_id == c_localL[0] -1)){
  READGAUGE(G1,gaugeTexAPE,2,pointPlus[0],c_surface[0]);}
 else{
   READGAUGE(G1,gaugeTexAPE,2,pointPlus[0],c_stride);}

if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G2,gaugeTexAPE,0,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G2,gaugeTexAPE,0,pointPlus[2],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexAPE,0,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////// mu = 2 , nu = 1 //////////////////////////////////////////////////////////
prp_position = 2*c_nSpin*c_nColor*c_nColor*c_stride + 1*c_nColor*c_nColor*c_stride;

READGAUGE(G1,gaugeTexAPE,2,sid,c_stride);
if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G2,gaugeTexAPE,1,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G2,gaugeTexAPE,1,pointPlus[2],c_stride);}

MUL3X3(g3_,g1_,g2_);

READGAUGE(G2,gaugeTexAPE,1,sid,c_stride);

MUL3X3(g1_,g2_T,g3_);

prp[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = G1_0;
prp[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = G1_1; 
prp[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = G1_2;
prp[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = G1_3; 
prp[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = G1_4; 
prp[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = G1_5; 
prp[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = G1_6; 
prp[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = G1_7; 
prp[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = G1_8; 



if(c_dimBreak[1] == true && (y_id == c_localL[1] -1)){
  READGAUGE(G1,gaugeTexAPE,2,pointPlus[1],c_surface[1]);}
 else{
   READGAUGE(G1,gaugeTexAPE,2,pointPlus[1],c_stride);}

if(c_dimBreak[2] == true && (z_id == c_localL[2] -1)){
  READGAUGE(G2,gaugeTexAPE,1,pointPlus[2],c_surface[2]);}
 else{
   READGAUGE(G2,gaugeTexAPE,1,pointPlus[2],c_stride);}

MUL3X3(g3_,g1_,g2_T);

READGAUGE(G2,gaugeTexAPE,1,sid,c_stride);

MUL3X3(g1_,g2_,g3_);

out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].x += G1_0.x;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].x += G1_1.x;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].x += G1_2.x;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].x += G1_3.x;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].x += G1_4.x;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].x += G1_5.x;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].x += G1_6.x;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].x += G1_7.x;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].x += G1_8.x;

out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].y += G1_0.y;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].y += G1_1.y;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].y += G1_2.y;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].y += G1_3.y;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].y += G1_4.y;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].y += G1_5.y;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].y += G1_6.y;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].y += G1_7.y;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].y += G1_8.y;

///////////////////////////////////// end all combinations /////////////////////////////


/*
if(sid == 0){
  for(int dir = 0 ; dir < 3 ; dir++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++)
	cuPrintf("%e %e\n",out[dir*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride].x , out[dir*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride].y);
 }
*/

/*
if(t_id == 0 && z_id == 0 && y_id == 0 && x_id ==0){
  for(int mu = 0 ; mu < 3 ; mu++)
  for(int nu = 0 ; nu < 3 ; nu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++)
	cuPrintf("%e %e\n",prp[mu*c_nSpin*c_nColor*c_nColor*c_stride + nu*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride + sid].x , prp[mu*c_nSpin*c_nColor*c_nColor*c_stride + nu*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride + sid].y);
 }



*/
