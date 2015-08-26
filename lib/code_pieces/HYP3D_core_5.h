
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

int pointMinus[4];

pointMinus[0] = LEXIC(t_id,z_id,y_id,(x_id-1+c_localL[0])%c_localL[0],c_localL);
pointMinus[1] = LEXIC(t_id,z_id,(y_id-1+c_localL[1])%c_localL[1],x_id,c_localL);
pointMinus[2] = LEXIC(t_id,(z_id-1+c_localL[2])%c_localL[2],y_id,x_id,c_localL);
pointMinus[3] = LEXIC((t_id-1+c_localL[3])%c_localL[3],z_id,y_id,x_id,c_localL);


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



 
double2 G1_0 , G1_1 , G1_2 , G1_3 , G1_4 , G1_5 , G1_6 , G1_7 , G1_8; // fetch input plaquette
double2 GP_0 , GP_1 , GP_2 , GP_3 , GP_4 , GP_5 , GP_6 , GP_7 , GP_8; // fetch from prop structure

//////////////// U_0 ////////////////////////////
if(c_dimBreak[2] == true && (z_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,2,0,pointMinus[2],c_surface[2]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,2,0,pointMinus[2],c_stride);}

out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].x += GP_0.x;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].x += GP_1.x;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].x += GP_2.x;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].x += GP_3.x;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].x += GP_4.x;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].x += GP_5.x;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].x += GP_6.x;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].x += GP_7.x;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].x += GP_8.x;

out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].y += GP_0.y;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].y += GP_1.y;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].y += GP_2.y;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].y += GP_3.y;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].y += GP_4.y;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].y += GP_5.y;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].y += GP_6.y;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].y += GP_7.y;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].y += GP_8.y;


/////////////// U_0 //////////////////
if(c_dimBreak[1] == true && (y_id == 0)){
  //  if(sid == 0) cuPrintf("%d\n",pointMinus[2] + ((0*4+2)*9 + 0 ) * c_surface[2]);
  READPROPGAUGE(GP,propagatorTexHYP,1,0,pointMinus[1],c_surface[1]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,1,0,pointMinus[1],c_stride);}

out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].x += GP_0.x;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].x += GP_1.x;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].x += GP_2.x;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].x += GP_3.x;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].x += GP_4.x;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].x += GP_5.x;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].x += GP_6.x;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].x += GP_7.x;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].x += GP_8.x;

out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].y += GP_0.y;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].y += GP_1.y;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].y += GP_2.y;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].y += GP_3.y;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].y += GP_4.y;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].y += GP_5.y;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].y += GP_6.y;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].y += GP_7.y;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].y += GP_8.y;


/////////////// U_1 /////////////////
if(c_dimBreak[2] == true && (z_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,2,1,pointMinus[2],c_surface[2]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,2,1,pointMinus[2],c_stride);}

out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].x += GP_0.x;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].x += GP_1.x;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].x += GP_2.x;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].x += GP_3.x;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].x += GP_4.x;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].x += GP_5.x;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].x += GP_6.x;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].x += GP_7.x;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].x += GP_8.x;

out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].y += GP_0.y;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].y += GP_1.y;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].y += GP_2.y;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].y += GP_3.y;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].y += GP_4.y;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].y += GP_5.y;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].y += GP_6.y;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].y += GP_7.y;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].y += GP_8.y;



////////////// U_1 //////////////
if(c_dimBreak[0] == true && (x_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,0,1,pointMinus[0],c_surface[0]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,0,1,pointMinus[0],c_stride);}


out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].x += GP_0.x;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].x += GP_1.x;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].x += GP_2.x;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].x += GP_3.x;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].x += GP_4.x;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].x += GP_5.x;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].x += GP_6.x;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].x += GP_7.x;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].x += GP_8.x;

out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].y += GP_0.y;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].y += GP_1.y;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].y += GP_2.y;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].y += GP_3.y;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].y += GP_4.y;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].y += GP_5.y;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].y += GP_6.y;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].y += GP_7.y;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].y += GP_8.y;



////////////// U_2 /////////////////////////////
if(c_dimBreak[1] == true && (y_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,1,2,pointMinus[1],c_surface[1]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,1,2,pointMinus[1],c_stride);}

out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].x += GP_0.x;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].x += GP_1.x;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].x += GP_2.x;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].x += GP_3.x;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].x += GP_4.x;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].x += GP_5.x;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].x += GP_6.x;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].x += GP_7.x;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].x += GP_8.x;

out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].y += GP_0.y;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].y += GP_1.y;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].y += GP_2.y;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].y += GP_3.y;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].y += GP_4.y;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].y += GP_5.y;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].y += GP_6.y;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].y += GP_7.y;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].y += GP_8.y;


///////////// U_2 //////////////
if(c_dimBreak[0] == true && (x_id == 0)){
  READPROPGAUGE(GP,propagatorTexHYP,0,2,pointMinus[0],c_surface[0]);}
 else{
   READPROPGAUGE(GP,propagatorTexHYP,0,2,pointMinus[0],c_stride);}

out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].x += GP_0.x;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].x += GP_1.x;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].x += GP_2.x;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].x += GP_3.x;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].x += GP_4.x;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].x += GP_5.x;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].x += GP_6.x;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].x += GP_7.x;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].x += GP_8.x;

out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid].y += GP_0.y;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid].y += GP_1.y;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid].y += GP_2.y;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid].y += GP_3.y;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid].y += GP_4.y;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid].y += GP_5.y;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid].y += GP_6.y;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid].y += GP_7.y;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid].y += GP_8.y;


/////////////// mu = 0 
READGAUGE(G1,gaugeTexHYP,0,sid,c_stride);
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] = (omega1/6.) * out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] + (1.-omega1)*G1_0;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] = (omega1/6.) * out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] + (1.-omega1)*G1_1;
out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] = (omega1/6.) * out[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] + (1.-omega1)*G1_2;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] = (omega1/6.) * out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] + (1.-omega1)*G1_3;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] = (omega1/6.) * out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] + (1.-omega1)*G1_4;
out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] = (omega1/6.) * out[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] + (1.-omega1)*G1_5;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] = (omega1/6.) * out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] + (1.-omega1)*G1_6;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] = (omega1/6.) * out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] + (1.-omega1)*G1_7;
out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] = (omega1/6.) * out[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] + (1.-omega1)*G1_8;
////////////// mu = 1 
READGAUGE(G1,gaugeTexHYP,1,sid,c_stride);
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] = (omega1/6.) * out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] + (1.-omega1)*G1_0;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] = (omega1/6.) * out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] + (1.-omega1)*G1_1;
out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] = (omega1/6.) * out[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] + (1.-omega1)*G1_2;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] = (omega1/6.) * out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] + (1.-omega1)*G1_3;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] = (omega1/6.) * out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] + (1.-omega1)*G1_4;
out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] = (omega1/6.) * out[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] + (1.-omega1)*G1_5;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] = (omega1/6.) * out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] + (1.-omega1)*G1_6;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] = (omega1/6.) * out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] + (1.-omega1)*G1_7;
out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] = (omega1/6.) * out[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] + (1.-omega1)*G1_8;

////////////// mu = 2
READGAUGE(G1,gaugeTexHYP,2,sid,c_stride);
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] = (omega1/6.) * out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] + (1.-omega1)*G1_0;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] = (omega1/6.) * out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] + (1.-omega1)*G1_1;
out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] = (omega1/6.) * out[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] + (1.-omega1)*G1_2;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] = (omega1/6.) * out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] + (1.-omega1)*G1_3;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] = (omega1/6.) * out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] + (1.-omega1)*G1_4;
out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] = (omega1/6.) * out[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] + (1.-omega1)*G1_5;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] = (omega1/6.) * out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] + (1.-omega1)*G1_6;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] = (omega1/6.) * out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] + (1.-omega1)*G1_7;
out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] = (omega1/6.) * out[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] + (1.-omega1)*G1_8;


//////////////////////////////////////////////////////////////// finish second phase /////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////// start SU3 projection //////////////////////////////////////////////////////////////

double2 ThirdRootOne[2];
double ThirdRoot_18,ThirdRoot_12,ThirdRoot_2_3;
double2 M[3][3] , H[3][3] , U[3][3];
double2 detM;
double phase;
double sum;
double e[3];
double trace;
double a;
double2 b,w;
double2 v[3][3] , vr[3][3];
// initialize some constants

ThirdRootOne[0].x = 1.;
ThirdRootOne[0].y = sqrt(3.);
ThirdRootOne[1].x = 1.;
ThirdRootOne[1].y = -sqrt(3.);
ThirdRoot_12 =pow(12.,1./3.);
ThirdRoot_18 =pow(18.,1./3.);
ThirdRoot_2_3=pow((2./3.),1./3.);

for(int dir = 0 ; dir < 3 ; dir++){

  for(int c1 = 0 ; c1 < 3 ; c1++)
    for(int c2 = 0 ; c2 < 3 ; c2++)
      M[c1][c2] = out[dir*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride + sid]; // load link to register

  detM = M[0][1]*M[1][2]*M[2][0] + M[0][2]*M[1][0]*M[2][1] + M[0][0]*M[1][1]*M[2][2] - M[0][2]*M[1][1]*M[2][0] - M[0][0]*M[1][2]*M[2][1] - M[0][1]*M[1][0]*M[2][2];

  phase = atan2(detM.y,detM.x)/3.;

  ADJOINTMUL3x3(H,M,M);

                                                                                                                                                                         
  //      Assureal() Hermiticity: 
  
  H[0][1].x = (H[0][1].x + H[1][0].x)/2.;                                                                                                                   
  H[0][1].y = (H[0][1].y - H[1][0].y)/2.;

  H[1][0] =  conj(H[0][1]);

  H[0][2].x = (H[0][2].x + H[2][0].x)/2.;
  H[0][2].y = (H[0][2].y - H[2][0].y)/2.;

  H[2][0] =  conj(H[0][2]);

  H[1][2].x = (H[1][2].x + H[2][1].x)/2.;
  H[1][2].y = (H[1][2].y - H[2][1].y)/2.;

  H[2][1] =  conj(H[1][2]);


  sum = norm(H[0][1])+norm(H[0][2])+norm(H[1][2]);

  if(sum <= 1e-08){

    e[0]=1./sqrt(H[0][0].x);
    e[1]=1./sqrt(H[1][1].x);                                                                                                                                         
    e[2]=1./sqrt(H[2][2].x);
    
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++)
	U[c1][c2] = e[c1] * M[c1][c2];

    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++)
	out[dir*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride + sid] = U[c1][c2];

  }
  else{

    trace=(H[0][0].x+H[1][1].x+H[2][2].x)/3.;

    H[0][0].x -= trace;
    H[1][1].x -= trace;
    H[2][2].x -= trace;

    a = - ( norm2(H[0][1]) + norm2(H[0][2]) + norm2(H[1][2]) + H[2][2].x * H[2][2].x - H[0][0].x * H[1][1].x ); 

    b.x = - H[0][0].x * H[1][1].x * H[2][2].x + H[2][2].x * norm2(H[0][1]) - (H[0][1] * H[1][2] * conj(H[0][2])).x + H[1][1].x * norm2(H[0][2]);
    b.y = H[2][2].x *(H[0][1] * conj(H[0][1])).y - (H[0][1] * H[1][2] * conj(H[0][2])).y + H[1][1].x * (H[0][2] * conj(H[0][2])).y;


    b.x +=   H[0][0].x * (H[1][2] * conj(H[1][2])).x - (H[0][2] * conj(H[0][1]) * conj(H[1][2]) ).x;                                 
    b.y +=   H[0][0].x * (H[1][2] * conj(H[1][2])).y - (H[0][2] * conj(H[0][1]) * conj(H[1][2]) ).y;

    double2 temp1;
    double2 D;

    temp1.x = 12. * a * a * a + 81. * (b * b).x ;
    temp1.y = 81. * (b * b).y;
    w = cpow(temp1,0.5);
    
    temp1.x = -9.0 * b.x + w.x;
    temp1.y = -9.0 * b.y + w.y;
    D = cpow(temp1,1./3.);

    temp1.x = a*ThirdRoot_2_3; temp1.y = 0.;
    e[0] = D.x / (ThirdRoot_18) - (temp1/D).x;
    temp1.x = D.x * ThirdRoot_12 ; temp1.y = D.y * ThirdRoot_12;
    e[1] = a * (ThirdRootOne[0] / temp1).x - (ThirdRootOne[1] * D).x / (ThirdRoot_18*2.);
    e[2] = -e[0]-e[1];

    e[0] += trace;
    e[1] += trace;
    e[2] += trace;

    H[0][0].x += trace;
    H[1][1].x += trace;
    H[2][2].x += trace;

    // eigenvectors

    v[0][0].x = -(e[0]*H[2][0].x - H[2][0].x*H[1][1].x + (H[1][0]*H[2][1]).x);                                                                 
    v[0][0].y = -(e[0]*H[2][0].y - H[2][0].y*H[1][1].x + (H[1][0]*H[2][1]).y);

    v[0][1].x = -((H[2][0]*H[0][1]).x + e[0]*H[2][1].x - H[0][0].x*H[2][1].x);
    v[0][1].y = -((H[2][0]*H[0][1]).y + e[0]*H[2][1].y - H[0][0].x*H[2][1].y);

    v[0][2].x =-e[0]*e[0] + e[0]*H[0][0].x + (H[0][1]*conj(H[0][1])).x + e[0]*H[1][1].x - H[0][0].x*H[1][1].x;
    v[0][2].y = 0.;

    v[1][0].x = -(e[1]*H[2][0].x - H[2][0].x*H[1][1].x + (H[1][0]*H[2][1]).x);
    v[1][0].y = -(e[1]*H[2][0].y - H[2][0].y*H[1][1].x + (H[1][0]*H[2][1]).y);

    v[1][1].x = -((H[2][0]*H[0][1]).x + e[1]*H[2][1].x - H[0][0].x*H[2][1].x);
    v[1][1].y = -((H[2][0]*H[0][1]).y + e[1]*H[2][1].y - H[0][0].x*H[2][1].y);

    v[1][2].x =-e[1]*e[1] + e[1]*H[0][0].x + (H[0][1]*conj(H[0][1])).x + e[1]*H[1][1].x - H[0][0].x*H[1][1].x;;
    v[1][2].y = 0.;

    double norma;


    norma = norm2(v[0][0]) + norm2(v[0][1]) + norm2(v[0][2]);
    w = v[0][0] * conj(v[1][0]) + v[0][1] * conj(v[1][1]) + v[0][2] * conj(v[1][2]);
    w.x /= norma;
    w.y /= norma;

    v[1][0].x-= (w * v[0][0]).x;
    v[1][0].y-= (w * v[0][0]).y;

    v[1][1].x-= (w * v[0][1]).x;
    v[1][1].y-= (w * v[0][1]).y;

    v[1][2].x-= (w * v[0][2]).x;
    v[1][2].y-= (w * v[0][2]).y;

    norma=1./sqrt(norma);

    v[0][0].x *= norma;
    v[0][0].y *= norma;

    v[0][1].x *= norma;
    v[0][1].y *= norma;

    v[0][2].x *= norma;
    v[0][2].y *= norma;

    //////////////////////
    norma = norm2(v[1][0]) + norm2(v[1][1]) + norm2(v[1][2]);

    norma=1./sqrt(norma);

    v[1][0].x *= norma;
    v[1][0].y *= norma;

    v[1][1].x *= norma;
    v[1][1].y *= norma;

    v[1][2].x *= norma;
    v[1][2].y *= norma;

    /////////////////////////////

    v[2][0].x =  (v[0][1]*v[1][2]).x - (v[0][2]*v[1][1]).x;                                                                                            
    v[2][0].y = -(v[0][1]*v[1][2]).y + (v[0][2]*v[1][1]).y;

    v[2][1].x = -(v[0][0]*v[1][2]).x + (v[0][2]*v[1][0]).x;
    v[2][1].y = +(v[0][0]*v[1][2]).y - (v[0][2]*v[1][0]).y;

    v[2][2].x =  (v[0][0]*v[1][1]).x - (v[0][1]*v[1][0]).x;
    v[2][2].y = -(v[0][0]*v[1][1]).y + (v[0][1]*v[1][0]).y;

    double de;

    //de = e[0]*e[1] + e[1]*e[2] + e[2]*e[0];

    de = 1./sqrt(e[0]);
    b.x = de*cos(phase);
    b.y =-de*sin(phase);
    vr[0][0] = (b*v[0][0]);
    vr[0][1] = (b*v[0][1]);
    vr[0][2] = (b*v[0][2]);

    de = 1./sqrt(e[1]);
    b.x = de*cos(phase);
    b.y =-de*sin(phase);

    vr[1][0] = (b*v[1][0]);
    vr[1][1] = (b*v[1][1]);
    vr[1][2] = (b*v[1][2]);

    de = 1./sqrt(e[2]);
    b.x = de*cos(phase);
    b.y =-de*sin(phase);

    vr[2][0] = (b*v[2][0]);
    vr[2][1] = (b*v[2][1]);
    vr[2][2] = (b*v[2][2]);
    
    MULADJOINT3x3(H,M,v);

    MUL3x3_2(U,H,vr);



    norma = norm2(U[0][0]) + norm2(U[1][0]) + norm2(U[2][0]);
    w = U[0][0] * conj(U[0][1]) + U[1][0] * conj(U[1][1]) + U[2][0] * conj(U[2][1]);

    w.x /= norma;
    w.y /= norma;


    U[0][1].x -= (w*U[0][0]).x;
    U[0][1].y -= (w*U[0][0]).y;

    U[1][1].x -= (w*U[1][0]).x;
    U[1][1].y -= (w*U[1][0]).y;

    U[2][1].x -= (w*U[2][0]).x;
    U[2][1].y -= (w*U[2][0]).y;

    norma = 1./sqrt(norma);

    U[0][0].x*= norma;
    U[0][0].y*= norma;
    U[1][0].x*= norma;
    U[1][0].y*= norma;
    U[2][0].x*= norma;
    U[2][0].y*= norma;


    norma = norm2(U[0][1]) + norm2(U[1][1]) + norm2(U[2][1]);
    norma = 1./sqrt(norma);

    U[0][1].x *= norma;
    U[0][1].y *= norma;
    U[1][1].x *= norma;
    U[1][1].y *= norma;
    U[2][1].x *= norma;
    U[2][1].y *= norma;


    U[0][2].x =  (U[1][0]*U[2][1]).x - (U[2][0]*U[1][1]).x;                                                                                            
    U[0][2].y = -(U[1][0]*U[2][1]).y + (U[2][0]*U[1][1]).y;

    U[1][2].x = -(U[0][0]*U[2][1]).x + (U[2][0]*U[0][1]).x;
    U[1][2].y =  (U[0][0]*U[2][1]).y - (U[2][0]*U[0][1]).y;

    U[2][2].x =  (U[0][0]*U[1][1]).x - (U[1][0]*U[0][1]).x;
    U[2][2].y = -(U[0][0]*U[1][1]).y + (U[1][0]*U[0][1]).y;

    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++)
        out[dir*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride + sid] = U[c1][c2];


  }

 } // close for over mu
