
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= c_stride) return;



int x[4];
int r1,r2;

r1 = sid/(c_localL[0]);
r2 = r1/(c_localL[1]);
x[0] = sid - r1*(c_localL[0]);
x[1] = r1 - r2*(c_localL[1]);
x[3] = r2/(c_localL[2]);
x[2] = r2 - x[3]*(c_localL[2]);


double2 gamma[4][4];
for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int nu = 0 ; nu < c_nSpin ; nu++){
    gamma[mu][nu].x = 0 ; gamma[mu][nu].y = 0;
  }

switch( iflag ){
  
 case 0: // g1
   gamma[3][0].y=-1.; gamma[2][1].y=-1.; gamma[1][2].y=1.; gamma[0][3].y=1.;     
   break;
 case 1: // g2
   gamma[3][0].x=1.; gamma[2][1].x=-1.; gamma[1][2].x=-1.; gamma[0][3].x=1.;
   break;
 case 2: // g3
   gamma[0][2].y=1.; gamma[1][3].y=-1.; gamma[2][0].y=-1.; gamma[3][1].y=1.;
   break;
 case 3: // g4
   gamma[0][0].x=1.; gamma[1][1].x=1.; gamma[2][2].x=-1.; gamma[3][3].x=-1.;
   break;
 case 4: // g5g1 
   gamma[0][1].y=-1.; gamma[1][0].y=-1.; gamma[2][3].y=1.; gamma[3][2].y=1.;
   break;
 case 5: // g5g2
   gamma[0][1].x=-1.; gamma[1][0].x=1.; gamma[2][3].x=1.; gamma[3][2].x=-1.;
   break;
 case 6: // g5g3
   gamma[0][0].y=-1.; gamma[1][1].y=1.; gamma[2][2].y=1.; gamma[3][3].y=-1.;   
   break;
 case 7: // g5g4
   gamma[0][2].x=-1.; gamma[1][3].x=-1.; gamma[2][0].x=1.; gamma[3][1].x=1.;
   break;

 }


double2 g5xg[4][4];
double2 g5[4][4];
for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int nu = 0 ; nu < c_nSpin ; nu++){
    g5xg[mu][nu].x = 0. ; g5xg[mu][nu].y = 0.;
    g5[mu][nu].x = 0. ; g5[mu][nu].y = 0.;
  }

g5[0][2].x=1.;
g5[1][3].x=1.;
g5[2][0].x=1.;
g5[3][1].x=1.;

for(int pi = 0 ; pi < c_nSpin ; pi++)
  for(int phi = 0 ; phi < c_nSpin ; phi++)
    for(int rho = 0 ; rho < c_nSpin ; rho++){
      g5xg[pi][phi] = g5xg[pi][phi] + g5[pi][rho]*gamma[rho][phi];
    }



int pointPlusVec[4];
int pointMinusVec[4];
int pointPlusProp[4];
int pointMinusProp[4];
int pointMinusGauge[4];

pointPlusVec[0] = LEXIC(x[3],x[2],x[1],(x[0]+1)%c_localL[0],c_localL);
pointPlusVec[1] = LEXIC(x[3],x[2],(x[1]+1)%c_localL[1],x[0],c_localL);
pointPlusVec[2] = LEXIC(x[3],(x[2]+1)%c_localL[2],x[1],x[0],c_localL);
pointPlusVec[3] = LEXIC((x[3]+1)%c_localL[3],x[2],x[1],x[0],c_localL);
pointMinusVec[0] = LEXIC(x[3],x[2],x[1],(x[0]-1+c_localL[0])%c_localL[0],c_localL);
pointMinusVec[1] = LEXIC(x[3],x[2],(x[1]-1+c_localL[1])%c_localL[1],x[0],c_localL);
pointMinusVec[2] = LEXIC(x[3],(x[2]-1+c_localL[2])%c_localL[2],x[1],x[0],c_localL);
pointMinusVec[3] = LEXIC((x[3]-1+c_localL[3])%c_localL[3],x[2],x[1],x[0],c_localL);

pointPlusProp[0] = pointPlusVec[0];
pointPlusProp[1] = pointPlusVec[1];
pointPlusProp[2] = pointPlusVec[2];
pointPlusProp[3] = pointPlusVec[3];

pointMinusProp[0] = pointMinusVec[0];
pointMinusProp[1] = pointMinusVec[1];
pointMinusProp[2] = pointMinusVec[2];
pointMinusProp[3] = pointMinusVec[3];

pointMinusGauge[0] = pointMinusVec[0];
pointMinusGauge[1] = pointMinusVec[1];
pointMinusGauge[2] = pointMinusVec[2];
pointMinusGauge[3] = pointMinusVec[3];

/*
if(x[3] == 0){
  cuPrintf("%d  %d %d %d %d  %d\n",sid,x[0],x[1],x[2],x[3],pointPlusVec[dir]);
 }
*/

// x direction
if(c_dimBreak[0] == true){
  if(x[0] == c_localL[0] -1){
    pointPlusProp[0] = c_plusGhost[0]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZY(x[3],x[2],x[1],c_localL);
    pointPlusVec[0] = c_plusGhost[0]*c_nSpin*c_nColor + LEXIC_TZY(x[3],x[2],x[1],c_localL);
  }

  if(x[0] == 0){
    pointMinusProp[0] = c_minusGhost[0]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZY(x[3],x[2],x[1],c_localL);
    pointMinusVec[0] = c_minusGhost[0]*c_nSpin*c_nColor + LEXIC_TZY(x[3],x[2],x[1],c_localL);
    pointMinusGauge[0] = c_minusGhost[0]*c_nDim*c_nColor*c_nColor + LEXIC_TZY(x[3],x[2],x[1],c_localL);
  }

 }
// y direction 
if(c_dimBreak[1] == true){
  if(x[1] == c_localL[1] -1){
    pointPlusProp[1] = c_plusGhost[1]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZX(x[3],x[2],x[0],c_localL);
    pointPlusVec[1] = c_plusGhost[1]*c_nSpin*c_nColor + LEXIC_TZX(x[3],x[2],x[0],c_localL);
  }
  if(x[1] == 0){
    pointMinusProp[1] = c_minusGhost[1]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZX(x[3],x[2],x[0],c_localL);
    pointMinusVec[1] = c_minusGhost[1]*c_nSpin*c_nColor + LEXIC_TZX(x[3],x[2],x[0],c_localL);
    pointMinusGauge[1] = c_minusGhost[1]*c_nDim*c_nColor*c_nColor +  LEXIC_TZX(x[3],x[2],x[0],c_localL);
  }
 }
// z direction
if(c_dimBreak[2] == true){
  if(x[2] == c_localL[2] -1){
    pointPlusProp[2] = c_plusGhost[2]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TYX(x[3],x[1],x[0],c_localL);
    pointPlusVec[2] = c_plusGhost[2]*c_nSpin*c_nColor + LEXIC_TYX(x[3],x[1],x[0],c_localL);
  }
  if(x[2] == 0){
    pointMinusProp[2] = c_minusGhost[2]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TYX(x[3],x[1],x[0],c_localL);
    pointMinusVec[2] = c_minusGhost[2]*c_nSpin*c_nColor + LEXIC_TYX(x[3],x[1],x[0],c_localL);
    pointMinusGauge[2] = c_minusGhost[2]*c_nDim*c_nColor*c_nColor +  LEXIC_TYX(x[3],x[1],x[0],c_localL);
  }  
 }
// t direction
if(c_dimBreak[3] == true){
  if(x[3] == c_localL[3] -1){
    pointPlusProp[3] = c_plusGhost[3]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_ZYX(x[2],x[1],x[0],c_localL);
    pointPlusVec[3] = c_plusGhost[3]*c_nSpin*c_nColor + LEXIC_ZYX(x[2],x[1],x[0],c_localL);
  }
  if(x[3] == 0){
    pointMinusProp[3] = c_minusGhost[3]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_ZYX(x[2],x[1],x[0],c_localL);
    pointMinusVec[3] = c_minusGhost[3]*c_nSpin*c_nColor + LEXIC_ZYX(x[2],x[1],x[0],c_localL);
    pointMinusGauge[3] = c_minusGhost[3]*c_nDim*c_nColor*c_nColor + LEXIC_ZYX(x[2],x[1],x[0],c_localL);
  }
 }



/*
if(x[3] == 0){                                                                                             
  for(int pi = 0 ; pi < 4 ; pi++)                                                                                                   
    for(int d = 0 ; d < 3 ; d++)                                                                                                  
      cuPrintf("%d %d %d %d  %d %d  %+e %+e\n",x[0],x[1],x[2],x[3],pointPlusVec[dir],pointPlusVec[dir] + (pi*3+d)*c_surface[dir],Phi[pi][d].x,Phi[pi][d].y);
}
*/


double2 prop[4][4][3][3];
double2 Phi[4][3];
double2 gauge[3][3];

double2 outIns[4][3];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int c1 = 0 ; c1 < 3 ; c1++){
    outIns[mu][c1].x=0.; outIns[mu][c1].y=0.;
  }


// fetch phi at x point



for(int pi = 0 ; pi < 4 ; pi++)
  for(int d = 0 ; d < 3 ; d++)
    Phi[pi][d] = fetch_double2(phiVectorStochTex,sid + (pi*3+d)*c_stride);

// x  x  x+dir (plus) /////////////////////////////////////////////////////////////////////////

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,sid + ( (dir*3*3 + c1*3 + c2) * c_stride ) );


if(c_dimBreak[dir] == true && (x[dir] == c_localL[dir]-1) ){
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = fetch_double2(propStochTex,pointPlusProp[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_surface[dir]);
      }  
 }
 else{

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = fetch_double2(propStochTex,pointPlusProp[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }  

 }


for(int d = 0 ; d < 3 ; d++)
  for(int e = 0 ; e < 3 ; e++)
    for(int cp = 0 ; cp < 3 ; cp++)
      for(int pi = 0 ; pi < 4 ; pi++)
	for(int phi = 0 ; phi < 4 ; phi++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++){
	    outIns[gammap][cp] = outIns[gammap][cp] + conj(Phi[pi][d]) * g5xg[pi][phi] * gauge[d][e] * prop[phi][gammap][e][cp];
	  }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// x  x-dir  x-dir (minus) ////////////////////////////////////////////////////////////////////////////////////////////////////////////

if(c_dimBreak[dir] == true && x[dir] == 0){

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,pointMinusGauge[dir] + ( (dir*3*3 + c1*3 + c2) * c_surface[dir] ) );

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = fetch_double2(propStochTex,pointMinusProp[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_surface[dir]);
      }  

 }
 else{

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,pointMinusGauge[dir] + ( (dir*3*3 + c1*3 + c2) * c_stride ) );

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = fetch_double2(propStochTex,pointMinusProp[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }  


 }



for(int d = 0 ; d < 3 ; d++)
  for(int e = 0 ; e < 3 ; e++)
    for(int cp = 0 ; cp < 3 ; cp++)
      for(int pi = 0 ; pi < 4 ; pi++)
	for(int phi = 0 ; phi < 4 ; phi++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++)
	    outIns[gammap][cp] = outIns[gammap][cp] - conj(Phi[pi][d]) * g5xg[pi][phi] * conj(gauge[e][d]) * prop[phi][gammap][e][cp];


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// --------------------------------------------------
for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	prop[mu][nu][c1][c2] = fetch_double2(propStochTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }
// --------------------------------------------------

// x+dir  x  x //////////////////////////////////////////////////////////////////////////////////////////////////////////////////



for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,sid + ( (dir*3*3 + c1*3 + c2) * c_stride ) );

if(c_dimBreak[dir] == true && (x[dir] == (c_localL[dir]-1)) ){

  for(int pi = 0 ; pi < 4 ; pi++)
    for(int d = 0 ; d < 3 ; d++)
      Phi[pi][d] = fetch_double2(phiVectorStochTex,pointPlusVec[dir] + (pi*3+d)*c_surface[dir]);

 }
 else{

  for(int pi = 0 ; pi < 4 ; pi++)
    for(int d = 0 ; d < 3 ; d++)
      Phi[pi][d] = fetch_double2(phiVectorStochTex,pointPlusVec[dir] + (pi*3+d)*c_stride);
 }



 
for(int d = 0 ; d < 3 ; d++)
  for(int e = 0 ; e < 3 ; e++)
    for(int cp = 0 ; cp < 3 ; cp++)
      for(int pi = 0 ; pi < 4 ; pi++)
	for(int phi = 0 ; phi < 4 ; phi++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++){
	    outIns[gammap][cp] = outIns[gammap][cp] - conj(Phi[pi][d]) * g5xg[pi][phi] * conj(gauge[e][d]) * prop[phi][gammap][e][cp];
	  }
 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 


// x-dir  x-dir  x ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////



if(c_dimBreak[dir] == true && x[dir] == 0){

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,pointMinusGauge[dir] + ( (dir*3*3 + c1*3 + c2) * c_surface[dir] ) );


  for(int pi = 0 ; pi < 4 ; pi++)
    for(int d = 0 ; d < 3 ; d++)
      Phi[pi][d] = fetch_double2(phiVectorStochTex,pointMinusVec[dir] + (pi*3+d)*c_surface[dir]);


 }
 else{

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,pointMinusGauge[dir] + ( (dir*3*3 + c1*3 + c2) * c_stride ) );

  for(int pi = 0 ; pi < 4 ; pi++)
    for(int d = 0 ; d < 3 ; d++)
      Phi[pi][d] = fetch_double2(phiVectorStochTex,pointMinusVec[dir] + (pi*3+d)*c_stride);

 }


for(int d = 0 ; d < 3 ; d++)
  for(int e = 0 ; e < 3 ; e++)
    for(int cp = 0 ; cp < 3 ; cp++)
      for(int pi = 0 ; pi < 4 ; pi++)
	for(int phi = 0 ; phi < 4 ; phi++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++)
	    outIns[gammap][cp] = outIns[gammap][cp] + conj(Phi[pi][d]) * g5xg[pi][phi] * gauge[d][e] * prop[phi][gammap][e][cp];


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

for(int mu = 0 ; mu < 4 ; mu++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
    out[sid + (mu*3+c1)*c_stride] = 0.25*outIns[mu][c1];



