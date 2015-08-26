
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


double2 OnePgamma[4][4];
double2 OneMgamma[4][4];

for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int nu = 0 ; nu < c_nSpin ; nu++){
    OnePgamma[mu][nu].x = 0 ; OnePgamma[mu][nu].y = 0;
    OneMgamma[mu][nu].x = 0 ; OneMgamma[mu][nu].y = 0;
  }

switch( dir ){
  
 case 0: // g1
   OnePgamma[0][0].x=1.; OnePgamma[1][1].x=1.; OnePgamma[2][2].x=1.; OnePgamma[3][3].x=1.; OnePgamma[0][3].y=1.;OnePgamma[1][2].y=1.;OnePgamma[2][1].y=-1.;OnePgamma[3][0].y=-1.;
   OneMgamma[0][0].x=1.; OneMgamma[1][1].x=1.; OneMgamma[2][2].x=1.; OneMgamma[3][3].x=1.; OneMgamma[0][3].y=-1.;OneMgamma[1][2].y=-1.;OneMgamma[2][1].y=1.;OneMgamma[3][0].y=1.;
   break;
 case 1: // g2
   OnePgamma[0][0].x=1.; OnePgamma[1][1].x=1.; OnePgamma[2][2].x=1.; OnePgamma[3][3].x=1.; OnePgamma[0][3].x=1.;OnePgamma[1][2].x=-1.;OnePgamma[2][1].x=-1.;OnePgamma[3][0].x=1.;
   OneMgamma[0][0].x=1.; OneMgamma[1][1].x=1.; OneMgamma[2][2].x=1.; OneMgamma[3][3].x=1.; OneMgamma[0][3].x=-1.;OneMgamma[1][2].x=1.;OneMgamma[2][1].x=1.;OneMgamma[3][0].x=-1.;
   break;
 case 2: // g3
   OnePgamma[0][0].x=1.; OnePgamma[1][1].x=1.; OnePgamma[2][2].x=1.; OnePgamma[3][3].x=1.; OnePgamma[0][2].y=1.;OnePgamma[1][3].y=-1.;OnePgamma[2][0].y=-1.;OnePgamma[3][1].y=1.;
   OneMgamma[0][0].x=1.; OneMgamma[1][1].x=1.; OneMgamma[2][2].x=1.; OneMgamma[3][3].x=1.; OneMgamma[0][2].y=-1.;OneMgamma[1][3].y=1.;OneMgamma[2][0].y=1.;OneMgamma[3][1].y=-1.;
   break;
 case 3: // g4
   OnePgamma[0][0].x=2.; OnePgamma[1][1].x=2.;
   OneMgamma[2][2].x=2.; OneMgamma[3][3].x=2.;
   break;

 }


int pointPlus[4];
int pointMinus[4];
int pointMinusG[4];

pointPlus[0] = LEXIC(x[3],x[2],x[1],(x[0]+1)%c_localL[0],c_localL);
pointPlus[1] = LEXIC(x[3],x[2],(x[1]+1)%c_localL[1],x[0],c_localL);
pointPlus[2] = LEXIC(x[3],(x[2]+1)%c_localL[2],x[1],x[0],c_localL);
pointPlus[3] = LEXIC((x[3]+1)%c_localL[3],x[2],x[1],x[0],c_localL);
pointMinus[0] = LEXIC(x[3],x[2],x[1],(x[0]-1+c_localL[0])%c_localL[0],c_localL);
pointMinus[1] = LEXIC(x[3],x[2],(x[1]-1+c_localL[1])%c_localL[1],x[0],c_localL);
pointMinus[2] = LEXIC(x[3],(x[2]-1+c_localL[2])%c_localL[2],x[1],x[0],c_localL);
pointMinus[3] = LEXIC((x[3]-1+c_localL[3])%c_localL[3],x[2],x[1],x[0],c_localL);

pointMinusG[0] = pointMinus[0];
pointMinusG[1] = pointMinus[1];
pointMinusG[2] = pointMinus[2];
pointMinusG[3] = pointMinus[3];

// x direction
if(c_dimBreak[0] == true){
  if(x[0] == c_localL[0] -1)
    pointPlus[0] = c_plusGhost[0]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZY(x[3],x[2],x[1],c_localL);
  if(x[0] == 0){
    pointMinus[0] = c_minusGhost[0]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZY(x[3],x[2],x[1],c_localL);
    pointMinusG[0] = c_minusGhost[0]*c_nDim*c_nColor*c_nColor + LEXIC_TZY(x[3],x[2],x[1],c_localL);
  }
 }
// y direction 
if(c_dimBreak[1] == true){
  if(x[1] == c_localL[1] -1)
    pointPlus[1] = c_plusGhost[1]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZX(x[3],x[2],x[0],c_localL);
  if(x[1] == 0){
    pointMinus[1] = c_minusGhost[1]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZX(x[3],x[2],x[0],c_localL);
    pointMinusG[1] = c_minusGhost[1]*c_nDim*c_nColor*c_nColor +  LEXIC_TZX(x[3],x[2],x[0],c_localL);
  }
 }
// z direction
if(c_dimBreak[2] == true){
  if(x[2] == c_localL[2] -1)
    pointPlus[2] = c_plusGhost[2]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TYX(x[3],x[1],x[0],c_localL);
  if(x[2] == 0){
    pointMinus[2] = c_minusGhost[2]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TYX(x[3],x[1],x[0],c_localL);
    pointMinusG[2] = c_minusGhost[2]*c_nDim*c_nColor*c_nColor +  LEXIC_TYX(x[3],x[1],x[0],c_localL);
  }  
 }
// t direction
if(c_dimBreak[3] == true){
  if(x[3] == c_localL[3] -1)
    pointPlus[3] = c_plusGhost[3]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_ZYX(x[2],x[1],x[0],c_localL);
  if(x[3] == 0){
    pointMinus[3] = c_minusGhost[3]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_ZYX(x[2],x[1],x[0],c_localL);
    pointMinusG[3] = c_minusGhost[3]*c_nDim*c_nColor*c_nColor + LEXIC_ZYX(x[2],x[1],x[0],c_localL);
  }
 }



// we have for different combination which have to do with forward and backward derivative
// the order is sequential propagator , gauge , forward propagator
double2 seqprop[4][4][3][3];
double2 prop[4][4][3][3];
double2 gauge[3][3];
double2 reduction;

reduction.x = 0.;
reduction.y = 0.;


//----------------------------------
for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	seqprop[mu][nu][c1][c2] = fetch_double2(seqPropagatorTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }
//-----------------------------------

//  x x x+dir (minus) ========================================================================================================================================
for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,sid + ( (dir*3*3 + c1*3 + c2) * c_stride ) );

if(c_dimBreak[dir] == true && (x[dir] == c_localL[dir]-1) ){
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = fetch_double2(fwdPropagatorTex,pointPlus[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_surface[dir]);
      }  
 }
 else{

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = fetch_double2(fwdPropagatorTex,pointPlus[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }  

 }

for(int ku = 0 ; ku < 4 ; ku++) // idx1
  for(int lu = 0 ; lu < 4 ; lu++) // idx3
    for(int pu = 0 ; pu < 4 ; pu++) // idx2
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++)
	  for(int c3 = 0 ; c3 < 3 ; c3++){
	    if(norm(OneMgamma[ku][lu]) > 1e-4 ) reduction =  reduction - seqprop[ku][pu][c1][c2] * OneMgamma[ku][lu] * gauge[c1][c3] * prop[lu][pu][c3][c2];
	  }

// x x-dir x-dir  (plus) =======================================================================================================================================
if(c_dimBreak[dir] == true && x[dir] == 0){

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,pointMinusG[dir] + ( (dir*3*3 + c1*3 + c2) * c_surface[dir] ) );

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = fetch_double2(fwdPropagatorTex,pointMinus[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_surface[dir]);
      }  

 }
 else{

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,pointMinusG[dir] + ( (dir*3*3 + c1*3 + c2) * c_stride ) );

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = fetch_double2(fwdPropagatorTex,pointMinus[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }  


 }


for(int ku = 0 ; ku < 4 ; ku++) // idx1
  for(int lu = 0 ; lu < 4 ; lu++) // idx3
    for(int pu = 0 ; pu < 4 ; pu++) // idx2
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++)
	  for(int c3 = 0 ; c3 < 3 ; c3++){
	    if(norm(OnePgamma[ku][lu]) > 1e-4 ) reduction =  reduction + seqprop[ku][pu][c1][c2] * OnePgamma[ku][lu] * conj(gauge[c3][c1]) * prop[lu][pu][c3][c2];
	  }

//-----------------------------------------------------
for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	prop[mu][nu][c1][c2] = fetch_double2(fwdPropagatorTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }
//-----------------------------------------------------

// x+dir x x (plus) =============================================================================================================================================

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,sid + ( (dir*3*3 + c1*3 + c2) * c_stride ) );

if(c_dimBreak[dir] == true && (x[dir] == c_localL[dir]-1) ){
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  seqprop[mu][nu][c1][c2] = fetch_double2(seqPropagatorTex,pointPlus[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_surface[dir]);
      }  
 }
 else{

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  seqprop[mu][nu][c1][c2] = fetch_double2(seqPropagatorTex,pointPlus[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }  

 }

for(int ku = 0 ; ku < 4 ; ku++) // idx1
  for(int lu = 0 ; lu < 4 ; lu++) // idx3
    for(int pu = 0 ; pu < 4 ; pu++) // idx2
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++)
	  for(int c3 = 0 ; c3 < 3 ; c3++){
	    if(norm(OnePgamma[ku][lu]) > 1e-4 ) reduction =  reduction + seqprop[ku][pu][c1][c2] * OnePgamma[ku][lu] * conj(gauge[c3][c1]) * prop[lu][pu][c3][c2];
	  }


// x-dir x-dir x (minus) ==========================================================================================================================================
if(c_dimBreak[dir] == true && x[dir] == 0){

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,pointMinusG[dir] + ( (dir*3*3 + c1*3 + c2) * c_surface[dir] ) );

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  seqprop[mu][nu][c1][c2] = fetch_double2(seqPropagatorTex,pointMinus[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_surface[dir]);
      }  

 }
 else{

for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,pointMinusG[dir] + ( (dir*3*3 + c1*3 + c2) * c_stride ) );

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  seqprop[mu][nu][c1][c2] = fetch_double2(seqPropagatorTex,pointMinus[dir] + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }  


 }


for(int ku = 0 ; ku < 4 ; ku++) // idx1
  for(int lu = 0 ; lu < 4 ; lu++) // idx3
    for(int pu = 0 ; pu < 4 ; pu++) // idx2
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++)
	  for(int c3 = 0 ; c3 < 3 ; c3++){
	    if(norm(OneMgamma[ku][lu]) > 1e-4 ) reduction =  reduction - seqprop[ku][pu][c1][c2] * OneMgamma[ku][lu] * gauge[c1][c3] * prop[lu][pu][c3][c2];
	  }

//==========================================================================================================================


/*
if(sid == 0){
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      cuPrintf("%+e %+e\n",prop1[mu][nu][0][0].x,prop1[mu][nu][0][0].y);
 }
*/

out[sid] = 0.25 * reduction;
