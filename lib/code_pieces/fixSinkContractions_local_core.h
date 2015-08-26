
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= c_stride) return;


double2 gamma[4][4];
for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int nu = 0 ; nu < c_nSpin ; nu++){
    gamma[mu][nu].x = 0 ; gamma[mu][nu].y = 0;
  }

switch( flag ){
  
 case 0: // 1 -> ig5 or -ig5

   if(partFlag == 1){
     if(testParticle == QKXTM_PROTON){
       gamma[0][2].y=1.; gamma[1][3].y=1.; gamma[2][0].y=1.; gamma[3][1].y=1.;}
     else{
       gamma[0][2].y=-1.; gamma[1][3].y=-1.; gamma[2][0].y=-1.; gamma[3][1].y=-1.;}
   }
   else{
     if(testParticle == QKXTM_PROTON){
       gamma[0][2].y=-1.; gamma[1][3].y=-1.; gamma[2][0].y=-1.; gamma[3][1].y=-1.;}
     else{
       gamma[0][2].y=1.; gamma[1][3].y=1.; gamma[2][0].y=1.; gamma[3][1].y=1.;}
   }
     
 
   break;
 case 1: // g1
   gamma[3][0].y=-1.; gamma[2][1].y=-1.; gamma[1][2].y=1.; gamma[0][3].y=1.;
   break;
 case 2: // g2
   gamma[3][0].x=1.; gamma[2][1].x=-1.; gamma[1][2].x=-1.; gamma[0][3].x=1.;
   break;
 case 3: // g3
   gamma[0][2].y=1.; gamma[1][3].y=-1.; gamma[2][0].y=-1.; gamma[3][1].y=1.;
   break;
 case 4: // g4 
   gamma[0][0].x=1.; gamma[1][1].x=1.; gamma[2][2].x=-1.; gamma[3][3].x=-1.;
   break;
 case 5: // g5 -> i or -i

   if(partFlag == 1){
     if(testParticle == QKXTM_PROTON){
       gamma[0][0].y=1.; gamma[1][1].y=1.; gamma[2][2].y=1.; gamma[3][3].y=1.;}
     else{
       gamma[0][0].y=-1.; gamma[1][1].y=-1.; gamma[2][2].y=-1.; gamma[3][3].y=-1.;}
   }
   else{
     if(testParticle == QKXTM_PROTON){
       gamma[0][0].y=-1.; gamma[1][1].y=-1.; gamma[2][2].y=-1.; gamma[3][3].y=-1.;}
     else{
       gamma[0][0].y=1.; gamma[1][1].y=1.; gamma[2][2].y=1.; gamma[3][3].y=1.;}
   }

   break;
 case 6: // g5g1
   gamma[0][1].y=-1.; gamma[1][0].y=-1.; gamma[2][3].y=1.; gamma[3][2].y=1.;
   break;
 case 7: // g5g2
   gamma[0][1].x=-1.; gamma[1][0].x=1.; gamma[2][3].x=1.; gamma[3][2].x=-1.;
   break;
 case 8: // g5g3
   gamma[0][0].y=-1.; gamma[1][1].y=1.; gamma[2][2].y=1.; gamma[3][3].y=-1.;
   break;
 case 9: // g5g4
   gamma[0][2].x=-1.; gamma[1][3].x=-1.; gamma[2][0].x=1.; gamma[3][1].x=1.;
   break;

 }





double2 seqprop[4][4][3][3];
double2 prop[4][4][3][3];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	seqprop[mu][nu][c1][c2] = fetch_double2(seqPropagatorTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
	prop[mu][nu][c1][c2] = fetch_double2(fwdPropagatorTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }

/*
if(sid == 0){
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      cuPrintf("%+e %+e\n",prop1[mu][nu][0][0].x,prop1[mu][nu][0][0].y);
 }
*/
double2 reduction;
reduction.x=0.;
reduction.y=0.;

for(int nu = 0 ; nu < c_nSpin ; nu++)
  for(int rho = 0 ; rho < c_nSpin ; rho++)
    for(int mup = 0 ; mup < c_nSpin ; mup++)
      for(int b = 0 ; b < c_nColor ; b++)
	for(int ap = 0 ; ap < c_nColor ; ap++){
	  reduction = reduction + gamma[nu][rho] * prop[rho][mup][b][ap] * seqprop[nu][mup][b][ap];
	}

out[sid] = reduction;
