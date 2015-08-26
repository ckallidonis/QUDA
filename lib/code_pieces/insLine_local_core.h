
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= c_stride) return;


double2 gamma[4][4];
for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int nu = 0 ; nu < c_nSpin ; nu++){
    gamma[mu][nu].x = 0 ; gamma[mu][nu].y = 0;
  }

switch( iflag ){
  
 case 0: // 1 -> ig5 or -ig5

   if(partFlag == 1){ // upart
     gamma[0][2].y=1.; gamma[1][3].y=1.; gamma[2][0].y=1.; gamma[3][1].y=1.;}
   else if(partFlag == 2){ // dpart
     gamma[0][2].y=-1.; gamma[1][3].y=-1.; gamma[2][0].y=-1.; gamma[3][1].y=-1.;}
   else if(partFlag == 3){                  // spart
     gamma[0][2].y=1.; gamma[1][3].y=1.; gamma[2][0].y=1.; gamma[3][1].y=1.;}
   


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

   if(partFlag == 1){ // upart
     gamma[0][0].y=1.; gamma[1][1].y=1.; gamma[2][2].y=1.; gamma[3][3].y=1.;}
   else if(partFlag == 2){ // dpart
     gamma[0][0].y=-1.; gamma[1][1].y=-1.; gamma[2][2].y=-1.; gamma[3][3].y=-1.;}
   else if(partFlag == 3){ // spart
     gamma[0][0].y=1.; gamma[1][1].y=1.; gamma[2][2].y=1.; gamma[3][3].y=1.;}

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



double2 prop[4][4][3][3];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	prop[mu][nu][c1][c2] = fetch_double2(propStochTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }

double2 Phi[4][3];

for(int pi = 0 ; pi < 4 ; pi++)
  for(int d = 0 ; d < 3 ; d++)
    Phi[pi][d] = fetch_double2(phiVectorStochTex,sid + (pi*3+d)*c_stride);

double2 outIns[4][3];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int c1 = 0 ; c1 < 3 ; c1++){
    outIns[mu][c1].x=0.; outIns[mu][c1].y=0.;
  }

double norma;

for(int pi = 0 ; pi < 4 ; pi++)
  for(int phi = 0 ; phi < 4 ; phi++){
    norma = norm(g5xg[pi][phi]);
    if(norma > 1e-3){
      for(int d = 0 ; d < 3 ; d++)
	for(int gammap = 0 ; gammap < 4 ; gammap++)
	  for(int cp = 0 ; cp < 3 ;cp++){
	    outIns[gammap][cp] = outIns[gammap][cp] + conj(Phi[pi][d]) * g5xg[pi][phi] * prop[phi][gammap][d][cp];
	  }

    }

  }


for(int mu = 0 ; mu < 4 ; mu++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
    out[sid + (mu*3+c1)*c_stride] = outIns[mu][c1]; 
