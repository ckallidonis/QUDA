
int space_stride = c_localL[0]*c_localL[1]*c_localL[2];

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= space_stride) return;


double2 projector[4][4];
for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int nu = 0 ; nu < c_nSpin ; nu++){
    projector[mu][nu].x = 0 ; projector[mu][nu].y = 0;
  }



if(typeProj == QKXTM_TYPE1){

  if(testParticle == QKXTM_PROTON){ //  0.5*(eye(4)+i*g5_uk)*0.25*( eye(4) + g4_uk ) * ( eye(4) + i*g5_uk)

    projector[0][0].x=0.25;
    projector[1][1].x=0.25;
    projector[2][2].x=-0.25;
    projector[3][3].x=-0.25;
    projector[0][2].y=0.25;
    projector[1][3].y=0.25;
    projector[2][0].y=0.25;
    projector[3][1].y=0.25;

  }
  else{                          // 0.5*(eye(4)-i*g5_uk)*0.25*( eye(4) + g4_uk ) * ( eye(4) - i*g5_uk)

    projector[0][0].x=0.25;
    projector[1][1].x=0.25;
    projector[2][2].x=-0.25;
    projector[3][3].x=-0.25;
    projector[0][2].y=-0.25;
    projector[1][3].y=-0.25;
    projector[2][0].y=-0.25;
    projector[3][1].y=-0.25;

  }
  //  projector[0][0].x = 0.5;
  // projector[1][1].x = 0.5;
 }
 else{
   //   projector[0][0].x = 0.5;
   // projector[0][1].x = 0.5;  projector[0][1].y = -0.5;
   // projector[1][0].x = 0.5;  projector[1][0].y = 0.5;
   // projector[1][1].x = -0.5;

   if(testParticle == QKXTM_PROTON){             //0.5*(eye(4)+i*g5_uk)*0.25*(eye(4)+g4_uk)*i*g5_uk*(g1_uk+g2_uk+g3_uk)*(eye(4)+i*g5_uk)
     projector[0][0].x = 0.25;
     projector[0][1].x = 0.25; projector[0][1].y = -0.25;
     projector[0][2].y = 0.25;
     projector[0][3].x = 0.25; projector[0][3].y = 0.25;
     projector[1][0].x = 0.25; projector[1][0].y = 0.25;
     projector[1][1].x = -0.25;
     projector[1][2].x = -0.25; projector[1][2].y = 0.25;
     projector[1][3].y = -0.25;
     projector[2][0].y = 0.25;
     projector[2][1].x = 0.25; projector[2][1].y = 0.25;
     projector[2][2].x = -0.25;
     projector[2][3].x = -0.25; projector[2][3].y = 0.25;
     projector[3][0].x = -0.25; projector[3][0].y = 0.25;
     projector[3][1].y = -0.25;
     projector[3][2].x = -0.25; projector[3][2].y = -0.25;
     projector[3][3].x = 0.25;
   }
   else{                                             //  0.5*(eye(4)-i*g5_uk)*0.25*(eye(4)+g4_uk)*i*g5_uk*(g1_uk+g2_uk+g3_uk)*(eye(4)-i*g5_uk)
     projector[0][0].x = 0.25;
     projector[0][1].x = 0.25; projector[0][1].y = -0.25;
     projector[0][2].y = -0.25;
     projector[0][3].x = -0.25; projector[0][3].y = -0.25;
     projector[1][0].x = 0.25; projector[1][0].y = 0.25;
     projector[1][1].x = -0.25;
     projector[1][2].x = 0.25; projector[1][2].y = -0.25;
     projector[1][3].y = 0.25;
     projector[2][0].y = -0.25;
     projector[2][1].x = -0.25; projector[2][1].y = -0.25;
     projector[2][2].x = -0.25;
     projector[2][3].x = -0.25; projector[2][3].y = 0.25;
     projector[3][0].x = 0.25; projector[3][0].y = -0.25;
     projector[3][1].y = 0.25;
     projector[3][2].x = -0.25; projector[3][2].y = -0.25;
     projector[3][3].x = 0.25;

   }


 }


/*
int x_id, y_id , z_id;
int r1,r2;

r1 = sid / c_localL[0];
x_id = sid - r1 * c_localL[0];
r2 = r1 / c_localL[1];
y_id = r1 - r2*c_localL[1];
z_id = r2;
*/
double2 Cg5[4][4];
double2 Cg5_bar[4][4];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++){
    Cg5[mu][nu].x = 0.; Cg5[mu][nu].y=0.;
    Cg5_bar[mu][nu].x = 0.; Cg5_bar[mu][nu].y=0.;
  }

Cg5[0][1].x = 1.;
Cg5[1][0].x = -1.;
Cg5[2][3].x = 1.;
Cg5[3][2].x = -1.;

Cg5_bar[0][1].x = -1.;
Cg5_bar[1][0].x = 1.;
Cg5_bar[2][3].x = -1.;
Cg5_bar[3][2].x = 1.;



double2 C_temp;
double2 Cg5Cg5bar_val[16*16];
unsigned short int Cg5Cg5bar_ind[16*16][4];
int counter = 0;
// adopt Thomas convention

for(unsigned short int mu = 0 ; mu < 4 ; mu++)
  for(unsigned short int nu = 0 ; nu < 4 ; nu++)
    for(unsigned short int ku = 0 ; ku < 4 ; ku++)
      for(unsigned short int lu = 0 ; lu < 4 ; lu++){
	C_temp = Cg5[mu][nu] * Cg5_bar[ku][lu];
	if( norm(C_temp) > 1e-3 ){
	  Cg5Cg5bar_val[counter] = C_temp;
	  Cg5Cg5bar_ind[counter][0] = mu;
	  Cg5Cg5bar_ind[counter][1] = nu;
	  Cg5Cg5bar_ind[counter][2] = ku;
	  Cg5Cg5bar_ind[counter][3] = lu;
	  counter++;
	}

      }

double2 prop1[4][4][3][3];
double2 prop2[4][4][3][3];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	prop1[mu][nu][c1][c2] = fetch_double2(propagator3DTex1,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * space_stride);
	prop2[mu][nu][c1][c2] = fetch_double2(propagator3DTex2,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * space_stride);
      }

/*
if(sid == 0){
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      cuPrintf("%+e %+e\n",prop1[mu][nu][0][0].x,prop1[mu][nu][0][0].y);
 }
*/

unsigned short int mu,gu,ku,ju,c1,c2,c3,c1p,c2p,c3p;
double2 factor;
double2 spinor[4][3];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int ic = 0 ; ic < 3 ; ic++){
    spinor[mu][ic].x = 0.; spinor[mu][ic].y = 0.;
  }
  

for(int cc1 = 0 ; cc1 < 6 ; cc1++){
  c1 = c_eps[cc1][0];
  c2 = c_eps[cc1][1];
  c3 = c_eps[cc1][2];
  for(int cc2 = 0 ; cc2 < 6 ; cc2++){
    c1p = c_eps[cc2][0];
    c2p = c_eps[cc2][1];
    c3p = c_eps[cc2][2];
    if(c3p == c_c2){
      for(int idx = 0 ; idx < counter ; idx++){
	mu = Cg5Cg5bar_ind[idx][0];
	gu = Cg5Cg5bar_ind[idx][1];
	ku = Cg5Cg5bar_ind[idx][2];
	ju = Cg5Cg5bar_ind[idx][3];
	
	for(int a = 0 ; a < 4 ; a++)
	  for(int b = 0 ; b < 4 ; b++)
	    if( norm(projector[b][a]) > 1e-3 ){
	      
	      factor = (-1)*c_sgn_eps[cc1]*c_sgn_eps[cc2]*projector[b][a]*Cg5Cg5bar_val[idx];
	      
	      for(int nu = 0 ; nu < 4 ; nu++){
		if( mu == nu && b == c_nu ) spinor[nu][c3] = spinor[nu][c3] + factor * prop2[gu][ju][c1][c1p] * prop1[a][ku][c2][c2p];
		if( mu == nu && ku == c_nu ) spinor[nu][c3] = spinor[nu][c3] + factor * prop2[gu][ju][c1][c1p] * prop1[a][b][c2][c2p];
		if( a == nu && b == c_nu ) spinor[nu][c3] = spinor[nu][c3] + factor * prop2[gu][ju][c1][c1p] * prop1[mu][ku][c2][c2p];
		if( a == nu && ku == c_nu ) spinor[nu][c3] = spinor[nu][c3] + factor * prop2[gu][ju][c1][c1p] * prop1[mu][b][c2][c2p];
	      }
	      
	    } // if statement
	
	
      } // close spin

    } // if statement      
  } // close color
 } // close color

/*
if(sid == 0){
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int ic = 0 ; ic < 3 ; ic++)
      cuPrintf("%+e %+e\n",spinor[mu][ic].x,spinor[mu][ic].y);
 }
*/
for(int mu = 0 ; mu < 4 ; mu++)
  for(int ic = 0 ; ic < 3 ; ic++)
    out[mu*c_nColor*c_stride + ic*c_stride + timeslice*space_stride + sid] = spinor[mu][ic];
