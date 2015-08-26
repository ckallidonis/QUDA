#define idx(dir,i,x,y,sid) ( dir*(c_totalL[0]/2)*c_nColor*c_nColor*c_stride + (i)*c_nColor*c_nColor*c_stride + (x)*c_nColor*c_stride + (y)*c_stride + sid)
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= c_stride) return;

int x_id, y_id, z_id, t_id;
int r1,r2;
r1 = sid/(c_localL[0]);
r2 = r1/(c_localL[1]);
x_id = sid - r1*(c_localL[0]);
y_id = r1 - r2*(c_localL[1]);
t_id = r2/(c_localL[2]);
z_id = r2 - t_id*(c_localL[2]);

int pointPlus;

switch(dir){
 case 0:
   pointPlus = LEXIC(t_id,z_id,y_id,(x_id+dl)%c_localL[0],c_localL);
   break;
 case 1:
   pointPlus = LEXIC(t_id,z_id,(y_id+dl)%c_localL[1],x_id,c_localL);
   break;
 case 2:
   pointPlus = LEXIC(t_id,(z_id+dl)%c_localL[2],y_id,x_id,c_localL);
   break;
 }

double2 gamma[4][4];
for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int nu = 0 ; nu < c_nSpin ; nu++){
    gamma[mu][nu].x = 0 ; gamma[mu][nu].y = 0;
  }

switch(dir){
 case 0:
   gamma[3][0].y=-1.; gamma[2][1].y=-1.; gamma[1][2].y=1.; gamma[0][3].y=1.; //g1
   break;
 case 1:
   gamma[3][0].x=1.; gamma[2][1].x=-1.; gamma[1][2].x=-1.; gamma[0][3].x=1.; //g2
   break;
 case 2:
   gamma[0][2].y=1.; gamma[1][3].y=-1.; gamma[2][0].y=-1.; gamma[3][1].y=1.; // g3
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
	prop[mu][nu][c1][c2] = fetch_double2(propStochTex,pointPlus + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }

double2 Phi[4][3];
for(int pi = 0 ; pi < 4 ; pi++)
  for(int d = 0 ; d < 3 ; d++)
    Phi[pi][d] = fetch_double2(phiVectorStochTex,sid + (pi*3+d)*c_stride);

double2 WilsonLink[3][3];
for(int c1 =0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    WilsonLink[c1][c2] =deviceWilsonLinks[idx(dir,dl,c1,c2,sid)] ;

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
	for(int gammap = 0 ; gammap < 4 ; gammap++)
	  for(int cp = 0 ; cp < 3 ;cp++){
	    for(int c1 = 0 ; c1 < 3 ; c1++)
	      for(int c2 = 0 ; c2 < 3 ; c2++)
		outIns[gammap][cp] = outIns[gammap][cp] + conj(Phi[pi][c1]) * g5xg[pi][phi] * WilsonLink[c1][c2] * prop[phi][gammap][c2][cp];
	  }

    }

  }


for(int mu = 0 ; mu < 4 ; mu++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
    out[sid + (mu*3+c1)*c_stride] = outIns[mu][c1]; 

#undef idx
