#define idx(i,x,y,sid) ((i)*c_nColor*c_nColor*c_stride + (x)*c_nColor*c_stride + (y)*c_stride + sid)
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

switch(direction){
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


switch(direction){
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


double2 seqprop[4][4][3][3];
double2 prop[4][4][3][3];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	seqprop[mu][nu][c1][c2] = fetch_double2(seqPropagatorTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
	prop[mu][nu][c1][c2] = fetch_double2(fwdPropagatorTex,pointPlus + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
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


double2 WilsonLink[3][3];
for(int c1 =0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    WilsonLink[c1][c2] =deviceWilsonLinks[idx(dl,c1,c2,sid)];


/*
double2 gauge[3][3];
int direction = 2;
for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
    gauge[c1][c2] = fetch_double2(gaugeDerivativeTex,sid + ( (direction*3*3 + c1*3 + c2) * c_stride ) );
*/

for(int nu = 0 ; nu < c_nSpin ; nu++)
  for(int rho = 0 ; rho < c_nSpin ; rho++)
    for(int mup = 0 ; mup < c_nSpin ; mup++)
      for(int b = 0 ; b < c_nColor ; b++)
	for(int ap = 0 ; ap < c_nColor ; ap++)
	  for(int c = 0 ; c < c_nColor ; c++){
	    reduction = reduction + gamma[nu][rho] * prop[rho][mup][c][ap] * WilsonLink[b][c] * seqprop[nu][mup][b][ap];
	    // reduction = reduction + gamma[nu][rho] * prop[rho][mup][c][ap] * conj(WilsonLink[c][b]) * seqprop[nu][mup][b][ap];
	  //	  reduction = reduction + gamma[nu][rho] * prop[rho][mup][b][ap] * seqprop[nu][mup][b][ap];
	}

out[sid] = reduction;

#undef idx
