
//#include <core_def.h>

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;


int prp_position;
 
double2 G1_0 , G1_1 , G1_2 , G1_3 , G1_4 , G1_5 , G1_6 , G1_7 , G1_8; // fetch input plaquette
double2 GP_0 , GP_1 , GP_2 , GP_3 , GP_4 , GP_5 , GP_6 , GP_7 , GP_8; // fetch from prop structure

//////////////// S20 //////////////////////
prp_position = 2*c_nSpin*c_nColor*c_nColor*c_stride + 0*c_nColor*c_nColor*c_stride;
READPROPGAUGE(GP,propagatorTexHYP,2,0,sid,c_stride);
READGAUGE(G1,gaugeTexHYP,2,sid,c_stride);

prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_0 + (omega2/4.)*GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_1 + (omega2/4.)*GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_2 + (omega2/4.)*GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_3 + (omega2/4.)*GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_4 + (omega2/4.)*GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_5 + (omega2/4.)*GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_6 + (omega2/4.)*GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_7 + (omega2/4.)*GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_8 + (omega2/4.)*GP_8;

///////////////////////////////////////////

//////////////// S21 //////////////////////
prp_position = 2*c_nSpin*c_nColor*c_nColor*c_stride + 1*c_nColor*c_nColor*c_stride;
READPROPGAUGE(GP,propagatorTexHYP,2,1,sid,c_stride);
READGAUGE(G1,gaugeTexHYP,2,sid,c_stride);

prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_0 + (omega2/4.)*GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_1 + (omega2/4.)*GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_2 + (omega2/4.)*GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_3 + (omega2/4.)*GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_4 + (omega2/4.)*GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_5 + (omega2/4.)*GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_6 + (omega2/4.)*GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_7 + (omega2/4.)*GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_8 + (omega2/4.)*GP_8;

///////////////////////////////////////////

//////////////// S12 //////////////////////
prp_position = 1*c_nSpin*c_nColor*c_nColor*c_stride + 2*c_nColor*c_nColor*c_stride;
READPROPGAUGE(GP,propagatorTexHYP,1,2,sid,c_stride);
READGAUGE(G1,gaugeTexHYP,1,sid,c_stride);

prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_0 + (omega2/4.)*GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_1 + (omega2/4.)*GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_2 + (omega2/4.)*GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_3 + (omega2/4.)*GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_4 + (omega2/4.)*GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_5 + (omega2/4.)*GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_6 + (omega2/4.)*GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_7 + (omega2/4.)*GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_8 + (omega2/4.)*GP_8;

///////////////////////////////////////////

//////////////// S10 //////////////////////
prp_position = 1*c_nSpin*c_nColor*c_nColor*c_stride + 0*c_nColor*c_nColor*c_stride;
READPROPGAUGE(GP,propagatorTexHYP,1,0,sid,c_stride);
READGAUGE(G1,gaugeTexHYP,1,sid,c_stride);

prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_0 + (omega2/4.)*GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_1 + (omega2/4.)*GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_2 + (omega2/4.)*GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_3 + (omega2/4.)*GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_4 + (omega2/4.)*GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_5 + (omega2/4.)*GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_6 + (omega2/4.)*GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_7 + (omega2/4.)*GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_8 + (omega2/4.)*GP_8;

///////////////////////////////////////////

//////////////// S02 //////////////////////
prp_position = 0*c_nSpin*c_nColor*c_nColor*c_stride + 2*c_nColor*c_nColor*c_stride;
READPROPGAUGE(GP,propagatorTexHYP,0,2,sid,c_stride);
READGAUGE(G1,gaugeTexHYP,0,sid,c_stride);

prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_0 + (omega2/4.)*GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_1 + (omega2/4.)*GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_2 + (omega2/4.)*GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_3 + (omega2/4.)*GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_4 + (omega2/4.)*GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_5 + (omega2/4.)*GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_6 + (omega2/4.)*GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_7 + (omega2/4.)*GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_8 + (omega2/4.)*GP_8;

///////////////////////////////////////////


//////////////// S01 //////////////////////
prp_position = 0*c_nSpin*c_nColor*c_nColor*c_stride + 1*c_nColor*c_nColor*c_stride;
READPROPGAUGE(GP,propagatorTexHYP,0,1,sid,c_stride);
READGAUGE(G1,gaugeTexHYP,0,sid,c_stride);

prp2[prp_position + 0*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_0 + (omega2/4.)*GP_0;
prp2[prp_position + 0*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_1 + (omega2/4.)*GP_1;
prp2[prp_position + 0*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_2 + (omega2/4.)*GP_2;
prp2[prp_position + 1*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_3 + (omega2/4.)*GP_3;
prp2[prp_position + 1*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_4 + (omega2/4.)*GP_4;
prp2[prp_position + 1*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_5 + (omega2/4.)*GP_5;
prp2[prp_position + 2*c_nColor*c_stride + 0*c_stride + sid] = (1-omega2)*G1_6 + (omega2/4.)*GP_6;
prp2[prp_position + 2*c_nColor*c_stride + 1*c_stride + sid] = (1-omega2)*G1_7 + (omega2/4.)*GP_7;
prp2[prp_position + 2*c_nColor*c_stride + 2*c_stride + sid] = (1-omega2)*G1_8 + (omega2/4.)*GP_8;

///////////////////////////////////////////

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

int mu[6];
int nu[6];

mu[0] = 2 ; nu[0] = 0;
mu[1] = 2 ; nu[1] = 1;
mu[2] = 1 ; nu[2] = 2;
mu[3] = 1 ; nu[3] = 0;
mu[4] = 0 ; nu[4] = 2;
mu[5] = 0 ; nu[5] = 1; 

for(int el = 0 ; el < 6 ; el++){
  prp_position = mu[el]*c_nSpin*c_nColor*c_nColor*c_stride + nu[el]*c_nColor*c_nColor*c_stride;
  for(int c1 = 0 ; c1 < 3 ; c1++)
    for(int c2 = 0 ; c2 < 3 ; c2++)
      M[c1][c2] = prp2[prp_position + c1*c_nColor*c_stride + c2*c_stride + sid]; // load link to register

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
	prp2[prp_position + c1*c_nColor*c_stride + c2*c_stride + sid] = U[c1][c2];

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
        prp2[prp_position + c1*c_nColor*c_stride + c2*c_stride + sid] = U[c1][c2];


  }

 } // close for over mu
