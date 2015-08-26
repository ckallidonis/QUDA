#define idx(i,x,y,sid) ((i)*c_nColor*c_nColor*c_stride + (x)*c_nColor*c_stride + (y)*c_stride + sid)
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= c_threads) return;

int x_id, y_id, z_id, t_id;
int r1,r2;
r1 = sid/(c_localL[0]);
r2 = r1/(c_localL[1]);
x_id = sid - r1*(c_localL[0]);
y_id = r1 - r2*(c_localL[1]);
t_id = r2/(c_localL[2]);
z_id = r2 - t_id*(c_localL[2]);

// this kernel works only if we dont break the z direction where we boost the particle

int pointPlus;
double2 unity;
double2 zero;

unity.x = 1.; unity.y = 0.;
zero.x = 0.; zero.y = 0.;

double2 G3_0 , G3_1 , G3_2 , G3_3 , G3_4 , G3_5 , G3_6 , G3_7 , G3_8;


for(int i = 0 ; i < (c_totalL[direction]/2) ; i++){

  if(i==0){
    deviceWilsonPath[idx(i,0,0,sid)] =unity; 
    deviceWilsonPath[idx(i,0,1,sid)] =zero; 
    deviceWilsonPath[idx(i,0,2,sid)] =zero; 
    deviceWilsonPath[idx(i,1,0,sid)] =zero; 
    deviceWilsonPath[idx(i,1,1,sid)] =unity; 
    deviceWilsonPath[idx(i,1,2,sid)] =zero; 
    deviceWilsonPath[idx(i,2,0,sid)] =zero; 
    deviceWilsonPath[idx(i,2,1,sid)] =zero; 
    deviceWilsonPath[idx(i,2,2,sid)] =unity; 
    }
  else if(i==1){
    READGAUGE(G3,gaugePath,direction,sid,c_stride); // direction is 2 for z direction
    deviceWilsonPath[idx(i,0,0,sid)] =G3_0; 
    deviceWilsonPath[idx(i,0,1,sid)] =G3_1; 
    deviceWilsonPath[idx(i,0,2,sid)] =G3_2; 
    deviceWilsonPath[idx(i,1,0,sid)] =G3_3; 
    deviceWilsonPath[idx(i,1,1,sid)] =G3_4; 
    deviceWilsonPath[idx(i,1,2,sid)] =G3_5; 
    deviceWilsonPath[idx(i,2,0,sid)] =G3_6; 
    deviceWilsonPath[idx(i,2,1,sid)] =G3_7; 
    deviceWilsonPath[idx(i,2,2,sid)] =G3_8; 

  }
  else{
    if(direction==0)
      pointPlus = LEXIC(t_id,z_id,y_id,(x_id+i-1)%c_localL[0],c_localL);
    if(direction==1)
      pointPlus = LEXIC(t_id,z_id,(y_id+i-1)%c_localL[1],x_id,c_localL);
    if(direction==2)
      pointPlus = LEXIC(t_id,(z_id+i-1)%c_localL[2],y_id,x_id,c_localL);
    if(direction==3)
      pointPlus = LEXIC((t_id+i-1)%c_localL[3],z_id,y_id,x_id,c_localL);

    READGAUGE(G3,gaugePath,2,pointPlus,c_stride);
    int stepBack = i-1;
    deviceWilsonPath[idx(i,0,0,sid)] = deviceWilsonPath[idx(stepBack,0,0,sid)] * G3_0 + deviceWilsonPath[idx(stepBack,0,1,sid)]*G3_3 + deviceWilsonPath[idx(stepBack,0,2,sid)] * G3_6;
    deviceWilsonPath[idx(i,0,1,sid)] = deviceWilsonPath[idx(stepBack,0,0,sid)] * G3_1 + deviceWilsonPath[idx(stepBack,0,1,sid)]*G3_4 + deviceWilsonPath[idx(stepBack,0,2,sid)] * G3_7;
    deviceWilsonPath[idx(i,0,2,sid)] = deviceWilsonPath[idx(stepBack,0,0,sid)] * G3_2 + deviceWilsonPath[idx(stepBack,0,1,sid)]*G3_5 + deviceWilsonPath[idx(stepBack,0,2,sid)] * G3_8;

    deviceWilsonPath[idx(i,1,0,sid)] = deviceWilsonPath[idx(stepBack,1,0,sid)] * G3_0 + deviceWilsonPath[idx(stepBack,1,1,sid)]*G3_3 + deviceWilsonPath[idx(stepBack,1,2,sid)] * G3_6;
    deviceWilsonPath[idx(i,1,1,sid)] = deviceWilsonPath[idx(stepBack,1,0,sid)] * G3_1 + deviceWilsonPath[idx(stepBack,1,1,sid)]*G3_4 + deviceWilsonPath[idx(stepBack,1,2,sid)] * G3_7;
    deviceWilsonPath[idx(i,1,2,sid)] = deviceWilsonPath[idx(stepBack,1,0,sid)] * G3_2 + deviceWilsonPath[idx(stepBack,1,1,sid)]*G3_5 + deviceWilsonPath[idx(stepBack,1,2,sid)] * G3_8;

    deviceWilsonPath[idx(i,2,0,sid)] = deviceWilsonPath[idx(stepBack,2,0,sid)] * G3_0 + deviceWilsonPath[idx(stepBack,2,1,sid)]*G3_3 + deviceWilsonPath[idx(stepBack,2,2,sid)] * G3_6;
    deviceWilsonPath[idx(i,2,1,sid)] = deviceWilsonPath[idx(stepBack,2,0,sid)] * G3_1 + deviceWilsonPath[idx(stepBack,2,1,sid)]*G3_4 + deviceWilsonPath[idx(stepBack,2,2,sid)] * G3_7;
    deviceWilsonPath[idx(i,2,2,sid)] = deviceWilsonPath[idx(stepBack,2,0,sid)] * G3_2 + deviceWilsonPath[idx(stepBack,2,1,sid)]*G3_5 + deviceWilsonPath[idx(stepBack,2,2,sid)] * G3_8;


  }


 }

#undef idx
