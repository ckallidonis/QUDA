
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;
extern __shared__ volatile double2 sm[];


#pragma unroll
for(int i = 0 ; i < 12 ; i++){
  int isid = i*c_threads + sid;

  volatile double2 sum;
  sum.x=0.;
  sum.y=0.;


  for(int i = 0 ; i < M ; i++){
    sm[threadIdx.x*M+i].x = mNxM[i*NL+isid].x;
    sm[threadIdx.x*M+i].y = mNxM[i*NL+isid].y;
  }

  
  if( transpose == true){
    for(int i = 0 ; i < L ; i++){
      sum.x=0.; sum.y=0.;
      for(int j = 0 ; j < M ; j++){
	sum.x = sum.x + sm[threadIdx.x*M+j].x * c_matrixQ[i*M+j];
	sum.y = sum.y + sm[threadIdx.x*M+j].y * c_matrixQ[i*M+j];
      }
      mNxM[i*NL+isid].x = sum.x;
      mNxM[i*NL+isid].y = sum.y;
    }
  }
  else{
    for(int i = 0 ; i < L ; i++){
      sum.x=0.; sum.y=0.;
      for(int j = 0 ; j < M ; j++){
	sum.x = sum.x + sm[threadIdx.x*M+j].x * c_matrixQ[j*M+i];
	sum.y = sum.y + sm[threadIdx.x*M+j].y * c_matrixQ[j*M+i];
      }
      mNxM[i*NL+isid].x = sum.x;
      mNxM[i*NL+isid].y = sum.y;
    }
    
  }
  
 }
