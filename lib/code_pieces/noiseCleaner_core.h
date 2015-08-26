int sid = blockIdx.x*blockDim.x + threadIdx.x;
extern __shared__ double shared_cache[];

shared_cache[sid] = norm(A[sid]);
int i = blockDim.x/2;

while (i != 0){
  if(threadIdx.x < i)
    shared_cache[threadIdx.x] += shared_cache[threadIdx.x + i];
  __syncthreads();
  i /= 2;
}

double matrixOrder = shared_cache[0];

if( norm(A[sid]) < c_tolArnoldi/matrixOrder ){
  A[sid].x = 0;
  A[sid].y = 0;
 }



if( fabs(A[sid].x) < c_tolArnoldi/matrixOrder) A[sid].x = 0;
if( fabs(A[sid].y) < c_tolArnoldi/matrixOrder) A[sid].y = 0;

