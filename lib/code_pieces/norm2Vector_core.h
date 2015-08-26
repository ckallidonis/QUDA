
__shared__ double shared_cache[THREADS_PER_BLOCK];

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

// take indices on 4d lattice
int cacheIndex = threadIdx.x;


double2 S_0 , S_1 , S_2 , S_3 , S_4 , S_5 , S_6 , S_7 , S_8 , S_9 , S_10 , S_11;


double norm = 0.;



  READVECTOR(S,vectorTexNorm2,sid,c_stride);
  
  norm += norm2(S_0);
  norm += norm2(S_1);
  norm += norm2(S_2);
  norm += norm2(S_3);
  norm += norm2(S_4);
  norm += norm2(S_5);
  norm += norm2(S_6);
  norm += norm2(S_7);
  norm += norm2(S_8);
  norm += norm2(S_9);
  norm += norm2(S_10);
  norm += norm2(S_11);


shared_cache[cacheIndex] = norm;

__syncthreads();

int i = blockDim.x/2;

while (i != 0){

  if(cacheIndex < i)
    shared_cache[cacheIndex] += shared_cache[cacheIndex + i];

  __syncthreads();
  i /= 2;

 }

if(cacheIndex == 0)
  cache[blockIdx.x] = shared_cache[0];   // write result back to global memory


/*
//////////// mu = 0
READGAUGE(G,gaugeTexNorm2,0,sid,c_stride);


////////////// mu = 1
READGAUGE(G,gaugeTexNorm2,1,sid,c_stride);

////////////// mu = 2
READGAUGE(G,gaugeTexNorm2,2,sid,c_stride);

/////////////// mu = 3

READGAUGE(G,gaugeTexNorm2,3,sid,c_stride);


*/
