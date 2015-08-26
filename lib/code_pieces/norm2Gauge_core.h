
__shared__ double shared_cache[THREADS_PER_BLOCK];

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

// take indices on 4d lattice
int cacheIndex = threadIdx.x;


double2 G_0 , G_1 , G_2 , G_3 , G_4 , G_5 , G_6 , G_7 , G_8;

double norm = 0.;

for(int dir = 0 ; dir < 4 ; dir++){

  READGAUGE(G,gaugeTexNorm2,dir,sid,c_stride);
  
  norm += norm2(G_0);
  norm += norm2(G_1);
  norm += norm2(G_2);
  norm += norm2(G_3);
  norm += norm2(G_4);
  norm += norm2(G_5);
  norm += norm2(G_6);
  norm += norm2(G_7);
  norm += norm2(G_8);

 }


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
