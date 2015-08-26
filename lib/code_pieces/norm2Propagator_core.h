
__shared__ double shared_cache[THREADS_PER_BLOCK];

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

// take indices on 4d lattice
int cacheIndex = threadIdx.x;


double2 P[144];


double norm = 0.;



for(int i = 0 ; i < 144 ; i++){
  P[i] = fetch_double2(propagatorTexNorm2, sid +  i  * c_stride);
 }
  
for(int i = 0 ; i < 144 ; i++){
  norm += norm2(P[i]);
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
