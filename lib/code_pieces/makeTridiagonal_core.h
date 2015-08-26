int ix = threadIdx.x;
int iy = blockIdx.x;

if( abs(ix-iy) > 1 ) A[iy*blockDim.x+ix] = 0.; 
