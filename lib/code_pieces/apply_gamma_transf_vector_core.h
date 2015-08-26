int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

double2 spinor[4][3];
double sqrt2 = sqrt(2.);

for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int c1 = 0 ; c1 < c_nColor ; c1++)
    spinor[mu][c1] = inOut[(mu*c_nColor+c1)*c_stride + sid];

for(int c1 = 0 ; c1 < c_nColor ; c1++){
  inOut[(0*c_nColor+c1)*c_stride + sid].x = (spinor[2][c1].x - spinor[0][c1].x)/sqrt2;
  inOut[(1*c_nColor+c1)*c_stride + sid].x = (spinor[3][c1].x - spinor[1][c1].x)/sqrt2;
  inOut[(2*c_nColor+c1)*c_stride + sid].x = (spinor[0][c1].x + spinor[2][c1].x)/sqrt2;
  inOut[(3*c_nColor+c1)*c_stride + sid].x = (spinor[1][c1].x + spinor[3][c1].x)/sqrt2;

  inOut[(0*c_nColor+c1)*c_stride + sid].y = (spinor[2][c1].y - spinor[0][c1].y)/sqrt2;
  inOut[(1*c_nColor+c1)*c_stride + sid].y = (spinor[3][c1].y - spinor[1][c1].y)/sqrt2;
  inOut[(2*c_nColor+c1)*c_stride + sid].y = (spinor[0][c1].y + spinor[2][c1].y)/sqrt2;
  inOut[(3*c_nColor+c1)*c_stride + sid].y = (spinor[1][c1].y + spinor[3][c1].y)/sqrt2;

 }
