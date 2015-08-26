int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;


int x_id, y_id, z_id, t_id;
int r1,r2;

r1 = sid/(c_localL[0]);
r2 = r1/(c_localL[1]);
x_id = sid - r1*(c_localL[0]);
y_id = r1 - r2*(c_localL[1]);
t_id = r2/(c_localL[2]);
z_id = r2 - t_id*(c_localL[2]);


int x,y,z;

x = x_id + c_procPosition[0] * c_localL[0] - c_sourcePosition[0];
y = y_id + c_procPosition[1] * c_localL[1] - c_sourcePosition[1];
z = z_id + c_procPosition[2] * c_localL[2] - c_sourcePosition[2];

double phase;

phase = ( ((double) nx*x)/c_totalL[0] + ((double) ny*y)/c_totalL[1] + ((double) nz*z)/c_totalL[2] ) * 2. * PI;

double2 expon;

expon.x = cos(phase);
expon.y = -sin(phase);


for(int gamma = 0 ; gamma < 4 ; gamma++)
  for(int c1 = 0 ; c1 < 3 ; c1++){
    vector[gamma*3*c_stride + c1*c_stride + sid] = vector[gamma*3*c_stride + c1*c_stride + sid] * expon;
  }
  
