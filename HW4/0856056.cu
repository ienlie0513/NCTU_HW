/**********************************************************************
 * DESCRIPTION:
 *   Serial Concurrent Wave Equation - C Version
 *   This program implements the concurrent wave equation
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

void check_param(void);
void init_line(void);
void update (void);
void printfinal (void);

int nsteps,                 	/* number of time steps */
    tpoints, 	     		/* total points along string */
    rcode;                  	/* generic return code */
float  values[MAXPOINTS]; 	/* values at time t */
float *old_d, /* values at time t on device*/
      *value_d, /* values at time (t-dt) on device*/
      *new_d; /* values at time (t+dt) on device*/
int size = (MAXPOINTS)*sizeof(float);


/**********************************************************************
 *	Checks input values from parameters
 *********************************************************************/
void check_param(void)
{
   char tchar[20];

   /* check number of points, number of iterations */
   while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
      printf("Enter number of points along vibrating string [%d-%d]: "
           ,MINPOINTS, MAXPOINTS);
      scanf("%s", tchar);
      tpoints = atoi(tchar);
      if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
         printf("Invalid. Please enter value between %d and %d\n", 
                 MINPOINTS, MAXPOINTS);
   }
   while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
      printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
      scanf("%s", tchar);
      nsteps = atoi(tchar);
      if ((nsteps < 1) || (nsteps > MAXSTEPS))
         printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
   }

   printf("Using points = %d, steps = %d\n", tpoints, nsteps);

}

__global__ void do_init(float *old_d, float *value_d, int tpoints)
{
   float x, fac, tmp;
   int k = blockIdx.x*blockDim.x+threadIdx.x;

   /* Calculate initial values based on sine curve */
   fac = 2.0 * PI;
   k = 0.0; 
   tmp = tpoints - 1;

   x = (float)k/tmp;
   value_d[k] = sin (fac * x);
   old_d[k] = value_d[k];
}

/**********************************************************************
 *     Initialize points on line
 *********************************************************************/
void init_line(void)
{
   // float x, fac, k, tmp;

   // /* Calculate initial values based on sine curve */
   // fac = 2.0 * PI;
   // k = 0.0; 
   // tmp = tpoints - 1;
   // for (int j = 0; j < tpoints; j++) {
   //    x = k/tmp;
   //    values[j] = sin (fac * x);
   //    k = k + 1.0;
   // } 

   // allocate input
   cudaMalloc(&value_d, size);
   // cudaMemcpy(value_d, values, size, cudaMemcpyHostToDevice);
   
   cudaMalloc(&old_d, size);
   // cudaMemcpy(old_d, values, size, cudaMemcpyHostToDevice);

   cudaMalloc(&new_d, size);

   dim3 dimBlock(1024);
   dim3 dimGrid(tpoints/1024);
   do_init<<<dimGrid, dimBlock>>>(old_d, value_d, tpoints);
}

/**********************************************************************
 *      Calculate new values using wave equation
 *********************************************************************/
__global__ void do_math(float *old_d, float *value_d, float *new_d, int tpoints)
{
   int i = blockIdx.x*blockDim.x+threadIdx.x;
   float dtime, c, dx, tau, sqtau;

   if((i==0) || (i == (tpoints-1)))
      new_d[i] = 0.0;
   else{
      dtime = 0.3;
      c = 1.0;
      dx = 1.0;
      tau = (c * dtime / dx);
      sqtau = tau * tau;
      new_d[i] = (2.0 * value_d[i]) - old_d[i] + (sqtau *  (-2.0)*value_d[i]);
   }

   /* Update old values with new values */
   old_d[i] = value_d[i];
   value_d[i] = new_d[i];
}

/**********************************************************************
 *     Update all values along line a specified number of times
 *********************************************************************/
void update()
{
   /* Update values for each time step */
   for (int i = 1; i<= nsteps; i++) {
      dim3 dimBlock(1000);
      dim3 dimGrid(tpoints/1000);
      do_math<<<dimGrid, dimBlock>>>(old_d, value_d, new_d, tpoints);
   }
}

/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal()
{
   cudaMemcpy(values, value_d, size, cudaMemcpyDeviceToHost);
   cudaFree(value_d); cudaFree(old_d); cudaFree(new_d);

   for (int i = 0; i < tpoints; i++) {
      printf("%6.4f ", values[i]);
      if (i%10 == 9)
         printf("\n");
   }
}

/**********************************************************************
 *	Main program
 *********************************************************************/
int main(int argc, char *argv[])
{
	sscanf(argv[1],"%d",&tpoints);
	sscanf(argv[2],"%d",&nsteps);
	check_param();
	printf("Initializing points on the line...\n");
	init_line();
	printf("Updating all points for all time steps...\n");
	update();
	printf("Printing final results...\n");
	printfinal();
	printf("\nDone.\n\n");
	
	return 0;
}
