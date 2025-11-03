// Name:Dillon Gatlin
// GPU random walk. 
// nvcc P_GPURandomWalk.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
// Defines
#define N_WALKS 20
#define N_STEPS 10000
// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
void cudaErrorCheck(const char*, int );
int main(int, char**);
__device__ int getRandomDirectionGPU(curandState *);
__global__ void randomWalkKernel(int *, int *, int );

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}
// This will be the layout of the parallel space we will be using.

int getRandomDirection()
{	
	int randomNumber = rand();
	int MidPoint = (float)RAND_MAX / 2.0f;
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

//---------------------------------GPU random direction function-------------------------
// I pretty much copied this from the function above but used curand to get the random number
//----------------------------------------------------------------------------------------
__device__ int getRandomDirectionGPU(curandState *state)
{
    float randomValue = curand(state); 
    int GPUMidpoint = (float)RAND_MAX / 2.0f;
    if (randomValue < GPUMidpoint) return -1;
    else return 1;
}

/*-------------------------random walk kernel--------------------------------
1- initilize thread index and starting positions
2- setup curand state with different seed for each thread, 
for this i use arbitary number (2025) + the idx *42925 (a large prime number) to get different seeds
3- loop over the number of steps and update the position using getRandomDirectionGPU
4- store the final positions in the unified memory arrays

State means the state of the random number generator for each thread. we 
use curand_init to initialize the state with a seed, sequence number, and offset. ensuring 
that each thread generates a unique sequence of random numbers.
-----------------------------------------------------------------------------

*/
__global__ void randomWalkKernel(int *positionX, int *positionY, int N_Steps)
{
    //1
    int idx = threadIdx.x;
    int x = 0;
    int y = 0;
    //2
    curandState state;
    curand_init(2025+(idx*42925), 0, 0, &state); 

    //3
    for (int step = 0; step < N_Steps; step++)
    {
        x += getRandomDirectionGPU(&state);
        y += getRandomDirectionGPU(&state);
    }

    //4
    positionX[idx] = x;
    positionY[idx] = y;
}

/*-------------------------main program------------------------------
1- Seed the random number generator on the CPU. printf the RAND_MAX value. and Nsteps and Nwalks
2- Allocate unified memory for the final positions arrays on the GPU.
3- Launch the random walk kernel with enough blocks and threads to cover N_WALKS.
4- Synchronize the device and print out the final positions.
5- Free the unified memory.
------------------------------------------------------------------
*/
int main(int argc, char** argv)
{
	srand(time(NULL));
	
	printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);
	
	int positionX = 0;
	int positionY = 0;
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		positionX += getRandomDirection();
		positionY += getRandomDirection();
	}
	
	printf("\n Final position = (%d,%d) \n", positionX, positionY);
    //---------------------------------------------------------------------------------------
    printf("\n-----------------GPU Random Walk-----------------\n");
    //1
    srand(time(NULL));

    printf(" RAND_MAX for this implementation = %d\n", RAND_MAX);
    printf(" Running %d random walks of %d steps each on the GPU:\n\n",
           N_WALKS, N_STEPS);

    // 2
    int *FPosX, *FPosY;
    cudaMallocManaged(&FPosX, N_WALKS * sizeof(int));
    cudaMallocManaged(&FPosY, N_WALKS * sizeof(int));
    cudaErrorCheck(__FILE__, __LINE__);

    // 3
    int threadsPerBlock = 20;
    int blocks = 1;

    randomWalkKernel<<<blocks, threadsPerBlock>>>(FPosX, FPosY, N_STEPS);
    cudaErrorCheck(__FILE__, __LINE__);

    cudaDeviceSynchronize();
   	cudaErrorCheck(__FILE__, __LINE__);

    // 4
    printf("GPU Random Walk Final Positions:\n");
    for (int i = 0; i < N_WALKS; i++)
    {
        printf(" Walk %2d -> Final Position = (%d, %d)\n", i, FPosX[i], FPosY[i]);
    }

    //5
    cudaFree(FPosX);
    cudaFree(FPosY);
	return 0;
}
