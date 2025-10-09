// Name: Dillon Gatlin
// Histogram useing atomics in global memory and shared memory.
// nvcc 12HistogramUseingAtomics.cu -o temp

/*
 What to do:
 This code generates a series of random numbers and places them into bins based on size ranges using the CPU.

 Your task:
 - Create a binning scheme that utilizes the GPU.
 - Take advantage of both global and shared memory atomic operations.
 - The function call has already been provided.
 - Set the block size to **twice** the number of multiprocessors on the GPU.
*/

/*
 Purpose:
 To learn how to use atomic operations at both the shared and global memory levels.
 Along the way, you'll also learn a bit about generating random numbers using `srand`,
 which will come in handy when we use `curand` in a later assignment.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

/*
 Note: The Max int value is 2,147,483,647
 The length of the sequence of random number that srand generates is 2^32,
 that is 4,294,967,296 which is bigger than the largest int but the max for an unsigned int.
*/

// Defines
#define NUMBER_OF_RANDOM_NUMBERS 2147483
#define NUMBER_OF_BINS 10
#define MAX_RANDOM_NUMBER 100.0f

// Global variables
float *RandomNumbersGPU;
int *HistogramGPU;
float *RandomNumbersCPU;
int *HistogramCPU;
int *HistogramCPUTemp; // Use it to hod the GPU histogram past back so we can compair to CPU histogram.
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//Function prototypes
void cudaErrorCheck(const char *, int);
void SetUpCudaDevices();
void AllocateMemory();
void Innitialize();
void CleanUp();
void fillHistogramCPU();
__global__ void fillHistogramGPU(float *, int *);
int main();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
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


/*-------------Summary of changes to SetupCUDA program-----------------
1: We set the block size to twice the number of multiprocessors on the GPU.
the reason for this is that each multiprocessor can handle more than one block at a time, so each multiprocessor
will work on two blocks at a time. this will help hide latency when waiting for memory.
-ex when one block is waiting for memory the other block can be working
(if you have 10 sm you will have 20 blocks working at a time)
-----------------------------------------------------------------------
*/
//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//BlockSize.x = 222;
	//we get the number of multiprocessors on the GPU and set the block size to twice that.
	//1
	BlockSize.x = prop.multiProcessorCount * 2;
	if(prop.maxThreadsDim[0] < BlockSize.x)
	{
		printf("\n You are trying to create more threads (%d) than your GPU can support on a block (%d).\n Good Bye\n", BlockSize.x, prop.maxThreadsDim[0]);
		exit(0);
	}
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (NUMBER_OF_RANDOM_NUMBERS - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	if(prop.maxGridSize[0] < GridSize.x)
	{
		printf("\n You are trying to create more blocks (%d) than your GPU can suppport (%d).\n Good Bye\n", GridSize.x, prop.maxGridSize[0]);
		exit(0);
	}
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&RandomNumbersGPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&HistogramGPU, NUMBER_OF_BINS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);

	//Allocate Host (CPU) Memory
	RandomNumbersCPU = (float*)malloc(NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	HistogramCPU = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	HistogramCPUTemp = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	
	//Setting the the histograms to zero.
	cudaMemset(HistogramGPU, 0, NUMBER_OF_BINS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);
	memset(HistogramCPU, 0, NUMBER_OF_BINS*sizeof(int));
}

//Loading random numbers.
void Innitialize()
{
	time_t t;
	srand((unsigned) time(&t));
	
	// rand() returns an int in [0, RAND_MAX] "end points included".
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++)
	{		
		RandomNumbersCPU[i] = MAX_RANDOM_NUMBER*(float)rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	cudaFree(RandomNumbersGPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(HistogramGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	free(RandomNumbersCPU); 
	free(HistogramCPU);
	free(HistogramCPUTemp);
	//printf("\n Cleanup Done.");
}

void fillHistogramCPU()
{
	float breakPoint;
	int k, done;
	float stepSize = MAX_RANDOM_NUMBER/(float)NUMBER_OF_BINS;
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++)
	{
		breakPoint = stepSize;
		k = 0;
		done =0;
		while(done == 0)
		{
			if(RandomNumbersCPU[i] < breakPoint)
			{
				HistogramCPU[k]++; 
				done = 1;
			}
			
			if(NUMBER_OF_BINS < k)
			{
				printf("\n k is too big\n");
				exit(0);
			}
			k++;
			breakPoint += stepSize;
		}
	}
}

//-------------------SUMMARY OF CHANGES TO KERNEL------------------------------
/* 1: Added shared memory histogramShared[] to hold the histogram for each block.
2: Initialized the shared memory histogram to zero to avoid garbage values.
3: global tid to get the global thread ID.
4: Each thread grabs its random number
5:computes which bin it belongs to, note we use integer division to get the bin since 
5.1: we need to deal with the max bin case value to store it in last bin. we just -1 from bin
	EX: bin 0 is 0-9, bin 1 is 10-19, etc. since we know bin size and max value ahead of time.
 	we use formual bin = (int)(randVal / stepSize); ex 24.5/10 = 2.45 which will be sort into bin 2
6:adds to the shared histogram. 
7: sync threads and then have 10 threads atomic add share histogram to global histogram.
*/

//This is the kernel. It is the function that will run on the GPU.
__global__ void fillHistogramGPU(float *randomNumbers, int *hist)
{
	//1
	__shared__ int histogramShared[NUMBER_OF_BINS];
	int tid = threadIdx.x;

	//2
	if(tid < NUMBER_OF_BINS)
		histogramShared[tid] = 0;
	__syncthreads();

	//3
	int gid = threadIdx.x + blockIdx.x * blockDim.x;

	//4-6
	if (gid < NUMBER_OF_RANDOM_NUMBERS)
    {
        // 4
        float randNum = randomNumbers[gid];

        // 5
		//instead of while loop we can use integer division
		//This step size can also be calculated outside the kernel and passed in as a parameter.
        float stepSize = MAX_RANDOM_NUMBER / (float)NUMBER_OF_BINS;
        int bin = (int)(randNum / stepSize);

		//5.1 will deal with the max bin case value. 
        if (bin >= NUMBER_OF_BINS)
            bin = NUMBER_OF_BINS - 1; 

		// 6.
        atomicAdd(&histogramShared[bin], 1);
    }

	__syncthreads();
	//7.
	if (tid < NUMBER_OF_BINS)
	{
		// add shared histogram to global histogram
		atomicAdd(&hist[tid], histogramShared[tid]);
	}
}

int main()
{
	float time;
	timeval start, end;
	
	long int test = NUMBER_OF_RANDOM_NUMBERS;
	if(2147483647 < test)
	{
		printf("\nThe length of your vector is longer than the largest integer value allowed of 2,147,483,647.\n");
		printf("You should check your code.\n Good Bye\n");
		exit(0);
	}
	
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using and padding with zero vector will be a factor of block size.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	gettimeofday(&start, NULL);
	fillHistogramCPU();
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\nTime on CPU = %.15f milliseconds\n", (time/1000.0));
	
	gettimeofday(&start, NULL);
	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(RandomNumbersGPU, RandomNumbersCPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	fillHistogramGPU<<<GridSize,BlockSize>>>(RandomNumbersGPU, HistogramGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(HistogramCPUTemp, HistogramGPU, NUMBER_OF_BINS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\nTime on GPU = %.15f milliseconds\n", (time/1000.0));
	
	//Check
	for(int i = 0; i < NUMBER_OF_BINS; i++)
	{
		printf("\n Deference in histogram bins %d is %d.", i, abs(HistogramCPUTemp[i] - HistogramCPU[i]));
	}
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	printf("\n\n");
	return(0);
}
