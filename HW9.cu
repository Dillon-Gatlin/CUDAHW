// Name: Dillon Gatlin
// Vector Dot product on many block and useing shared memory
// nvcc HW9.cu -o temp
/*
 What to do:
 This code is the solution to HW8. It finds the dot product of vectors that are smaller than the block size.
 Extend this code so that it sets as many blocks as needed for a set thread count and vector length.
 Use shared memory in your blocks to speed up your code.
 You will have to do the final reduction on the CPU.
 Set your thread count to 200 (block size = 200). Set N to different values to check your code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 600000 // Length of the vector
#define BLOCK_SIZE 200 //the size of our block in case we wanted to change it

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

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

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = BLOCK_SIZE; //we defined it above to be 200
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//---------------use this formula to get the correct amount of blocks to account for N---------------
	GridSize.x = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
	GridSize.y = 1;
	GridSize.z = 1;

	//---------------prints out the amount of blocks and the size of each
	printf("Grids: %d,  Blocks: %d", GridSize.x, BlockSize.x);
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

/*------------------------SUMMARY OF CHANGES---------------
1- changed storage size to block size (since we account for odds later)
2- added blcN which accounts for how many N are in a block,added Global Thread ID so threads work on correct value
3-used globalTid < n and used it for indexs since per block it is different
4-same as last hw but we change n to blcN since we only account for N in block
5-store output into blockIdx.x
----------------------------------------------------
*/
// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
	//1
	__shared__ float storage[BLOCK_SIZE]; 

	//2
    int tid = threadIdx.x;
	int blcN = blockDim.x; //the # of element in this block
	int globaTid = blockIdx.x * blockDim.x +tid; 

    //3 
    if (globaTid < n)
	{
		storage[tid] = a[globaTid] * b[globaTid];
	}
	else
	{
		storage[tid] = 0.0;
	}
    __syncthreads();

    //4 this is same as last hw but change n to blcN since we only account for n in block
    while (blcN > 1)
    {
		//this takes care of the odd protion by adding and using integer division to achieve correct folding
        int half = (blcN + 1) / 2;
		
		
        if (tid < half && tid + half < blcN)
        {
            storage[tid] += storage[tid + half];
        }
        __syncthreads();
        blcN = half; // halve number of active elements
    }

    //5
    if (tid == 0)
	{
        c[blockIdx.x] = storage[0];
		//added print statment for double checking and bug fixing
		//printf("\n The dot product of block: %d on the GPU is %f",blockIdx.x, c[blockIdx.x]);
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	//--------------------------CHANGED TO SHOW NUMBER OF BLOCKS AND SIZE-----------
	if(BlockSize.x < N)
	{
		printf("\n\n Multiple blocks used, Blocks: %d, of size %d", GridSize.x, BlockSize.x);
		
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	
	cudaMemcpyAsync(C_CPU, C_GPU, GridSize.x*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	//----------------------------------------FINAL SUMMATION--------
	//final summation will add each output from the blocks into DotGPU
	DotGPU= 0.0f;
	for(int i =0; i< GridSize.x;i++)
		DotGPU += C_CPU[i]; // C_GPU was copied into C_CPU.


	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}


