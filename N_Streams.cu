// Name: Dillon Gatlin
// Setting up a stream
// nvcc N_Streams.cu -o temp

/*
 What to do:
 Read about CUDA streams. Look at all the ???s in the code and remove the ???s that need to be removed so the code will run.
*/

/*
 Purpose:
 To learn how to setup and work with CUDA streams.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define DATA_CHUNKS (1024*1024) 
#define ENTIRE_DATA_SET (20*DATA_CHUNKS)
#define MAX_RANDOM_NUMBER 1000
#define BLOCK_SIZE 256

//Globals
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
cudaEvent_t StartEvent, StopEvent;
// ??? Notice that we have to define a stream 
//-----------------------------------------------------------------------------------------
// a stream is a sequence of operations that are performed in order on the GPU
//-----------------------------------------------------------------------------------------
cudaStream_t Stream0;

//Function prototypes
void cudaErrorCheck(const char*, int);
void setUpCudaDevices();
void allocateMemory();
void loadData();
void cleanUp();
__global__ void trigAdditionGPU(float *, float *, float *, int );

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

//This will be the layout of the parallel space we will be using.
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaDeviceProp prop;
	int whichDevice;
	
	cudaGetDevice(&whichDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaGetDeviceProperties(&prop, whichDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	if(prop.deviceOverlap != 1)
	{
		printf("\n GPU will not handle overlaps so no speedup from streams");
		printf("\n Good bye.");
		exit(0);
	}
	
	// ??? Notice that we have to create the stream
	//we create it here so it is available to all functions
	//the cudaStreamCreate function creates the stream
	cudaStreamCreate(&Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(DATA_CHUNKS%BLOCK_SIZE != 0)
	{
		printf("\n Data chunks do not divide evenly by block size, sooo this program will not work.");
		printf("\n Good bye.");
		exit(0);
	}
	GridSize.x = DATA_CHUNKS/BLOCK_SIZE;
	GridSize.y = 1;
	GridSize.z = 1;	
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	//??? Notice that we are using host page locked memory
	//Allocate page locked Host (CPU) Memory
	cudaHostAlloc(&A_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&B_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&C_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
}

void loadData()
{
	time_t t;
	srand((unsigned) time(&t));
	
	for(int i = 0; i < ENTIRE_DATA_SET; i++)
	{		
		A_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;
		B_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	// ??? Notice that we have to free this memory with cudaFreeHost
	//-----------------------------------------------------------------------------------------
	// this is because it was allocated with cudaHostAlloc
	//-----------------------------------------------------------------------------------------
	cudaFreeHost(A_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(B_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(C_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// ??? Notice that we have to kill the stream.
	//-----------------------------------------------------------------------------------------
	// we do this since we created it with cudaStreamCreate which allocates memory for the stream
	//-----------------------------------------------------------------------------------------
	cudaStreamDestroy(Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
}

__global__ void trigAdditionGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n)
	{
		c[id] = sin(a[id]) + cos(b[id]);
	}
}

int main()
{
	float timeEvent;
	
	setUpCudaDevices();
	allocateMemory();
	loadData();
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	//-----------------------------------------------------------------------------------------
	//we must  break it into chunks so that it can fit in GPU
	// cudaMemcpy will be asynchronous because we are using streams, which allows the CPU to continue
	// wihtout the need for it to finish the copy first
	// we split it into 2 chunks to allow for one to start computing while theo other is being copied
	// after we pass the to trigAdditionGPU kernel passing our 2 chunks to it with the output going to C_GPU
	// then we copy the result back to the CPU asynchronously. 
	// (asynchronous meaning the CPU does not wait for it to finish since we are using streams which allow for that)
	//-----------------------------------------------------------------------------------------
	/*Why we use streams:
	without streams the  operations are done one afte another, which means waiting for each to finish before starting the next.
	with streams we overlap the opeartion allowing the GPU to compute while the CPU is copying data, thus speeding up the overall process
	It basically allows for a parallelism between data transfer and computation. (this is like pipelining in a sense)
	main goal is to keep gpu busy even if data transfer is taking time.
	//-----------------------------------------------------------------------------------------
	*/
	for(int i = 0; i < ENTIRE_DATA_SET; i += DATA_CHUNKS)
	{
		cudaMemcpyAsync(A_GPU, &A_CPU[i], DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(B_GPU, &B_CPU[i], DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);

		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream0>>>(A_GPU, B_GPU, C_GPU, DATA_CHUNKS);
		cudaErrorCheck(__FILE__, __LINE__);
		
		cudaMemcpyAsync(&C_CPU[i], C_GPU, DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);


	}
	
	// ??? Notice that we have to make the CPU wait until the GPU has finished stream0
	//-----------------------------------------------------------------------------------------
	// this synch needs to happen because the cudaMemcpyAsync calls are asynchronous so the cpu 
	//coud continue on without waiting for them to finish, so we need to make sure everything is done
	//-----------------------------------------------------------------------------------------
	cudaStreamSynchronize(Stream0); 
	
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	// Make the CPU wiat until this event finishes so the timing will be correct.
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU = %3.1f milliseconds", timeEvent);
	
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
