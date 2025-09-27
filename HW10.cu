// Name: Dillon Gatlin
// Robust Vector Dot product 
// nvcc HW10.cu -o temp
/*
 What to do:
 This code is the solution to HW9. It computes the dot product of vectors of any length and uses shared memory to 
 reduce the number of calls to global memory. However, because blocks can't sync, it must perform the final reduction 
 on the CPU. 
 To make this code a little less complicated on the GPU let do some pregame stuff and use atomic adds.
 1. Make sure the number of threads on a block are a power of 2 so we don't have to see if the fold is going to be
    even. Because if it is not even we had to add the last element to the first reduce the fold by 1 and then fold. 
    If it is not even tell your client what is wrong and exit. 
	***********************[DONE]***********************[

 2. Find the right number of blocks to finish the job. But, it is possible that the grid demention is too big. I know
    it is a large number but it is finite. So use device properties to see if the grid is too big for the machine 
    you are on and while you are at it make sure the blocks are not to big too. Maybe you wrote the code on a new GPU 
    but your client is using an old GPU. Check both and if either is out of bound report it to your client then kindly
    exit the program.
	###############################[DONE]########################[


 3. Always checking to see if you have threads working past your vector is a real pain and adds a bunch of time consumming
    if statments to your GPU code. To get around this findout how much you would have to add to your vector to make it 
    perfectly fit in your block and grid layout and pad it with zeros. Multipying zeros and adding zero do nothing to a 
    dot product. If you were luck on HW8 you kind of did this but you just got lucky because most of the time the GPU sets
    everything to zero at start up. But!!!, you don't want to put code out where you are just lucky soooo do a cudaMemset
    so you know everything is zero. Then copy up the now zero values.
	###############################[DONE]########################[

 4. In HW9 we had to do the final add "reduction' on the CPU because we can't sync block. Use atomic add to get around 
    this and finish the job on the GPU. Also you will have to copy this final value down to the CPU with a cudaMemCopy.
    But!!! We are working with floats and atomics with floats can only be done on GPUs with major compute capability 3 
    or higher. Use device properties to check if this is true. And, while you are at it check to see if you have more
    than 1 GPU and if you do select the best GPU based on compute capablity.
	###############################[DONE]########################

 5. Add any additional bells and whistles to the code that you thing would make the code better and more foolproof.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 60000 // Length of the vector
//#define OPTIMAL_BLOCK_SIZE 1024 // Threads in a block ------------------------QUESTION 1 --------------------

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

//------------------------------VARIABLES ADDED--------------------------------
int BLOCK_SIZE; //this will be changed at runtime to try and match limits of the GPU
int PADDED_N; // will be what we use for N after padding with zeros
float *dotGPU_result;// for atomic add result on GPU


// Function prototypes
void cudaErrorCheck(const char *, int);
//--------------------------------function added in order to get best device and highest power of 2----------
int powerOfTwo(int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int, int, float*);
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
//------------------------Question 1 setting best number of threads-------------------
//this function will find the highest power of 2 that is less than or equal to x
int powerOfTwo(int x)
{
	int powerTwo = 1;
	while (powerTwo*2 < x)
		powerTwo *= 2;
	return powerTwo;
}

//------------------------Question 4 finding best device and checking compute capability-------------------
/*this function will find the best GPU based on compute capability and set it to be the one we use, we go based
of the major version first then the minor version if there is a tie in major version number.
*/
void chooseGPU()
{
	int count = 0;
	int best =0;
	//GPUS have a major and minor version number that tells you what features they have, so
	int majorVersion = -1; 
	int minorVersion = -1;
	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		printf("Cannot find a CUDA Devce, goodbye.\n");
		exit(0); 
	}
	//if one of the device have a higher major version it is better
	//if they have the same major version the one with the higher minor version is better
	for (int i = 0; i < count; i++)
	{
		cudaDeviceProp propDev;
		cudaGetDeviceProperties(&propDev, i);
		if (propDev.major > majorVersion) {
			majorVersion = propDev.major;
			minorVersion = propDev.minor;
			best = i;
		} 
		else if (propDev.major == majorVersion) {
			if (propDev.minor > minorVersion) {
				minorVersion = propDev.minor;
				best = i;
			}
		}
	}
	cudaSetDevice(best);
	printf("Using the best GPU, index: %d\n", best);
}
/*------------------------Question 1,2 and 3 setting up device, sizes, and padding-------------------
 This will be the layout of the parallel space we will be using.
*/
void setUpDevices()
{
	chooseGPU(); 
	cudaDeviceProp propDev;
	cudaGetDeviceProperties(&propDev, 0);

    //Q1 highest power of 2 not going over deivice limit
	//this gets the max threads per block and finds the highest power of 2 that is less than or equal to it
	BLOCK_SIZE = powerOfTwo(propDev.maxThreadsPerBlock);
	if (BLOCK_SIZE < 1)
	{
		printf("\nNot able to get valid block size, goodbye!\n");
		exit(0);
	}

	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//--------------------------Q3 GRID size check, shared memory check, atomic add check and padding N------------------------
	//use this formula to get the correct amount of blocks to account for N since N may not be a multiple of BLOCK_SIZE
	//so we round up to make sure we have enough blocks to cover all elements

	GridSize.x = (N +BLOCK_SIZE -1)/BLOCK_SIZE; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;

	if (BLOCK_SIZE % 2 != 0)
	{
		printf("\n\n The block size is not even. Goodbye.\n\n");
		exit(0);
	}
	//max grid size check
	printf("Max grid size for this GPU is %d\n", propDev.maxGridSize[0]);
	//--------------------------------------------------------Q3 GRID SIZE CHECK------------------------------
	//my gpud max on my pc is 2,147,483,647 so this will pass for large values but percent error will be high
	//if you have a older GPU it may be smaller so this check is important
	//i made extra check for 65,535 since that is the standard max grid size for older GPUs to say high percent error expected
	if(GridSize.x > propDev.maxGridSize[0])
	{
		printf("\nToo many blocks on the grid for this GPU, goodbye!\n");
		exit(0);
	}
	if (GridSize.x > 65535)
	{
		printf("\nYOU HAVE A LOT OF BLOCKS OVER STANDARD OF 65,535. HIGH PERCENT ERROR EXPECTED!\n");
	}
	//Checking shared memory requirement
	if (BLOCK_SIZE * sizeof(float) > propDev.sharedMemPerBlock)
	{
		printf("\n Not enough shared memory on this GPU, goodbye!\n");
		exit(0);
	}
	// version check to make sure atomic add on floats is supported
	if (propDev.major < 3)
	{
		printf("\n Atomic Add with floats not availabled on this GPU, goobye!\n");
		exit(0);
	}

	PADDED_N = GridSize.x * BLOCK_SIZE; //this is the new N we will use after padding

	printf("Our Block size is %d, Grid size: %d, and PaddedN =%d\n", BLOCK_SIZE, GridSize.x, PADDED_N);

}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,PADDED_N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,PADDED_N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,GridSize.x*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	//-------------------Question 4 memset for padded N and atomic add result-----------------------------
	
	cudaMalloc(&dotGPU_result, sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMemset(A_GPU,0,PADDED_N*sizeof(float));
    cudaMemset(B_GPU,0,PADDED_N*sizeof(float));
    cudaMemset(C_GPU,0,GridSize.x*sizeof(float));
    cudaMemset(dotGPU_result,0,sizeof(float));

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

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n, float *dotGPU_result)
{
	
	//__shared__ float storage[BLOCK_SIZE]; 
	//to use the global size of the block i have to pass out when calling the kernel
	//so it will still be passes and essentially is the same as this: 
	//__shared__ float storage[BLOCK_SIZE]; 
	extern __shared__ float storage[];

	
    int tid = threadIdx.x;
	int blcN = blockDim.x; //the # of element in this block
	int globaTid = blockIdx.x * blockDim.x +tid; 
    
    if (globaTid < n)
	{
		storage[tid] = a[globaTid] * b[globaTid];
	}
	else
	{
		storage[tid] = 0.0;
	}
    __syncthreads();

    // sane as before but get rid of odd check since we guarentee power of 2
    while (blcN > 1)
    {
        int half = blcN / 2; //no longer need to see if odd since guaranteed power of 2
		
        if (tid < half )
        {
            storage[tid] += storage[tid + half];
        }
        __syncthreads();
        blcN = half; // halve number of active elements
    }

    // now we use atomic add to get around not being able to sync blocks
    if (tid == 0)
	{
        c[blockIdx.x] = storage[0];
		atomicAdd(dotGPU_result, storage[0]); //stores final result on gpu
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
	cudaFree(dotGPU_result);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	cudaErrorCheck(__FILE__, __LINE__);
	// Allocating the memory you will need.
	allocateMemory();
	cudaErrorCheck(__FILE__, __LINE__);
	// Putting values in the vectors.
	innitialize();
	cudaErrorCheck(__FILE__, __LINE__);
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	cudaErrorCheck(__FILE__, __LINE__);
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//---------------------------------------Q1~ WE PASS THE BLOCK SIZE TO THE KERNEL------------------------------
	//Need to pass as extern shared memory since BLOCK_SIZE is not a compile time constant so the GPU wont allow it unless
	//we pass it as extern shared memory
	dotProductGPU<<<GridSize,BlockSize, BLOCK_SIZE * sizeof(float)>>>(A_GPU, B_GPU, C_GPU, PADDED_N, dotGPU_result);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	/*cudaMemcpy(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	*/
	//------------------------------------SINCE WE DONT COPY TO CPU WE CAN STORE DIRECTLY TO DotGPU--------------
	cudaMemcpy(&DotGPU, dotGPU_result, sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
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


