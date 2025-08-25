// Name: Dillon Gatlin
// Vector addition on the CPU, with timer and error checking
// To compile: nvcc HW1.cu -o temp
/*
 What to do:
 1. Understand every line of code and be able to explain it in class.
 2. Compile, run, and play around with the code.
 3. Also play around with the Pointerstest.cu code to understand pointers.  
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
//the stdio.h is standard input output header file
// sys/time.h is used for struct timeval and gettime of day() it helps in measuring the time elapsed 

// Defines
#define N 1000 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; 
float Tolerance = 0.01;
//CPU is the pointer arrays, tolerance is the percent error allowed when checking if correct

// Function prototypes
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
bool  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

//standard procedure declaring the function but not defining what they do.

//Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	//malloc dynamically gives memory to heap for each of our vector and cast types to float
}

//Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
		//B_CPU[i] = (float)(7*i); //USED WHEN TESTING CODE
	}
	//simple for loop from 0 < N (in our casse N= 1,000)
	//A_CPU is loaded with repsentation of i, 0,1,2,3 etc
	//B_CPU is loaded with repsentation of i*2 0,2,4,6 etc
	//modified B_CPU to be 7*1 loaded with repsentation of i*7 0,7,14,21 etc
}

//Adding vectors a and b then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
	//simple for loop from 0 to n (in our case n = 1,000)
	// each array element is added together and stored inside c[id]
	// Ex (a[1] = 1 b[1] = 2 c[1] = 3)
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	//m equals 1,000 -1 so 999
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	//my answer will find the sum of the results by adding each element in the array
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	//hardcoded true answer works because a[i] = i, b[i] = 2i, so adding them means c[i] =3i
	//to find the sum we can do 3.0(summation formula)

	//--------------TESTING AUGMATION TO THE CODE-----------------
	//trueAnswer = 8.0*(m*(m+1))/2.0;
	//for testing I also used 8.0 when modifying the b array to i*7. making c[i] = 8i	


	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	//percent error is the difference between my answer and the true answer divided by the true answer,
	// this shows how close we are from the expected value

	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
	//if else statment to return if the percent error is less than the tolerance
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds
	//TV_SEC is in seconds so we multiply by 1,000,000 to get microseconds
	//the additional end.tv_usec account for the microseconds that occur between each second
	//essentially the fractions of a second (ex: 123.778 seconds) 778 is the fractions

	return endTime - startTime;
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	//frees the allocated memory for the CPU vectors
}

int main()
{
	timeval start, end;
	//declares struct variables for start ans end. 
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();

	// Starting the timer.	
	gettimeofday(&start, NULL);

	// Add the two vectors.
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);

	// Stopping the timer.
	gettimeofday(&end, NULL);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the CPU");
		printf("\n The time it took was %ld microseconds", elaspedTime(start, end));
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}
