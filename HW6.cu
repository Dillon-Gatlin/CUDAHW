// Name: Dillon Gatlin
// Simple Julia CPU.
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/


// ------------------------------------*****Quick Summary of changes made to the original code:********-----------------------------------
// 1. Added CUDA memory management to allocate and free memory on the GPU.
// 2. Created a CUDA kernel to perform the escape time algorithm in parallel on the GPU.
// 3. Modified the display function to call the CUDA kernel and copy the results back to the CPU for rendering.
// 4. Changed the color scheme to green to differentiate between CPU and GPU rendering.
// 5. Ensured proper error checking for CUDA calls.
// 6. displayed the GPU-rendered fractal in a separate window and kept the original CPU-rendered fractal window for comparison.



// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

//---------------------------------------------------added pointers for cuda---------------------------------------------------
float *d_pixels; // Pointer to device memory
float *h_pixels; // Pointer to host memory

// Function prototypes
void cudaErrorCheck(const char*, int);
float escapeOrNotColor(float, float);

//---------------------------------------------------added protypers for cuda---------------------------------------------------
__global__ void gpuEscapeOrNotColor(float *, int, int, float, float, float, float);
void cudaDisplay(void);

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

float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}
void display(void) 
{ 
	float *pixels; 
	float x, y, stepSizeX, stepSizeY;
	int k;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	k=0;
	y = YMin;
	while(y < YMax) 
	{
		x = XMin;
		while(x < XMax) 
		{
			pixels[k] = escapeOrNotColor(x,y);	//Red on or off returned from color
			pixels[k+1] = 0.0; 	//Green off
			pixels[k+2] = 0.0;	//Blue off
			k=k+3;			//Skip to next pixel (3 float jump)
			x += stepSizeX;
		}
		y += stepSizeY;
	}

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}

//---------------------------------------------------added functions for cuda---------------------------------------------------
__global__ void gpuEscapeOrNotColor(float *pixels, int width, int height,
                            float XMin, float XMax, float YMin, float YMax) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // x index this will go from 0 to 1023 in x direction
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // y index this will go from 0 to 1023 in y direction

    if (i >= width || j >= height) return; //check we are not out of bounds, if we are, just return

    float stepSizeX = (XMax - XMin) / (float)width; // how much to increment x for each pixel, depends on width of window
    float stepSizeY = (YMax - YMin) / (float)height;// how much to increment y for each pixel, depends on height of window

	//this will give us the x and y values for this pixel needed for the calculations to see if it escapes
    float x = XMin + i * stepSizeX; // x value for this pixel, this will go from -2 to 2 in x direction
    float y = YMin + j * stepSizeY; // y value for this pixel, this will go from -2 to 2 in y direction 

	//Now do the escape time algorithm
	//these variables are used to determine if the point escapes or not
    float mag, tempX;// tempX is used to hold the old value of x when we update x and y, mag is the magnitude of the complex number
    int count = 0;


	//--------------------********ESCAPE LOOP HAS MOVED INTO THE KERNEL**************--------------------
	//we did this so it can be done in parallel on the GPU for each pixel, causing a huge speedup


	//do the escape time algorithm
    mag = sqrtf(x * x + y * y); // initial magnitude by taking the square root of x^2 + y^2
	//while the magnitude is less than MAXMAG and we have not reached MAXITERATIONS
    while (mag < MAXMAG && count < MAXITERATIONS) {
        tempX = x; // store the old value of x
        x = x * x - y * y + A; // update x using the formula x = x^2 - y^2 + A
        y = 2.0f * tempX * y + B; // update y using the formula y = 2xy + B
        mag = sqrtf(x * x + y * y); // update the magnitude
        count++; // increment the count
    }

	//if the count is less than MAXITERATIONS, it means it escaped, so color it black (0.0f), otherwise color it white (1.0f)
	//it escaping means it is in the set, so we color it black, if it is not escaping, it is outside the set, so we color it white
    float color = (count < MAXITERATIONS) ? 0.0f : 1.0f;

	// Store the color in the pixel array
	// each pixel has 3 values (R, G, B), so we need to multiply by 3 to get the correct index
	//I will color it green instead of red to show that it was done on the GPU

	//--------------------------------------************Pixel Indexing***************--------------------------------------
	//we change from a loop to a direct calculation of the index since we are in a kernel and each thread is responsible for one pixel
	//this is done by calculating the index based on the thread and block indices

    int idx = (j * width + i) * 3; // index in the pixel array
    pixels[idx + 0] = 0.0f; // red 
    pixels[idx + 1] = color;  // green
    pixels[idx + 2] = 0.0f;  // blue
}
//---------------------------------------------------added function for cuda---------------------------------------------------
void cudaDisplay(void) {
    // Define the block size, 16x16 threads per block since 16*16=256 which is a good number of threads per block
	//this is also a common block size for 2D problems
	dim3 block(16, 16);
	// Calculate the grid size to cover the entire window based on the block size
	// we use (WindowWidth + block.x - 1) / block.x to round up to the next whole number of blocks
	// this ensures that we have enough blocks to cover the entire window
    dim3 grid((WindowWidth + block.x - 1) / block.x,
              (WindowHeight + block.y - 1) / block.y);

	
	//----------------------Display function changes for CUDA----------------------
	//since we are using CUDA, we need to copy the pixel data to the device, run the kernel, and then copy the data back to the host



    // Run kernel on GPU
	// launch the kernel with the grid and block dimensions defined above
    gpuEscapeOrNotColor<<<grid, block>>>(d_pixels, WindowWidth, WindowHeight,
                                 XMin, XMax, YMin, YMax);
    cudaDeviceSynchronize();// wait for the GPU to finish
	cudaErrorCheck(__FILE__, __LINE__);

    // Copy results back to CPU
	// copy the pixel data from device memory to host memory so we can use it with OpenGL
    cudaMemcpy(h_pixels, d_pixels,
               WindowWidth * WindowHeight * 3 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

	
    // Draw pixels using OpenGL 
	// use glDrawPixels to draw the pixel data to the window
    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, h_pixels);
    glFlush();// flush the OpenGL commands to ensure they get executed, this clears the buffer and draws everything to the screen
}

int main(int argc, char** argv)
{ 
	// Allocate memory for CUDA
	// 3 for RGB this is needed for cude because we need to allocate memory for the pixel data
    size_t size = WindowWidth * WindowHeight * 3 * sizeof(float);
    h_pixels = (float *)malloc(size); // Allocate host memory
    cudaMalloc((void **)&d_pixels, size);// Allocate device memory
	cudaErrorCheck(__FILE__, __LINE__);

	// Initialize GLUT
   	glutInit(&argc, argv); // Initialize GLUT
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE); // Set the display mode


   	glutInitWindowSize(WindowWidth, WindowHeight); // Set the window size
	glutInitWindowPosition(100, 100); // Set the window position on screen
	glutCreateWindow("Fractals--Man--Fractals"); // Create the window with a title
   	glutDisplayFunc(display);// Register the display function


	glutInitWindowSize(WindowWidth, WindowHeight); // Set the window size
	glutInitWindowPosition(700, 100); // Set the window position on screen
	glutCreateWindow("GPU--Fractals--Man");
	glutDisplayFunc(cudaDisplay); 
	// Register the display function to use the CUDA version
	// Start the GLUT main loop
	//this starts the event loop and will call the display function whenever the window needs to be redrawn
	//the loop will run indefinitely until the window is closed
   	glutMainLoop();

	//free memory once the window is closed
	free(h_pixels);// Free host memory
	cudaFree(d_pixels);// Free device memory
	cudaErrorCheck(__FILE__, __LINE__);
}

