// Name: Dillon gatlin
// Not simple Julia Set on the GPU
// nvcc HW7.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artisct flare. 
 Don't cute off your ear or anything but make Vincent wish he had, had a GPU.
*/

//----------------------------SUMMARY OF CHANGES FOR WINDOW SIZE----------------------------
//1. Added parameters width and height to the kernel to allow it to run on any window size.
//2. Calculated pixel coordinates in the kernel using blockIdx, blockDim, and threadIdx.
//3. Added boundary checks in the kernel to ensure we don't access out-of-bounds memory.
//4. Calculated step sizes in the display function based on the current window size.
//5. Calculated grid size in the display function based on the window size and block size.
//6. Allocated pixel buffers outside of the display function to avoid repeated allocations and deallocations. (needed for dynamic color)
//7. Updated the display function to use the new parameters and calculations.
//--------------------------------------------------------------------------------------------

//----------------------------SUMMAY OF CHANGES FOR DYNAMIC COLOR----------------------------
//1. Made A and B global variables so they can be changed over time.
//2. Added a time variable t to change A and B over time.
//3. Changed the way pixels are colored in the kernel to use pixel coordinates and time t using sine and cosine functions.
//4. Added an updateColor function that increments t and updates A and B based on sine and cosine functions.
//5. Set the GLUT idle function to call updateColor so the display updates continuously.
//6. Allocated pixel buffers outside of the display function to avoid repeated allocations and deallocations.
//--------------------------------------------------------------------------------------------

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
//#define A  -0.824	//Real part of C
//#define B  -0.1711	//Imaginary part of C

//----------------------------CHANGES/ADDED FOR DYNAMIC COLOR//----------------------------
//Here we make A and B global variables that can be changed over time. since we cant change the #define values.
float t = 0.0f;
float A = -0.824f;  // Real part of C
float B = -0.1711f;  // Imaginary part of C
float *pixelsCPU = nullptr;
float *pixelsGPU = nullptr;
//--------------------------------------------------------------------------------------------
// Global variables
unsigned int WindowWidth = 1400;
unsigned int WindowHeight = 1400;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float*, int, int, float, float, float, float, float, float);
//--------------------------------ADDED PROTOTYPE FOR DYNAMIC COLOR//--------------------------------
void updateColor();

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

//added width, height, A,B, and t to the kernel parameters
//width and height are used to calculate pixel coordinates
//A and B are used in the fractal equation
//t is used to change A and B over time for dynamic coloring
__global__ void colorPixels(float *pixels, int width, int height, float xMin, float yMin, float dx, float dy, float A, float B, float t) 
{
	//this is the standard way to calculate pixel coordinates in a kernel.
	//this allows us to run on any window size. by passing in width and height
	//the math works since take blockIdx.x * blockDim.x + threadIdx.x gives us the pixel x coordinate
	//ex if we have a block size of 16 and we are in block 2, thread 3, we get (2*16)+3 = 35
	int px = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int py = blockIdx.y * blockDim.y + threadIdx.y; // pixel y

    if (px >= width || py >= height) return; // boundary check
    int id = 3 * (py * width + px);
    float x = xMin + dx * px;
    float y = yMin + dy * py;
    float mag;
    int count = 0;
	
	//Getting the offset into the pixel buffer. 
	//We need the 3 because each pixel has a red, green, and blue value.
	do {
        float tempX = x;
        x = x * x - y * y + A;
        y = 2.0f * tempX * y + B;
        mag = sqrtf(x * x + y * y);
        count++;
    } while (mag < MAXMAG && count < MAXITERATIONS);
	
	// //-------------------------THIS CODE WILL COLOR THE INTERIOR BY PIXEL COORDINATES.-------------------------
	//using pixel coordinates px and py along with time t to create a dynamic color pattern.
	//the sin and cos functions create smooth transitions in color over time and space.
	if (count >= MAXITERATIONS) {
		
		pixels[id]     = 0.5f + 0.5f * sinf(t + px * 0.01f); // R
    	pixels[id + 1] = 0.5f + 0.5f * cosf(t + py * 0.01f); // G
    	pixels[id + 2] = 0.5f + 0.5f * sinf(t);              // B
	} else {
    	// Exterior = black
    	pixels[id]     = 0.0f;
    	pixels[id + 1] = 0.0f;
    	pixels[id + 2] = 0.0f;
	}
}
    

void display(void) 
{ 
	//-------------------------this is used to account for any window size-------------------
	//float *pixelsCPU, *pixelsGPU; 
	//Calculating the step size in each direction. needed to map pixels to coordinates.
	//to find step size we take the range and divide by the number of pixels in that direction.
	float stepSizeX = (XMax - XMin) / (float)WindowWidth;
    float stepSizeY = (YMax - YMin) / (float)WindowHeight;
	//16x16 is a common block size that works well on most GPUs. we calculate the grid size based on the window size.
	dim3 blockSize(16, 16);
	dim3 gridSize((WindowWidth + blockSize.x - 1) / blockSize.x,(WindowHeight + blockSize.y - 1) / blockSize.y);
	//--------------------------------------------------------------------------------------------------------------

	//Allocating memory for the pixels on the CPU and GPU. we added widht, height, for the new window size
	//we added A, B, and t for the dynamic color
	colorPixels<<<gridSize, blockSize>>>(pixelsGPU,WindowWidth, WindowHeight, XMin, YMin, stepSizeX, stepSizeY, A, B,t);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	//-----------------------------------------------CHANGE TO SYNC FOR THE DYNAMIC COLOR----------------------------
	cudaMemcpy(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
	glFlush(); 
}


//-----------------------THIS FUNCTION WILL UPDATE OUR COLOR AND CAN ALSO UPDATE A AND B OVER TIME-----------------------
//incrementing t (time)	will change the color pattern over time.
//we also update A and B using sine and cosine functions to create a dynamic fractal pattern
//glutPostRedisplay() tells GLUT to call the display function again to update the screen
//uncomment the A and B lines if you want to see the fractal change as well as the color
void updateColor(){
	t += 0.01f; //increment time. increasing value will speed up the color change
	//A = -0.8f + 0.2f * sinf(t);
	//B = 0.17f + 0.2f * cosf(t);
	glutPostRedisplay();
}

int main(int argc, char** argv)
{ 
	//-------------Needed for DYNAMIC COLOR----------------
	//we store this outside of display now so we dont have to keep allocating and freeing memory.
	// when i put it in display it caused a black screen after a few seconds.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
	//-----------------------------ADDED FOR DYNAMIC COLOR----------------
	//idle function will call updateColor to change the color over time
	glutIdleFunc(updateColor);
   	glutMainLoop();
}



