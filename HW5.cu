// Name: Dillon gatlin
// Device query
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);

		//added memory information-----------------------------------------------------------------------------------------------------------------------------
		printf("---new memory information for device %d ---\n",i);
		size_t freeMem, totalMem; // size_t is an unsigned integer type used to represent sizes
		cudaMemGetInfo(&freeMem, &totalMem); // this function gets the amount of free and total memory on the GPU
		printf("Free memory: %zu, Total memory: %zu\n", freeMem, totalMem); // this gives you the free memory meaning how much memory is available for use, 
		// and the total memory which is the total amount of memory on the GPU
		// this is given in bytes, so to convert to gigabytes you can divide by 1024^3
		// this is just so we can  see the memory in a more understandable way
		double freeMemGB  = (double)freeMem  / (1024.0 * 1024.0 * 1024.0);
		double totalMemGB = (double)totalMem / (1024.0 * 1024.0 * 1024.0);

		printf("Free memory: %.2f GB, Total memory: %.2f GB\n", freeMemGB, totalMemGB);
		
		
		printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth); // the memory bus width is the number of bits that can be sent to the memory in parallel
		printf("L2 Cache Size: %d\n", prop.l2CacheSize); // cache size in bytes
		printf("Unified virtual address space with host: %s\n", prop.unifiedAddressing? "Enabled" : "Disabled"); // if the device shares a unified address space with the host
		printf("Supports managed memory: %s\n", prop.managedMemory ? "Yes" : "No"); // if the device supports allocating managed memory on this system, which is memory that is automatically managed by the Unified Memory system
		printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);// maximum number of threads per multiprocessor
		printf("Concurrent kernels: %s\n", prop.concurrentKernels? "Enabled" : "Disabled"); // if the device supports executing multiple kernels within the same context simultaneously
		printf("Async engine count: %d\n", prop.asyncEngineCount);// number of asynchronous engines. which are used for copy and execution
		printf("Pageable memory access: %s\n", prop.pageableMemoryAccess ? "Enabled" : "Disabled"); // if the device supports coherently accessing pageable memory without calling cudaHostRegister on it
		//-------------------------------------------------------------------------------------------------------------------------------------------------------
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		//----------added PCI and MISC information-------------------------------------------------------------------------------------------------------------
		printf("-------PCI info-------\n");
		//PCI means Peripheral Component Interconnect, it is a local computer bus for attaching hardware devices in a computer.
		//these ID are used to identify where the GPU is located in the computer
		printf("PCI Bus: %d \n",prop.pciBusID );// PCI Bus ID, 1 means the GPU is on the first bus
		printf("PCI Device: %d \n", prop.pciDeviceID);// Device ID 0 means the GPU is the first device on the bus
		printf("PCI Domain: %d \n", prop.pciDomainID);// Domain ID 0 means the GPU is on the first domain
		printf("TCC Driver: %s\n", prop.tccDriver ? "Enabled" : "Disabled");
		// TCC stands for Tesla Compute Cluster, it is a special driver for NVIDIA Tesla GPUs used in workstations and servers

		//printf("Multi-GPU Board: %s\n", prop.multiGpuBoard ? "Yes" : "No");
		// this tells you if the GPU is on a multi-GPU board, which means there are multiple GPUs on the same board ( SAdly this doesnt work on my system)

		printf("-------Misc-------\n");
		int driverVersion = 0, runtimeVersion = 0;// we need to set to 0 so that the function can write to them
		cudaDriverGetVersion(&driverVersion); // gets the version of the installed CUDA driver
		cudaRuntimeGetVersion(&runtimeVersion); // gets the version of the installed CUDA runtime
		printf("Driver Version: %d, Runtime Version: %d\n", driverVersion, runtimeVersion); // these two should be the same number 
		// because they are both the version of CUDA installed on your machine, if they are not the same then you may have a problem with your installation
		//-------------------------------------------------------------------------------------------------------------------------------------------------------
		printf("\n");
	}	
	return(0);
}

