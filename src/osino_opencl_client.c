#include <stdint.h>

#include "osino_client.h"
#include "osino_opencl_client.h"

#if defined(OSX)
#	include <OpenCL/opencl.h>
#else
#	include <CL/cl.h>
#endif

#include <string.h>
#include <math.h>

#include "cl_error_string.h"

#include "logx.h"

typedef signed short value_t;

#define BLKRES	(1<<BLKMAG)
#define BLKSIZ	(BLKRES*BLKRES*BLKRES)

#define NUMSTREAMS 16

static const size_t fieldsize = BLKSIZ * sizeof(value_t);
static const size_t casessize = BLKSIZ * sizeof(uint8_t);


static cl_device_id device_id;			// compute device id
static cl_context context;			// compute context
static cl_program program;			// compute program
static cl_kernel kernel_dens;			// compute kernel for density
static cl_kernel kernel_matt;			// compute kernel for matter type
static cl_kernel kernel_clas;			// compute kernel for classification
static cl_command_queue queues[NUMSTREAMS];	// compute command queues
static cl_mem fieldbufs[NUMSTREAMS];		// compute kernel outputs
static cl_mem casesbufs[NUMSTREAMS];		// compute kernel outputs
static value_t* stagingareas_f[NUMSTREAMS];	// staging areas for GPU->CPU copy.
static uint8_t* stagingareas_c[NUMSTREAMS];	// staging areas for GPU->CPU copy.
static size_t workgroup_size;			// OpenCL workgroup size
static size_t preferred_multiple;		// Workgroup should be a multiple of this.

#define CHECK_CL \
{ \
	const char* s = clErrorString( err ); \
	if ( err != CL_SUCCESS ) \
		LOGE( "CL Error %s:%d %s(%d)", __FILE__, __LINE__, s, err ); \
}


static int callcount=0;
static int usedcount=0;

int osino_opcl_client_usage(void)
{
	return usedcount;
}


void osino_opcl_client_release(int slot)
{
	usedcount--;
	assert(usedcount>=0);
}


static void opencl_notify
(
	const char *errinfo,
	const void *private_info,
	size_t cb,
	void *user_data
)
{
	LOGE( "OpenCL called back with error: %s", errinfo );
	//ASSERT( 0 );
}


int osino_opcl_num_platforms( void )
{
	cl_int err;
	cl_platform_id platforms[16];
	cl_uint num_platforms=-1;
	err = clGetPlatformIDs
	(
		16,
		platforms,
		&num_platforms
	);
	CHECK_CL
	return num_platforms;
}


int osino_opcl_client_init( void )
{
	cl_int err;

	cl_platform_id platforms[ 16 ];
	cl_uint num_platforms=-1;
	err = clGetPlatformIDs
	(
		16,
		platforms,
		&num_platforms
	);
	CHECK_CL
	LOGI( "OSINO found %d OpenCL platforms.", num_platforms );

	if ( num_platforms == 0 )
		return 0;

	for ( unsigned int i=0; i<num_platforms; ++i )
	{
		const cl_platform_id platform = platforms[ i ];
		LOGI( "OSINO: Platform nr %d:", i );
		char prof[ 128 ];
		char name[ 128 ];
		char vend[ 128 ];
		char vers[ 128 ];
		err = clGetPlatformInfo( platform, CL_PLATFORM_PROFILE, sizeof(prof), prof, 0 );
		CHECK_CL
		err = clGetPlatformInfo( platform, CL_PLATFORM_VERSION, sizeof(vers), vers, 0 );
		CHECK_CL
		err = clGetPlatformInfo( platform, CL_PLATFORM_NAME,    sizeof(name), name, 0 );
		CHECK_CL
		err = clGetPlatformInfo( platform, CL_PLATFORM_VENDOR,  sizeof(vend), vend, 0 );
		CHECK_CL

		cl_device_id devices[ 16 ];
		cl_uint num_devices = -1;
		LOGI( "OSINO: Getting device ids for this platform." );
		err = clGetDeviceIDs
		(
			platform,
			CL_DEVICE_TYPE_ALL,
			16,
			devices,
			&num_devices
		);
		CHECK_CL

		LOGI( "OSINO: Platform %s %s %s %s has %d device%s:", prof, vers, name, vend, num_devices, num_devices==1?"":"s" );
		for ( unsigned int j=0; j<num_devices; ++j )
		{
			cl_device_id device = devices[ j ];
			cl_uint units = -1;
			cl_device_type type;
			size_t lmem = -1;
			cl_uint dims = -1;
			size_t wisz[ 3 ];
			size_t wgsz = -1;
			size_t gmsz = -1;
			err = clGetDeviceInfo( device, CL_DEVICE_NAME, sizeof(name), name, 0 );
			err = clGetDeviceInfo( device, CL_DEVICE_VENDOR, sizeof(vend), vend, 0 );
			err = clGetDeviceInfo( device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, 0 );
			err = clGetDeviceInfo( device, CL_DEVICE_TYPE, sizeof(type), &type, 0 );
			err = clGetDeviceInfo( device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, 0 );
			err = clGetDeviceInfo( device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dims), &dims, 0 );
			err = clGetDeviceInfo( device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(wisz), &wisz, 0 );
			err = clGetDeviceInfo( device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgsz), &wgsz, 0 );
			CHECK_CL
			err = clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmsz), &gmsz, 0 );
			CHECK_CL
			if ( type == CL_DEVICE_TYPE_GPU )
				device_id = device;
			LOGI( "  %s %s with [%d units] localmem=%zuKiB globalmem=%zuMiB dims=%d(%zux%zux%zu) max workgrp sz %zu", name, vend, units, lmem/1024, gmsz/(1024*1024), dims, wisz[0], wisz[1], wisz[2], wgsz );
		}
	}

	context = clCreateContext( 0, 1, &device_id, opencl_notify, NULL, &err );
	CHECK_CL
	if ( !context )
	{
		LOGE( "OSINO: Failed to create CL context. err=0x%x", err );
		return 0;
	}

	cl_command_queue_properties queue_properties =
//		CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
//		CL_QUEUE_PROFILING_ENABLE |
		0;

	for (int i=0; i<NUMSTREAMS; ++i)
	{
		queues[i] = clCreateCommandQueue( context, device_id, queue_properties, &err );
		CHECK_CL
		if ( !queues[i] )
		{
			LOGE( "OSINO: Failed to create command queue nr %d out of %d. err=0x%x", i, NUMSTREAMS, err );
			return 0;
		}
	}

	char sourcecode0[ 32768 ];
	char sourcecode1[ 32768 ];
	memset( sourcecode0, 0, sizeof(sourcecode0) );
	memset( sourcecode1, 0, sizeof(sourcecode1) );
	FILE* f;
	int bytesread;
	f = fopen( "data/kernels-cl/computefield.cl", "r" );
	ASSERT( f );
	bytesread = fread( sourcecode0, 1, sizeof( sourcecode0 ), f );
	fclose( f );
	ASSERT( bytesread > 0 && bytesread < (int)sizeof( sourcecode0 ) );
	f = fopen( "data/kernels-cl/classifyfield.cl", "r" );
	ASSERT( f );
	bytesread = fread( sourcecode1, 1, sizeof( sourcecode1 ), f );
	fclose( f );
	ASSERT( bytesread > 0 && bytesread < (int)sizeof( sourcecode1 ) );

	const char* kernel_source[ 3 ] =
	{
		sourcecode0,
		sourcecode1,
		0,
	};

	program = clCreateProgramWithSource( context, 2, (const char **) kernel_source, NULL, &err );
	CHECK_CL
	if (!program)
	{
		LOGE( "OSINO: clCeateProgramWithSource() failed." );
		return 0;
	}
 
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CHECK_CL
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[8192];
		LOGE( "OSINO: clBuildProgram failed." );
		clGetProgramBuildInfo( program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len );
		printf("buildlog:\n");
		printf("%s\n", buffer);
		return 0;
	}

	kernel_dens = clCreateKernel(program, "osino_computefield", &err);
	CHECK_CL
	if (!kernel_dens || err != CL_SUCCESS)
	{
		LOGE( "OSINO: clCreateKernel() for computefield failed with err %0xx", err );
		return 0;
	}
	kernel_matt = clCreateKernel(program, "osino_computematter", &err);
	CHECK_CL
	if (!kernel_matt || err != CL_SUCCESS)
	{
		LOGE( "OSINO: clCreateKernel() for computematter failed with err %0xx", err );
		return 0;
	}
	kernel_clas = clCreateKernel(program, "osino_classifyfield", &err);
	CHECK_CL
	if (!kernel_clas || err != CL_SUCCESS)
	{
		LOGE( "OSINO: clCreateKernel() for classifyfield failed with err %0xx", err );
		return 0;
	}

	LOGI( "OSINO: All compute kernels successfuly created." );

	// Create the buffers, that reside GPU-side and will contain the compute kernel output.
	for (int i=0; i<NUMSTREAMS; ++i)
	{
		fieldbufs[i] = clCreateBuffer( context, CL_MEM_READ_WRITE, fieldsize, NULL, &err );
		CHECK_CL
		casesbufs[i] = clCreateBuffer( context, CL_MEM_WRITE_ONLY, casessize, NULL, &err );
		CHECK_CL
	}

	// Create the staging areas.
	for (int i=0; i<NUMSTREAMS; ++i)
	{
		stagingareas_f[i] = (value_t*)malloc( fieldsize );
		stagingareas_c[i] = (uint8_t*)malloc( casessize );
	}

	err = clGetKernelWorkGroupInfo( kernel_dens, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL );
	CHECK_CL
	err = clGetKernelWorkGroupInfo( kernel_dens, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(preferred_multiple), &preferred_multiple, NULL );
	CHECK_CL
	LOGI( "Workgroup size for computefield is %zu, preferred multiple is %zu", workgroup_size, preferred_multiple );

	callcount=0;
	usedcount=0;

	return 1;
}


void osino_opcl_client_sync(int slot)
{
	cl_int err;
	err = clFinish( queues[slot] );
}


int  osino_opcl_client_computefield (int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence)
{
	ASSERT( usedcount != NUMSTREAMS );
	const int slot = callcount++ % NUMSTREAMS;
	usedcount++;

	cl_int err;
	err = clSetKernelArg( kernel_dens,  0, sizeof(cl_mem), fieldbufs+slot);
	err = clSetKernelArg( kernel_dens,  1, sizeof(int),    &stride);
	err = clSetKernelArg( kernel_dens,  2, sizeof(int),    gridOff+0);
	err = clSetKernelArg( kernel_dens,  3, sizeof(int),    gridOff+1);
	err = clSetKernelArg( kernel_dens,  4, sizeof(int),    gridOff+2);
	err = clSetKernelArg( kernel_dens,  5, sizeof(int),    &fullres);
	err = clSetKernelArg( kernel_dens,  6, sizeof(float),  offsets+0);
	err = clSetKernelArg( kernel_dens,  7, sizeof(float),  offsets+1);
	err = clSetKernelArg( kernel_dens,  8, sizeof(float),  offsets+2);
	err = clSetKernelArg( kernel_dens,  9, sizeof(float),  &domainwarp);
	err = clSetKernelArg( kernel_dens, 10, sizeof(float),  &freq);
	err = clSetKernelArg( kernel_dens, 11, sizeof(float),  &lacunarity);
	err = clSetKernelArg( kernel_dens, 12, sizeof(float),  &persistence);
	CHECK_CL

	size_t glb_sz = BLKSIZ;
	size_t lcl_sz = workgroup_size;
	err = clEnqueueNDRangeKernel
	(
		queues[slot],
		kernel_dens,
		1,
		NULL,
		&glb_sz,
		&lcl_sz,
		0,
		NULL,
		NULL
	);
	CHECK_CL
	if (err != CL_SUCCESS) LOGW("Failed to queue computefield kernel.");
	return  slot;
}


void osino_opcl_client_stagefield(int slot)
{
	// Trigger a transfer here!
	cl_int err;
	const cl_bool blocking_rd = CL_FALSE;
	err = clEnqueueReadBuffer( queues[slot], fieldbufs[slot], blocking_rd, 0, fieldsize, stagingareas_f[slot], 0, NULL, NULL );
	CHECK_CL
}


void osino_opcl_client_stagecases(int slot)
{
	// Trigger a transfer here!
	cl_int err;
	const cl_bool blocking_rd = CL_FALSE;
	err = clEnqueueReadBuffer( queues[slot], casesbufs[slot], blocking_rd, 0, casessize, stagingareas_c[slot], 0, NULL, NULL );
	CHECK_CL
}


void osino_opcl_client_collectfield(int slot, value_t* output)
{
	memcpy( output, stagingareas_f[slot], fieldsize );
}


void osino_opcl_client_collectcases(int slot, uint8_t* output)
{
	memcpy( output, stagingareas_c[slot], casessize );
}


extern int  osino_opcl_client_computematter(int stride, int gridOff[3], int fullres, const float offsets[3], float domainwarp, float freq, float lacunarity, float persistence)
{
	ASSERT( usedcount != NUMSTREAMS );
	const int slot = callcount++ % NUMSTREAMS;
	usedcount++;

	cl_int err;
	err = clSetKernelArg( kernel_matt,  0, sizeof(cl_mem), fieldbufs+slot);	CHECK_CL
	err = clSetKernelArg( kernel_matt,  1, sizeof(int),    &stride);	CHECK_CL
	err = clSetKernelArg( kernel_matt,  2, sizeof(int),    gridOff+0);	CHECK_CL
	err = clSetKernelArg( kernel_matt,  3, sizeof(int),    gridOff+1);	CHECK_CL
	err = clSetKernelArg( kernel_matt,  4, sizeof(int),    gridOff+2);	CHECK_CL
	err = clSetKernelArg( kernel_matt,  5, sizeof(int),    &fullres);	CHECK_CL
	err = clSetKernelArg( kernel_matt,  6, sizeof(float),  offsets+0);	CHECK_CL
	err = clSetKernelArg( kernel_matt,  7, sizeof(float),  offsets+1);	CHECK_CL
	err = clSetKernelArg( kernel_matt,  8, sizeof(float),  offsets+2);	CHECK_CL
	err = clSetKernelArg( kernel_matt,  9, sizeof(float),  &domainwarp);	CHECK_CL
	err = clSetKernelArg( kernel_matt, 10, sizeof(float),  &freq);		CHECK_CL
	err = clSetKernelArg( kernel_matt, 11, sizeof(float),  &lacunarity);	CHECK_CL
	err = clSetKernelArg( kernel_matt, 12, sizeof(float),  &persistence);	CHECK_CL

	size_t glb_sz = BLKSIZ;
	size_t lcl_sz = workgroup_size;
	err = clEnqueueNDRangeKernel
	(
		queues[slot],
		kernel_matt,
		1,
		NULL,
		&glb_sz,
		&lcl_sz,
		0,
		NULL,
		NULL
	);
	CHECK_CL
	if (err != CL_SUCCESS) LOGW("Failed to queue computematter kernel.");
	return slot;
}


void osino_opcl_client_classifyfield
(
	int slot, 
	value_t isoval
)
{
	cl_int err;
	err = clSetKernelArg( kernel_clas, 0, sizeof(value_t), &isoval        ); CHECK_CL
	err = clSetKernelArg( kernel_clas, 1, sizeof(cl_mem),  fieldbufs+slot ); CHECK_CL
	err = clSetKernelArg( kernel_clas, 2, sizeof(cl_mem),  casesbufs+slot ); CHECK_CL
	size_t glb_sz = BLKSIZ;
	size_t lcl_sz = workgroup_size;
	err = clEnqueueNDRangeKernel
	(
		queues[slot],
		kernel_clas,
		1,
		NULL,
		&glb_sz,
		&lcl_sz,
		0,
		NULL,
		NULL
	);
	CHECK_CL
	if (err != CL_SUCCESS) LOGW("Failed to queue classifyfield kernel.");
}


void osino_opcl_client_exit( void )
{
	cl_int err;
	for ( int i=0; i<NUMSTREAMS; ++i )
	{
		free(stagingareas_f[i]);
		stagingareas_f[i]=0;
		free(stagingareas_c[i]);
		stagingareas_c[i]=0;
		err = clReleaseMemObject( fieldbufs[i] );
		CHECK_CL
		fieldbufs[i] = 0;
		err = clReleaseMemObject( casesbufs[i] );
		CHECK_CL
		casesbufs[i] = 0;
	}
	err = clReleaseProgram( program );
	CHECK_CL
	program = 0;
	err = clReleaseKernel( kernel_matt );
	CHECK_CL
	kernel_matt = 0;
	err = clReleaseKernel( kernel_dens );
	CHECK_CL
	kernel_dens = 0;
	for ( int i=0; i<NUMSTREAMS; ++i )
	{
		err = clReleaseCommandQueue( queues[i] );
		CHECK_CL
		queues[i] = 0;
	}
	err = clReleaseContext( context );
	CHECK_CL
	context = 0;
	LOGW("OpenCL client has been shut down.");
}


