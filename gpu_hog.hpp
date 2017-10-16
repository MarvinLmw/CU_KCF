#ifndef GPU_HOG_CUH
#define GPU_HOG_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui/highgui.hpp>

//Modified from latentsvm module's "lsvmc_featurepyramid.cpp".

//#include "precomp.hpp"
//#include "_lsvmc_latentsvm.h"
//#include "_lsvmc_resizeimg.h"


typedef struct{
    int sizeX;
    int sizeY;
    int numFeatures;
    float *map;
} CvLSVMFeatureMapCaskade;


#include "float.h"

#define PI    CV_PI

#define EPS 0.000001

#define F_MAX FLT_MAX
#define F_MIN -FLT_MAX

// The number of elements in bin
// The number of sectors in gradient histogram building
#define NUM_SECTOR 9

// The number of levels in image resize procedure
// We need Lambda levels to resize image twice
#define LAMBDA 10

// Block size. Used in feature pyramid building procedure
#define SIDE_LENGTH 8

#define VAL_OF_TRUNCATE 0.2f 


//modified from "_lsvm_error.h"
#define LATENT_SVM_OK 0
#define LATENT_SVM_MEM_NULL 2
#define DISTANCE_TRANSFORM_OK 1
#define DISTANCE_TRANSFORM_GET_INTERSECTION_ERROR -1
#define DISTANCE_TRANSFORM_ERROR -2
#define DISTANCE_TRANSFORM_EQUAL_POINTS -3
#define LATENT_SVM_GET_FEATURE_PYRAMID_FAILED -4
#define LATENT_SVM_SEARCH_OBJECT_FAILED -5
#define LATENT_SVM_FAILED_SUPERPOSITION -6
#define FILTER_OUT_OF_BOUNDARIES -7
#define LATENT_SVM_TBB_SCHEDULE_CREATION_FAILED -8
#define LATENT_SVM_TBB_NUMTHREADS_NOT_CORRECT -9
#define FFT_OK 2
#define FFT_ERROR -10
#define LSVM_PARSER_FILE_NOT_FOUND -11

#include <iostream>

inline int iDivUp(int a, int b)
{
    return (a + b - 1)/b;
}
__global__ void GetmapofHOG(uchar* in, int width, int height, int channel, float *map, int numFeatures, int stringSize);

__global__ void getpartOfNorm(float *partOfNorm, float *map, int sizeX);

__global__ void PCANTFeatureMaps(float *partOfNorm, float *map, float *newData, int sizeX, int xp0);
__global__ void MultiGetmapofHOG(uchar* in0, int width, int height, int channel, float *map0, int numFeatures, int stringSize, int page_partOfNorm);

__global__ void MultigetpartOfNorm(float *partOfNorm, float *map, int sizeX, int page_partOfNorm);

__global__ void MultiPCANTFeatureMaps(float *partOfNorm, float *map, float *newData, int sizeX, int xp0, int page_partOfNorm);


__global__ void MultiPCANTFeatureMapsPitch(float *partOfNorm, float *map, float *newData, int sizeX, int xp0,
										int pitch_partOfNorm, int pitch_map);

__global__ void MultigetpartOfNormPitch(float *partOfNorm, float *map, int sizeX, 
										int pitch_partOfNorm, int pitch_map);

__global__ void MultiGetmapofHOGPitch(uchar* in0, int width, int height, int channel, 
								 float *map0, int numFeatures, int stringSize, int pitch_in, int pitch_map);
int allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj, const int sizeX, const int sizeY,
                          const int p);

int freeFeatureMapObject (CvLSVMFeatureMapCaskade **obj);

#endif


#ifdef HAVE_TBB
#include <tbb/tbb.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#endif
/*
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif
*/
