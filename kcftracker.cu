/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "labdata.hpp"
#include "gpu_hog.hpp"
#endif

//#include <sys/time.h>
#include <iostream>

#include "cudautils.h"

using namespace std;

extern int part_time[16];


cv::KalmanFilter KF(1, 1, 0); 

// Constructor
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{
    // Parameters equal in all cases
    lambda = 0.0001;
    padding = 2.5;
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;

    //Added by LMW
    average = 0.0f;
    roi_zero = cv::Rect(0,0,0,0);
    roi_lose = cv::Rect(0,0,0,0);
    nFrame = 0;
    nFrame_lose = 0;
    target_lose = false;
	initial = true;

    if (hog) {    // HOG
        // VOT
        interp_factor = 0.012;
        sigma = 0.6;
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5;
        cell_size = 4;
        _hogfeatures = true;

        if (lab) {
            interp_factor = 0.005;
            sigma = 0.4;
            //output_sigma_factor = 0.025;
            output_sigma_factor = 0.1;

            _labfeatures = true;
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);
            cell_sizeQ = cell_size*cell_size;
        }
        else{
            _labfeatures = false;
        }
    }
    else {   // RAW
        interp_factor = 0.075;
        sigma = 0.2;
        cell_size = 1;
        _hogfeatures = false;

        if (lab) {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }


    if (multiscale) { // multiscale
        template_size = 96;//lmw
        //template_size = 100;
        scale_step = 1.1;//1.05;
        scale_weight = 0.95;
        if (!fixed_window) {
            //printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
    }
    else {
        template_size = 1;
        scale_step = 1;
    }
}

// Initialize tracker
int KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;
    if(roi.width < 0 || roi.height < 0)//assert
      return 1;

    int padded_w = _roi.width * padding;
    int padded_h = _roi.height * padding;

    if (template_size > 1) {  // Fit largest dimension to the given template size
        if (padded_w >= padded_h)  //fit to width
            _scale = padded_w / (float) template_size;
        else
            _scale = padded_h / (float) template_size;

        _tmpl_sz.width = padded_w / _scale;//lmw*2
        _tmpl_sz.height = padded_h / _scale;
    }
    else {  //No template size given, use ROI size
        _tmpl_sz.width = padded_w;//lmw*2
        _tmpl_sz.height = padded_h;
        _scale = 1;

    }
	// Round to cell size and also make it even
	_tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
	_tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
	
    size_patch[0] = _tmpl_sz.height/4-2;
    size_patch[1] = _tmpl_sz.width/4-2;
    size_patch[2] = NUM_SECTOR * 3 + 4;
    createHanningMats();

    _tmpl = getFeatures(image, 0);
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame
    return 0;
 }


void KCFTracker::Allocate(){
	pp    = NUM_SECTOR * 3 + 4;
	safeCall(cudaMalloc((void**)&mul_d_map, sizeof (float) * (sizeX * sizeY  * pp*5)));
	safeCall(cudaMalloc((void**)&mul_partOfNorm, sizeof (float) * (sizeX * sizeY*5)));
	safeCall(cudaMalloc((void**)&mul_finData, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR * 12*5)));

	
	int page = _tmpl_sz.height*3*_tmpl_sz.width;
	safeCall(cudaMalloc((void**)&mul_in, 5*page));
}
// Obtain sub-window from image, with replication-padding and extract features
cv::Point2f KCFTracker::getMultiFeaturesResultPitch(const cv::Mat & image, bool inithann, float d_scale_adjust, float &peak_value_result)
{
	cv::Point2f res;
	cv::Point2f new_res;
    float new_peak_value = 0;
	float tmp_scale;
	cv::Rect tmp_roi;

	float cx = _roi.x + _roi.width / 2;
	float cy = _roi.y + _roi.height / 2;
    float peak_value = 0;float scale_adjust = 1/(d_scale_adjust*d_scale_adjust);
	
	cv::Rect extracted_roi;
	cv::Mat z(_tmpl_sz.height*5, _tmpl_sz.width, CV_8UC3, cv::Scalar(255,255,255));
	int page = _tmpl_sz.height*3*_tmpl_sz.width;
	extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
	extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data, image, extracted_roi, _tmpl_sz);

	extracted_roi.width *=d_scale_adjust;
	extracted_roi.height *=d_scale_adjust;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data+page, image, extracted_roi, _tmpl_sz);

	extracted_roi.width *=d_scale_adjust;
	extracted_roi.height *=d_scale_adjust;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data+page*2, image, extracted_roi, _tmpl_sz);

	extracted_roi.width *=d_scale_adjust;
	extracted_roi.height *=d_scale_adjust;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data+page*3, image, extracted_roi, _tmpl_sz);

	extracted_roi.width *=d_scale_adjust;
	extracted_roi.height *=d_scale_adjust;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data+page*4, image, extracted_roi, _tmpl_sz);

	p0     = 3 * NUM_SECTOR;
	p  = NUM_SECTOR;
	xp = NUM_SECTOR * 3;
	pp    = NUM_SECTOR * 3 + 4;
	height = _tmpl_sz.height;
	width  = _tmpl_sz.width ;
	sizeX = (int)width  / 4;
	sizeY = (int)height / 4;
	stringSize = sizeX * p0;
	block1 = dim3((sizeX+3)/4, sizeY,5);
	thread1 = dim3(16, 4);
	block2 = dim3((sizeX+2)/5, sizeY-2,5);
	thread2 = dim3(10*NUM_SECTOR, 1);
	block3 = dim3((sizeX+6)/7, sizeY,5);
	thread3 = dim3(63, 1);

	cudaMalloc((void**)&mul_d_map, sizeof (float) * (sizeX * sizeY  * pp*5));
	cudaMalloc((void**)&mul_partOfNorm, sizeof (float) * (sizeX * sizeY*5));
	cudaMalloc((void**)&mul_finData, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR * 12*5));
	cudaMalloc((void**)&mul_in, 5*page);
	size_t pitch_d_map;
	size_t pitch_partOfNorm;
	size_t pitch_finData;
	size_t pitch_in;
	/*safeCall(cudaMallocPitch((void **)&d_map, (size_t*)&pitch_d_map,
							(size_t)(sizeof(float)*sizeX*sizeY*pp), (size_t)5));
	safeCall(cudaMallocPitch((void **)&mul_partOfNorm, (size_t*)&pitch_partOfNorm, 
							(size_t)(sizeof(float)*sizeX*sizeY), (size_t)5));
	safeCall(cudaMallocPitch((void **)&mul_finData, (size_t*)&pitch_finData, 
							(size_t)(sizeof(float)*sizeX*sizeY*NUM_SECTOR*12), (size_t)5));
	safeCall(cudaMallocPitch((void **)&mul_in, (size_t*)&pitch_in, 
							(size_t)(sizeof(uchar)*page), (size_t)5));*/
	//Allocate();
	//safeCall(cudaMemset((void**)&d_map, 0, sizeof (float) * (sizeX * sizeY  * pp*5)));
	//safeCall(cudaMemset((void**)&partOfNorm, 0, sizeof (float) * (sizeX * sizeY*5)));
	//safeCall(cudaMemset((void**)&finData, 0, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR * 12*5)));
	scale_now = false;
	int page_partOfNorm = sizeX * sizeY;
	//safeCall(cudaMemcpy2D(mul_in, pitch_in,  z.data, sizeof(uchar)*page, sizeof(uchar)*page, 5, cudaMemcpyHostToDevice));
	//MultiGetmapofHOGPitch<<< block1, thread1 >>>(mul_in, width, height, 3, mul_d_map, p0, stringSize,  pitch_in, pitch_d_map/4);
	//MultigetpartOfNormPitch<<< block3, thread3 >>>(mul_partOfNorm, mul_d_map, sizeX, pitch_partOfNorm/4, pitch_d_map/4);
	//MultiPCANTFeatureMapsPitch<<< block2, thread2 >>>(mul_partOfNorm, mul_d_map, mul_finData, sizeX-2, xp, pitch_partOfNorm/4, pitch_d_map/4);
	safeCall( cudaMemcpy(mul_in, z.data, 5*page, cudaMemcpyHostToDevice));
	MultiGetmapofHOGPitch<<< block1, thread1 >>>(mul_in, width, height, 3, mul_d_map, p0, stringSize, page, sizeX * sizeY  * pp);
	MultigetpartOfNormPitch<<< block3, thread3 >>>(mul_partOfNorm, mul_d_map, sizeX, sizeX * sizeY, sizeX * sizeY  * pp*5);
	MultiPCANTFeatureMapsPitch<<< block2, thread2 >>>(mul_partOfNorm, mul_d_map, mul_finData, sizeX-2, xp, sizeX * sizeY, sizeX * sizeY  * pp*5);
	
		tmp_scale = _scale;
		tmp_roi.width = _roi.width;
		tmp_roi.height = _roi.height;
	for(int i=0;i<5;i++){
        CvLSVMFeatureMapCaskade *map;
		allocFeatureMapObject(&map, sizeX-2, sizeY-2, pp);
		safeCall( cudaMemcpy(map->map, mul_finData+i*sizeX * sizeY  * NUM_SECTOR * 12, 
			sizeof (float) * ((sizeX-2)* (sizeY-2)* pp), cudaMemcpyDeviceToHost));
		//safeCall(cudaMemcpy2D(map->map, sizeof (float) * ((sizeX-2)* (sizeY-2)* pp), mul_finData+i*pitch_finData, pitch_finData, sizeof (float) * ((sizeX-2)* (sizeY-2)* pp), 1, cudaMemcpyDeviceToHost));

        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;

		cv::Mat FeaturesMap;
        FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);
		FeaturesMap = hann.mul(FeaturesMap);

	
		new_res = detect(_tmpl, FeaturesMap, new_peak_value);
		if (scale_weight * new_peak_value > (1-0.05*(i!=2))*peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			tmp_scale = _scale*scale_adjust;
			scale_now = i!=2;
			tmp_roi.width = tmp_roi.width*scale_adjust;
			tmp_roi.height = tmp_roi.height*scale_adjust;
		}
		scale_adjust*=d_scale_adjust;
	}


	peak_value_result = peak_value;
    _scale = tmp_scale;
    _roi.width = tmp_roi.width;
    _roi.height = tmp_roi.height;
    return res;
}
// Obtain sub-window from image, with replication-padding and extract features
cv::Point2f KCFTracker::getMultiFeaturesResult(const cv::Mat & image, bool inithann, float d_scale_adjust, float &peak_value_result)
{
	cv::Point2f res;
	cv::Point2f new_res;
    float new_peak_value = 0;
	float tmp_scale;
	cv::Rect tmp_roi;

	float cx = _roi.x + _roi.width / 2;
	float cy = _roi.y + _roi.height / 2;
    float peak_value = 0;float scale_adjust = 1/(d_scale_adjust*d_scale_adjust);
	
	cv::Rect extracted_roi;
	cv::Mat z(_tmpl_sz.height*5, _tmpl_sz.width, CV_8UC3, cv::Scalar(255,255,255));
	int page = _tmpl_sz.height*3*_tmpl_sz.width;
	extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
	extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data, image, extracted_roi, _tmpl_sz);

	extracted_roi.width *=d_scale_adjust;
	extracted_roi.height *=d_scale_adjust;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data+page, image, extracted_roi, _tmpl_sz);

	extracted_roi.width *=d_scale_adjust;
	extracted_roi.height *=d_scale_adjust;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data+page*2, image, extracted_roi, _tmpl_sz);

	extracted_roi.width *=d_scale_adjust;
	extracted_roi.height *=d_scale_adjust;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data+page*3, image, extracted_roi, _tmpl_sz);

	extracted_roi.width *=d_scale_adjust;
	extracted_roi.height *=d_scale_adjust;
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;
	RectTools::subresize_data(z.data+page*4, image, extracted_roi, _tmpl_sz);

	p0     = 3 * NUM_SECTOR;
	p  = NUM_SECTOR;
	xp = NUM_SECTOR * 3;
	pp    = NUM_SECTOR * 3 + 4;
	height = _tmpl_sz.height;
	width  = _tmpl_sz.width ;
	sizeX = (int)width  / 4;
	sizeY = (int)height / 4;
	stringSize = sizeX * p0;
	block1 = dim3((sizeX+3)/4, sizeY,5);
	thread1 = dim3(16, 4);
	block2 = dim3((sizeX+2)/5, sizeY-2,5);
	thread2 = dim3(10*NUM_SECTOR, 1);
	block3 = dim3((sizeX+6)/7, sizeY,5);
	thread3 = dim3(63, 1);


	block1 = dim3((sizeX+3)/4, sizeY, 5);
	thread1 = dim3(16, 4);
	block2 = dim3((sizeX+2)/5, sizeY-2, 5);
	thread2 = dim3(10*NUM_SECTOR, 1);
	block3 = dim3((sizeX+6)/7, sizeY, 5);
	thread3 = dim3(63, 1);
	cudaMalloc((void**)&d_map, sizeof (float) * (sizeX * sizeY  * pp*5));
	cudaMalloc((void**)&partOfNorm, sizeof (float) * (sizeX * sizeY*5));
	cudaMalloc((void**)&finData, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR * 12*5));
		cudaMalloc((void**)&in, page*5);
		cudaMemcpy(in, z.data, page*5, cudaMemcpyHostToDevice);

/*
	//cudaMalloc((void**)&mul_d_map, sizeof (float) * (sizeX * sizeY  * pp*5));
	//cudaMalloc((void**)&mul_partOfNorm, sizeof (float) * (sizeX * sizeY*5));
	//cudaMalloc((void**)&mul_finData, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR * 12*5));
	//cudaMalloc((void**)&mul_in, 5*page);
	size_t pitch_d_data;
	size_t pitch_partOfNorm;
	size_t pitch_finData;
	size_t pitch_in;
	safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&	size_t pitch_d_data,]
							(size_t)(sizeof(float)*width), (size_t)height));
	safeCall(cudaMallocPitch((void **)&mul_partOfNorm, (size_t*)&pitch_partOfNorm, 
							(size_t)(sizeof(float)*width), (size_t)height));
	safeCall(cudaMallocPitch((void **)&mul_finData, (size_t*)&pitch_finData, 
							(size_t)(sizeof(float)*width), (size_t)height));
	safeCall(cudaMallocPitch((void **)&mul_in, (size_t*)&pitch_in, 
							(size_t)(sizeof(float)*width), (size_t)height));*/
	//Allocate();
	//safeCall(cudaMemset((void**)&d_map, 0, sizeof (float) * (sizeX * sizeY  * pp*5)));
	//safeCall(cudaMemset((void**)&partOfNorm, 0, sizeof (float) * (sizeX * sizeY*5)));
	//safeCall(cudaMemset((void**)&finData, 0, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR * 12*5)));
			scale_now = false;
	int page_partOfNorm = sizeX * sizeY;
		safeCall( cudaMemcpy(mul_in, z.data, 5*page, cudaMemcpyHostToDevice));
		MultiGetmapofHOG<<< block1, thread1 >>>(mul_in, width, height, 3, mul_d_map, p0, stringSize, page_partOfNorm);
		MultigetpartOfNorm<<< block3, thread3 >>>(mul_partOfNorm, mul_d_map, sizeX, page_partOfNorm);
		MultiPCANTFeatureMaps<<< block2, thread2 >>>(mul_partOfNorm, mul_d_map, mul_finData, sizeX-2, xp, page_partOfNorm);
		tmp_scale = _scale;
		tmp_roi.width = _roi.width;
		tmp_roi.height = _roi.height;
	for(int i=0;i<5;i++){

        CvLSVMFeatureMapCaskade *map;
		allocFeatureMapObject(&map, sizeX-2, sizeY-2, pp);
		safeCall( cudaMemcpy(map->map, mul_finData+i*sizeX * sizeY  * NUM_SECTOR * 12, 
			sizeof (float) * ((sizeX-2)* (sizeY-2)* pp), cudaMemcpyDeviceToHost));

        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;

		cv::Mat FeaturesMap;
        FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);
		FeaturesMap = hann.mul(FeaturesMap);

	
		new_res = detect(_tmpl, FeaturesMap, new_peak_value);
		new_peak_value = (1-0.05*(i!=2))*new_peak_value;
		tmp_scale = _scale;
		if (new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			tmp_scale = _scale*scale_adjust;
			scale_now = i!=2;
			tmp_roi.width = _roi.width*scale_adjust;
			tmp_roi.height = _roi.height*scale_adjust;
		}
		scale_adjust*=d_scale_adjust;
	}


	peak_value_result = peak_value;
    _scale = tmp_scale;
    _roi.width = tmp_roi.width;
    _roi.height = tmp_roi.height;
    return res;
}


// Obtain sub-window from image, with replication-padding and extract features
cv::Mat KCFTracker::getMultiFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{


    cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;

    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    cv::Mat FeaturesMap;

    cv::Mat z(_tmpl_sz.height, _tmpl_sz.width, CV_8UC3, cv::Scalar(255,255,255));
    RectTools::subresize_3(z, image, extracted_roi, _tmpl_sz);
        IplImage z_ipl = z;
        CvLSVMFeatureMapCaskade *map;
		
		p0     = 3 * NUM_SECTOR;
		p  = NUM_SECTOR;
		xp = NUM_SECTOR * 3;
		pp    = NUM_SECTOR * 3 + 4;
		height = _tmpl_sz.height;
		width  = _tmpl_sz.width ;
		sizeX = (int)width  / 4;
		sizeY = (int)height / 4;
		stringSize = sizeX * p0;
		

		block1 = dim3((sizeX+3)/4, sizeY);
		thread1 = dim3(16, 4);
		block2 = dim3((sizeX+2)/5, sizeY-2);
		thread2 = dim3(10*NUM_SECTOR, 1);
		block3 = dim3((sizeX+6)/7, sizeY);
		thread3 = dim3(63, 1);
		 safeCall( cudaMalloc((void**)&in, height * width * 3 * sizeof(uchar)));
		cudaMalloc((void**)&d_map, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR*12));
		cudaMalloc((void**)&partOfNorm, sizeof (float) * (sizeX * sizeY));
		cudaMalloc((void**)&finData, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR * 12));

		cudaMemcpy(in, z_ipl.imageData, height * width * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

		GetmapofHOG<<< block1, thread1 >>>(in, width, height, 3, d_map, p0, stringSize);
		cudaDeviceSynchronize();
		getpartOfNorm<<< block3, thread3 >>>(partOfNorm, d_map, sizeX);

		PCANTFeatureMaps<<< block2, thread2 >>>(partOfNorm, d_map, finData, sizeX-2, xp);
		cudaDeviceSynchronize();

		allocFeatureMapObject(&map, sizeX-2, sizeY-2, pp);
		cudaMemcpy(map->map, finData, sizeof (float) * ((sizeX-2)* (sizeY-2)* pp), cudaMemcpyDeviceToHost);

        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;

        FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);
    cudaFree(in);
    cudaFree(d_map);
    cudaFree(partOfNorm);
    cudaFree(finData);

    FeaturesMap = hann.mul(FeaturesMap);

    return FeaturesMap;
}

// Update position based on the new frame
float KCFTracker::detectmultiscales(cv::Mat image, cv::Point2f &res)
{
    float peak_value;

	cv::Mat hogfeature;

	res = getMultiFeaturesResult(image, 0, scale_step, peak_value);

	return peak_value;
}

// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat image)
{
	TimerGPU timeG;
    nFrame++;
    //struct timeval tv, tz;

    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;


    //gettimeofday(&tv, NULL);//time1:detect
	cv::Point2f res;
	
	float peak_value;
    if ((scale_step != 1)&&(!target_lose)) {
#if GPU_MODEL
		peak_value = detectmultiscales(image, res);
#else GPU_MODEL
		
		res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);
        // Test at a smaller _scale:-1
        float new_peak_value;
        cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale /= scale_step;
            _roi.width /= scale_step;
            _roi.height /= scale_step;
        }

        // Test at a bigger _scale:1
        new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale *= scale_step;
            _roi.width *= scale_step;
            _roi.height *= scale_step;
        }

        // Test at a smaller _scale:-2
        new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / (scale_step*scale_step)), new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale /= (scale_step*scale_step);
            _roi.width /= (scale_step*scale_step);
            _roi.height /= (scale_step*scale_step);
        }

        // Test at a bigger _scale:2
        new_res = detect(_tmpl, getFeatures(image, 0, scale_step*scale_step), new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale *= scale_step*scale_step;
            _roi.width *= scale_step*scale_step;
            _roi.height *= scale_step*scale_step;
        }
#endif
    }
    else{
      std::cout<<"lose_size="<<_roi.width<<", "<<_roi.height<<std::endl;
		res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value);
    
      float new_peak_value;
      cv::Rect roi_save;
      cv::Point2f new_res;
      for(int dx=-1;dx<2;dx++){
        for(int dy=-1;dy<2;dy++){

	  if(!(dx||dy))
	    continue;
	  

	  _roi.x = roi_lose.x + dx * _scale * _tmpl_sz.width*0.6;
	  _roi.y = roi_lose.y + dy * _scale * _tmpl_sz.height*0.6;
	  if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	  if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	  if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	  if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;


          new_res = detect(_tmpl, getFeatures(image, 0, 1), new_peak_value);
          cout<<(dx+1)*3+dy+1<<"="<<new_peak_value<<" ";
          if (scale_weight * new_peak_value > peak_value) {
              res = new_res;
              peak_value = new_peak_value;
	      roi_save = _roi;
	      cx = _roi.x + _roi.width / 2.0f;
	      cy = _roi.y + _roi.height / 2.0f;
          }
	}
      }
      _roi = roi_save;
      _roi.width = roi_lose.width;
      _roi.height = roi_lose.height;
    }

    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;
/*
    if(_roi.width >= 0 && _roi.height >= 0){//;assert
      _roi = roi_lose;
      target_lose = true;
      if(target_lose)
        nFrame_lose++;
      std::cout<<" nothing"<<std::endl;
      return roi_zero;
    }
*/
    cv::Mat x = getFeatures(image, 0);

	cv::Mat  measurement = cv::Mat::zeros(1, 1, CV_32F); ;
	if(nFrame==1){
		KF.transitionMatrix = *(cv::Mat_<float>(1, 1) <<1);				//转移矩阵A  
		setIdentity(KF.measurementMatrix);								//测量矩阵H  
		setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));         //系统噪声方差矩阵Q  
		setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));     //测量噪声方差矩阵R  
		setIdentity(KF.errorCovPost, cv::Scalar::all(1)); 
		
		//初始状态值
		KF.statePost = *(cv::Mat_<float>(1, 1) <<peak_value);//,A[0][1],A[0][2]); 
	}
	
	    cv::Mat prediction = KF.predict(); 			
            //计算测量值
	    measurement.at<float>(0) = peak_value;  
	    //measurement.at<float>(1) = (float)A[i][1]; 
	   // measurement.at<float>(2) = (float)A[i][2]; 
	    //更新
	    KF.correct(measurement);  
	    float kalman = KF.statePost.at<float>(0);//<<endl<<"\t"<<KF.statePost.at<float>(1)<<endl;//<<"\t"<<KF.statePost.at<float>(2)<<endl;
		float tmp_detect = kalman*(0.6-0.1*(scale_now==true));
    if(peak_value > tmp_detect || (peak_value > average*0.4 && target_lose==true) ){
      train(x, interp_factor);
      average = (average*(nFrame-1)+peak_value)/nFrame;
      roi_lose = _roi;
      target_lose = false;
      nFrame_lose = 0;
    std::cout<<nFrame<<" = "<<timeG.read()<<"ms, = "<<peak_value;
    std::cout<<" = "<<tmp_detect;
      std::cout<<"; size="<<_roi.width<<", "<<_roi.height<<std::endl;
      return _roi;
      
    }
    else{// if(peak_value<average*0.4)
      _roi = roi_lose;
      target_lose = true;
      if(target_lose)
        nFrame_lose++;
    std::cout<<nFrame<<" = "<<timeG.read()<<"ms, = "<<peak_value;
    std::cout<<" = "<<tmp_detect;
      std::cout<<" nothing"<<std::endl;
      return roi_lose;
    }
}


// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, z);
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float) pv;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols-1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1), peak_value, res.at<float>(pi.y, pi.x+1));
    }

    if (pi.y > 0 && pi.y < res.rows-1) {
        p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x), peak_value, res.at<float>(pi.y+1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p;
}

// train tracker with a single image
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, x);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;


    /*cv::Mat kf = fftd(gaussianCorrelation(x, x));
    cv::Mat num = complexMultiplication(kf, _prob);
    cv::Mat den = complexMultiplication(kf, kf + lambda);

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
    _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

    _alphaf = complexDivision(_num, _den);*/

}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, 
// which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat( cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0) );
    // HOG features
    if (_hogfeatures) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);
            x2aux = x2.row(i).reshape(1, size_patch[0]);
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true);
            rearrange(caux);
            caux.convertTo(caux,CV_32F);
            c = c + real(caux);
        }
    }
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }
    cv::Mat d;
    cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0])- 2. * c) / (size_patch[0]*size_patch[1]*size_patch[2]) , 0, d);

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
    return k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat KCFTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{


    cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;

    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;


    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    cv::Mat FeaturesMap;


    int loss_flag = 0;
    cv::Mat z;
    if(image.channels()==3){
      cv::Mat zz(_tmpl_sz.height, _tmpl_sz.width, CV_8UC3, cv::Scalar(255,255,255));
      loss_flag = RectTools::subresize_3(zz, image, extracted_roi, _tmpl_sz);
      z = zz;
    }
    else if(image.channels()==1){
      cv::Mat zz(_tmpl_sz.height, _tmpl_sz.width, CV_8UC1, cv::Scalar(255));
      loss_flag = RectTools::subresize_1(zz, image, extracted_roi, _tmpl_sz);
      z = zz;
    }
    if(loss_flag){
      //return NULL;
    }
    cv::imwrite("cry.bmp",z);
    //getchar();
	//z = cv::imread("cry22.bmp");
    // HOG features
    if (_hogfeatures) {
        IplImage z_ipl = z;
        CvLSVMFeatureMapCaskade *map;
		/*
        getFeatureMaps(&z_ipl, cell_size, &map, z.channels());
        //IplImage z_ipl = z, cell_size = 4, CvLSVMFeatureMapCaskade *map//Error?
        normalizeAndTruncate(map,0.2f);
        PCAFeatureMaps(map);*/
		//getPcaHogFeatureMaps(&z_ipl, cell_size, &map, z.channels());
		
		p0     = 3 * NUM_SECTOR;
		p  = NUM_SECTOR;
		xp = NUM_SECTOR * 3;
		pp    = NUM_SECTOR * 3 + 4;
		height = _tmpl_sz.height;
		width  = _tmpl_sz.width ;
		sizeX = (int)width  / 4;
		sizeY = (int)height / 4;
		stringSize = sizeX * p0;
		

		block1 = dim3((sizeX+3)/4, sizeY);
		thread1 = dim3(16, 4);
		block2 = dim3((sizeX+2)/5, sizeY-2);
		thread2 = dim3(10*NUM_SECTOR, 1);
		block3 = dim3((sizeX+6)/7, sizeY);
		thread3 = dim3(63, 1);
		cudaMalloc((void**)&in, height * width * 3 * sizeof(uchar));
		cudaMalloc((void**)&d_map, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR*12));
		cudaMalloc((void**)&partOfNorm, sizeof (float) * (sizeX * sizeY));
		cudaMalloc((void**)&finData, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR * 12));

		cudaMemcpy(in, z_ipl.imageData, height * width * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

		GetmapofHOG<<< block1, thread1 >>>(in, width, height, 3, d_map, p0, stringSize);
		cudaDeviceSynchronize();
		getpartOfNorm<<< block3, thread3 >>>(partOfNorm, d_map, sizeX);

		PCANTFeatureMaps<<< block2, thread2 >>>(partOfNorm, d_map, finData, sizeX-2, xp);
		cudaDeviceSynchronize();

		allocFeatureMapObject(&map, sizeX-2, sizeY-2, pp);
		cudaMemcpy(map->map, finData, sizeof (float) * ((sizeX-2)* (sizeY-2)* pp), cudaMemcpyDeviceToHost);

        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;

        FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);
    cudaFree(in);
    cudaFree(d_map);
    cudaFree(partOfNorm);
    cudaFree(finData);
    }

    //gettimeofday(&tz, NULL);
    //part_time[6] = part_time[6] -1*(tv.tv_sec*1000+tv.tv_usec/1000-tz.tv_sec*1000-tz.tv_usec/1000);//time3:getFeatureMaps

    if (inithann) {
        createHanningMats();
    }
    FeaturesMap = hann.mul(FeaturesMap);/*
	for(int i=0;i<FeaturesMap.rows;i++){
		for(int j=0;j<FeaturesMap.cols;j++){
			printf("%f,",FeaturesMap.at<float> (i, j));
		}
			printf("\n");
	}
	cv::Mat write;
	FeaturesMap.convertTo(write,CV_8UC1);
	cv::imwrite("ԭFeaturesMap.bmp", FeaturesMap);
	getchar();
	exit(-1);*/
    //gettimeofday(&tz0, NULL);
    //part_time[2] = part_time[2] -1*(tv0.tv_sec*1000+tv0.tv_usec/1000-tz0.tv_sec*1000-tz0.tv_usec/1000);//time3:getFeatureMaps

    return FeaturesMap;
}

// Initialize Hanning window. Function called only in the first frame.
void KCFTracker::createHanningMats()
{
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[0]), CV_32F, cv::Scalar(0));

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures) {
        cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug

        hann = cv::Mat(cv::Size(size_patch[0]*size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++) {
            for (int j = 0; j<size_patch[0]*size_patch[1]; j++) {
                hann.at<float>(i,j) = hann1d.at<float>(0,j);
            }
        }
    }
    // Gray features
    else {
        hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;

    return 0.5 * (right - left) / divisor;
}

/*
int KCFTracker::getMaps(const IplImage* image, const int k, CvLSVMFeatureMapCaskade **map, int channels){
	
    p0     = 3 * NUM_SECTOR;
    p  = NUM_SECTOR;
    xp = NUM_SECTOR * 3;
    pp    = NUM_SECTOR * 3 + 4;
    height = _tmpl_sz.height;
    width  = _tmpl_sz.width ;
    sizeX = (int)width  / 4;
    sizeY = (int)height / 4;
    stringSize = sizeX * p0;
		

	block1 = dim3((sizeX+3)/4, sizeY);
	thread1 = dim3(16, 4);
	block2 = dim3((sizeX+2)/5, sizeY-2);
	thread2 = dim3(10*NUM_SECTOR, 1);
	block3 = dim3((sizeX+6)/7, sizeY);
	thread3 = dim3(63, 1);
	cudaMalloc((void**)&in, height * width * 3 * sizeof(uchar));
	cudaMalloc((void**)&d_map, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR*12));
	cudaMalloc((void**)&partOfNorm, sizeof (float) * (sizeX * sizeY));
	cudaMalloc((void**)&finData, sizeof (float) * (sizeX * sizeY  * NUM_SECTOR * 12));

	cudaMemcpy(in, image->imageData, height * width * channels * sizeof(uchar), cudaMemcpyHostToDevice);

	GetmapofHOG<<< block1, thread1 >>>(in, width, height, channels, d_map, p0, stringSize);
    cudaDeviceSynchronize();
	getpartOfNorm<<< block3, thread3 >>>(partOfNorm, d_map, sizeX);

	PCANTFeatureMaps<<< block2, thread2 >>>(partOfNorm, d_map, finData, sizeX-2, xp);
    cudaDeviceSynchronize();

    allocFeatureMapObject(map, sizeX-2, sizeY-2, pp);
	cudaMemcpy((*map)->map, finData, sizeof (float) * ((sizeX-2)* (sizeY-2)* pp), cudaMemcpyDeviceToHost);

    return LATENT_SVM_OK;

}*/

KCFTracker::~KCFTracker(){
		cudaFree(mul_in);
		cudaFree(mul_d_map);
		cudaFree(mul_partOfNorm);
		cudaFree(mul_finData);
}