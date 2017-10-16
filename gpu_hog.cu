#include "gpu_hog.hpp"

__constant__ const float gpu_boundary_x[NUM_SECTOR + 1] = {1.00000000, 0.939692616, 0.766044438, 0.499999970, 0.173648104,
															-0.173648298, -0.500000060, -0.766044617, -0.939692676, -1.00000000};
__constant__ const float gpu_boundary_y[NUM_SECTOR + 1] = {0.000000000, 0.342020154, 0.642787635, 0.866025448, 0.984807789,
															0.984807730, 0.866025388, 0.642787457, 0.342020005, -8.74227766e-008};

__constant__ int gpu_nearest[4] = {-1,-1,1,1};
__constant__ float gpu_w[8] = {0.625, 0.375, 0.875, 0.125, 
							0.875, 0.125 , 0.625 , 0.375};
__global__ void GetmapofHOG(uchar* in, int width, int height, int channel, float *map, int numFeatures, int stringSize){
	int k = 4;
	int i = blockIdx.y;
	int ii = threadIdx.y;
	int jj = threadIdx.x%k;
	int j = blockIdx.x*k+threadIdx.x/k;//x->j
	int dx = j*k+jj;//blockIdx.x*blockDim.x+threadIdx.x;//j * k + jj;
	int dy = i*k+ii;//i * k + ii;
	int alfa0,alfa1;
	int offset = (dx+dy*width)*channel;
	int dw = width*channel;
	float x2,y2,sqrt2;
	float x3,y3,sqrt3;
	float x,y,tmp_sqrt;


    if (dy > 0 && dy < height - 1 && dx > 0 && dx < width  - 1)
    {
		x = in[offset+3] - in[offset-3];y = in[offset+dw] - in[offset-dw];
		tmp_sqrt = sqrtf(x * x + y * y);
		x2 = in[offset+4] - in[offset-2];y2 = in[offset+1+dw] - in[offset+1-dw];
		sqrt2 = sqrtf(x2 * x2 + y2 * y2);
		x3 = in[offset+5] - in[offset-1];y3 = in[offset+2+dw] - in[offset+2-dw];
		sqrt3 = sqrtf(x3 * x3 + y3 * y3);

		if(sqrt2>tmp_sqrt){
			tmp_sqrt = sqrt2;
			x = x2;y = y2;
		}
		if(sqrt3>tmp_sqrt){
			tmp_sqrt = sqrt3;
			x = x3;y = y3;
		}

        float Gmax  = gpu_boundary_x[0] * x + gpu_boundary_y[0] * y;
        int Gmaxi = 0;
		float dotProd;
        for (int kk = 0; kk < NUM_SECTOR; kk++)
        {
            dotProd = gpu_boundary_x[kk] * x + gpu_boundary_y[kk] * y;
            if (dotProd > Gmax)
            {
                Gmax  = dotProd;
                Gmaxi = kk;
            }
            else
            {
                if (-dotProd > Gmax)
                {
                    Gmax  = -dotProd;
                    Gmaxi = kk + NUM_SECTOR;
                }
            }
        }
        alfa0 = Gmaxi % NUM_SECTOR;
        alfa1 = Gmaxi + NUM_SECTOR;
		float rd = tmp_sqrt;
		float *mapoffset = map+i * stringSize + j * numFeatures;
		int ns = gpu_nearest[ii] * stringSize;
		int nn = gpu_nearest[jj] * numFeatures;
		float tmp0 = rd * gpu_w[ii * 2] * gpu_w[jj * 2];
        atomicAdd(mapoffset + alfa0,tmp0);
		atomicAdd(mapoffset + alfa1,tmp0);
		int flagi = i + gpu_nearest[ii];
		int flagj = j + gpu_nearest[jj];
		if ((flagi >= 0) && (flagi <= gridDim.y - 1))
        {
			tmp0 = rd * gpu_w[ii * 2 + 1] * gpu_w[jj * 2 ];
			atomicAdd(mapoffset + ns + alfa0,tmp0);
			atomicAdd(mapoffset + ns + alfa1,tmp0);
		}
        if ((flagj >= 0) && (flagj <= width/4 - 1))
        {
			tmp0 = rd * gpu_w[ii * 2] * gpu_w[jj * 2 + 1];
			atomicAdd(mapoffset + nn + alfa0,tmp0);
			atomicAdd(mapoffset + nn + alfa1,tmp0);
		}
        if ((flagi >= 0) &&
            (flagi <= gridDim.y - 1) &&
            (flagj >= 0) &&
            (flagj <= width/4 - 1))
        {
			tmp0 = rd * gpu_w[ii * 2 + 1] * gpu_w[jj * 2 + 1];
			atomicAdd(mapoffset + ns + nn + alfa0,tmp0);
			atomicAdd(mapoffset + ns + nn + alfa1,tmp0);
		}
    }
}

__global__ void getpartOfNorm(float *partOfNorm, float *map, int sizeX){
    int p  = NUM_SECTOR;
	int jj = threadIdx.x%p;//for(ii = 0; ii < 2 * p; ii++)
	int i = blockIdx.y;//(i = 1; i <= sizeY; i++)
	int djj = threadIdx.x/p;
	int j = blockIdx.x*7+djj;//(j = 0; j < sizeX; j++)
	int pos1, pos2;
	__shared__ float val_vec[63];
	pos2 = i*sizeX+j;
	pos1 = pos2*3*p;
	float tmp = map[pos1 + jj];
	int readset = 9*djj;
	int tmpoffset = readset+jj;
	val_vec[tmpoffset] = tmp*tmp;
	__syncthreads();
	if(jj<4) val_vec[tmpoffset] += val_vec[tmpoffset+4]; 
	__syncthreads();
	if(jj<2) val_vec[tmpoffset] += val_vec[tmpoffset+2]; 
	__syncthreads();
	if(jj==2){
		float val = val_vec[readset] + val_vec[readset+1] + val_vec[readset+8];
		partOfNorm[pos2] = val;
	}

}


__global__ void PCANTFeatureMaps(float *partOfNorm, float *map, float *newData, int sizeX, int xp0){
	int jj = threadIdx.x%18;//for(ii = 0; ii < 2 * p; ii++)
	int i = blockIdx.y;//(i = 1; i <= sizeY; i++)
	int djj = threadIdx.x/18;
	int j = blockIdx.x*5+djj;//(j = 0; j < sizeX; j++)
	int i1 = i+1;
	int j1 = j+1;
	float valOfNorm;
	int pos01, pos2;
    int p  = NUM_SECTOR;
	__shared__ float val_vec[640];

	int p_partOfNorm = (i1    )*(sizeX + 2) + (j1    );
	float p00 = partOfNorm[p_partOfNorm];
	float p01 = partOfNorm[p_partOfNorm+1];
	float p0i = partOfNorm[p_partOfNorm-1];
	float p10 = partOfNorm[p_partOfNorm+sizeX + 2];
	float pi0 = partOfNorm[p_partOfNorm-sizeX - 2];
    pos01 = p_partOfNorm * xp0 + jj;
	float map1 = map[pos01    ] ;
	float map2 = map[pos01 + p] ;
	float nD0, nD4, nD1, nD6, nD2, nD8, nD3, nD10;
	float nDMax = 0.2f;float tmp;
	
    valOfNorm = sqrtf(p00+p01+p10+partOfNorm[p_partOfNorm+sizeX+3]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD0 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD4 = tmp>nDMax?nDMax:tmp;
	int tmpoffset = djj*32+jj;
	val_vec[tmpoffset] = nD4;

    valOfNorm = sqrtf(p00+p01+pi0+partOfNorm[p_partOfNorm-sizeX-1]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD1 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD6 = tmp>nDMax?nDMax:tmp;
	val_vec[160+tmpoffset] = nD6;

    valOfNorm = sqrtf(p00+p0i+p10+partOfNorm[p_partOfNorm+sizeX+1]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD2 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD8 = tmp>nDMax?nDMax:tmp;
	val_vec[320+tmpoffset] = nD8;

    valOfNorm = sqrtf(p00+p0i+pi0+partOfNorm[p_partOfNorm-sizeX-3]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD3 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD10 = tmp>nDMax?nDMax:tmp;
	val_vec[480+tmpoffset] = nD10;

	int pp2 = NUM_SECTOR * 3 + 4;
    int yp = 4;
    int xp = NUM_SECTOR;
	int k=0;
	float val = 0.0f;
    pos2 = ((i)*sizeX + j)*pp2;
	k = jj;
	val = nD4+nD6+nD8+nD10;
    newData[pos2 + k]= val * 0.5;
    if(jj< xp)
    {
		k = xp * 2 + jj;
		val = nD0+nD1+nD2+nD3;
        newData[pos2 + k]= val * 0.5;
    }
		k = xp * 3;
		__syncthreads();
		int readset = djj*32;
	for(int ii=0;ii<yp;ii++){
		if(jj<9) val_vec[tmpoffset] += val_vec[tmpoffset+9]; 
		__syncthreads();
		if(jj<4) val_vec[tmpoffset] += val_vec[tmpoffset+4]; 
		__syncthreads();
		if(jj<2) val_vec[tmpoffset] += val_vec[tmpoffset+2]; 
		__syncthreads();
		if(jj==0){
			val = val_vec[readset] + val_vec[readset+1] + val_vec[readset+8];
			newData[pos2 + k]=val * 0.2357226;
		}
		k++;
		tmpoffset += 160;
		readset += 160;
	}
}

__global__ void MultiGetmapofHOGPitch(uchar* in0, int width, int height, int channel, 
								 float *map0, int numFeatures, int stringSize, int pitch_in, int pitch_map){
	int k = 4;
	int i = blockIdx.y;
	int ii = threadIdx.y;
	int jj = threadIdx.x%k;
	int j = blockIdx.x*k+threadIdx.x/k;//x->j
	int dx = j*k+jj;//blockIdx.x*blockDim.x+threadIdx.x;//j * k + jj;
	int dy = i*k+ii;//i * k + ii;
	int alfa0,alfa1;
	int offset = (dx+dy*width)*channel;
	int dw = width*channel;
	float x2,y2,sqrt2;
	float x3,y3,sqrt3;
	float x,y,tmp_sqrt;
	uchar *in = in0+blockIdx.z*pitch_in;
	float *map = map0+blockIdx.z*pitch_map;

    if (dy > 0 && dy < height - 1 && dx > 0 && dx < width  - 1)
    {
		x = in[offset+3] - in[offset-3];y = in[offset+dw] - in[offset-dw];
		tmp_sqrt = sqrtf(x * x + y * y);
		x2 = in[offset+4] - in[offset-2];y2 = in[offset+1+dw] - in[offset+1-dw];
		sqrt2 = sqrtf(x2 * x2 + y2 * y2);
		x3 = in[offset+5] - in[offset-1];y3 = in[offset+2+dw] - in[offset+2-dw];
		sqrt3 = sqrtf(x3 * x3 + y3 * y3);

		if(sqrt2>tmp_sqrt){
			tmp_sqrt = sqrt2;
			x = x2;y = y2;
		}
		if(sqrt3>tmp_sqrt){
			tmp_sqrt = sqrt3;
			x = x3;y = y3;
		}

        float Gmax  = gpu_boundary_x[0] * x + gpu_boundary_y[0] * y;
        int Gmaxi = 0;
		float dotProd;
        for (int kk = 0; kk < NUM_SECTOR; kk++)
        {
            dotProd = gpu_boundary_x[kk] * x + gpu_boundary_y[kk] * y;
            if (dotProd > Gmax)
            {
                Gmax  = dotProd;
                Gmaxi = kk;
            }
            else
            {
                if (-dotProd > Gmax)
                {
                    Gmax  = -dotProd;
                    Gmaxi = kk + NUM_SECTOR;
                }
            }
        }
        alfa0 = Gmaxi % NUM_SECTOR;
        alfa1 = Gmaxi + NUM_SECTOR;
		float rd = tmp_sqrt;
		float *mapoffset = map+i * stringSize + j * numFeatures;
		int ns = gpu_nearest[ii] * stringSize;
		int nn = gpu_nearest[jj] * numFeatures;
		float tmp0 = rd * gpu_w[ii * 2] * gpu_w[jj * 2];
        atomicAdd(mapoffset + alfa0,tmp0);
		atomicAdd(mapoffset + alfa1,tmp0);
		int flagi = i + gpu_nearest[ii];
		int flagj = j + gpu_nearest[jj];
		if ((flagi >= 0) && (flagi <= gridDim.y - 1))
        {
			tmp0 = rd * gpu_w[ii * 2 + 1] * gpu_w[jj * 2 ];
			atomicAdd(mapoffset + ns + alfa0,tmp0);
			atomicAdd(mapoffset + ns + alfa1,tmp0);
		}
        if ((flagj >= 0) && (flagj <= width/4 - 1))
        {
			tmp0 = rd * gpu_w[ii * 2] * gpu_w[jj * 2 + 1];
			atomicAdd(mapoffset + nn + alfa0,tmp0);
			atomicAdd(mapoffset + nn + alfa1,tmp0);
		}
        if ((flagi >= 0) &&
            (flagi <= gridDim.y - 1) &&
            (flagj >= 0) &&
            (flagj <= width/4 - 1))
        {
			tmp0 = rd * gpu_w[ii * 2 + 1] * gpu_w[jj * 2 + 1];
			atomicAdd(mapoffset + ns + nn + alfa0,tmp0);
			atomicAdd(mapoffset + ns + nn + alfa1,tmp0);
		}
    }
}

__global__ void MultigetpartOfNormPitch(float *partOfNorm, float *map, int sizeX, 
										int pitch_partOfNorm, int pitch_map){
    int p  = NUM_SECTOR;
	int jj = threadIdx.x%p;//for(ii = 0; ii < 2 * p; ii++)
	int i = blockIdx.y;//(i = 1; i <= sizeY; i++)
	int djj = threadIdx.x/p;
	int j = blockIdx.x*7+djj;//(j = 0; j < sizeX; j++)
	int pos1, pos2;
	__shared__ float val_vec[63];
	pos2 = i*sizeX+j + pitch_partOfNorm*blockIdx.z;
	pos1 = (i*sizeX+j)*3*p + pitch_map*blockIdx.z;
	float tmp = map[pos1 + jj];
	int readset = 9*djj;
	int tmpoffset = readset+jj;
	val_vec[tmpoffset] = tmp*tmp;
	__syncthreads();
	if(jj<4) val_vec[tmpoffset] += val_vec[tmpoffset+4]; 
	__syncthreads();
	if(jj<2) val_vec[tmpoffset] += val_vec[tmpoffset+2]; 
	__syncthreads();
	if(jj==2){
		float val = val_vec[readset] + val_vec[readset+1] + val_vec[readset+8];
		partOfNorm[pos2] = val;
	}

}


__global__ void MultiPCANTFeatureMapsPitch(float *partOfNorm, float *map, float *newData, int sizeX, int xp0,
										int pitch_partOfNorm, int pitch_map){
	int jj = threadIdx.x%18;//for(ii = 0; ii < 2 * p; ii++)
	int i = blockIdx.y;//(i = 1; i <= sizeY; i++)
	int djj = threadIdx.x/18;
	int j = blockIdx.x*5+djj;//(j = 0; j < sizeX; j++)
	int i1 = i+1;
	int j1 = j+1;
	float valOfNorm;
	int pos01, pos2;
    int p  = NUM_SECTOR;
	__shared__ float val_vec[640];

	int p_partOfNorm = (i1    )*(sizeX + 2) + (j1    )+ pitch_partOfNorm*blockIdx.z;
	float p00 = partOfNorm[p_partOfNorm];
	float p01 = partOfNorm[p_partOfNorm+1];
	float p0i = partOfNorm[p_partOfNorm-1];
	float p10 = partOfNorm[p_partOfNorm+sizeX + 2];
	float pi0 = partOfNorm[p_partOfNorm-sizeX - 2];
    pos01 = (i1*(sizeX + 2) + j1) * xp0 + jj + pitch_map*blockIdx.z;
	float map1 = map[pos01    ] ;
	float map2 = map[pos01 + p] ;
	float nD0, nD4, nD1, nD6, nD2, nD8, nD3, nD10;
	float nDMax = 0.2f;float tmp;
	
    valOfNorm = sqrtf(p00+p01+p10+partOfNorm[p_partOfNorm+sizeX+3]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD0 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD4 = tmp>nDMax?nDMax:tmp;
	int tmpoffset = djj*32+jj;
	val_vec[tmpoffset] = nD4;

    valOfNorm = sqrtf(p00+p01+pi0+partOfNorm[p_partOfNorm-sizeX-1]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD1 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD6 = tmp>nDMax?nDMax:tmp;
	val_vec[160+tmpoffset] = nD6;

    valOfNorm = sqrtf(p00+p0i+p10+partOfNorm[p_partOfNorm+sizeX+1]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD2 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD8 = tmp>nDMax?nDMax:tmp;
	val_vec[320+tmpoffset] = nD8;

    valOfNorm = sqrtf(p00+p0i+pi0+partOfNorm[p_partOfNorm-sizeX-3]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD3 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD10 = tmp>nDMax?nDMax:tmp;
	val_vec[480+tmpoffset] = nD10;

	int pp2 = NUM_SECTOR * 3 + 4;
    int yp = 4;
    int xp = NUM_SECTOR;
	int k=0;
	float val = 0.0f;
    pos2 = ((i)*sizeX + j)*pp2+ pitch_partOfNorm*blockIdx.z*12*xp;
	k = jj;
	val = nD4+nD6+nD8+nD10;
    newData[pos2 + k]= val * 0.5;
    if(jj< xp)
    {
		k = xp * 2 + jj;
		val = nD0+nD1+nD2+nD3;
        newData[pos2 + k]= val * 0.5;
    }
		k = xp * 3;
		__syncthreads();
		int readset = djj*32;
	for(int ii=0;ii<yp;ii++){
		if(jj<9) val_vec[tmpoffset] += val_vec[tmpoffset+9]; 
		__syncthreads();
		if(jj<4) val_vec[tmpoffset] += val_vec[tmpoffset+4]; 
		__syncthreads();
		if(jj<2) val_vec[tmpoffset] += val_vec[tmpoffset+2]; 
		__syncthreads();
		if(jj==0){
			val = val_vec[readset] + val_vec[readset+1] + val_vec[readset+8];
			newData[pos2 + k]=val * 0.2357226;
		}
		k++;
		tmpoffset += 160;
		readset += 160;
	}
}


__global__ void MultiGetmapofHOG(uchar* in0, int width, int height, int channel, 
								 float *map0, int numFeatures, int stringSize, int page_partOfNorm){
	int k = 4;
	int i = blockIdx.y;
	int ii = threadIdx.y;
	int jj = threadIdx.x%k;
	int j = blockIdx.x*k+threadIdx.x/k;//x->j
	int dx = j*k+jj;//blockIdx.x*blockDim.x+threadIdx.x;//j * k + jj;
	int dy = i*k+ii;//i * k + ii;
	int alfa0,alfa1;
	int offset = (dx+dy*width)*channel;
	int dw = width*channel;
	float x2,y2,sqrt2;
	float x3,y3,sqrt3;
	float x,y,tmp_sqrt;
	uchar *in = in0+blockIdx.z*dw*height;
	float *map = map0+blockIdx.z*page_partOfNorm*31;

    if (dy > 0 && dy < height - 1 && dx > 0 && dx < width  - 1)
    {
		x = in[offset+3] - in[offset-3];y = in[offset+dw] - in[offset-dw];
		tmp_sqrt = sqrtf(x * x + y * y);
		x2 = in[offset+4] - in[offset-2];y2 = in[offset+1+dw] - in[offset+1-dw];
		sqrt2 = sqrtf(x2 * x2 + y2 * y2);
		x3 = in[offset+5] - in[offset-1];y3 = in[offset+2+dw] - in[offset+2-dw];
		sqrt3 = sqrtf(x3 * x3 + y3 * y3);

		if(sqrt2>tmp_sqrt){
			tmp_sqrt = sqrt2;
			x = x2;y = y2;
		}
		if(sqrt3>tmp_sqrt){
			tmp_sqrt = sqrt3;
			x = x3;y = y3;
		}

        float Gmax  = gpu_boundary_x[0] * x + gpu_boundary_y[0] * y;
        int Gmaxi = 0;
		float dotProd;
        for (int kk = 0; kk < NUM_SECTOR; kk++)
        {
            dotProd = gpu_boundary_x[kk] * x + gpu_boundary_y[kk] * y;
            if (dotProd > Gmax)
            {
                Gmax  = dotProd;
                Gmaxi = kk;
            }
            else
            {
                if (-dotProd > Gmax)
                {
                    Gmax  = -dotProd;
                    Gmaxi = kk + NUM_SECTOR;
                }
            }
        }
        alfa0 = Gmaxi % NUM_SECTOR;
        alfa1 = Gmaxi + NUM_SECTOR;
		float rd = tmp_sqrt;
		float *mapoffset = map+i * stringSize + j * numFeatures;
		int ns = gpu_nearest[ii] * stringSize;
		int nn = gpu_nearest[jj] * numFeatures;
		float tmp0 = rd * gpu_w[ii * 2] * gpu_w[jj * 2];
        atomicAdd(mapoffset + alfa0,tmp0);
		atomicAdd(mapoffset + alfa1,tmp0);
		int flagi = i + gpu_nearest[ii];
		int flagj = j + gpu_nearest[jj];
		if ((flagi >= 0) && (flagi <= gridDim.y - 1))
        {
			tmp0 = rd * gpu_w[ii * 2 + 1] * gpu_w[jj * 2 ];
			atomicAdd(mapoffset + ns + alfa0,tmp0);
			atomicAdd(mapoffset + ns + alfa1,tmp0);
		}
        if ((flagj >= 0) && (flagj <= width/4 - 1))
        {
			tmp0 = rd * gpu_w[ii * 2] * gpu_w[jj * 2 + 1];
			atomicAdd(mapoffset + nn + alfa0,tmp0);
			atomicAdd(mapoffset + nn + alfa1,tmp0);
		}
        if ((flagi >= 0) &&
            (flagi <= gridDim.y - 1) &&
            (flagj >= 0) &&
            (flagj <= width/4 - 1))
        {
			tmp0 = rd * gpu_w[ii * 2 + 1] * gpu_w[jj * 2 + 1];
			atomicAdd(mapoffset + ns + nn + alfa0,tmp0);
			atomicAdd(mapoffset + ns + nn + alfa1,tmp0);
		}
    }
}

__global__ void MultigetpartOfNorm(float *partOfNorm, float *map, int sizeX, int page_partOfNorm){
    int p  = NUM_SECTOR;
	int jj = threadIdx.x%p;//for(ii = 0; ii < 2 * p; ii++)
	int i = blockIdx.y;//(i = 1; i <= sizeY; i++)
	int djj = threadIdx.x/p;
	int j = blockIdx.x*7+djj;//(j = 0; j < sizeX; j++)
	int pos1, pos2;
	__shared__ float val_vec[63];
	pos2 = i*sizeX+j + page_partOfNorm*blockIdx.z;
	pos1 = (i*sizeX+j)*3*p + page_partOfNorm*blockIdx.z*(3*p+4);
	float tmp = map[pos1 + jj];
	int readset = 9*djj;
	int tmpoffset = readset+jj;
	val_vec[tmpoffset] = tmp*tmp;
	__syncthreads();
	if(jj<4) val_vec[tmpoffset] += val_vec[tmpoffset+4]; 
	__syncthreads();
	if(jj<2) val_vec[tmpoffset] += val_vec[tmpoffset+2]; 
	__syncthreads();
	if(jj==2){
		float val = val_vec[readset] + val_vec[readset+1] + val_vec[readset+8];
		partOfNorm[pos2] = val;
	}

}


__global__ void MultiPCANTFeatureMaps(float *partOfNorm, float *map, float *newData, int sizeX, int xp0, int page_partOfNorm){
	int jj = threadIdx.x%18;//for(ii = 0; ii < 2 * p; ii++)
	int i = blockIdx.y;//(i = 1; i <= sizeY; i++)
	int djj = threadIdx.x/18;
	int j = blockIdx.x*5+djj;//(j = 0; j < sizeX; j++)
	int i1 = i+1;
	int j1 = j+1;
	float valOfNorm;
	int pos01, pos2;
    int p  = NUM_SECTOR;
	__shared__ float val_vec[640];

	int p_partOfNorm = (i1    )*(sizeX + 2) + (j1    )+ page_partOfNorm*blockIdx.z;
	float p00 = partOfNorm[p_partOfNorm];
	float p01 = partOfNorm[p_partOfNorm+1];
	float p0i = partOfNorm[p_partOfNorm-1];
	float p10 = partOfNorm[p_partOfNorm+sizeX + 2];
	float pi0 = partOfNorm[p_partOfNorm-sizeX - 2];
    pos01 = (i1*(sizeX + 2) + j1) * xp0 + jj + page_partOfNorm*blockIdx.z*(3*p+4);
	float map1 = map[pos01    ] ;
	float map2 = map[pos01 + p] ;
	float nD0, nD4, nD1, nD6, nD2, nD8, nD3, nD10;
	float nDMax = 0.2f;float tmp;
	
    valOfNorm = sqrtf(p00+p01+p10+partOfNorm[p_partOfNorm+sizeX+3]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD0 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD4 = tmp>nDMax?nDMax:tmp;
	int tmpoffset = djj*32+jj;
	val_vec[tmpoffset] = nD4;

    valOfNorm = sqrtf(p00+p01+pi0+partOfNorm[p_partOfNorm-sizeX-1]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD1 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD6 = tmp>nDMax?nDMax:tmp;
	val_vec[160+tmpoffset] = nD6;

    valOfNorm = sqrtf(p00+p0i+p10+partOfNorm[p_partOfNorm+sizeX+1]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD2 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD8 = tmp>nDMax?nDMax:tmp;
	val_vec[320+tmpoffset] = nD8;

    valOfNorm = sqrtf(p00+p0i+pi0+partOfNorm[p_partOfNorm-sizeX-3]) + FLT_EPSILON;
	if(jj<NUM_SECTOR) {tmp = fdividef(map1, valOfNorm);nD3 = tmp>nDMax?nDMax:tmp;}
    tmp = fdividef(map2, valOfNorm);nD10 = tmp>nDMax?nDMax:tmp;
	val_vec[480+tmpoffset] = nD10;

	int pp2 = NUM_SECTOR * 3 + 4;
    int yp = 4;
    int xp = NUM_SECTOR;
	int k=0;
	float val = 0.0f;
    pos2 = ((i)*sizeX + j)*pp2+ page_partOfNorm*blockIdx.z*12*xp;
	k = jj;
	val = nD4+nD6+nD8+nD10;
    newData[pos2 + k]= val * 0.5;
    if(jj< xp)
    {
		k = xp * 2 + jj;
		val = nD0+nD1+nD2+nD3;
        newData[pos2 + k]= val * 0.5;
    }
		k = xp * 3;
		__syncthreads();
		int readset = djj*32;
	for(int ii=0;ii<yp;ii++){
		if(jj<9) val_vec[tmpoffset] += val_vec[tmpoffset+9]; 
		__syncthreads();
		if(jj<4) val_vec[tmpoffset] += val_vec[tmpoffset+4]; 
		__syncthreads();
		if(jj<2) val_vec[tmpoffset] += val_vec[tmpoffset+2]; 
		__syncthreads();
		if(jj==0){
			val = val_vec[readset] + val_vec[readset+1] + val_vec[readset+8];
			newData[pos2 + k]=val * 0.2357226;
		}
		k++;
		tmpoffset += 160;
		readset += 160;
	}
}

int getPcaHogFeatureMaps(const IplImage* image, const int k, CvLSVMFeatureMapCaskade **map, int channels)
{
    int sizeX, sizeY;
    int p, p0, pp, stringSize;
    int height, width;
    int pos, xp;
	uchar *in;
    float * partOfNorm; // norm of C(i, j)
	float *d_map;
    float * finData;
    height = image->height;
    width  = image->width ;
    sizeX = (int)width  / k;
    sizeY = (int)height / k;
    p0     = 3 * NUM_SECTOR;
    stringSize = sizeX * p0;
    p  = NUM_SECTOR;
    xp = NUM_SECTOR * 3;
    pp    = NUM_SECTOR * 3 + 4;
		

	dim3 block1((sizeX+3)/4, sizeY);
	dim3 thread1(16, 4);
	dim3 block2((sizeX+2)/5, sizeY-2);
	dim3 thread2(10*NUM_SECTOR, 1);
	dim3 block3((sizeX+6)/7, sizeY);
	dim3 thread3(63, 1);
	cudaMalloc((void**)&in, height * width * channels * sizeof(uchar));
	cudaMalloc((void**)&partOfNorm, sizeof (float) * (sizeX * sizeY));
	cudaMalloc((void**)&d_map, sizeof (float) * (sizeX * sizeY  * p0));
	cudaMalloc((void**)&finData, sizeof (float) * ((sizeX-2)* (sizeY-2)  * NUM_SECTOR * 12));

	cudaMemcpy(in, image->imageData, height * width * channels * sizeof(uchar), cudaMemcpyHostToDevice);

	GetmapofHOG<<< block1, thread1 >>>(in, width, height, channels, d_map, p0, stringSize);
    cudaDeviceSynchronize();
	getpartOfNorm<<< block3, thread3 >>>(partOfNorm, d_map, sizeX);

	PCANTFeatureMaps<<< block2, thread2 >>>(partOfNorm, d_map, finData, sizeX-2, xp);
    cudaDeviceSynchronize();

    allocFeatureMapObject(map, sizeX-2, sizeY-2, pp);
	cudaMemcpy((*map)->map, finData, sizeof (float) * ((sizeX-2)* (sizeY-2)* pp), cudaMemcpyDeviceToHost);

	
    cudaFree(in);
    cudaFree(d_map);
    cudaFree(partOfNorm);
    cudaFree(finData);

    return LATENT_SVM_OK;
}
/*
PCAHOGMaps::PCAHOGMaps(cv::Size _tmpl_sz){
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

}
using namespace KCFTracker
{
int KCFTracker::getMaps(const IplImage* image, const int k, CvLSVMFeatureMapCaskade **map, int channels){


	cudaMemcpy(in, image->imageData, height * width * channels * sizeof(uchar), cudaMemcpyHostToDevice);

	GetmapofHOG<<< block1, thread1 >>>(in, width, height, channels, d_map, p0, stringSize);
    cudaDeviceSynchronize();
	getpartOfNorm<<< block3, thread3 >>>(partOfNorm, d_map, sizeX);

	PCANTFeatureMaps<<< block2, thread2 >>>(partOfNorm, d_map, finData, sizeX-2, xp);
    cudaDeviceSynchronize();

    allocFeatureMapObject(map, sizeX-2, sizeY-2, pp);
	cudaMemcpy((*map)->map, finData, sizeof (float) * ((sizeX-2)* (sizeY-2)* pp), cudaMemcpyDeviceToHost);

    return LATENT_SVM_OK;

}

KCFTracker::~KCFTracker(){
    cudaFree(in);
    cudaFree(d_map);
    cudaFree(partOfNorm);
    cudaFree(finData);
}
}*/