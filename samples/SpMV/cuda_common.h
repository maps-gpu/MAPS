#pragma once

#include <vector>

//#include <vector_types.h>

template<typename T >
void copyBufferToHost(const T *d_inputBuf, int inputBufSize, T **h_Buf)
{
	MAPS_CUDA_CHECK(cudaMemcpy(*h_Buf, d_inputBuf, inputBufSize * sizeof(T), cudaMemcpyDeviceToHost));
} 

template<typename T >
void copyBufferToDevice( T **d_inputBuf, int inputBufSize,const T *h_Buf)
{
	MAPS_CUDA_CHECK(cudaMemcpy(*d_inputBuf, h_Buf, inputBufSize * sizeof(T), cudaMemcpyHostToDevice));
} 

template<typename T >
void initHostAndCopyBufferToHost(const T *d_inputBuf, int inputBufSize, T **h_Buf)
{
	*h_Buf = new T[inputBufSize];
	copyBufferToHost(d_inputBuf, inputBufSize, h_Buf);
}

template<typename T >
bool notZero(const T &dat)
{
	return dat != 0;
}

template<>
bool notZero(const uint2 &dat);

template<>
bool notZero(const int &dat);

template<>
bool notZero(const unsigned int &dat);

template<typename T >
float calcAbsDiff(const T &dat1, const T &dat2)
{
	return (float)abs((float)dat1-(float)dat2);
}

template<typename T >
bool checkForInvalidValue(const T &dat)
{
	printf("warning ! using default implementation ! spcilazation needed !!!\n");
	return true;
}

template<typename T>
float computeAvg(const T* dat, const unsigned int size)
{
	printf("No method to compute avg val \n");
	return 1.f;
}

template<>
float computeAvg(const float* dat, const unsigned int size);

template<typename T >
bool compareCpuCpuBuffers(const T *HostInputA, const T *HostInputB, const int size, const char* name, float toll=0.00001, bool compare_to_avg = false)
{	
	float diffSum = 0.0f;
	float diffMax = 0.0f;
	int diffNum = 0;
	float allSum = 0.0f;
	int agreements = 0;
	int nnzB = 0, nnzA = 0;
	//cuComplex a,b;

	float A_avg = 1.0f;

	if (compare_to_avg)
	{
		A_avg = abs(computeAvg(HostInputA,size));
		printf("avg val of A is %f\n",A_avg);
		float B_avg = abs(computeAvg(HostInputB,size));
		printf("avg val of B is %f\n",B_avg);
	}

	for(int i=0; i<size; i++)
	{
		//allSum += HostInputA[i].x + HostInputA[i].y + HostInputB[i].x + HostInputB[i].y ;
		if (checkForInvalidValue(HostInputA[i]) || checkForInvalidValue(HostInputB[i]))
		{
			printf("[%s] QNAN value !!!\n",name);
			return false;
		}

		float tmpDiff = calcAbsDiff(HostInputA[i],HostInputB[i])/A_avg;//abs(HostInputA[i].x-HostInputB[i].x)+abs(HostInputA[i].y-HostInputB[i].y);
		if (tmpDiff > toll)
		{
			diffNum++;

			if (diffNum ==1)
				printDat(name,tmpDiff,HostInputA[i],HostInputB[i],i);

			diffSum += tmpDiff;
			if (tmpDiff > diffMax)
			{
				diffMax = tmpDiff;
				//printDat(name,diffMax,HostInputA[i],HostInputB[i]);
				//std::cout << name << ": newMax " << diffMax << " valA " << (HostInputA[i]) << " valB " << HostInputB[i] << std::endl;
				//a = HostInputA[i];
				// b = HostInputB[i];
			}
		}
		else if (notZero<T>(HostInputA[i]) || notZero<T>(HostInputB[i]))
		{
			agreements++;
		}	

		if (notZero(HostInputB[i]))
			nnzB++;

		if (notZero(HostInputA[i]))
			nnzA++;
	}

	float allAvg = allSum/float(4*size);

	if (diffNum == 0)
		printf("%s: no diffs !!! ag %d\n",name,agreements);
	else
		printf("%s: diff num %d size %d avg %f max %f allAvg %f, ag %d, nnzB %d, nnzA %d\n",name ,diffNum,
		size,diffSum/((float)diffNum), diffMax,allAvg,agreements,nnzB,nnzA);//, abs(a.x) + abs(a.y) + abs(b.x) + abs(b.y) );                        

	return true;
} 

//template<>
//bool compareCpuCpuBuffers(const float *HostInputA, const float *HostInputB, const int size, const char* name, float toll);

template<typename T >
void printDat(const char* name , const float diff, const T &dat1, const T &dat2, int index)
{
	printf("[%s] no printdat for this type, index %d diff %f\n",name,index,diff);
}

template<typename T>
bool compareGpuCpuVecs(T *d_data, T *h_data, const unsigned int size, const char* name, float toll=0.00001, bool compare_to_avg = false)
{
	cudaThreadSynchronize();

	T *tmp_dat;

	initHostAndCopyBufferToHost<T>(d_data,size,&tmp_dat);

	compareCpuCpuBuffers<T>(h_data,tmp_dat,size,name,toll,compare_to_avg);

	delete[] tmp_dat;

	return true;
}

template<typename T>
bool compareGpuGpuVecs(T *d_data1, T *d_data2, const unsigned int size, const char* name, float toll=0.00001)
{
	cudaThreadSynchronize();

	T *tmp_dat1;

	initHostAndCopyBufferToHost<T>(d_data1,size,&tmp_dat1);

	T *tmp_dat2;

	initHostAndCopyBufferToHost<T>(d_data2,size,&tmp_dat2);

	compareCpuCpuBuffers<T>(tmp_dat1,tmp_dat2,size,name,toll);

	delete[] tmp_dat1;
	delete[] tmp_dat2;

	return true;
}

template<typename T>
bool compareCpuCpuStdVecs(std::vector<T> &vec1, std::vector<T> &vec2, const char* name, float toll=0.00001)
{
	if (vec1.size() != vec2.size())
	{
		printf("vector sizes do not matche !!!\n");
		return false;
	}

	return	compareCpuCpuBuffers<T>((T*)&(*vec1.begin()),(T*)&(*vec2.begin()),vec1.size(),name,toll);
}

template<>
bool notZero(const float4 &dat);

template<>
float calcAbsDiff(const float &dat1, const float &dat2);


template<>
float calcAbsDiff(const unsigned int &dat1, const unsigned int &dat2);

template<>
float calcAbsDiff(const uint2 &dat1, const uint2 &dat2);

template<>
float calcAbsDiff(const float4 &dat1, const float4 &dat2);

template<>
bool checkForInvalidValue(const float &dat);

template<>
bool checkForInvalidValue(const float4 &dat);

template<>
bool checkForInvalidValue(const unsigned int &dat);

template<>
bool checkForInvalidValue(const int &dat);

template<>
bool checkForInvalidValue(const uint2 &dat);

