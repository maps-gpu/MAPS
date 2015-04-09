#include "cuda_runtime.h"
//#include "cutil_inline.h"

//#include "cloth_sim_defs.h"

//#include "Vec3D.h"

#include "cuda_common.h"

template<>
bool notZero(const float4 &dat)
{
	if (dat.x != 0.0f || dat.y != 0.0f || dat.z != 0.0f)
		return true;
	else
		return false;
}


template<>
bool notZero(const uint2 &dat)
{
	return true;
}

template<>
bool notZero(const int &dat)
{
	return true;
}

template<>
bool notZero(const unsigned int &dat)
{
	return true;
}

template<>
float calcAbsDiff(const float &dat1, const float &dat2)
{
	return abs(dat1-dat2);
}


template<>
float calcAbsDiff(const unsigned int &dat1, const unsigned int &dat2)
{
	return (float)abs((float)(dat1-dat2));
}

template<>
float calcAbsDiff(const uint2 &dat1, const uint2 &dat2)
{
	return (float)abs((int)dat1.x-(int)dat2.x)+abs((int)dat1.y-(int)dat2.y);
}


template<>
float calcAbsDiff(const float4 &dat1, const float4 &dat2)
{
	return sqrtf(
		(dat1.x-dat2.x)*(dat1.x-dat2.x)+
		(dat1.y-dat2.y)*(dat1.y-dat2.y)+
		(dat1.z-dat2.z)*(dat1.z-dat2.z)
		);
}

//#include <cmath>

template<>
bool checkForInvalidValue(const float &dat)
{
	if (_isnan(dat))
		return true;

	return false;
}

template<>
bool checkForInvalidValue(const float4 &dat)
{
	if (_isnan(dat.x) || _isnan(dat.y) || _isnan(dat.z))
		return true;

	return false;
}

template<>
bool checkForInvalidValue(const unsigned int &dat)
{
	return false;
}

template<>
bool checkForInvalidValue(const int &dat)
{
	return false;
}

template<>
bool checkForInvalidValue(const uint2 &dat)
{
	return false;
}

template<>
float computeAvg(const float* dat, const unsigned int size)
{
	float sum=0.f;

	for (unsigned int i=0; i<size; i++)
	{
		sum += dat[i];
	}

	return sum/(float)size;
}

//template<>
//bool compareCpuCpuBuffers(const float *HostInputA, const float *HostInputB, const int size, const char* name, float toll)
//{	
//	float diffSum = 0.0f;
//	float diffMax = 0.0f;
//	int diffNum = 0;
//	float allSum = 0.0f;
//	float A_sum = 0.0f;
//	int agreements = 0;
//	int nnzB = 0, nnzA = 0;
//	//cuComplex a,b;
//
//	for(int i=0; i<size; i++)
//	{
//		A_sum += HostInputA[i];
//	}
//
//	float A_avg = A_sum/size;
//
//	for(int i=0; i<size; i++)
//	{
//		//allSum += HostInputA[i].x + HostInputA[i].y + HostInputB[i].x + HostInputB[i].y ;
//		if (checkForInvalidValue<float>(HostInputA[i]) || checkForInvalidValue<float>(HostInputB[i]))
//		{
//			printf("[%s] QNAN value !!!\n",name);
//			return false;
//		}
//
//		float tmpDiff = calcAbsDiff(HostInputA[i],HostInputB[i])/A_avg;//abs(HostInputA[i].x-HostInputB[i].x)+abs(HostInputA[i].y-HostInputB[i].y);
//		if (tmpDiff > toll)
//		{
//			diffNum++;
//
//			//if (diffNum ==1)
//			//	printDat(name,tmpDiff,HostInputA[i],HostInputB[i],i);
//
//			diffSum += tmpDiff;
//			if (tmpDiff > diffMax)
//			{
//				diffMax = tmpDiff;
//				//printDat(name,diffMax,HostInputA[i],HostInputB[i]);
//				//std::cout << name << ": newMax " << diffMax << " valA " << (HostInputA[i]) << " valB " << HostInputB[i] << std::endl;
//				//a = HostInputA[i];
//				// b = HostInputB[i];
//			}
//		}
//		else if (notZero(HostInputA[i]) || notZero(HostInputB[i]))
//		{
//			agreements++;
//		}	
//
//		if (notZero(HostInputB[i]))
//			nnzB++;
//
//		if (notZero(HostInputA[i]))
//			nnzA++;
//	}
//
//	float allAvg = allSum/float(4*size);
//
//	if (diffNum == 0)
//		printf("%s: no diffs !!! ag %d\n",name,agreements);
//	else
//		printf("%s: diff num %d size %d avg %f max %f allAvg %f, ag %d, nnzB %d, nnzA %d\n",name ,diffNum,
//		size,diffSum/((float)diffNum), diffMax,allAvg,agreements,nnzB,nnzA);//, abs(a.x) + abs(a.y) + abs(b.x) + abs(b.y) );                        
//
//	return true;
//} 
