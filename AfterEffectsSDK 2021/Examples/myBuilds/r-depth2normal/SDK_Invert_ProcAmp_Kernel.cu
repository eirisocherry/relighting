#ifndef SDK_INVERT_PROC_AMP
#	define SDK_INVERT_PROC_AMP

#	include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#	include "PrGPU/KernelSupport/KernelMemory.h"
#	include "..\..\Headers\Math_Utils.h"
#	include "Structures.h"

#	if GF_DEVICE_TARGET_DEVICE

		GF_KERNEL_FUNCTION(
			firstPassCUDAKernel,
			((const GF_PTR(float4))(src1))
			((GF_PTR(float4))(src3)),

			((int)(pitch1))
			((unsigned int)(width1))
			((unsigned int)(height1))
			((int)(in16f1))

			((int)(pitch3))
			((unsigned int)(width3))
			((unsigned int)(height3))
			((int)(in16f3))

			((const GF_PTR(InvertProcAmpParams))(params)),

			((uint2)(inXY)(KERNEL_XY)))
		{
			if (inXY.x < width1 && inXY.y < height1)
			{

				// Camera

				float3 camVx = { params->camVx1, params->camVx2, params->camVx3 };
				float3 camVy = { params->camVy1, params->camVy2, params->camVy3 };
				float3 camVz = { params->camVz1, params->camVz2, params->camVz3 };
				float3 camPos = { params->camPos1, params->camPos2, params->camPos3 };
				float downsample = params->cameraWidth / (float)width1;
				
				// Normal

				float3 normal = { 0.0f,  0.0f,  0.0f };

				if (params->improve) {
					normal = getNormalImproved(
						src1, inXY, pitch1, width1, height1, in16f1, params->depthBlackIsNear, params->depthFar,
						false, camVx, camVy, camVz, camPos, params->cameraZoom, downsample
					);
				}
				else {
					normal = getNormal(
						src1, inXY, pitch1, width1, height1, in16f1, params->depthBlackIsNear, params->depthFar,
						false, camVx, camVy, camVz, camPos, params->cameraZoom, downsample
					);
				}

				if (params->normalize) {
					normal = addf3(mulf3(normal, 0.5f), { 0.5f, 0.5f, 0.5f });
				}

				// Output

				WriteFloat4({ normal.z, normal.y, normal.x, 1.0f }, src3, inXY.y * pitch3 + inXY.x, !!in16f3);

			}
		}


		GF_KERNEL_FUNCTION(
			secondPassCUDAKernel,
			((const GF_PTR(float4))(src1))
			((const GF_PTR(float4))(src2))
			((GF_PTR(float4))(src3)),

			((int)(pitch1))
			((unsigned int)(width1))
			((unsigned int)(height1))
			((int)(in16f1))

			((int)(pitch2))
			((unsigned int)(width2))
			((unsigned int)(height2))
			((int)(in16f2))

			((int)(pitch3))
			((unsigned int)(width3))
			((unsigned int)(height3))
			((int)(in16f3))

			((const GF_PTR(InvertProcAmpParams))(params)),

			((uint2)(inXY)(KERNEL_XY)))
		{
			if (inXY.x < width1 && inXY.y < height1)
			{

				// Smooth

				bool horizontalBlur = true;
				float3 smoothResult = smoothNormal(
					inXY,
					src1, pitch1, width1, height1, in16f1, // normal
					src2, pitch2, width2, height2, in16f2, // depth
					params->depthBlackIsNear, 1.0f,
					horizontalBlur, params->radius, params->normalThreshold, params->depthWeight // smooth normal
				);

				// Output

				WriteFloat4({ smoothResult.z, smoothResult.y, smoothResult.x, 1.0f }, src3, inXY.y * pitch3 + inXY.x, !!in16f3);

			}
		}


		GF_KERNEL_FUNCTION(
			thirdPassCUDAKernel,
			((const GF_PTR(float4))(src1))
			((const GF_PTR(float4))(src2))
			((GF_PTR(float4))(src3)),

			((int)(pitch1))
			((unsigned int)(width1))
			((unsigned int)(height1))
			((int)(in16f1))

			((int)(pitch2))
			((unsigned int)(width2))
			((unsigned int)(height2))
			((int)(in16f2))

			((int)(pitch3))
			((unsigned int)(width3))
			((unsigned int)(height3))
			((int)(in16f3))

			((const GF_PTR(InvertProcAmpParams))(params)),

			((uint2)(inXY)(KERNEL_XY)))
		{
			if (inXY.x < width1 && inXY.y < height1)
			{

				// Smooth
				bool horizontalBlur = false;
				float3 smoothResult = smoothNormal(
					inXY,
					src1, pitch1, width1, height1, in16f1, // normal
					src2, pitch2, width2, height2, in16f2, // depth
					params->depthBlackIsNear, 1.0f,
					horizontalBlur, params->radius, params->normalThreshold, params->depthWeight // smooth normal
				);

				// Output
				WriteFloat4({ smoothResult.z, smoothResult.y, smoothResult.x, 1.0f }, src3, inXY.y * pitch3 + inXY.x, !!in16f3);

			}
		}


#	endif

#	if __NVCC__

		void firstPassCUDA(
			float const* src1,
			float* src3,

			unsigned int pitch1,
			unsigned int width1,
			unsigned int height1,
			int	is16f1,

			unsigned int pitch3,
			unsigned int width3,
			unsigned int height3,
			int	is16f3,

			InvertProcAmpParams* d_infoP
		)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width1 + blockDim.x - 1)/ blockDim.x, (height1 + blockDim.y - 1) / blockDim.y, 1 );

			firstPassCUDAKernel <<< gridDim, blockDim, 0 >>> (
				(float4 const*)src1,
				(float4*)src3,

				pitch1,
				width1,
				height1,
				is16f1,

				pitch3,
				width3,
				height3,
				is16f3,

				d_infoP
			);

			cudaDeviceSynchronize();
		}


		void secondPassCUDA(
			float const* src1,
			float const* src2,
			float* src3,

			unsigned int pitch1,
			unsigned int width1,
			unsigned int height1,
			int	is16f1,

			unsigned int pitch2,
			unsigned int width2,
			unsigned int height2,
			int	is16f2,

			unsigned int pitch3,
			unsigned int width3,
			unsigned int height3,
			int	is16f3,

			InvertProcAmpParams* d_infoP
		)
		{
			dim3 blockDim(16, 16, 1);
			dim3 gridDim((width1 + blockDim.x - 1) / blockDim.x, (height1 + blockDim.y - 1) / blockDim.y, 1);

			secondPassCUDAKernel << < gridDim, blockDim, 0 >> > (
				(float4 const*)src1,
				(float4 const*)src2,
				(float4*)src3,

				pitch1,
				width1,
				height1,
				is16f1,

				pitch2,
				width2,
				height2,
				is16f2,

				pitch3,
				width3,
				height3,
				is16f3,

				d_infoP
				);

			cudaDeviceSynchronize();
		}

		void thirdPassCUDA(
			float const* src1,
			float const* src2,
			float* src3,

			unsigned int pitch1,
			unsigned int width1,
			unsigned int height1,
			int	is16f1,

			unsigned int pitch2,
			unsigned int width2,
			unsigned int height2,
			int	is16f2,

			unsigned int pitch3,
			unsigned int width3,
			unsigned int height3,
			int	is16f3,

			InvertProcAmpParams* d_infoP
		)
		{
			dim3 blockDim(16, 16, 1);
			dim3 gridDim((width1 + blockDim.x - 1) / blockDim.x, (height1 + blockDim.y - 1) / blockDim.y, 1);

			thirdPassCUDAKernel << < gridDim, blockDim, 0 >> > (
				(float4 const*)src1,
				(float4 const*)src2,
				(float4*)src3,

				pitch1,
				width1,
				height1,
				is16f1,

				pitch2,
				width2,
				height2,
				is16f2,

				pitch3,
				width3,
				height3,
				is16f3,

				d_infoP
			);

			cudaDeviceSynchronize();
		}

#	endif //GF_DEVICE_TARGET_HOST

#endif