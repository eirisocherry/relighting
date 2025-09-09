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
				// Camera

				float3 camVx = { params->camVx1, params->camVx2, params->camVx3 };
				float3 camVy = { params->camVy1, params->camVy2, params->camVy3 };
				float3 camVz = { params->camVz1, params->camVz2, params->camVz3 };
				float3 camPos = { params->camPos1, params->camPos2, params->camPos3 };
				float downsample = { params->cameraWidth / (float)width1 };

				// Draw

				int samples = params->samples;
				if (samples < 1) { samples = 1; }
				float zRadius = params->zRadius;
				if (zRadius == 0.0f) { zRadius = 0.0001f; }

				float drawAmbientOcclusion = ambientOcclusion(
					// Ambient Occlusion Inputs
					params->intensity, params->threshold, params->xyRadius, zRadius, samples,

					// Coords
					inXY,

					// Depth
					src1, pitch1, width1, height1, in16f1, params->depthBlackIsNear, params->depthFar,

					// Camera
					camVx, camVy, camVz, camPos, params->cameraZoom, downsample,

					// Normal Pass
					src2, pitch2, width2, height2, in16f2
				);

				// Output
				WriteFloat4({ drawAmbientOcclusion, drawAmbientOcclusion, drawAmbientOcclusion, 1.0f }, src3, inXY.y * pitch3 + inXY.x, !!in16f3);

			}
		}
#	endif

#	if __NVCC__

		void firstPassCUDA(
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
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width1 + blockDim.x - 1)/ blockDim.x, (height1 + blockDim.y - 1) / blockDim.y, 1 );

			firstPassCUDAKernel <<< gridDim, blockDim, 0 >>> (
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