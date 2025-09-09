#ifndef SDK_INVERT_PROC_AMP
#	define SDK_INVERT_PROC_AMP

#	include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#	include "PrGPU/KernelSupport/KernelMemory.h"
#	include "..\..\Headers\Math_Utils.h"
#	include "Structures.h"

#	if GF_DEVICE_TARGET_DEVICE

		GF_KERNEL_FUNCTION(
			ProcAmp2Kernel,
			((const GF_PTR(float4))(inSrc))
			((GF_PTR(float4))(outDst)),
			((int)(inSrcPitch))
			((int)(inDstPitch))
			((int)(in16f))
			((unsigned int)(inWidth))
			((unsigned int)(inHeight))

			((const GF_PTR(InvertProcAmpParams))(params)),

			((uint2)(inXY)(KERNEL_XY)))
		{
			if (inXY.x < inWidth && inXY.y < inHeight)
			{

				// Camera

				float3 camVx = { params->camVx1, params->camVx2, params->camVx3 };
				float3 camVy = { params->camVy1, params->camVy2, params->camVy3 };
				float3 camVz = { params->camVz1, params->camVz2, params->camVz3 };
				float3 camPos = { params->camPos1, params->camPos2, params->camPos3 };
				float downsample = { params->cameraWidth / (float)inWidth };

				// World Position

				float3 worldPos = getPosition(
					inSrc, inXY, inSrcPitch, inWidth, inHeight, in16f, params->depthBlackIsNear, params->depthFar,
					false, camVx, camVy, camVz, camPos, params->cameraZoom, downsample
				);

				// Output

				WriteFloat4({ worldPos.z,  worldPos.y,  worldPos.x, 1.0f }, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);

			}
		}
#	endif

#	if __NVCC__

		void ProcAmp_CUDA (
			float const* src,
			float* dst,
			unsigned int srcPitch,
			unsigned int dstPitch,
			int	is16f,
			unsigned int width,
			unsigned int height,

			InvertProcAmpParams* d_infoP
		)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );

			ProcAmp2Kernel <<< gridDim, blockDim, 0 >>> (
				(float4 const*)src,
				(float4*)dst,
				srcPitch,
				dstPitch,
				is16f,
				width,
				height,

				d_infoP
			);

			cudaDeviceSynchronize();
		}

#	endif //GF_DEVICE_TARGET_HOST

#endif