#ifndef SDK_INVERT_PROC_AMP
#	define SDK_INVERT_PROC_AMP

#	include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#	include "PrGPU/KernelSupport/KernelMemory.h"
#	include "..\..\Headers\Math_Utils.h"

#	if GF_DEVICE_TARGET_DEVICE

		GF_KERNEL_FUNCTION(ProcAmp2Kernel,
			((const GF_PTR(float4))(inSrc))
			((GF_PTR(float4))(outDst)),
			((int)(inSrcPitch))
			((int)(inDstPitch))
			((int)(in16f))
			((unsigned int)(inWidth))
			((unsigned int)(inHeight))
			((bool)(clampToggle)),
			((uint2)(inXY)(KERNEL_XY)))
		{
			if (inXY.x < inWidth && inXY.y < inHeight)
			{
				// BGRA
				float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
				// BGRA to RGBA
				float4 inTexture = { pixel.z, pixel.y, pixel.x, pixel.w };

				float4 screenColor = { 0.0f, 0.0f, 0.0f, 0.0f};
				if (clampToggle) {
					screenColor = screenClamp(inTexture);
				}
				else {
					screenColor = screen(inTexture);
				}

				// RGB to BGR
				pixel = { screenColor.z, screenColor.y, screenColor.x, screenColor.w };
				// Output
				WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
			}
		}
#	endif

#	if __NVCC__

		void ProcAmp_CUDA (
			float const *src,
			float *dst,
			unsigned int srcPitch,
			unsigned int dstPitch,
			int	is16f,
			unsigned int width,
			unsigned int height,
			bool clampToggle
		)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );

			ProcAmp2Kernel <<< gridDim, blockDim, 0 >>> (
				(float4 const*) src, (float4*) dst, srcPitch, dstPitch, is16f, width, height,
				clampToggle
			);

			cudaDeviceSynchronize();
		}

#	endif //GF_DEVICE_TARGET_HOST

#endif