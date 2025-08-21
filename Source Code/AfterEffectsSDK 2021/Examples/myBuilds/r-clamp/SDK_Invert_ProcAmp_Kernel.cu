#ifndef SDK_INVERT_PROC_AMP
#	define SDK_INVERT_PROC_AMP

#	include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#	include "PrGPU/KernelSupport/KernelMemory.h"
#	include "..\..\Headers\Math_Utils.h"

#	if GF_DEVICE_TARGET_DEVICE

		GF_KERNEL_FUNCTION(

			// Pixels

			ProcAmp2Kernel,
			((const GF_PTR(float4))(inSrc))
			((GF_PTR(float4))(outDst)),
			((int)(inSrcPitch))
			((int)(inDstPitch))
			((int)(in16f))
			((unsigned int)(inWidth))
			((unsigned int)(inHeight))

			// Checkboxes

			((bool)(keepAlphaToggle))
			((bool)(clampToggle)),

			// Output

			((uint2)(inXY)(KERNEL_XY))

		)
		{
			if (inXY.x < inWidth && inXY.y < inHeight)
			{
				
				// Input

				float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f); // BGRA
				float4 inColor = { pixel.z, pixel.y, pixel.x, pixel.w }; // BGRA to RGBA

				// Main

				float4 clampedColor = { 0.0f, 0.0f, 0.0f, 0.0f };

				if (keepAlphaToggle && !clampToggle) {
					clampedColor = inColor;
				}

				if (keepAlphaToggle && clampToggle) {
					clampedColor = keepAlpha(inColor);
					clampedColor = clampf4(clampedColor, 0.0f, 1.0f);
				}

				if (!keepAlphaToggle && clampToggle) {
					clampedColor = removeAlpha(inColor);
					clampedColor = clampf4(clampedColor, 0.0f, 1.0f);
				}

				if (!keepAlphaToggle && !clampToggle) {
					clampedColor = removeAlpha(inColor);
				}

				// Output
				
				pixel = { clampedColor.z, clampedColor.y, clampedColor.x, clampedColor.w }; // RGB to BGR
				WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
			}
		}
#	endif

#	if __NVCC__

		void ProcAmp_CUDA (

			// Pixels

			float const *src,
			float *dst,
			unsigned int srcPitch,
			unsigned int dstPitch,
			int	is16f,
			unsigned int width,
			unsigned int height,

			// Checkboxes

			bool keepAlphaToggle,
			bool clampToggle

		)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );

			ProcAmp2Kernel <<< gridDim, blockDim, 0 >>> (

				// Pixels

				(float4 const*) src,
				(float4*) dst,
				srcPitch,
				dstPitch,
				is16f,
				width,
				height,

				// Checkboxes

				keepAlphaToggle,
				clampToggle

			);

			cudaDeviceSynchronize();
		}

#	endif //GF_DEVICE_TARGET_HOST

#endif