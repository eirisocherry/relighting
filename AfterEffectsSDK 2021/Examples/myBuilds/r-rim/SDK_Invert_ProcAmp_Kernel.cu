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

				// Normal

				float3 normal = takeXYZf4(samplePixel(inSrc, inXY, inSrcPitch, inWidth, inHeight, in16f)); //normalized normal
				normal = subf3(mulf3(normal, 2.0f), { 1.0f, 1.0f, 1.0f }); //unnormalized normal

				// Draw

				float3 draw = { 0.0f, 0.0f, 0.0f };

				for (int i = 0; i < 5; i++) {

					bool rimToggle = params->rimToggle[i];
					if (!rimToggle) { continue; }

					float3 drawRim = rimLight(
						normal,
						{ params->rimLightPositionX[i], params->rimLightPositionY[i], params->rimLightPositionZ[i] },
						{ params->rimLightLookAtPositionX[i], params->rimLightLookAtPositionY[i], params->rimLightLookAtPositionZ[i] },
						params->rimStart[i],
						params->rimEnd[i],
						params->rimIntensity[i] * params->intensityMultiplier,
						params->rimSaturation[i] * params->saturationMultiplier,
						{ params->rimColorNearR[i], params->rimColorNearG[i], params->rimColorNearB[i] },
						params->rimColorFarToggle[i],
						{ params->rimColorFarR[i], params->rimColorFarG[i], params->rimColorFarB[i] },
						params->rimColorFalloff[i]
					);

					draw = {
						draw.x + drawRim.x,
						draw.y + drawRim.y,
						draw.z + drawRim.z
					};

				}

				// Output

				WriteFloat4({ draw.z,  draw.y,  draw.x, 1.0f }, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);

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