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

				// Input

				float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f); // BGRA
				float4 inTexture = { pixel.z, pixel.y, pixel.x, pixel.w }; // BGRA to RGBA
				float2 uv = { (float)inXY.x / (float)inWidth,
							  (float)inXY.y / (float)inHeight };

				// Camera

				float4 camVx = { params->camVx1, params->camVx2, params->camVx3, params->camVx4 };
				float4 camVy = { params->camVy1, params->camVy2, params->camVy3, params->camVy4 };
				float4 camVz = { params->camVz1, params->camVz2, params->camVz3, params->camVz4 };
				float4 camPos = { params->camPos1, params->camPos2, params->camPos3, params->camPos4 };

				// Depth

				float depth = getDepth(inTexture, params->depthBlackIsNear, params->depthFar);

				// World Position

				float3 curPos = getPosition(uv, true, depth, camVx, camVy, camVz, camPos, params->cameraZoom, params->cameraWidth, params->cameraHeight);


				// Draw

				float3 draw = { 0.0f, 0.0f, 0.0f };

				for (int i = 0; i <= 9; i++) {

					// Light Settings

					bool lightToggle = params->lightToggle[i];
					float3 lightPos = { params->lightPosX[i], params->lightPosY[i], params->lightPosZ[i] };

					// ----- Shape

					bool invertToggle = params->invertToggle[i];
					float radius = params->radius[i] * params->radiusMultiplier;
					float falloff = params->falloff[i] * params->falloffMultiplier;

					// ----- Colors

					float intensity = params->intensity[i] * params->intensityMultiplier;
					float saturation = params->saturation[i] * params->saturationMultiplier;
					float colorFalloff = params->colorFalloff[i] * params->colorFalloffMultiplier;

					float3 colorNear = { params->colorNearR[i] / 255.0f, params->colorNearG[i] / 255.0f, params->colorNearB[i] / 255.0f };
					bool colorFarToggle = params->colorFarToggle[i];
					float3 colorFar = { params->colorFarR[i] / 255.0f, params->colorFarG[i] / 255.0f, params->colorFarB[i] / 255.0f };

					// Draw

					float3 drawTemp = { 0.0f, 0.0f, 0.0f };
					if (lightToggle) {
						drawTemp = point(curPos, lightPos, radius, falloff, intensity, saturation, colorNear, colorFarToggle, colorFar, colorFalloff, invertToggle);
					}

					draw = {
						draw.x + drawTemp.x,
						draw.y + drawTemp.y,
						draw.z + drawTemp.z
					};

				}

				// Output

				float4 outTexture = { draw.x, draw.y, draw.z, pixel.w };
				pixel = { outTexture.z, outTexture.y, outTexture.x, pixel.w }; // RGBA to BGRA
				WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);

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