#ifndef SDK_INVERT_PROC_AMP
#	define SDK_INVERT_PROC_AMP

#	include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#	include "PrGPU/KernelSupport/KernelMemory.h"
#	include "..\..\Headers\Math_Utils.h"
#	include "Structures.h"


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

				float3 curPos = getPosition(uv, false, depth, camVx, camVy, camVz, camPos, params->cameraZoom, params->cameraWidth, params->cameraHeight);

				// Draw

				float3 draw = { 0.0f, 0.0f, 0.0f };

				for (int i = 0; i < 10; i++) {

					bool lightToggle = params->lightToggle[i];

					float3 pos1 = { params->posX1[i], params->posY1[i], params->posZ1[i] };
					float3 vX1 = { params->vXx1[i], params->vXy1[i], params->vXz1[i] };
					float3 vY1 = { params->vYx1[i], params->vYy1[i], params->vYz1[i] };
					float3 vZ1 = { params->vZx1[i], params->vZy1[i], params->vZz1[i] };
					float3 res1 = { params->resX1[i], params->resY1[i], params->resZ1[i] };
					float3 scale1 = { params->scaleX1[i], params->scaleY1[i], params->scaleZ1[i] };

					float3 pos2 = { params->posX2[i], params->posY2[i], params->posZ2[i] };
					float3 vX2 = { params->vXx2[i], params->vXy2[i], params->vXz2[i] };
					float3 vY2 = { params->vYx2[i], params->vYy2[i], params->vYz2[i] };
					float3 vZ2 = { params->vZx2[i], params->vZy2[i], params->vZz2[i] };
					float3 res2 = { params->resX2[i], params->resY2[i], params->resZ2[i] };
					float3 scale2 = { params->scaleX2[i], params->scaleY2[i], params->scaleZ2[i] };

					// Shape

					bool invertToggle = params->invertToggle[i];
					bool featherNormalized = params->featherNormalize[i];
					float2 featherX = { params->featherX1[i] * params->featherMultiplier, params->featherX2[i] * params->featherMultiplier };
					float2 featherY = { params->featherY1[i] * params->featherMultiplier, params->featherY2[i] * params->featherMultiplier };
					float2 featherZ = { params->featherZ1[i] * params->featherMultiplier, params->featherZ2[i] * params->featherMultiplier };
					if (featherNormalized) {
						featherX = { featherX.x / 2.0f, featherX.y / 2.0f };
						featherY = { featherY.x / 2.0f, featherY.y / 2.0f };
						featherZ = { featherZ.x / 2.0f, featherZ.y / 2.0f };
					}
					float falloff = params->falloff[i] * params->falloffMultiplier;

					// Colors

					float intensity = params->intensity[i] * params->intensityMultiplier;
					float saturation = params->saturation[i] * params->saturationMultiplier;
					float3 colorNear = { params->colorNearR[i] / 255.0f, params->colorNearG[i] / 255.0f, params->colorNearB[i] / 255.0f };
					bool colorFarToggle = params->colorFarToggle[i];
					float3 colorFar = { params->colorFarR[i] / 255.0f, params->colorFarG[i] / 255.0f, params->colorFarB[i] / 255.0f };
					float colorFalloff = params->colorFalloff[i] * params->colorFalloffMultiplier;

					float3 drawTemp = { 0.0f, 0.0f, 0.0f };
					if (lightToggle) {
						drawTemp = rect(curPos, pos1, vX1, vY1, vZ1, res1, scale1, pos2, vX2, vY2, vZ2, res2, scale2, invertToggle, featherX, featherY, featherZ, featherNormalized, falloff, intensity, saturation, colorNear, colorFarToggle, colorFar, colorFalloff);
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

			// Pixels

			float const *src,
			float *dst,
			unsigned int srcPitch,
			unsigned int dstPitch,
			int	is16f,
			unsigned int width,
			unsigned int height,
			
			// Parameters

			InvertProcAmpParams* d_infoP

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

				// Parameters

				d_infoP

			);

			cudaDeviceSynchronize();
		}

#	endif //GF_DEVICE_TARGET_HOST

#endif