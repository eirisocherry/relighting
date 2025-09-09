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

				// Camera

				float3 camVx = { params->camVx1, params->camVx2, params->camVx3 };
				float3 camVy = { params->camVy1, params->camVy2, params->camVy3 };
				float3 camVz = { params->camVz1, params->camVz2, params->camVz3 };
				float3 camPos = { params->camPos1, params->camPos2, params->camPos3 };
				float downsample = { params->cameraWidth / (float)inWidth };

				// World Position

				float3 curPos = getPosition(
					inSrc, inXY, inSrcPitch, inWidth, inHeight, in16f, params->depthBlackIsNear, params->depthFar,
					false, camVx, camVy, camVz, camPos, params->cameraZoom, downsample
				);

				// Draw

				float3 draw = { 0.0f, 0.0f, 0.0f };

				for (int i = 0; i < 10; i++) {

					bool lightToggle = params->lightToggle[i];
					if (!lightToggle) { continue; }

					float3 pos1 = { params->posX1[i] * downsample, params->posY1[i] * downsample, params->posZ1[i] * downsample };
					float3 vX1 = { params->vXx1[i] * downsample, params->vXy1[i] * downsample, params->vXz1[i] * downsample };
					float3 vY1 = { params->vYx1[i] * downsample, params->vYy1[i] * downsample, params->vYz1[i] * downsample };
					float3 vZ1 = { params->vZx1[i] * downsample, params->vZy1[i] * downsample, params->vZz1[i] * downsample };
					float3 res1 = { params->resX1[i] * downsample, params->resY1[i] * downsample, params->resZ1[i] * downsample };
					float3 scale1 = { params->scaleX1[i] * downsample, params->scaleY1[i] * downsample, params->scaleZ1[i] * downsample };

					float3 pos2 = { params->posX2[i] * downsample, params->posY2[i] * downsample, params->posZ2[i] * downsample };
					float3 vX2 = { params->vXx2[i] * downsample, params->vXy2[i] * downsample, params->vXz2[i] * downsample };
					float3 vY2 = { params->vYx2[i] * downsample, params->vYy2[i] * downsample, params->vYz2[i] * downsample };
					float3 vZ2 = { params->vZx2[i] * downsample, params->vZy2[i] * downsample, params->vZz2[i] * downsample };
					float3 res2 = { params->resX2[i] * downsample, params->resY2[i] * downsample, params->resZ2[i] * downsample };
					float3 scale2 = { params->scaleX2[i] * downsample, params->scaleY2[i] * downsample, params->scaleZ2[i] * downsample };

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

					// Color

					float intensity = params->intensity[i] * params->intensityMultiplier;
					float saturation = params->saturation[i] * params->saturationMultiplier;
					float3 colorNear = { params->colorNearR[i], params->colorNearG[i], params->colorNearB[i] };
					bool colorFarToggle = params->colorFarToggle[i];
					float3 colorFar = { params->colorFarR[i], params->colorFarG[i], params->colorFarB[i] };
					float colorFalloff = params->colorFalloff[i] * params->colorFalloffMultiplier;

					float3 drawTemp = rect(
						curPos,
						// Light Start
						pos1, vX1, vY1, vZ1, res1, scale1,
						// Light End
						pos2, vX2, vY2, vZ2, res2, scale2,
						// Shape
						invertToggle, featherX, featherY, featherZ, featherNormalized, falloff,
						// Color
						intensity, saturation, colorNear, colorFarToggle, colorFar, colorFalloff
					);

					draw = {
						draw.x + drawTemp.x,
						draw.y + drawTemp.y,
						draw.z + drawTemp.z
					};

				}

				// Output

				WriteFloat4({ draw.z, draw.y, draw.x, 1.0f }, outDst, inXY.y* inDstPitch + inXY.x, !!in16f);

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