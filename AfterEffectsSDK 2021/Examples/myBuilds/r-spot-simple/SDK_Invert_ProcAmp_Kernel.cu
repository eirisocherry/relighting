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

				// Inputs

				float3 curPos = getPosition(
					inSrc, inXY, inSrcPitch, inWidth, inHeight, in16f, params->depthBlackIsNear, params->depthFar,
					false, camVx, camVy, camVz, camPos, params->cameraZoom, downsample
				);

				// Draw

				float3 draw = { 0.0f, 0.0f, 0.0f };

				for (int i = 0; i < 10; i++) {

					bool lightToggle = params->lightToggle[i];
					if (!lightToggle) { continue; }

					float angleX = clamp(params->angleX[i] * params->angleXmultiplier / 2.0f, 0.0f, 90.0f);
					float angleY = clamp(params->angleY[i] * params->angleYmultiplier / 2.0f, 0.0f, 90.0f);

					float4 iesChosenPreset1 = { params->iesBrightness1[i], params->iesBrightness1[i], params->iesBrightness1[i], params->iesDistance1[i] };
					float4 iesChosenPreset2 = { params->iesBrightness2[i], params->iesBrightness2[i], params->iesBrightness2[i], params->iesDistance2[i] };
					float4 iesChosenPreset3 = { params->iesBrightness3[i], params->iesBrightness3[i], params->iesBrightness3[i], params->iesDistance3[i] };
					float4 iesChosenPreset4 = { params->iesBrightness4[i], params->iesBrightness4[i], params->iesBrightness4[i], params->iesDistance4[i] };
					float4 iesChosenPreset5 = { params->iesBrightness5[i], params->iesBrightness5[i], params->iesBrightness5[i], params->iesDistance5[i] };
					float4 iesChosenPreset6 = { params->iesBrightness6[i], params->iesBrightness6[i], params->iesBrightness6[i], params->iesDistance6[i] };

					if (params->ies[i] != 5) {
						iesChosenPreset1 = getIESpreset(params->ies[i], 0);
						iesChosenPreset2 = getIESpreset(params->ies[i], 1);
						iesChosenPreset3 = getIESpreset(params->ies[i], 2);
						iesChosenPreset4 = getIESpreset(params->ies[i], 3);
						iesChosenPreset5 = getIESpreset(params->ies[i], 4);
						iesChosenPreset6 = getIESpreset(params->ies[i], 5);
					}

					// Colors

					float3 drawSpotSimple = spot(
						curPos,
						{ params->lightPosX[i] * downsample, params->lightPosY[i] * downsample, params->lightPosZ[i] * downsample },
						{ params->lightVxX[i] * downsample, params->lightVxY[i] * downsample, params->lightVxZ[i] * downsample },
						{ params->lightVyX[i] * downsample, params->lightVyY[i] * downsample, params->lightVyZ[i] * downsample },
						{ params->lightVzX[i] * downsample, params->lightVzY[i] * downsample, params->lightVzZ[i] * downsample },
						params->invertToggle[i],
						params->length[i] * params->lengthMultiplier,
						{ angleX, angleY },
						params->curvature[i] * params->curvatureMultiplier,
						params->falloff[i] * params->falloffMultiplier,
						params->feather[i] * params->featherMultiplier,
						params->intensity[i] * params->intensityMultiplier,
						params->saturation[i] * params->saturationMultiplier,
						{ params->colorNearR[i], params->colorNearG[i], params->colorNearB[i] },
						params->colorFarToggle[i],
						{ params->colorFarR[i], params->colorFarG[i], params->colorFarB[i] },
						params->colorFalloff[i] * params->colorFalloffMultiplier,
						iesChosenPreset1, iesChosenPreset2, iesChosenPreset3, iesChosenPreset4, iesChosenPreset5, iesChosenPreset6
					);
					
					draw = {
						draw.x + drawSpotSimple.x,
						draw.y + drawSpotSimple.y,
						draw.z + drawSpotSimple.z
					};

				}

			

				// Output

				WriteFloat4({ draw.z, draw.y, draw.x, 1.0f }, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);

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