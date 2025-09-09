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

				// Layers

				float3 worldPos = getPosition(
					src1, inXY, pitch1, width1, height1, in16f1, params->depthBlackIsNear, params->depthFar,
					false, camVx, camVy, camVz, camPos, params->cameraZoom, downsample
				);

				float3 normal = { 0.0f, 0.0f, 0.0f };
				if (params->normalExistToggle) {
					normal = takeXYZf4(samplePixel(src2, inXY, pitch2, width2, height2, in16f2)); //normalized normal
					normal = subf3(mulf3(normal, 2.0f), { 1.0f, 1.0f, 1.0f }); //unnormalized normal
				}

				// Draw

				float3 draw = { 0.0f, 0.0f, 0.0f };
				float shadowsMask = 1.0f;
				
				for (int i = 0; i < 10; i++) {

					// Light Settings

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

					float shadowSampleStep = fmaxf(params->shadowSampleStep[i], 0.2f);
					float3 shadowColor = { params->shadowColorR[i], params->shadowColorG[i], params->shadowColorB[i] };

					float3 drawSpot = { 0.0f, 0.0f, 0.0f };
					float shadows = 1.0f;
					float3 shadowsColored = { 0.0f, 0.0f, 0.0f };


					// Draw

					drawSpot = spotAdvanced(
						// Inputs
						params->renderMode, camPos, worldPos, normal,

						// Spot
						{ params->lightPosX[i] * downsample, params->lightPosY[i] * downsample, params->lightPosZ[i] * downsample },
						{ params->lightVxX[i] * downsample, params->lightVxY[i] * downsample, params->lightVxZ[i] * downsample },
						{ params->lightVyX[i] * downsample, params->lightVyY[i] * downsample, params->lightVyZ[i] * downsample },
						{ params->lightVzX[i] * downsample, params->lightVzY[i] * downsample, params->lightVzZ[i] * downsample },
						params->length[i] * params->lengthMultiplier,
						{ angleX, angleY },
						params->curvature[i] * params->curvatureMultiplier,
						params->falloff[i] * params->falloffMultiplier,
						params->feather[i] * params->featherMultiplier,
						params->ambientIntensity[i] * params->intensityMultiplier,
						params->ambientSaturation[i] * params->saturationMultiplier,
						{ params->ambientColorNearR[i], params->ambientColorNearG[i], params->ambientColorNearB[i] },
						params->ambientColorFarToggle[i],
						{ params->ambientColorFarR[i], params->ambientColorFarG[i], params->ambientColorFarB[i] },
						params->ambientColorFalloff[i] * params->colorFalloffMultiplier,
						iesChosenPreset1, iesChosenPreset2, iesChosenPreset3, iesChosenPreset4, iesChosenPreset5, iesChosenPreset6,

						// Ambient
						params->ambientToggle[i],
						params->ambientIntensity[i] * params->intensityMultiplier,
						params->ambientSaturation[i] * params->saturationMultiplier,
						{ params->ambientColorNearR[i], params->ambientColorNearG[i], params->ambientColorNearB[i] },
						params->ambientColorFarToggle[i],
						{ params->ambientColorFarR[i], params->ambientColorFarG[i], params->ambientColorFarB[i] },
						params->ambientColorFalloff[i],

						// Diffuse
						params->diffuseToggle[i],
						params->diffuseIntensity[i] * params->intensityMultiplier,
						params->diffuseSaturation[i] * params->saturationMultiplier,
						{ params->diffuseColorNearR[i], params->diffuseColorNearG[i], params->diffuseColorNearB[i] },
						params->diffuseColorFarToggle[i],
						{ params->diffuseColorFarR[i], params->diffuseColorFarG[i], params->diffuseColorFarB[i] },
						params->diffuseColorFalloff[i],

						// Specular
						params->specularToggle[i],
						params->specularSize[i],
						params->specularIntensity[i] * params->intensityMultiplier,
						params->specularSaturation[i] * params->saturationMultiplier,
						{ params->specularColorNearR[i], params->specularColorNearG[i], params->specularColorNearB[i] },
						params->specularColorFarToggle[i],
						{ params->specularColorFarR[i], params->specularColorFarG[i], params->specularColorFarB[i] },
						params->specularColorFalloff[i],

						// Shadows Toggle
						params->shadowToggle[i], params->shadowIgnoreAmbientToggle[i], params->shadowIgnoreDiffuseToggle[i], params->shadowIgnoreSpecularToggle[i], params->shadowClipToLightToggle[i],
						// Soft Shadows
						params->shadowSoftnessRadius[i], params->shadowSoftnessSamples[i],
						// Shadows
						shadowSampleStep, params->shadowImprovedSampleRadius[i], params->shadowMaxLength[i], params->shadowThresholdStart[i], params->shadowThresholdEnd[i],
						// Shadow Visualize
						params->shadowIntensity[i], shadowColor,
						// Depth
						src1, inXY, pitch1, width1, height1, in16f1, params->depthBlackIsNear, params->depthFar,
						// Camera
						camVx, camVy, camVz, params->cameraZoom, params->cameraWidth, params->cameraHeight, downsample
					);

					if (params->renderMode == 5) {
						shadows = drawSpot.x;
						shadows = mix(shadows, 1.0f, 1.0f - params->shadowIntensity[i]);
						//shadowsColored = mulf3(shadowColor, (1.0f - shadows));
						shadows = clamp(shadows, 0.0f, 1.0f);
						shadowsMask = shadowsMask * shadows;
						drawSpot = { 0.0f, 0.0f, 0.0f };
					}

					draw = {
						draw.x + drawSpot.x,
						draw.y + drawSpot.y,
						draw.z + drawSpot.z
					};

				}

				if (params->renderMode == 5) {
					draw = addf3(draw, { shadowsMask, shadowsMask, shadowsMask });
				}

				// Output

				WriteFloat4({ draw.z, draw.y, draw.x, 1.0f }, src3, inXY.y * pitch3 + inXY.x, !!in16f3);

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