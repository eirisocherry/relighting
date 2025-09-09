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

					float3 lightPos = { params->lightPosX[i] * downsample, params->lightPosY[i] * downsample, params->lightPosZ[i] * downsample };
					float radius = params->radius[i] * params->radiusMultiplier;
					float distance = distancef3(lightPos, worldPos);

					float3 drawAmbient = { 0.0f, 0.0f, 0.0f };
					float3 drawDiffuse = { 0.0f, 0.0f, 0.0f };
					float3 drawSpecular = { 0.0f, 0.0f, 0.0f };
					float shadows = 1.0f;
					float3 shadowsColored = { 0.0f, 0.0f, 0.0f };

					// Ambient, Diffuse, Specular

					if ((distance <= radius) && (params->renderMode != 5)) {

						if (params->ambientToggle[i] && ((params->renderMode == 1) || (params->renderMode == 2))) {
							drawAmbient = pointLightAmbient(
								worldPos, lightPos, radius,
								params->ambientFalloff[i],
								params->ambientIntensity[i] * params->intensityMultiplier,
								params->ambientSaturation[i] * params->saturationMultiplier,
								{ params->ambientColorNearR[i], params->ambientColorNearG[i], params->ambientColorNearB[i] },
								params->ambientColorFarToggle[i],
								{ params->ambientColorFarR[i], params->ambientColorFarG[i], params->ambientColorFarB[i] },
								params->ambientColorFalloff[i]
							);
						}

						if (params->diffuseToggle[i] && ((params->renderMode == 1) || (params->renderMode == 3))) {
							drawDiffuse = pointLightDiffuse(
								worldPos, lightPos, normal, radius,
								params->diffuseFalloff[i],
								params->diffuseIntensity[i] * params->intensityMultiplier,
								params->diffuseSaturation[i] * params->saturationMultiplier,
								{ params->diffuseColorNearR[i], params->diffuseColorNearG[i], params->diffuseColorNearB[i] },
								params->diffuseColorFarToggle[i],
								{ params->diffuseColorFarR[i], params->diffuseColorFarG[i], params->diffuseColorFarB[i] },
								params->diffuseColorFalloff[i]
							);
						}

						if (params->specularToggle[i] && ((params->renderMode == 1) || (params->renderMode == 4))) {
							drawSpecular = pointLightSpecular(
								worldPos, lightPos, normal, camPos, radius,
								params->specularSize[i],
								params->specularFalloff[i],
								params->specularIntensity[i] * params->intensityMultiplier,
								params->specularSaturation[i] * params->saturationMultiplier,
								{ params->specularColorNearR[i], params->specularColorNearG[i], params->specularColorNearB[i] },
								params->specularColorFarToggle[i],
								{ params->specularColorFarR[i], params->specularColorFarG[i], params->specularColorFarB[i] },
								params->specularColorFalloff[i]
							);
						}

					}

					// Shadows

					if (params->shadowToggle[i] && ((params->renderMode == 1) || (params->renderMode == 5))) {

						float shadowSampleStep = fmaxf(params->shadowSampleStep[i], 0.2f);
						
						if (params->shadowClipToLightToggle[i]) {
							if (distance <= radius) {
								//shadows = getSoftShadows(shadowSoftnessRadius, shadowSoftnessSamples, _PixelsWorld_uv, lightPos, shadowSampleStep, shadowStepsLimit, shadowThresholdStart, shadowThresholdEnd);
								shadows = getSoftShadows(
									params->shadowSoftnessRadius[i], params->shadowSoftnessSamples[i],
									lightPos, shadowSampleStep, params->shadowImprovedSampleRadius[i], params->shadowMaxLength[i], params->shadowThresholdStart[i], params->shadowThresholdEnd[i],
									src1, inXY, pitch1, width1, height1, in16f1, params->depthBlackIsNear, params->depthFar,
									camVx, camVy, camVz, camPos, params->cameraZoom, params->cameraWidth, params->cameraHeight, downsample
								);
							}
						}
						else {
							shadows = getSoftShadows(
								params->shadowSoftnessRadius[i], params->shadowSoftnessSamples[i],
								lightPos, shadowSampleStep, params->shadowImprovedSampleRadius[i], params->shadowMaxLength[i], params->shadowThresholdStart[i], params->shadowThresholdEnd[i],
								src1, inXY, pitch1, width1, height1, in16f1, params->depthBlackIsNear, params->depthFar,
								camVx, camVy, camVz, camPos, params->cameraZoom, params->cameraWidth, params->cameraHeight, downsample
							);
						}

						shadows = mix(shadows, 1.0f, 1.0f - params->shadowIntensity[i]);
						float3 shadowColor = { params->shadowColorR[i], params->shadowColorG[i], params->shadowColorB[i] };
						shadowsColored = mulf3(shadowColor, (1.0f - shadows));
						shadows = clamp(shadows, 0.0f, 1.0f);
						shadowsMask = shadowsMask * shadows;

						if (!params->shadowIgnoreAmbientToggle[i]) {
							drawAmbient = mulf3(drawAmbient, shadows);
						}
						if (!params->shadowIgnoreDiffuseToggle[i]) {
							drawDiffuse = mulf3(drawDiffuse, shadows);
						}
						if (!params->shadowIgnoreSpecularToggle[i]) {
							drawSpecular = mulf3(drawSpecular, shadows);
						}
						
					
					}

					draw = {
						draw.x + drawAmbient.x + drawDiffuse.x + drawSpecular.x + shadowsColored.x,
						draw.y + drawAmbient.y + drawDiffuse.y + drawSpecular.y + shadowsColored.y,
						draw.z + drawAmbient.z + drawDiffuse.z + drawSpecular.z + shadowsColored.z
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