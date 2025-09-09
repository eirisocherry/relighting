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

					// Shadow

					float shadowSampleStep = fmaxf(params->shadowSampleStep[i], 0.2f);
					float3 shadowColor = { params->shadowColorR[i], params->shadowColorG[i], params->shadowColorB[i] };

					// Draw

					float3 drawRect = { 0.0f, 0.0f, 0.0f };
					float shadows = 1.0f;
					float3 shadowsColored = { 0.0f, 0.0f, 0.0f };

					drawRect = rectAdvanced(
						// Inputs
						params->renderMode, camPos, worldPos, normal,
					 
						// Light Start
						pos1, vX1, vY1, vZ1, res1, scale1,
						// Light End
						pos2, vX2, vY2, vZ2, res2, scale2,
						// Shape
						featherX, featherY, featherZ, featherNormalized, falloff,

						// Ambient
						params->ambientToggle[i],
						params->ambientIntensity[i] * params->intensityMultiplier,
						params->ambientSaturation[i] * params->saturationMultiplier,
						{ params->ambientColorNearR[i], params->ambientColorNearG[i], params->ambientColorNearB[i] },
						params->ambientColorFarToggle[i],
						{ params->ambientColorFarR[i], params->ambientColorFarG[i], params->ambientColorFarB[i] },
						params->ambientColorFalloff[i] * params->colorFalloffMultiplier,

						// Diffuse
						params->diffuseToggle[i],
						params->diffuseIntensity[i] * params->intensityMultiplier,
						params->diffuseSaturation[i] * params->saturationMultiplier,
						{ params->diffuseColorNearR[i], params->diffuseColorNearG[i], params->diffuseColorNearB[i] },
						params->diffuseColorFarToggle[i],
						{ params->diffuseColorFarR[i], params->diffuseColorFarG[i], params->diffuseColorFarB[i] },
						params->diffuseColorFalloff[i] * params->colorFalloffMultiplier,

						// Specular
						params->specularToggle[i],
						params->specularSize[i],
						params->specularIntensity[i] * params->intensityMultiplier,
						params->specularSaturation[i] * params->saturationMultiplier,
						{ params->specularColorNearR[i], params->specularColorNearG[i], params->specularColorNearB[i] },
						params->specularColorFarToggle[i],
						{ params->specularColorFarR[i], params->specularColorFarG[i], params->specularColorFarB[i] },
						params->specularColorFalloff[i] * params->colorFalloffMultiplier,

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
						shadows = drawRect.x;
						shadows = mix(shadows, 1.0f, 1.0f - params->shadowIntensity[i]);
						//shadowsColored = mulf3(shadowColor, (1.0f - shadows));
						shadows = clamp(shadows, 0.0f, 1.0f);
						shadowsMask = shadowsMask * shadows;
						drawRect = { 0.0f, 0.0f, 0.0f };
					}

					draw = {
						draw.x + drawRect.x,
						draw.y + drawRect.y,
						draw.z + drawRect.z
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