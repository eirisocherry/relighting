#ifndef SDK_INVERT_PROC_AMP
#	define SDK_INVERT_PROC_AMP

#	include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#	include "PrGPU/KernelSupport/KernelMemory.h"
#	include "Math_Utils.h"

#	if GF_DEVICE_TARGET_DEVICE

		GF_KERNEL_FUNCTION(ProcAmp2Kernel,
			((const GF_PTR(float4))(inSrc))
			((GF_PTR(float4))(outDst)),
			((int)(inSrcPitch))
			((int)(inDstPitch))
			((int)(in16f))
			((unsigned int)(inWidth))
			((unsigned int)(inHeight))
			((float)(depthFar))
			((bool)(depthBlackIsNear))
			((float)(camVx1))
			((float)(camVx2))
			((float)(camVx3))
			((float)(camVx4))
			((float)(camVy1))
			((float)(camVy2))
			((float)(camVy3))
			((float)(camVy4))
			((float)(camVz1))
			((float)(camVz2))
			((float)(camVz3))
			((float)(camVz4))
			((float)(camPos1))
			((float)(camPos2))
			((float)(camPos3))
			((float)(camPos4))
			((float)(cameraZoom))
			((float)(cameraWidth))
			((float)(cameraHeight))
			((float)(lightRadius))
			((float)(lightFalloff))
			((float)(lightPosX1))
			((float)(lightPosY1))
			((float)(lightPosZ1)),
			((uint2)(inXY)(KERNEL_XY)))
		{
			if (inXY.x < inWidth && inXY.y < inHeight)
			{
				// BGRA
				float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
				// BGR to RGB
				//float3 inTexture = { pixel.z, pixel.y, pixel.x };

				// Depth
				float depth = (1.0f - pixel.x) * depthFar;

				// Position
				float2 uv = make_float2((float)inXY.x / (float)inWidth,
										(float)inXY.y / (float)inHeight);
				float2 fragCoord = { uv.x * cameraWidth, uv.y * cameraHeight };

				float3 screenPos = {
					fragCoord.x - 0.5f * cameraWidth,  // [-halfRes..halfRes]
					fragCoord.y - 0.5f * cameraHeight,
					cameraZoom
				};

				float3 localPos = { 0.0f,  0.0f,  0.0f };
				localPos.z = depth;
				float diff = localPos.z / screenPos.z;
				localPos.x = screenPos.x * diff;
				localPos.y = screenPos.y * diff;


				// Camera matrix
				float4 camVx = { camVx1, camVx2, camVx3, camVx4 };
				float4 camVy = { camVy1, camVy2, camVy3, camVy4 };
				float4 camVz = { camVz1, camVz2, camVz3, camVz4 };
				float4 camPos = { camPos1, camPos2, camPos3, camPos4 };
				//float4x4 camMatrix = make_float4x4(camVx, camVy, camVz, camPos);
				//float4 worldPos = mulMatrixVector(camMatrix, localPos);


				//float3 worldPos{
				//	camVx1 * localPos.x + camVy1 * localPos.y + camVz1 * localPos.z + camPos1 * 1.0f,
				//	camVx2 * localPos.x + camVy2 * localPos.y + camVz2 * localPos.z + camPos2 * 1.0f,
				//	camVx3 * localPos.x + camVy3 * localPos.y + camVz3 * localPos.z + camPos3 * 1.0f
				//};

				// Point
				float3 lightPos = { lightPosX1, lightPosY1, lightPosZ1 };

				float3 draw = point(localPos, lightPos, lightRadius, lightFalloff);

				float3 outTexture = { draw.x, draw.y, draw.z };
				// RGB to BGR
				pixel = { outTexture.z, outTexture.y, outTexture.x, pixel.w };
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
			float depthFar,
			bool depthBlackIsNear,
			float camVx1, float camVx2, float camVx3, float camVx4,
			float camVy1, float camVy2, float camVy3, float camVy4,
			float camVz1, float camVz2, float camVz3, float camVz4,
			float camPos1, float camPos2, float camPos3, float camPos4,
			float cameraZoom, float cameraWidth, float cameraHeight,
			float lightRadius,
			float lightFalloff,
			float lightPosX1,
			float lightPosY1,
			float lightPosZ1)
		{
			dim3 blockDim (16, 16, 1);
			dim3 gridDim ( (width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1 );

			ProcAmp2Kernel <<< gridDim, blockDim, 0 >>> (
				(float4 const*) src, (float4*) dst, srcPitch, dstPitch, is16f, width, height,
				depthFar,
				depthBlackIsNear,
				camVx1, camVx2, camVx3, camVx4,
				camVy1, camVy2, camVy3, camVy4,
				camVz1, camVz2, camVz3, camVz4,
				camPos1, camPos2, camPos3, camPos4,
				cameraZoom, cameraWidth, cameraHeight,
				lightRadius,
				lightFalloff,
				lightPosX1,
				lightPosY1,
				lightPosZ1
			);

			cudaDeviceSynchronize();
		}

#	endif //GF_DEVICE_TARGET_HOST

#endif