#pragma once
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#ifndef __MATH_UTILS_DECL__
#define __MATH_UTILS_DECL__ __inline__ __device__ __forceinline__
#endif

// Math Functions

__MATH_UTILS_DECL__ float radians(float a)
{
    return 0.017453292f * a;
}

__MATH_UTILS_DECL__ float3 absf3(float3 a) {

    return {
        (a.x < 0) ? -a.x : a.x,
        (a.y < 0) ? -a.y : a.y,
        (a.z < 0) ? -a.z : a.z
    };


}

__MATH_UTILS_DECL__ float2 absf2(float2 a) {

    return {
        (a.x < 0) ? -a.x : a.x,
        (a.y < 0) ? -a.y : a.y
    };

}


__MATH_UTILS_DECL__ float fract(float x) {

    return x - floor(x);

}

__MATH_UTILS_DECL__ float3 fractf3(float3 a) {

    return {
        a.x - floor(a.x),
        a.y - floor(a.y),
        a.z - floor(a.z)
    };

}


__MATH_UTILS_DECL__ float clamp(float x, float minVal, float maxVal) {

    return fminf(fmaxf(x, minVal), maxVal);

}

__MATH_UTILS_DECL__ float3 clampf3(float3 a, float minVal, float maxVal) {

    return {
        fminf(fmaxf(a.x, minVal), maxVal),
        fminf(fmaxf(a.y, minVal), maxVal),
        fminf(fmaxf(a.z, minVal), maxVal)
    };

}

__MATH_UTILS_DECL__ float4 clampf4(float4 a, float minVal, float maxVal) {

    return {
        fminf(fmaxf(a.x, minVal), maxVal),
        fminf(fmaxf(a.y, minVal), maxVal),
        fminf(fmaxf(a.z, minVal), maxVal),
        fminf(fmaxf(a.w, minVal), maxVal)
    };

}


__MATH_UTILS_DECL__ float dotf3(float3 a, float3 b) {

    return a.x * b.x + a.y * b.y + a.z * b.z;

}

__MATH_UTILS_DECL__ float dotf2(float2 a, float2 b) {

    return a.x * b.x + a.y * b.y;

}

__MATH_UTILS_DECL__ float3 cross(float3 a, float3 b) {

    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };

}

__MATH_UTILS_DECL__ float distancef4(float4 a, float4 b) {

    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    float dw = a.w - b.w;
    return sqrtf(dx * dx + dy * dy + dz * dz + dw * dw);

}

__MATH_UTILS_DECL__ float distancef3(float3 a, float3 b) {

    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);

}

__MATH_UTILS_DECL__ float distancef2(float2 a, float2 b) {

    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);

}


__MATH_UTILS_DECL__ float lengthf4(float4 a) {

    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);

}

__MATH_UTILS_DECL__ float lengthf3(float3 a) {

    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);

}

__MATH_UTILS_DECL__ float lengthf2(float2 a) {

    return sqrtf(a.x * a.x + a.y * a.y);

}


__MATH_UTILS_DECL__ float2 invertf2(float2 a) {

    return { -a.x, -a.y };

}

__MATH_UTILS_DECL__ float3 invertf3(float3 a) {

    return { -a.x, -a.y, -a.z };

}

__MATH_UTILS_DECL__ float4 invertf4(float4 a) {

    return { -a.x, -a.y, -a.z, -a.w };

}



__MATH_UTILS_DECL__ float4 writef4(float4 a) {
    return { a.x, a.y, a.z, a.w };
}

__MATH_UTILS_DECL__ float3 writef3(float3 a) {
    return { a.x, a.y, a.z };
}

__MATH_UTILS_DECL__ float2 writef2(float2 a) {
    return { a.x, a.y };
}


__MATH_UTILS_DECL__ float2 takeXYf2(float2 a) {
    return { a.x, a.y };
}

__MATH_UTILS_DECL__ float2 takeXYf3(float3 a) {
    return { a.x, a.y };
}

__MATH_UTILS_DECL__ float2 takeXYf4(float4 a) {
    return { a.x, a.y };
}

__MATH_UTILS_DECL__ float3 takeXYZf3(float3 a) {
    return { a.x, a.y, a.z };
}

__MATH_UTILS_DECL__ float3 takeXYZf4(float4 a) {
    return { a.x, a.y, a.z };
}

__MATH_UTILS_DECL__ float4 takeXYZWf4(float4 a) {
    return { a.x, a.y, a.z, a.w };
}


__MATH_UTILS_DECL__ float4 subf4(float4 a, float4 b) {

    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
        a.w - b.w
    };

}

__MATH_UTILS_DECL__ float3 subf3(float3 a, float3 b) {

    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };

}

__MATH_UTILS_DECL__ float2 subf2(float2 a, float2 b) {

    return {
        a.x - b.x,
        a.y - b.y
    };

}


__MATH_UTILS_DECL__ float4 addf4(float4 a, float4 b) {

    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        a.w + b.w
    };

}

__MATH_UTILS_DECL__ float3 addf3(float3 a, float3 b) {

    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };

}

__MATH_UTILS_DECL__ float2 addf2(float2 a, float2 b) {

    return {
        a.x + b.x,
        a.y + b.y
    };

}


__MATH_UTILS_DECL__ float4 mulf4(float4 a, float w) {

    return {
        a.x * w,
        a.y * w,
        a.z * w,
        a.w * w
    };

}

__MATH_UTILS_DECL__ float3 mulf3(float3 a, float w) {

    return {
        a.x * w,
        a.y * w,
        a.z * w
    };

}

__MATH_UTILS_DECL__ float2 mulf2(float2 a, float w) {

    return {
        a.x * w,
        a.y * w
    };

}


__MATH_UTILS_DECL__ float4 divf4(float4 a, float w) {

    if (w != 0.0f) {
        return {
            a.x / w,
            a.y / w,
            a.z / w,
            a.w / w
        };
    }
    else {
        return {
            0.0f,
            0.0f,
            0.0f,
            0.0f
        };
    }

}

__MATH_UTILS_DECL__ float3 divf3(float3 a, float w) {

    if (w != 0.0f) {
        return {
            a.x / w,
            a.y / w,
            a.z / w
        };
    }
    else {
        return {
            0.0f,
            0.0f,
            0.0f
        };
    }

}

__MATH_UTILS_DECL__ float2 divf2(float2 a, float w) {

    if (w != 0.0f) {
        return {
            a.x / w,
            a.y / w
        };
    }
    else {
        return {
            0.0f,
            0.0f
        };
    }

}


__MATH_UTILS_DECL__ float4 mixf4(float4 a, float4 b, float w) {

    return {
        a.x + w * (b.x - a.x),
        a.y + w * (b.y - a.y),
        a.z + w * (b.z - a.z),
        a.w + w * (b.w - a.w)
    };

}

__MATH_UTILS_DECL__ float3 mixf3(float3 a, float3 b, float w) {

    return {
        a.x + w * (b.x - a.x),
        a.y + w * (b.y - a.y),
        a.z + w * (b.z - a.z)
    };

}

__MATH_UTILS_DECL__ float mix(float a, float b, float w) {

    return a + w * (b - a);

}


__MATH_UTILS_DECL__ float step(float edge, float x) {

    if (x < edge)
        return 0.0f;
    else
        return 1.0f;

}

__MATH_UTILS_DECL__ float smoothstep(float edge0, float edge1, float x) {

    if (edge0 == edge1) {
        return 0.0f;
    }

    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);

}


__MATH_UTILS_DECL__ float3 normalizef3(float3 a) {
    float len = lengthf3(a);
    if (len > 0.0f) {
        return divf3(a, len);
    }
    return { 0.0f, 0.0f, 0.0f };
}

__MATH_UTILS_DECL__ float3 reflectf3(float3 i, float3 n) {

    return subf3(i, mulf3(n, 2.0f * dotf3(i, n)));

}

// Color

__MATH_UTILS_DECL__ float3 rgb2hsv(float3 c)
{
    float4 K = { 0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f };
    float4 p = mixf4({ c.z, c.y, K.w, K.z }, { c.y, c.z, K.x, K.y }, step(c.z, c.y));
    float4 q = mixf4({ p.x, p.y, p.w, c.x }, { c.x, p.y, p.z, p.x }, step(p.x, c.x));

    float d = q.x - fminf(q.w, q.y);
    float e = 1.0e-10f;

    return {
        abs(q.z + (q.w - q.y) / (6.0f * d + e)),
        d / (q.x + e),
        q.x
    };
}

__MATH_UTILS_DECL__ float3 hsv2rgb(float3 c) {

    float4 K = { 1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f };
    float3 p = absf3(
        subf3(
            mulf3(
                fractf3(
                    addf3({ c.x, c.x, c.x }, { K.x, K.y, K.z })
                ),
                6.0f
            ), 
            { K.w, K.w, K.w }
        )
    );

    return mulf3(
        mixf3({ K.x, K.x, K.x }, clampf3(subf3(p, { K.x, K.x, K.x }), 0.0f, 1.0f), c.y),
        c.z
    );

}

__MATH_UTILS_DECL__ float3 rgbMultiplySaturation(float3 color, float saturationMultiplier) {

    float3 colorHSV = rgb2hsv(color);
    float3 colorRGB = hsv2rgb({ colorHSV.x, clamp(colorHSV.y * saturationMultiplier, 0.0f, 1.0f), colorHSV.z });

    return colorRGB;

}

// Unmult

__MATH_UTILS_DECL__ float luminance(float3 color) {

    return dotf3(color, { 0.299f, 0.587f, 0.114f });
}

__MATH_UTILS_DECL__ float4 screen(float4 color) {

    float luma = luminance({ color.x, color.y, color.z });

    float alpha = clamp(luma, 0.0f, 1.0f);

    float4 screen = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (alpha != 0.0f) {
        screen = {
            color.x / alpha,
            color.y / alpha,
            color.z / alpha,
            alpha * color.w
        };
    }

    return screen;

}

__MATH_UTILS_DECL__ float4 screenClamp(float4 color) {

    color.x = clamp(color.x * color.w, 0.0f, 1.0f);
    color.y = clamp(color.y * color.w, 0.0f, 1.0f);
    color.z = clamp(color.z * color.w, 0.0f, 1.0f);

    float maxChannel = fmaxf(fmaxf(color.x, color.y), color.z);

    float alpha = maxChannel;

    float4 screen = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (alpha != 0.0f) {
        screen = {
            color.x / alpha,
            color.y / alpha,
            color.z / alpha,
            alpha
        };
    }

    return screen;

}

// Clamp

__MATH_UTILS_DECL__ float4 removeAlpha(float4 color) {

    float4 clampedColor = { 0.0f,  0.0f,  0.0f,  0.0f };
    if (color.w == 0.0f) {
        clampedColor = { 0.0f, 0.0f, 0.0f, 1.0f };
    }
    else {
        float3 multipliedColor = mulf3({ color.x, color.y, color.z }, color.w);
        clampedColor = { multipliedColor.x, multipliedColor.y, multipliedColor.z, 1.0f };
    }

    return clampedColor;

}

__MATH_UTILS_DECL__ float4 keepAlpha(float4 color) {

    float maxChannel = fmaxf(fmaxf(color.x, color.y), color.z);
    if (maxChannel > 1.0f) {
        float realValue = maxChannel * color.w;

        if (realValue <= 1.0f) {
            color.x /= maxChannel;
            color.y /= maxChannel;
            color.z /= maxChannel;

            color.w = color.w * maxChannel;
        }
        else {

            color.x = color.x / maxChannel * realValue;
            color.y = color.y / maxChannel * realValue;
            color.z = color.z / maxChannel * realValue;

            color.w = color.w * maxChannel;
        }

    }

    return color;

}

// Get IES Preset

__MATH_UTILS_DECL__ float4 getIESpreset(int preset, int index) {

    if (preset == 1) {
        switch (index) {
        case 0: return { 1.0f, 1.0f, 1.0f, 0.0f };
        case 1: return { 1.0f, 1.0f, 1.0f, 0.2f };
        case 2: return { 1.0f, 1.0f, 1.0f, 0.4f };
        case 3: return { 1.0f, 1.0f, 1.0f, 0.6f };
        case 4: return { 1.0f, 1.0f, 1.0f, 0.8f };
        case 5: return { 1.0f, 1.0f, 1.0f, 1.0f };
        }
    }
    if (preset == 2) {
        switch (index) {
        case 0: return { 1.0f, 1.0f, 1.0f, 0.0f };
        case 1: return { 0.0f, 0.0f, 0.0f, 1.0f };
        case 2: return { 0.0f, 0.0f, 0.0f, 1.0f };
        case 3: return { 0.0f, 0.0f, 0.0f, 1.0f };
        case 4: return { 0.0f, 0.0f, 0.0f, 1.0f };
        case 5: return { 0.0f, 0.0f, 0.0f, 1.0f };
        }
    }
    if (preset == 3) {
        switch (index) {
        case 0: return { 0.5f, 0.5f, 0.5f, 0.0f };
        case 1: return { 1.0f, 1.0f, 1.0f, 0.4f };
        case 2: return { 0.0f, 0.0f, 0.0f, 1.0f };
        case 3: return { 0.0f, 0.0f, 0.0f, 1.0f };
        case 4: return { 0.0f, 0.0f, 0.0f, 1.0f };
        case 5: return { 0.0f, 0.0f, 0.0f, 1.0f };
        }
    }
    if (preset == 4) {
        switch (index) {
        case 0: return { 0.9f, 0.9f, 0.9f, 0.0f };
        case 1: return { 0.4f, 0.4f, 0.4f, 0.3f };
        case 2: return { 0.7f, 0.7f, 0.7f, 0.4f };
        case 3: return { 0.35f, 0.35f, 0.35f, 0.6f };
        case 4: return { 0.0f, 0.0f, 0.0f, 0.9f };
        case 5: return { 0.0f, 0.0f, 0.0f, 1.0f };
        }
    }

    return { 1.0f, 1.0f, 1.0f, 0.0f };

}

// Map 6 color gradient to 0-1 float 

__MATH_UTILS_DECL__ float3 ramp(
    float4 iesChosenPreset11,
    float4 iesChosenPreset21,
    float4 iesChosenPreset31,
    float4 iesChosenPreset41,
    float4 iesChosenPreset51,
    float4 iesChosenPreset61,
    float x) {

    x = clamp(x, 0.0f, 1.0f);

    float4 p1 = iesChosenPreset11;
    float4 p2 = iesChosenPreset21;
    float4 p3 = iesChosenPreset31;
    float4 p4 = iesChosenPreset41;
    float4 p5 = iesChosenPreset51;
    float4 p6 = iesChosenPreset61;

    float3 c1 = { p1.x, p1.y, p1.z };
    float3 c2 = { p2.x, p2.y, p2.z };
    float3 c3 = { p3.x, p3.y, p3.z };
    float3 c4 = { p4.x, p4.y, p4.z };
    float3 c5 = { p5.x, p5.y, p5.z };
    float3 c6 = { p6.x, p6.y, p6.z };

    float d1 = p1.w, d2 = p2.w, d3 = p3.w, d4 = p4.w, d5 = p5.w, d6 = p6.w;

    if (x <= d2) {
        float denom = d2 - d1;
        if (denom == 0.0f) return c1;
        float t = (x - d1) / denom;
        t = smoothstep(0.0f, 1.0f, t);
        return mixf3(c1, c2, t);
    }
    else if (x <= d3) {
        float denom = d3 - d2;
        if (denom == 0.0f) return c2;
        float t = (x - d2) / denom;
        t = smoothstep(0.0f, 1.0f, t);
        return mixf3(c2, c3, t);
    }
    else if (x <= d4) {
        float denom = d4 - d3;
        if (denom == 0.0f) return c3;
        float t = (x - d3) / denom;
        t = smoothstep(0.0f, 1.0f, t);
        return mixf3(c3, c4, t);
    }
    else if (x <= d5) {
        float denom = d5 - d4;
        if (denom == 0.0f) return c4;
        float t = (x - d4) / denom;
        t = smoothstep(0.0f, 1.0f, t);
        return mixf3(c4, c5, t);
    }
    else if (x <= d6) {
        float denom = d6 - d5;
        if (denom == 0.0f) return c5;
        float t = (x - d5) / denom;
        t = smoothstep(0.0f, 1.0f, t);
        return mixf3(c5, c6, t);
    }
    else {
        return c6;
    }

}


// Layers

__MATH_UTILS_DECL__ float4 samplePixel(GF_PTR(float4 const) inSrc, uint2 inXY, int inSrcPitch, unsigned int inWidth, unsigned int inHeight, int in16f) {

    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float4 bgra = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f); //BGRA
        return { bgra.z, bgra.y, bgra.x, bgra.w }; // RGBA
    }
    return { 0.0f, 0.0f, 0.0f, 0.0f };

}

__MATH_UTILS_DECL__ float4 samplePixelUV(GF_PTR(float4 const) inSrc, float2 uv, int inSrcPitch, unsigned int inWidth, unsigned int inHeight, int in16f) {

    uint2 inXY = { static_cast<unsigned int>(uv.x * inWidth), static_cast<unsigned int>(uv.y * inHeight) };

    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float4 bgra = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f); //BGRA
        return { bgra.z, bgra.y, bgra.x, bgra.w }; // RGBA
    }
    return { 0.0f, 0.0f, 0.0f, 0.0f };

}


__MATH_UTILS_DECL__ float getDepth(
    GF_PTR(float4 const) inSrc, uint2 inXY, int inSrcPitch, unsigned int inWidth, unsigned int inHeight, int in16f, bool depthBlackIsNear, float depthFar
) {

    float depth = samplePixel(inSrc, inXY, inSrcPitch, inWidth, inHeight, in16f).x;

    if (depthBlackIsNear) {
        depth = depth * depthFar;
    }
    else {
        depth = (1.0f - depth) * depthFar;
    }

    return depth;

}

__MATH_UTILS_DECL__ float3 getPosition(
    GF_PTR(float4 const) inSrc, uint2 inXY, int inSrcPitch, unsigned int inWidth, unsigned int inHeight, int in16f, bool depthBlackIsNear, float depthFar,
    bool screenSpace, float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float downsample
) {

    float2 fragCoord = { (float)inXY.x * downsample, (float)inXY.y * downsample };

    float3 screenPos = {
        fragCoord.x - 0.5f * (float)inWidth * downsample,  // [-halfRes..halfRes]
        fragCoord.y - 0.5f * (float)inHeight * downsample,
        cameraZoom
    };

    float3 localPos = { 0.0f,  0.0f,  0.0f };
    localPos.z = getDepth( inSrc, inXY, inSrcPitch, inWidth, inHeight, in16f, depthBlackIsNear, depthFar);
    float diff = localPos.z / screenPos.z;
    localPos.x = screenPos.x * diff;
    localPos.y = screenPos.y * diff;

    if (screenSpace) {
        return localPos;
    }
    else {
        //float4x4 camMatrix = make_float4x4(camVx, camVy, camVz, camPos);
        //float4 worldPos = mulMatrixVector(camMatrix, localPos);
        return {
            camVx.x * localPos.x + camVy.x * localPos.y + camVz.x * localPos.z + camPos.x * 1.0f,
            camVx.y * localPos.x + camVy.y * localPos.y + camVz.y * localPos.z + camPos.y * 1.0f,
            camVx.z * localPos.x + camVy.z * localPos.y + camVz.z * localPos.z + camPos.z * 1.0f
        };
    }

}

__MATH_UTILS_DECL__ float3 getPositionNoDepth(uint2 inXY, unsigned int inWidth, unsigned int inHeight, float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float downsample) {

    float2 fragCoord = { (float)inXY.x * downsample, (float)inXY.y * downsample };

    float3 screenPos = {
        fragCoord.x - 0.5f * (float)inWidth * downsample,  // [-halfRes..halfRes]
        fragCoord.y - 0.5f * (float)inHeight * downsample,
        cameraZoom
    };

    return {
        camVx.x * screenPos.x + camVy.x * screenPos.y + camVz.x * screenPos.z + camPos.x * 1.0f,
        camVx.y * screenPos.x + camVy.y * screenPos.y + camVz.y * screenPos.z + camPos.y * 1.0f,
        camVx.z * screenPos.x + camVy.z * screenPos.y + camVz.z * screenPos.z + camPos.z * 1.0f
    };
    
}

__MATH_UTILS_DECL__ float3 worldPosToLocalPos(float3 worldPos, float3 camVx, float3 camVy, float3 camVz, float3 camPos) {

    float3 localPos = { 0.0f, 0.0f, 0.0f };
    float3 difff3 = subf3(worldPos, camPos);
    localPos.x = dotf3(difff3, camVx);
    localPos.y = dotf3(difff3, camVy);
    localPos.z = dotf3(difff3, camVz);
    return localPos;

}

__MATH_UTILS_DECL__ float2 localPosToScreenPos(float3 localPos, float cameraZoom, float cameraWidth, float cameraHeight) {

    float diff = localPos.z / cameraZoom;
    float2 screenPosHalf = {
        localPos.x / diff,
        localPos.y / diff
    };
    float2 screenPos = {
        screenPosHalf.x + 0.5f * cameraWidth,
        screenPosHalf.y + 0.5f * cameraHeight
    };
    return screenPos;

}

__MATH_UTILS_DECL__ float2 localPosToUV(float3 localPos, float cameraZoom, float cameraWidth, float cameraHeight) {

    float diff = localPos.z / cameraZoom;
    float2 uv = {
        (localPos.x / diff + 0.5f * cameraWidth) / cameraWidth,
        (localPos.y / diff + 0.5f * cameraHeight) / cameraHeight
    };
    uv.y = 1.0f - uv.y;
    return uv;

}

__MATH_UTILS_DECL__ float2 worldPosToScreenPos(
    float3 worldPos,
    float3 camVx, float3 camVy, float3 camVz, float3 camPos,
    float cameraZoom, float cameraWidth, float cameraHeight
) {

    float3 localPos = worldPosToLocalPos(worldPos, camVx, camVy, camVz, camPos);
    float2 screenPos = localPosToScreenPos(localPos, cameraZoom, cameraWidth, cameraHeight);
    return screenPos;

}

__MATH_UTILS_DECL__ float3 getNormal(
    GF_PTR(float4 const) inSrc, uint2 inXY, int inSrcPitch, unsigned int inWidth, unsigned int inHeight, int in16f,
    bool depthBlackIsNear, float depthFar,
    bool screenSpace, float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float downsample
) {

    float3 right = getPosition(
        inSrc, { inXY.x + 1, inXY.y }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );

    float3 left = getPosition(
        inSrc, { inXY.x - 1, inXY.y }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );

    float3 top = getPosition(
        inSrc, { inXY.x, inXY.y - 1 }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );

    float3 bottom = getPosition(
        inSrc, { inXY.x, inXY.y + 1 }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );

    float3 dx = subf3(right, left);
    float3 dy = subf3(top, bottom);

    return normalizef3(cross(dx, dy));

}

__MATH_UTILS_DECL__ float3 getNormalImproved(
    GF_PTR(float4 const) inSrc, uint2 inXY, int inSrcPitch, unsigned int inWidth, unsigned int inHeight, int in16f,
    bool depthBlackIsNear, float depthFar,
    bool screenSpace, float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float downsample
) {

    float3 c = getPosition(
        inSrc, { inXY.x, inXY.y }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );

    float3 l1 = getPosition(
        inSrc, { inXY.x - 1, inXY.y }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );
    float3 l2 = getPosition(
        inSrc, { inXY.x - 2, inXY.y }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );
    float3 r1 = getPosition(
        inSrc, { inXY.x + 1, inXY.y }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );
    float3 r2 = getPosition(
        inSrc, { inXY.x + 2, inXY.y }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );
    float3 t1 = getPosition(
        inSrc, { inXY.x, inXY.y - 1 }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );
    float3 t2 = getPosition(
        inSrc, { inXY.x, inXY.y - 2 }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );
    float3 b1 = getPosition(
        inSrc, { inXY.x, inXY.y + 1 }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );
    float3 b2 = getPosition(
        inSrc, { inXY.x, inXY.y + 2 }, inSrcPitch, inWidth, inHeight, in16f,
        depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );

    float dl = abs(l1.x * l2.x / (2.0f * l2.x - l1.x) - c.x);
    float dr = abs(r1.x * r2.x / (2.0f * r2.x - r1.x) - c.x);
    float db = abs(b1.y * b2.y / (2.0f * b2.y - b1.y) - c.y);
    float dt = abs(t1.y * t2.y / (2.0f * t2.y - t1.y) - c.y);

    float3 dpdx = (dl < dr) ? subf3(c, l1) : subf3(r1, c);
    float3 dpdy = (db < dt) ? subf3(c, b1) : subf3(t1, c);

    return normalizef3(cross(dpdx, dpdy));

}

__MATH_UTILS_DECL__ float3 smoothNormal(
    uint2 inXY,
    GF_PTR(float4 const) normalSrc, int normalPitch, unsigned int normalWidth, unsigned int normalHeight, int normal16f, // normal
    GF_PTR(float4 const) depthSrc, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depth16f, bool depthBlackIsNear, float depthFar, // depth
    bool horizontalBlur, int radius, float normalThreshold, float depthWeight // smooth normal
) {

    float4 centerNormalF4 = samplePixel(normalSrc, inXY, normalPitch, normalWidth, normalHeight, normal16f);
    float3 centerNormal = { centerNormalF4.x, centerNormalF4.y , centerNormalF4.z };
    centerNormal = subf3(mulf3(centerNormal, 2.0f), { 1.0f, 1.0f, 1.0f });
    float centerDepth = getDepth(
        depthSrc, inXY, depthPitch, depthWidth, depthHeight, depth16f,
        depthBlackIsNear, depthFar
    );

    float3 blurred = { 0.0f, 0.0f, 0.0f };
    float weightSum = 0.0f;

    for (int i = -radius; i <= radius; i++) {

        int2 offset = { 0, 0 };
        if (horizontalBlur) {
            offset = { i, 0 };
        }
        else {
            offset = { 0, i };
        }

        float4 sampleNormalF4 = samplePixel(normalSrc, { inXY.x + offset.x, inXY.y + offset.y }, normalPitch, normalWidth, normalHeight, normal16f);
        float3 sampleNormal = { sampleNormalF4.x, sampleNormalF4.y , sampleNormalF4.z };
        sampleNormal = subf3(mulf3(sampleNormal, 2.0f), { 1.0f, 1.0f, 1.0f });

        float sampleDepth = getDepth(
            depthSrc, { inXY.x + offset.x, inXY.y + offset.y }, depthPitch, depthWidth, depthHeight, depth16f,
            depthBlackIsNear, depthFar
        );

        float diffNormal = distancef3(centerNormal, sampleNormal) * normalThreshold;
        float diffDepth = abs(centerDepth - sampleDepth) * depthWeight;
        float diff = diffNormal + diffDepth;

        float weight = fmaxf(0.0f, 1.0f - diff);
        blurred = addf3(blurred, mulf3(sampleNormal, weight));
        weightSum += weight;

    }

    if (weightSum > 0.0001f) {
        blurred = divf3(blurred, weightSum);
    }

    blurred = addf3(mulf3(blurred, 0.5f), { 0.5f, 0.5f, 0.5f });

    return blurred;

}

// Draw

__MATH_UTILS_DECL__ float3 point(float3 curPos, float3 lightPos, float radius, float fallOff, float intensityMultiplier, float saturation, float3 lightColorNear, bool lightColorFarToggle, float3 lightColorFar, float colorFalloff, bool invertToggle) {

    float dist = distancef3(curPos, lightPos);

    if (!invertToggle) {
        if (radius <= 0.0f || dist > radius) {
            return { 0.0f, 0.0f, 0.0f };
        }
    }

    // Falloff
    float intensity = 1.0f;
    float colorIntensity = 1.0f;
    float t = clamp(dist / radius, 0.0f, 1.0f);
    if (invertToggle) { t = 1.0 - t; }
    if (fallOff > 0.0f) {
        intensity *= powf(1.0f - t, fallOff);
    }
    if (colorFalloff > 0.0f) {
        colorIntensity *= powf(1.0f - t, colorFalloff);
    }

    // Mix Color
    float3 color = { 0.0f, 0.0f, 0.0f };
    if (lightColorFarToggle) {
        color = mixf3(lightColorFar, lightColorNear, colorIntensity);
    }
    else {
        color = lightColorNear;
    }

    // Saturation
    float3 colorHSV = rgb2hsv(color);
    float3 colorRGB = hsv2rgb({ colorHSV.x, clamp(colorHSV.y * saturation, 0.0f, 1.0f), colorHSV.z });
    float3 colorFinal = mulf3(colorRGB, intensity * intensityMultiplier);

    return colorFinal;

}

__MATH_UTILS_DECL__ float3 spot(float3 worldPos, float3 lightPos, float3 vX, float3 vY, float3 vZ, bool invertToggle, float z, float2 angles, float curvature, float falloff, float feather, float intensityMultiplier, float saturation, float3 lightColorNear, bool lightColorFarToggle, float3 lightColorFar, float colorFalloff, float4 iesChosenPreset1, float4 iesChosenPreset2, float4 iesChosenPreset3, float4 iesChosenPreset4, float4 iesChosenPreset5, float4 iesChosenPreset6) {

    if (z == 0.0f) return { 0.0f, 0.0f, 0.0f };

    float3 localPos = subf3(worldPos, lightPos);

    float curZ = dotf3(localPos, vZ);

    float intensity = 1.0f;

    if (curZ < 0.0f || curZ > z) {
        if (invertToggle) {
            intensity = 0.0f;
        }
        else {
            return { 0.0f, 0.0f, 0.0f };
        }
    }

    float x = dotf3(localPos, vX);
    float y = dotf3(localPos, vY);

    float t = clamp(curZ / z, 0.0f, 1.0f);

    float endX = z * tan(radians(angles.x));
    float endY = z * tan(radians(angles.y));
    float curEndX = endX * powf(t, curvature);
    float curEndY = endY * powf(t, curvature);

    // Ellipse check
    float inside = (x * x) / (curEndX * curEndX) +
                   (y * y) / (curEndY * curEndY);

    // IES
    float2 curXY = { x / curEndX,
                     y / curEndY };
    float curXYlength = lengthf2(curXY);

    intensity *= ramp(iesChosenPreset1, iesChosenPreset2, iesChosenPreset3, iesChosenPreset4, iesChosenPreset5, iesChosenPreset6, curXYlength).x;
    //float intensity = 1.0f;

    // float intensity = 1.0 - length(curXY);

    // Feather
    if (feather == 0.0f) {
        intensity *= step(inside, 1.0f);
    } else {
        float fOut = 1.0f;
        float fIn = 1.0f - feather;
        intensity *= smoothstep(fOut, fIn, inside);
    }

    // Falloff
    float colorIntensity = 1.0f;
    if (falloff > 0.0f) {
        intensity *= powf(1.0f - t, falloff);
    }

    // Invert
    if (invertToggle) {
        intensity = 1.0f - intensity;
        t = 1.0f - intensity;
    }

    // Color Falloff
    if (colorFalloff > 0.0f) {
        colorIntensity *= powf(1.0f - t, colorFalloff);
    }

    // Mix Color
    float3 color = { 0.0f, 0.0f, 0.0f };
    if (lightColorFarToggle) {
        color = mixf3(lightColorFar, lightColorNear, colorIntensity);
    }
    else {
        color = lightColorNear;
    }

    // Saturation
    float3 colorHSV = rgb2hsv(color);
    float3 colorRGB = hsv2rgb({ colorHSV.x, clamp(colorHSV.y * saturation, 0.0f, 1.0f), colorHSV.z });
    float3 colorFinal = mulf3(colorRGB, intensity * intensityMultiplier);

    return colorFinal;

}

__MATH_UTILS_DECL__ float3 rect(float3 curPos, float3 pos1, float3 vX1, float3 vY1, float3 vZ1, float3 res1, float3 scale1, float3 pos2, float3 vX2, float3 vY2, float3 vZ2, float3 res2, float3 scale2, bool invertToggle, float2 featherX, float2 featherY, float2 featherZ, bool featherNormalized, float falloff, float intensityMultiplier, float saturation, float3 lightColorNear, bool lightColorFarToggle, float3 lightColorFar, float colorFalloff) {

    // Sizes
    float sizeX1 = res1.x * abs(scale1.x);
    float sizeY1 = res1.y * abs(scale1.y);
    float sizeX2 = res2.x * abs(scale2.x);
    float sizeY2 = res2.y * abs(scale2.y);

    // Vertices A
    float3 TL1 = addf3(pos1, mulf3(vX1, sizeX1));
    float3 TR1 = writef3(pos1);
    float3 BR1 = subf3(pos1, mulf3(vY1, sizeY1));
    float3 BL1 = addf3(pos1, subf3(mulf3(vX1, sizeX1), mulf3(vY1, sizeY1)));


    // Vertices B Temp
    float3 TL2_Temp = addf3(pos2, mulf3(vX2, sizeX2));
    float3 TR2_Temp = writef3(pos2);
    float3 BR2_Temp = subf3(pos2, mulf3(vY2, sizeY2));
    float3 BL2_Temp = addf3(pos2, subf3(mulf3(vX2, sizeX2), mulf3(vY2, sizeY2)));


    // ----- Match vertices -----
    // Pattern that has the smallest total distance is the right one

    float3 rect1_vertices[4] = { TL1, TR1, BR1, BL1 };

    // Predefine 8 possible patterns
    float3 config_patterns[32] = {
        // config 0: BL2, TL2, TR2, BR2
        BL2_Temp, TL2_Temp, TR2_Temp, BR2_Temp,
        // config 1: BL2, BR2, TR2, TL2
        BL2_Temp, BR2_Temp, TR2_Temp, TL2_Temp,
        // config 2: BR2, BL2, TL2, TR2
        BR2_Temp, BL2_Temp, TL2_Temp, TR2_Temp,
        // config 3: BR2, TR2, TL2, BL2
        BR2_Temp, TR2_Temp, TL2_Temp, BL2_Temp,
        // config 4: TL2, BL2, BR2, TR2
        TL2_Temp, BL2_Temp, BR2_Temp, TR2_Temp,
        // config 5: TL2, TR2, BR2, BL2
        TL2_Temp, TR2_Temp, BR2_Temp, BL2_Temp,
        // config 6: TR2, TL2, BL2, BR2
        TR2_Temp, TL2_Temp, BL2_Temp, BR2_Temp,
        // config 7: TR2, BR2, BL2, TL2
        TR2_Temp, BR2_Temp, BL2_Temp, TL2_Temp
    };

    float min_distance = 1e6;
    int best_config = 0;

    // Check what pattern has the smallest total distance
    for (int config = 0; config < 8; config++) {
        float total_dist = 0.0;
        for (int j = 0; j < 4; j++) {
            float3 target_vertex = config_patterns[config * 4 + j];  // j-th vertex in config
            total_dist += distancef3(rect1_vertices[j], target_vertex);
        }

        if (total_dist < min_distance) {
            min_distance = total_dist;
            best_config = config;
        }
    }

    // Apply the correct pattern
    float3 TL2 = config_patterns[best_config * 4 + 0];
    float3 TR2 = config_patterns[best_config * 4 + 1];
    float3 BR2 = config_patterns[best_config * 4 + 2];
    float3 BL2 = config_patterns[best_config * 4 + 3];



    // ----- Rect Light Projection -----

    // 6 faces x 3 vertices
    float3 faceVertices[18] = {
        // Face 0: Bottom (pos1) — BL1, BR1, TL1
        BL1, BR1, TL1,

        // Face 1: Top (pos2) — BL2, TL2, BR2
        BL2, TL2, BR2,

        // Face 2: Front — BL1, BR1, BR2
        BL1, BR1, BR2,

        // Face 3: Right — BR1, TR1, TR2
        BR1, TR1, TR2,

        // Face 4: Back — TR1, TL1, TL2
        TR1, TL1, TL2,

        // Face 5: Left — TL1, BL1, BL2
        TL1, BL1, BL2
    };

    float3 center = {
        (BL1.x + BR1.x + TL1.x + TR1.x + BL2.x + BR2.x + TL2.x + TR2.x) * 0.125f,
        (BL1.y + BR1.y + TL1.y + TR1.y + BL2.y + BR2.y + TL2.y + TR2.y) * 0.125f,
        (BL1.z + BR1.z + TL1.z + TR1.z + BL2.z + BR2.z + TL2.z + TR2.z) * 0.125f
    };

    float distanceToLeftFace = 1e6;   // Face 5 (Left) - X
    float distanceToRightFace = 1e6;  // Face 3 (Right) - X
    float distanceToFrontFace = 1e6;  // Face 2 (Front) - Y
    float distanceToBackFace = 1e6;   // Face 4 (Back) - Y
    float distanceToBottomFace = 1e6; // Face 0 (Bottom) - Z
    float distanceToTopFace = 1e6;    // Face 1 (Top) - Z

    bool outsideLeft = false, outsideRight = false;
    bool outsideFront = false, outsideBack = false;
    bool outsideBottom = false, outsideTop = false;

    for (int i = 0; i < 6; i++) {
        float3 v1 = faceVertices[i * 3 + 0];
        float3 v2 = faceVertices[i * 3 + 1];
        float3 v3 = faceVertices[i * 3 + 2];

        float3 edge1 = subf3(v2, v1);
        float3 edge2 = subf3(v3, v1);
        float3 normal = normalizef3(cross(edge1, edge2));

        float3 toPoint = subf3(curPos, v1);
        float3 toCenter = subf3(center, v1);

        float distanceToPlane = dotf3(normal, toPoint);
        float sideCenter = dotf3(normal, toCenter);

        bool outsideThisFace = (distanceToPlane * sideCenter < 0.0f);

        if (i == 0) { // Bottom face
            distanceToBottomFace = abs(distanceToPlane);
            outsideBottom = outsideThisFace;
        }
        else if (i == 1) { // Top face
            distanceToTopFace = abs(distanceToPlane);
            outsideTop = outsideThisFace;
        }
        else if (i == 2) { // Front face
            distanceToFrontFace = abs(distanceToPlane);
            outsideFront = outsideThisFace;
        }
        else if (i == 3) { // Right face 
            distanceToRightFace = abs(distanceToPlane);
            outsideRight = outsideThisFace;
        }
        else if (i == 4) { // Back face 
            distanceToBackFace = abs(distanceToPlane);
            outsideBack = outsideThisFace;
        }
        else if (i == 5) { // Left face 
            distanceToLeftFace = abs(distanceToPlane);
            outsideLeft = outsideThisFace;
        }
    }

    // Parameters for feather normalization
    float halfWidthX = (sizeX1 + sizeX2) * 0.5f;
    float halfWidthY = (sizeY1 + sizeY2) * 0.5f;
    float halfWidthZ = distancef3(pos2, pos1) * 0.5f;

    float featherValue = 1.0f;

    // Feather X Right
    if (outsideLeft) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToLeftFace / halfWidthX;
            featherValue *= (featherX.x > 0.0f) ? smoothstep(0.0f, featherX.x, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherX.x > 0.0f) ? smoothstep(0.0f, featherX.x, distanceToLeftFace) : 1.0f;
        }
    }

    // Feather X Left
    if (outsideRight) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToRightFace / halfWidthX;
            featherValue *= (featherX.y > 0.0f) ? smoothstep(0.0f, featherX.y, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherX.y > 0.0f) ? smoothstep(0.0f, featherX.y, distanceToRightFace) : 1.0f;
        }
    }

    // Feather Y Down
    if (outsideFront) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToFrontFace / halfWidthY;
            featherValue *= (featherY.x > 0.0f) ? smoothstep(0.0f, featherY.x, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherY.x > 0.0f) ? smoothstep(0.0f, featherY.x, distanceToFrontFace) : 1.0f;
        }
    }

    // Feather Y Up
    if (outsideBack) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToBackFace / halfWidthY;
            featherValue *= (featherY.y > 0.0f) ? smoothstep(0.0f, featherY.y, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherY.y > 0.0f) ? smoothstep(0.0f, featherY.y, distanceToBackFace) : 1.0f;
        }
    }

    // Feather Z Near
    if (outsideBottom) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToBottomFace / halfWidthZ;
            featherValue *= (featherZ.x > 0.0f) ? smoothstep(0.0f, featherZ.x, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherZ.x > 0.0f) ? smoothstep(0.0f, featherZ.x, distanceToBottomFace) : 1.0f;
        }
    }

    // Feather Z Far
    if (outsideTop) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToTopFace / halfWidthZ;
            featherValue *= (featherZ.y > 0.0f) ? smoothstep(0.0f, featherZ.y, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherZ.y > 0.0f) ? smoothstep(0.0f, featherZ.y, distanceToTopFace) : 1.0f;
        }
    }

    // Falloff Z
    float normalizedDistance = distanceToTopFace / (halfWidthZ * 2.0f);
    float t = clamp(normalizedDistance, 0.0f, 1.0f);

    float intensity = featherValue;
    if (falloff > 0.0f) {
        intensity *= powf(t, falloff);
    }

    float colorIntensity = 1.0f;
    if (colorFalloff > 0.0f) {
        colorIntensity *= powf(t, colorFalloff);
    }

    // Invert
    if (invertToggle) {
        intensity = 1.0f - intensity;
        colorIntensity = intensity;
    }

    // Mix Color
    float3 color = { 0.0f, 0.0f, 0.0f };
    if (lightColorFarToggle) {
        color = mixf3(lightColorFar, lightColorNear, colorIntensity);
    }
    else {
        color = lightColorNear;
    }

    // Saturation
    float3 colorHSV = rgb2hsv(color);
    float3 colorRGB = hsv2rgb({ colorHSV.x, clamp(colorHSV.y * saturation, 0.0f, 1.0f), colorHSV.z });
    float3 colorFinal = mulf3(colorRGB, intensity * intensityMultiplier);

    return colorFinal;

}



__MATH_UTILS_DECL__ float3 pointLightAdvanced(
    int renderMode, float3 camPos, float3 worldPos, float3 normal, float3 lightPos, float radius,
    bool ambientToggle, float ambientFalloff, float ambientIntensity, float ambientSaturation, float3 ambientColorNear, bool ambientColorFarToggle, float3 ambientColorFar, float ambientColorFalloff,
    bool diffuseToggle, float diffuseFalloff, float diffuseIntensity, float diffuseSaturation, float3 diffuseColorNear, bool diffuseColorFarToggle, float3 diffuseColorFar, float diffuseColorFalloff,
    bool specularToggle, float specularSize, float specularFalloff, float specularIntensity, float specularSaturation, float3 specularColorNear, bool specularColorFarToggle, float3 specularColorFar, float specularColorFalloff,
    bool shadowToggle, bool shadowIgnoreAmbientToggle, bool shadowIgnoreDiffuseToggle, bool shadowIgnoreSpecularToggle, float shadows, float3 coloredShadows
) {

    // Inputs

    float3 viewDir = normalizef3(subf3(camPos, worldPos));
    float3 lightDir = normalizef3(subf3(lightPos, worldPos));
    float3 reflectLightDir = normalizef3(reflectf3(invertf3(lightDir), normal));

    // Falloff

    float ambientFalloffIntensity = 1.0f;
    float diffuseFalloffIntensity = 1.0f;
    float specularFalloffIntensity = 1.0f;
    float ambientColorIntensity = 1.0f;
    float diffuseColorIntensity = 1.0f;
    float specularColorIntensity = 1.0f;
    float distance = distancef3(lightPos, worldPos);

    if (distance >= radius) {
        ambientFalloffIntensity *= 0.0f;
        diffuseFalloffIntensity *= 0.0f;
        specularFalloffIntensity *= 0.0f;
        ambientColorIntensity *= 0.0f;
        diffuseColorIntensity *= 0.0f;
        specularColorIntensity *= 0.0f;
    }
    else {
        float t = clamp(distance / radius, 0.0f, 1.0f);
        ambientFalloffIntensity *= powf(1.0f - t, ambientFalloff);
        diffuseFalloffIntensity *= powf(1.0f - t, diffuseFalloff);
        specularFalloffIntensity *= powf(1.0f - t, specularFalloff);
        ambientColorIntensity *= powf(1.0f - t, ambientColorFalloff);
        diffuseColorIntensity *= powf(1.0f - t, diffuseColorFalloff);
        specularColorIntensity *= powf(1.0f - t, specularColorFalloff);
    }

    // Ambient

    float3 ambientColor = { 0.0f, 0.0f, 0.0f };
    float3 ambient = { 0.0f, 0.0f, 0.0f };
    if (renderMode == 1 || renderMode == 2) {
        if (ambientToggle) {
            if (ambientColorFarToggle) {
                ambientColor = mixf3(ambientColorFar, ambientColorNear, ambientColorIntensity);
            }
            else {
                ambientColor = takeXYZf3(ambientColorNear);
            }
            ambientColor = rgbMultiplySaturation(ambientColor, ambientSaturation);
            ambient = mulf3(ambientColor, ambientIntensity * ambientFalloffIntensity);
            if (renderMode == 1 && shadowToggle && !shadowIgnoreAmbientToggle) {
                ambient = mulf3(ambient, shadows);
            }
        }
    }


    // Diffuse

    float3 diffuseColor = { 0.0f, 0.0f, 0.0f };
    float3 diffuse = { 0.0f, 0.0f, 0.0f };
    if (renderMode == 1 || renderMode == 3) {
        if (diffuseToggle) {
            if (diffuseColorFarToggle) {
                diffuseColor = mixf3(diffuseColorFar, diffuseColorNear, diffuseColorIntensity);
            }
            else {
                diffuseColor = takeXYZf3(diffuseColorNear);
            }
            diffuseColor = rgbMultiplySaturation(diffuseColor, diffuseSaturation);
            float diffuseStrength = fmaxf(dotf3(normal, lightDir), 0.0f);
            diffuse = mulf3(diffuseColor, diffuseStrength * diffuseIntensity * diffuseFalloffIntensity);
            if (renderMode == 1 && shadowToggle && !shadowIgnoreDiffuseToggle) {
                diffuse = mulf3(diffuse, shadows);
            }
        }
    }

    // Specular

    float3 specularColor = { 0.0f, 0.0f, 0.0f };
    float3 specular = { 0.0f, 0.0f, 0.0f };
    if (renderMode == 1 || renderMode == 4) {
        if (specularToggle) {
            if (specularColorFarToggle) {
                specularColor = mixf3(specularColorFar, specularColorNear, specularColorIntensity);
            }
            else {
                specularColor = takeXYZf3(specularColorNear);
            }
            specularColor = rgbMultiplySaturation(specularColor, specularSaturation);
            float specularStrength = fmaxf(0.0f, dotf3(viewDir, reflectLightDir));
            specularStrength = powf(specularStrength, specularSize);
            specular = mulf3(specularColor, specularStrength * specularIntensity * specularFalloffIntensity);
            if (renderMode == 1 && shadowToggle && !shadowIgnoreSpecularToggle) {
                specular = mulf3(specular, shadows);
            }
        }
    }

    // Combine

    float3 color = addf3(addf3(addf3(ambient, diffuse), specular), coloredShadows);

    return color;
}

__MATH_UTILS_DECL__ float3 pointLightAmbient(
    float3 worldPos, float3 lightPos, float radius,
    float ambientFalloff, float ambientIntensity, float ambientSaturation, float3 ambientColorNear, bool ambientColorFarToggle, float3 ambientColorFar, float ambientColorFalloff
) {

    // Falloff

    float ambientFalloffIntensity = 1.0f;
    float ambientColorIntensity = 1.0f;
    float distance = distancef3(lightPos, worldPos);
    float t = clamp(distance / radius, 0.0f, 1.0f);
    ambientFalloffIntensity *= powf(1.0f - t, ambientFalloff);
    ambientColorIntensity *= powf(1.0f - t, ambientColorFalloff);

    // Ambient

    float3 ambientColor = { 0.0f, 0.0f, 0.0f };
    float3 ambient = { 0.0f, 0.0f, 0.0f };

    if (ambientColorFarToggle) {
        ambientColor = mixf3(ambientColorFar, ambientColorNear, ambientColorIntensity);
    }
    else {
        ambientColor = takeXYZf3(ambientColorNear);
    }
    ambientColor = rgbMultiplySaturation(ambientColor, ambientSaturation);
    ambient = mulf3(ambientColor, ambientIntensity * ambientFalloffIntensity);

    return ambient;
}

__MATH_UTILS_DECL__ float3 pointLightDiffuse(
    float3 worldPos, float3 lightPos, float3 normal, float radius,
    float diffuseFalloff, float diffuseIntensity, float diffuseSaturation, float3 diffuseColorNear, bool diffuseColorFarToggle, float3 diffuseColorFar, float diffuseColorFalloff
) {

    // Inputs

    float3 lightDir = normalizef3(subf3(lightPos, worldPos));

    // Falloff

    float diffuseFalloffIntensity = 1.0f;
    float diffuseColorIntensity = 1.0f;
    float distance = distancef3(lightPos, worldPos);
    float t = clamp(distance / radius, 0.0f, 1.0f);
    diffuseFalloffIntensity *= powf(1.0f - t, diffuseFalloff);
    diffuseColorIntensity *= powf(1.0f - t, diffuseColorFalloff);
    

    // Diffuse

    float3 diffuseColor = { 0.0f, 0.0f, 0.0f };
    float3 diffuse = { 0.0f, 0.0f, 0.0f };
    if (diffuseColorFarToggle) {
        diffuseColor = mixf3(diffuseColorFar, diffuseColorNear, diffuseColorIntensity);
    }
    else {
        diffuseColor = takeXYZf3(diffuseColorNear);
    }
    diffuseColor = rgbMultiplySaturation(diffuseColor, diffuseSaturation);
    float diffuseStrength = fmaxf(dotf3(normal, lightDir), 0.0f);
    diffuse = mulf3(diffuseColor, diffuseStrength * diffuseIntensity * diffuseFalloffIntensity);

    return diffuse;
}

__MATH_UTILS_DECL__ float3 pointLightSpecular(
    float3 worldPos, float3 lightPos, float3 normal, float3 camPos, float radius,
    float specularSize, float specularFalloff, float specularIntensity, float specularSaturation, float3 specularColorNear, bool specularColorFarToggle, float3 specularColorFar, float specularColorFalloff
) {

    // Inputs

    float3 viewDir = normalizef3(subf3(camPos, worldPos));
    float3 lightDir = normalizef3(subf3(lightPos, worldPos));
    float3 reflectLightDir = normalizef3(reflectf3(invertf3(lightDir), normal));

    // Falloff

    float specularFalloffIntensity = 1.0f;
    float specularColorIntensity = 1.0f;
    float distance = distancef3(lightPos, worldPos);
    float t = clamp(distance / radius, 0.0f, 1.0f);
    specularFalloffIntensity *= powf(1.0f - t, specularFalloff);
    specularColorIntensity *= powf(1.0f - t, specularColorFalloff);
    
    // Specular

    float3 specularColor = { 0.0f, 0.0f, 0.0f };
    float3 specular = { 0.0f, 0.0f, 0.0f };

    if (specularColorFarToggle) {
        specularColor = mixf3(specularColorFar, specularColorNear, specularColorIntensity);
    }
    else {
        specularColor = takeXYZf3(specularColorNear);
    }
    specularColor = rgbMultiplySaturation(specularColor, specularSaturation);
    float specularStrength = fmaxf(0.0f, dotf3(viewDir, reflectLightDir));
    specularStrength = powf(specularStrength, specularSize);
    specular = mulf3(specularColor, specularStrength * specularIntensity * specularFalloffIntensity);

    return specular;
}


__MATH_UTILS_DECL__ float getShadows(
    float3 lightPos, float shadowSampleStep, float shadowImprovedSampleRadius, float shadowMaxLength, float shadowThresholdStart, float shadowThresholdEnd,
    GF_PTR(float4 const) depthSrc, uint2 inXY, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depthIn16f, bool depthBlackIsNear, float depthFar,
    float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float cameraWidth, float cameraHeight, float downsample
) {

    // Local Pos
    float3 lightLocalPos = worldPosToLocalPos(lightPos, camVx, camVy, camVz, camPos);
    float3 curLocalPos = getPosition(
        depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
        true, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );
    float3 rayDir = normalizef3(subf3(lightLocalPos, curLocalPos));
    float rayLength = distancef3(lightLocalPos, curLocalPos);

    // Screen Pos
    float2 lightScreenPos = localPosToScreenPos(lightLocalPos, cameraZoom, cameraWidth, cameraHeight);
    float2 curScreenPos = localPosToScreenPos(curLocalPos, cameraZoom, cameraWidth, cameraHeight);
    float rayScreenLength = distancef2(lightScreenPos, curScreenPos);

    // Adaptive Step
    float eps = 0.0f;
    float3 posA = { 0.0f, 0.0f, 0.0f };
    float3 posB = { 0.0f, 0.0f, 0.0f };
    float2 screenA = { 0.0f, 0.0f };
    float2 screenB = { 0.0f, 0.0f };
    float pixelSpeed = 0.0f;
    float adaptiveStep = 0.0f;

    if (rayScreenLength > shadowImprovedSampleRadius) {
        eps = 0.001f;
        posA = { curLocalPos.x, curLocalPos.y, curLocalPos.z };
        posB = addf3(curLocalPos, mulf3(rayDir, eps));
        screenA = localPosToScreenPos(posA, cameraZoom, cameraWidth, cameraHeight);
        screenB = localPosToScreenPos(posB, cameraZoom, cameraWidth, cameraHeight);
        pixelSpeed = distancef2(screenB, screenA);
        adaptiveStep = fmaxf(eps / pixelSpeed * downsample, shadowSampleStep);
    }
    else {
        adaptiveStep = shadowSampleStep;
    }


    float shadow = 1.0f;
    float minLength = fminf(rayLength, shadowMaxLength);
    float step = 0.0f;


    int2 prevScreenPosInt = { 
        static_cast<int>(screenA.x),
        static_cast<int>(screenA.y)
    };
    while (step < minLength) {

        // Go along ray

        step += adaptiveStep;

        float3 rayLocalPos = addf3(curLocalPos, mulf3(rayDir, step));

        // Local Pos to Screen Pos

        float2 sampleScreenPos = localPosToScreenPos(rayLocalPos, cameraZoom, cameraWidth, cameraHeight);
        int2 sampleScreenPosInt = {
            static_cast<int>(sampleScreenPos.x / downsample + 0.5f),
            static_cast<int>(sampleScreenPos.y / downsample + 0.5f)
        };

        if (rayScreenLength > shadowImprovedSampleRadius) {
            if ((sampleScreenPosInt.x == prevScreenPosInt.x) && (sampleScreenPosInt.y == prevScreenPosInt.y)) {
                continue;
            }
        }

        prevScreenPosInt = { sampleScreenPosInt.x, sampleScreenPosInt.y };

        if (sampleScreenPosInt.x < 0 || sampleScreenPosInt.x > depthWidth || sampleScreenPosInt.y < 0 || sampleScreenPosInt.y > depthHeight) {
            break;
        }

        uint2 curXY = { (unsigned int)(sampleScreenPosInt.x), (unsigned int)(sampleScreenPosInt.y) };

        // Check depths

        float depth = getDepth(depthSrc, curXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar);
        float depthDifference = abs(rayLocalPos.z - depth);
        bool inThreshold = depthDifference > shadowThresholdStart && depthDifference < shadowThresholdEnd;
        if (rayLocalPos.z > depth && inThreshold) {
            shadow = 0.0f;
            break;
        }

        // Recalc Adaptive Step
        if (rayScreenLength > shadowImprovedSampleRadius) {
            posA = { rayLocalPos.x, rayLocalPos.y, rayLocalPos.z };
            posB = addf3(rayLocalPos, mulf3(rayDir, eps));
            screenA = { sampleScreenPos.x, sampleScreenPos.y };
            screenB = localPosToScreenPos(posB, cameraZoom, cameraWidth, cameraHeight);
            pixelSpeed = distancef2(screenB, screenA);
            adaptiveStep = fmaxf(eps / pixelSpeed * downsample, shadowSampleStep);
        }
        else {
            adaptiveStep = shadowSampleStep;
        }

    }
    
    return shadow;

}

__MATH_UTILS_DECL__ float getSoftShadows(
    float shadowSoftness, int shadowSamples,
    float3 lightPos, float shadowSampleStep, float shadowImprovedSampleRadius, float shadowMaxLength, float shadowThresholdStart, float shadowThresholdEnd,
    GF_PTR(float4 const) depthSrc, uint2 inXY, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depthIn16f, bool depthBlackIsNear, float depthFar,
    float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float cameraWidth, float cameraHeight, float downsample
) {

    float shadow = 0.0f;

    if (shadowSoftness != 0.0f) {
        for (int i = 0; i < shadowSamples; i++) {
            // Offsets around the light
            float angle = float(i) * 6.28318530718f / float(shadowSamples);
            float3 offset = { cos(angle) * shadowSoftness, sin(angle) * shadowSoftness, 0.0f };
            float3 offsetLightPos = addf3(lightPos, offset);

            shadow += getShadows(
                offsetLightPos, shadowSampleStep, shadowImprovedSampleRadius, shadowMaxLength, shadowThresholdStart, shadowThresholdEnd,
                depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
                camVx, camVy, camVz, camPos, cameraZoom, cameraWidth, cameraHeight, downsample
            );
        }
        return clamp(shadow / float(shadowSamples), 0.0f, 1.0f);
    }
    else {
        shadow = getShadows(
            lightPos, shadowSampleStep, shadowImprovedSampleRadius, shadowMaxLength, shadowThresholdStart, shadowThresholdEnd,
            depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
            camVx, camVy, camVz, camPos, cameraZoom, cameraWidth, cameraHeight, downsample
        );
        return shadow;
    }

}


__MATH_UTILS_DECL__ float getShadowsDir(
    float3 lightPos, float3 lightLookAt, float shadowSampleStep, float shadowImprovedSampleRadius, float shadowMaxLength, float shadowThresholdStart, float shadowThresholdEnd,
    GF_PTR(float4 const) depthSrc, uint2 inXY, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depthIn16f, bool depthBlackIsNear, float depthFar,
    float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float cameraWidth, float cameraHeight, float downsample
) {

    // Local Pos
    float3 curLocalPos = getPosition(
        depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
        true, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );
    float3 lightPosLocal = worldPosToLocalPos(lightPos, camVx, camVy, camVz, camPos);
    float3 lightLookAtLocal = worldPosToLocalPos(lightLookAt, camVx, camVy, camVz, camPos);
    float3 rayDir = normalizef3(subf3(lightPosLocal, lightLookAtLocal));

    // Adaptive Step
    float eps = 0.0f;
    float3 posA = { 0.0f, 0.0f, 0.0f };
    float3 posB = { 0.0f, 0.0f, 0.0f };
    float2 screenA = { 0.0f, 0.0f };
    float2 screenB = { 0.0f, 0.0f };
    float pixelSpeed = 0.0f;
    float adaptiveStep = 0.0f;
    float shadow = 1.0f;
    float step = 0.0f;

    if (step < shadowImprovedSampleRadius) {
        adaptiveStep = shadowSampleStep;
    }
    else {
        eps = 0.001f;
        posA = { curLocalPos.x, curLocalPos.y, curLocalPos.z };
        posB = addf3(curLocalPos, mulf3(rayDir, eps));
        screenA = localPosToScreenPos(posA, cameraZoom, cameraWidth, cameraHeight);
        screenB = localPosToScreenPos(posB, cameraZoom, cameraWidth, cameraHeight);
        pixelSpeed = distancef2(screenB, screenA);
        adaptiveStep = fmaxf(eps / pixelSpeed * downsample, shadowSampleStep);
    }

    int2 prevScreenPosInt = {
        static_cast<int>(screenA.x),
        static_cast<int>(screenA.y)
    };
    while (step < shadowMaxLength) {

        // Go along ray

        step += adaptiveStep;

        float3 rayLocalPos = addf3(curLocalPos, mulf3(rayDir, step));

        // Local Pos to Screen Pos

        float2 sampleScreenPos = localPosToScreenPos(rayLocalPos, cameraZoom, cameraWidth, cameraHeight);
        int2 sampleScreenPosInt = {
            static_cast<int>(sampleScreenPos.x / downsample + 0.5f),
            static_cast<int>(sampleScreenPos.y / downsample + 0.5f)
        };

        if (step > shadowImprovedSampleRadius) {
            if ((sampleScreenPosInt.x == prevScreenPosInt.x) && (sampleScreenPosInt.y == prevScreenPosInt.y)) {
                continue;
            }
        }

        prevScreenPosInt = { sampleScreenPosInt.x, sampleScreenPosInt.y };

        if (sampleScreenPosInt.x < 0 || sampleScreenPosInt.x > depthWidth || sampleScreenPosInt.y < 0 || sampleScreenPosInt.y > depthHeight) {
            break;
        }

        uint2 curXY = { (unsigned int)(sampleScreenPosInt.x), (unsigned int)(sampleScreenPosInt.y) };

        // Check depths

        float depth = getDepth(depthSrc, curXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar);
        float depthDifference = abs(rayLocalPos.z - depth);
        bool inThreshold = depthDifference > shadowThresholdStart && depthDifference < shadowThresholdEnd;
        if (rayLocalPos.z > depth && inThreshold) {
            shadow = 0.0f;
            break;
        }

        // Recalc Adaptive Step
        if (step < shadowImprovedSampleRadius) {
            adaptiveStep = shadowSampleStep;
        }
        else {
            posA = { rayLocalPos.x, rayLocalPos.y, rayLocalPos.z };
            posB = addf3(rayLocalPos, mulf3(rayDir, eps));
            screenA = { sampleScreenPos.x, sampleScreenPos.y };
            screenB = localPosToScreenPos(posB, cameraZoom, cameraWidth, cameraHeight);
            pixelSpeed = distancef2(screenB, screenA);
            adaptiveStep = fmaxf(eps / pixelSpeed * downsample, shadowSampleStep);
        }

    }

    return shadow;

}


__MATH_UTILS_DECL__ float getSoftShadowsDir(
    float shadowSoftness, int shadowSamples,
    float3 lightPos, float3 lightLookAt, float shadowSampleStep, float shadowImprovedSampleRadius, float shadowMaxLength, float shadowThresholdStart, float shadowThresholdEnd,
    GF_PTR(float4 const) depthSrc, uint2 inXY, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depthIn16f, bool depthBlackIsNear, float depthFar,
    float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float cameraWidth, float cameraHeight, float downsample
) {

    float shadow = 0.0f;

    if (shadowSoftness != 0.0f) {
        for (int i = 0; i < shadowSamples; i++) {
            // Offsets around the light
            float angle = float(i) * 6.28318530718f / float(shadowSamples);
            float3 offset = { cos(angle) * shadowSoftness, sin(angle) * shadowSoftness, 0.0f };
            float3 offsetLightPos = addf3(lightLookAt, offset);

            shadow += getShadowsDir(
                lightPos, offsetLightPos, shadowSampleStep, shadowImprovedSampleRadius, shadowMaxLength, shadowThresholdStart, shadowThresholdEnd,
                depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
                camVx, camVy, camVz, camPos, cameraZoom, cameraWidth, cameraHeight, downsample
            );
        }
        return clamp(shadow / float(shadowSamples), 0.0f, 1.0f);
    }
    else {
        shadow = getShadowsDir(
            lightPos, lightLookAt, shadowSampleStep, shadowImprovedSampleRadius, shadowMaxLength, shadowThresholdStart, shadowThresholdEnd,
            depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
            camVx, camVy, camVz, camPos, cameraZoom, cameraWidth, cameraHeight, downsample
        );
        return shadow;
    }

}

__MATH_UTILS_DECL__ float3 spotAdvanced(
    int renderMode, float3 camPos, float3 worldPos, float3 normal,
    float3 lightPos, float3 vX, float3 vY, float3 vZ, float z, float2 angles, float curvature, float falloff, float feather, float intensityMultiplier, float saturation, float3 lightColorNear, bool lightColorFarToggle, float3 lightColorFar, float colorFalloff, float4 iesChosenPreset1, float4 iesChosenPreset2, float4 iesChosenPreset3, float4 iesChosenPreset4, float4 iesChosenPreset5, float4 iesChosenPreset6,
    bool ambientToggle, float ambientIntensity, float ambientSaturation, float3 ambientColorNear, bool ambientColorFarToggle, float3 ambientColorFar, float ambientColorFalloff,
    bool diffuseToggle, float diffuseIntensity, float diffuseSaturation, float3 diffuseColorNear, bool diffuseColorFarToggle, float3 diffuseColorFar, float diffuseColorFalloff,
    bool specularToggle, float specularSize, float specularIntensity, float specularSaturation, float3 specularColorNear, bool specularColorFarToggle, float3 specularColorFar, float specularColorFalloff,

    bool shadowToggle, bool shadowIgnoreAmbientToggle, bool shadowIgnoreDiffuseToggle, bool shadowIgnoreSpecularToggle, bool shadowClipToLightToggle,
    float shadowSoftness, int shadowSamples,
    float shadowSampleStep, float shadowImprovedSampleRadius, float shadowMaxLength, float shadowThresholdStart, float shadowThresholdEnd,
    float shadowIntensity, float3 shadowColor,
    GF_PTR(float4 const) depthSrc, uint2 inXY, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depthIn16f, bool depthBlackIsNear, float depthFar,
    float3 camVx, float3 camVy, float3 camVz, float cameraZoom, float cameraWidth, float cameraHeight, float downsample
) {

    float3 localPos = subf3(worldPos, lightPos);
    float curZ = dotf3(localPos, vZ);

    if (z == 0.0f || curZ < 0.0f || curZ > z) {
        if (shadowToggle && (renderMode == 1 || renderMode == 5) && !shadowClipToLightToggle) {
            float shadows = getSoftShadows(
                shadowSoftness, shadowSamples,
                lightPos, shadowSampleStep, shadowImprovedSampleRadius, shadowMaxLength, shadowThresholdStart, shadowThresholdEnd,
                depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
                camVx, camVy, camVz, camPos, cameraZoom, cameraWidth, cameraHeight, downsample
            );

            if (renderMode == 1) {
                shadows = mix(shadows, 1.0f, 1.0f - shadowIntensity);
                float3 shadowsColored = mulf3(shadowColor, (1.0f - shadows));
                shadows = clamp(shadows, 0.0f, 1.0f);
                return shadowsColored;
            }
            else {
                return { shadows, shadows, shadows };
            }

        }
        else {
            return { 0.0f, 0.0f, 0.0f };
        }
    }

    // Inputs for diffuse and specular

    float3 lightDir = normalizef3(subf3(lightPos, worldPos));
    float3 viewDir = normalizef3(subf3(camPos, worldPos));
    float3 reflectLightDir = normalizef3(reflectf3(invertf3(lightDir), normal));

    // Spot

    float intensity = 1.0f;

    float x = dotf3(localPos, vX);
    float y = dotf3(localPos, vY);

    float t = clamp(curZ / z, 0.0f, 1.0f);

    float endX = z * tan(radians(angles.x));
    float endY = z * tan(radians(angles.y));
    float curEndX = endX * powf(t, curvature);
    float curEndY = endY * powf(t, curvature);

    // Ellipse check

    float inside = (x * x) / (curEndX * curEndX) +
        (y * y) / (curEndY * curEndY);

    // IES

    float2 curXY = { x / curEndX,
                     y / curEndY };
    float curXYlength = lengthf2(curXY);

    intensity *= ramp(iesChosenPreset1, iesChosenPreset2, iesChosenPreset3, iesChosenPreset4, iesChosenPreset5, iesChosenPreset6, curXYlength).x;

    // Feather

    if (feather == 0.0f) {
        intensity *= step(inside, 1.0f);
    }
    else {
        float fOut = 1.0f;
        float fIn = 1.0f - feather;
        intensity *= smoothstep(fOut, fIn, inside);
    }

    // Falloff

    if (falloff > 0.0f) {
        intensity *= powf(1.0f - t, falloff);
    }

    // Ambient

    float ambientColorIntensity = 1.0f;
    float3 ambientColor = { 0.0f, 0.0f, 0.0f };
    float3 ambient = { 0.0f, 0.0f, 0.0f };
    if (ambientToggle && (renderMode == 1 || renderMode == 2)) {
        if (ambientColorFalloff > 0.0f) {
            ambientColorIntensity *= powf(1.0f - t, ambientColorFalloff);
        }

        if (ambientColorFarToggle) {
            ambientColor = mixf3(ambientColorFar, ambientColorNear, ambientColorIntensity);
        }
        else {
            ambientColor = ambientColorNear;
        }

        ambientColor = rgbMultiplySaturation(ambientColor, ambientSaturation);

        ambient = mulf3(ambientColor, ambientIntensity * intensity);
    }


    // Diffuse

    float diffuseColorIntensity = 1.0f;
    float3 diffuseColor = { 0.0f, 0.0f, 0.0f };
    float3 diffuse = { 0.0f, 0.0f, 0.0f };
    if (diffuseToggle && (renderMode == 1 || renderMode == 3)) {
        if (diffuseColorFalloff > 0.0f) {
            diffuseColorIntensity *= powf(1.0f - t, diffuseColorFalloff);
        }

        if (diffuseColorFarToggle) {
            diffuseColor = mixf3(diffuseColorFar, diffuseColorNear, diffuseColorIntensity);
        }
        else {
            diffuseColor = diffuseColorNear;
        }

        diffuseColor = rgbMultiplySaturation(diffuseColor, diffuseSaturation);

        float diffuseStrength = fmaxf(dotf3(normal, lightDir), 0.0f);
        diffuse = mulf3(diffuseColor, diffuseStrength * diffuseIntensity * intensity);
    }

    // Specular 

    float specularColorIntensity = 1.0f;
    float3 specular = { 0.0f, 0.0f, 0.0f };
    if (specularToggle && (renderMode == 1 || renderMode == 4)) {
        if (specularColorFalloff > 0.0f) {
            specularColorIntensity *= powf(1.0f - t, specularColorFalloff);
        }

        float3 specularColor = { 0.0f, 0.0f, 0.0f };
        if (specularColorFarToggle) {
            specularColor = mixf3(specularColorFar, specularColorNear, specularColorIntensity);
        }
        else {
            specularColor = specularColorNear;
        }

        specularColor = rgbMultiplySaturation(specularColor, specularSaturation);

        float specularStrength = fmaxf(0.0f, dotf3(viewDir, reflectLightDir));
        specularStrength = powf(specularStrength, specularSize);
        specular = mulf3(specularColor, specularStrength * specularIntensity * intensity);
    }

    // Shadow

    float shadows = 1.0f;
    float3 shadowsColored = { 0.0f, 0.0f, 0.0f };
    if (shadowToggle && (renderMode == 1 || renderMode == 5) && intensity > 0.0) {
        shadows = getSoftShadows(
            shadowSoftness, shadowSamples,
            lightPos, shadowSampleStep, shadowImprovedSampleRadius, shadowMaxLength, shadowThresholdStart, shadowThresholdEnd,
            depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
            camVx, camVy, camVz, camPos, cameraZoom, cameraWidth, cameraHeight, downsample
        );

        if (renderMode == 5) {
            return { shadows, shadows, shadows };
        }

        shadows = mix(shadows, 1.0f, 1.0f - shadowIntensity);
        shadowsColored = mulf3(shadowColor, (1.0f - shadows));
        shadows = clamp(shadows, 0.0f, 1.0f);
    
        if (!shadowIgnoreAmbientToggle) {
            ambient = mulf3(ambient, shadows);
        }
        if (!shadowIgnoreDiffuseToggle) {
            diffuse = mulf3(diffuse, shadows);
        }
        if (!shadowIgnoreSpecularToggle) {
            specular = mulf3(specular, shadows);
        }

    }

    // Result

    float3 color = addf3(addf3(addf3(ambient, diffuse), specular), shadowsColored);

    return color;

}

__MATH_UTILS_DECL__ float3 rectAdvanced(
    int renderMode, float3 camPos, float3 worldPos, float3 normal,

    float3 pos1, float3 vX1, float3 vY1, float3 vZ1, float3 res1, float3 scale1, float3 pos2, float3 vX2, float3 vY2, float3 vZ2, float3 res2, float3 scale2, float2 featherX, float2 featherY, float2 featherZ, bool featherNormalized, float falloff,

    bool ambientToggle, float ambientIntensity, float ambientSaturation, float3 ambientColorNear, bool ambientColorFarToggle, float3 ambientColorFar, float ambientColorFalloff,
    bool diffuseToggle, float diffuseIntensity, float diffuseSaturation, float3 diffuseColorNear, bool diffuseColorFarToggle, float3 diffuseColorFar, float diffuseColorFalloff,
    bool specularToggle, float specularSize, float specularIntensity, float specularSaturation, float3 specularColorNear, bool specularColorFarToggle, float3 specularColorFar, float specularColorFalloff,

    bool shadowToggle, bool shadowIgnoreAmbientToggle, bool shadowIgnoreDiffuseToggle, bool shadowIgnoreSpecularToggle, bool shadowClipToLightToggle,
    float shadowSoftness, int shadowSamples,
    float shadowSampleStep, float shadowImprovedSampleRadius, float shadowMaxLength, float shadowThresholdStart, float shadowThresholdEnd,
    float shadowIntensity, float3 shadowColor,
    GF_PTR(float4 const) depthSrc, uint2 inXY, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depthIn16f, bool depthBlackIsNear, float depthFar,
    float3 camVx, float3 camVy, float3 camVz, float cameraZoom, float cameraWidth, float cameraHeight, float downsample

) {

    // Sizes
    float sizeX1 = res1.x * abs(scale1.x);
    float sizeY1 = res1.y * abs(scale1.y);
    float sizeX2 = res2.x * abs(scale2.x);
    float sizeY2 = res2.y * abs(scale2.y);

    // Vertices A
    float3 TL1 = addf3(pos1, mulf3(vX1, sizeX1));
    float3 TR1 = writef3(pos1);
    float3 BR1 = subf3(pos1, mulf3(vY1, sizeY1));
    float3 BL1 = addf3(pos1, subf3(mulf3(vX1, sizeX1), mulf3(vY1, sizeY1)));

    // Inputs for diffuse and specular
    float3 lightPos = addf3(pos1, subf3(mulf3(vX1, sizeX1 * 0.5f), mulf3(vY1, sizeY1 * 0.5f))); // Center
    float3 lightDir = normalizef3(subf3(lightPos, worldPos));
    float3 viewDir = normalizef3(subf3(camPos, worldPos));
    float3 reflectLightDir = normalizef3(reflectf3(invertf3(lightDir), normal));


    // Vertices B Temp
    float3 TL2_Temp = addf3(pos2, mulf3(vX2, sizeX2));
    float3 TR2_Temp = writef3(pos2);
    float3 BR2_Temp = subf3(pos2, mulf3(vY2, sizeY2));
    float3 BL2_Temp = addf3(pos2, subf3(mulf3(vX2, sizeX2), mulf3(vY2, sizeY2)));


    // ----- Match vertices -----
    // Pattern that has the smallest total distance is the right one

    float3 rect1_vertices[4] = { TL1, TR1, BR1, BL1 };

    // Predefine 8 possible patterns
    float3 config_patterns[32] = {
        // config 0: BL2, TL2, TR2, BR2
        BL2_Temp, TL2_Temp, TR2_Temp, BR2_Temp,
        // config 1: BL2, BR2, TR2, TL2
        BL2_Temp, BR2_Temp, TR2_Temp, TL2_Temp,
        // config 2: BR2, BL2, TL2, TR2
        BR2_Temp, BL2_Temp, TL2_Temp, TR2_Temp,
        // config 3: BR2, TR2, TL2, BL2
        BR2_Temp, TR2_Temp, TL2_Temp, BL2_Temp,
        // config 4: TL2, BL2, BR2, TR2
        TL2_Temp, BL2_Temp, BR2_Temp, TR2_Temp,
        // config 5: TL2, TR2, BR2, BL2
        TL2_Temp, TR2_Temp, BR2_Temp, BL2_Temp,
        // config 6: TR2, TL2, BL2, BR2
        TR2_Temp, TL2_Temp, BL2_Temp, BR2_Temp,
        // config 7: TR2, BR2, BL2, TL2
        TR2_Temp, BR2_Temp, BL2_Temp, TL2_Temp
    };

    float min_distance = 1e6;
    int best_config = 0;

    // Check what pattern has the smallest total distance
    for (int config = 0; config < 8; config++) {
        float total_dist = 0.0;
        for (int j = 0; j < 4; j++) {
            float3 target_vertex = config_patterns[config * 4 + j];  // j-th vertex in config
            total_dist += distancef3(rect1_vertices[j], target_vertex);
        }

        if (total_dist < min_distance) {
            min_distance = total_dist;
            best_config = config;
        }
    }

    // Apply the correct pattern
    float3 TL2 = config_patterns[best_config * 4 + 0];
    float3 TR2 = config_patterns[best_config * 4 + 1];
    float3 BR2 = config_patterns[best_config * 4 + 2];
    float3 BL2 = config_patterns[best_config * 4 + 3];



    // ----- Rect Light Projection -----

    // 6 faces x 3 vertices
    float3 faceVertices[18] = {
        // Face 0: Bottom (pos1) — BL1, BR1, TL1
        BL1, BR1, TL1,

        // Face 1: Top (pos2) — BL2, TL2, BR2
        BL2, TL2, BR2,

        // Face 2: Front — BL1, BR1, BR2
        BL1, BR1, BR2,

        // Face 3: Right — BR1, TR1, TR2
        BR1, TR1, TR2,

        // Face 4: Back — TR1, TL1, TL2
        TR1, TL1, TL2,

        // Face 5: Left — TL1, BL1, BL2
        TL1, BL1, BL2
    };

    float3 center = {
        (BL1.x + BR1.x + TL1.x + TR1.x + BL2.x + BR2.x + TL2.x + TR2.x) * 0.125f,
        (BL1.y + BR1.y + TL1.y + TR1.y + BL2.y + BR2.y + TL2.y + TR2.y) * 0.125f,
        (BL1.z + BR1.z + TL1.z + TR1.z + BL2.z + BR2.z + TL2.z + TR2.z) * 0.125f
    };

    float distanceToLeftFace = 1e6;   // Face 5 (Left) - X
    float distanceToRightFace = 1e6;  // Face 3 (Right) - X
    float distanceToFrontFace = 1e6;  // Face 2 (Front) - Y
    float distanceToBackFace = 1e6;   // Face 4 (Back) - Y
    float distanceToBottomFace = 1e6; // Face 0 (Bottom) - Z
    float distanceToTopFace = 1e6;    // Face 1 (Top) - Z

    bool outsideLeft = false, outsideRight = false;
    bool outsideFront = false, outsideBack = false;
    bool outsideBottom = false, outsideTop = false;

    for (int i = 0; i < 6; i++) {
        float3 v1 = faceVertices[i * 3 + 0];
        float3 v2 = faceVertices[i * 3 + 1];
        float3 v3 = faceVertices[i * 3 + 2];

        float3 edge1 = subf3(v2, v1);
        float3 edge2 = subf3(v3, v1);
        float3 normal = normalizef3(cross(edge1, edge2));

        float3 toPoint = subf3(worldPos, v1);
        float3 toCenter = subf3(center, v1);

        float distanceToPlane = dotf3(normal, toPoint);
        float sideCenter = dotf3(normal, toCenter);

        bool outsideThisFace = (distanceToPlane * sideCenter < 0.0f);

        if (i == 0) { // Bottom face
            distanceToBottomFace = abs(distanceToPlane);
            outsideBottom = outsideThisFace;
        }
        else if (i == 1) { // Top face
            distanceToTopFace = abs(distanceToPlane);
            outsideTop = outsideThisFace;
        }
        else if (i == 2) { // Front face
            distanceToFrontFace = abs(distanceToPlane);
            outsideFront = outsideThisFace;
        }
        else if (i == 3) { // Right face 
            distanceToRightFace = abs(distanceToPlane);
            outsideRight = outsideThisFace;
        }
        else if (i == 4) { // Back face 
            distanceToBackFace = abs(distanceToPlane);
            outsideBack = outsideThisFace;
        }
        else if (i == 5) { // Left face 
            distanceToLeftFace = abs(distanceToPlane);
            outsideLeft = outsideThisFace;
        }
    }

    // Parameters for feather normalization
    float halfWidthX = (sizeX1 + sizeX2) * 0.5f;
    float halfWidthY = (sizeY1 + sizeY2) * 0.5f;
    float halfWidthZ = distancef3(pos2, pos1) * 0.5f;

    float featherValue = 1.0f;

    // Feather X Right
    if (outsideLeft) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToLeftFace / halfWidthX;
            featherValue *= (featherX.x > 0.0f) ? smoothstep(0.0f, featherX.x, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherX.x > 0.0f) ? smoothstep(0.0f, featherX.x, distanceToLeftFace) : 1.0f;
        }
    }

    // Feather X Left
    if (outsideRight) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToRightFace / halfWidthX;
            featherValue *= (featherX.y > 0.0f) ? smoothstep(0.0f, featherX.y, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherX.y > 0.0f) ? smoothstep(0.0f, featherX.y, distanceToRightFace) : 1.0f;
        }
    }

    // Feather Y Down
    if (outsideFront) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToFrontFace / halfWidthY;
            featherValue *= (featherY.x > 0.0f) ? smoothstep(0.0f, featherY.x, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherY.x > 0.0f) ? smoothstep(0.0f, featherY.x, distanceToFrontFace) : 1.0f;
        }
    }

    // Feather Y Up
    if (outsideBack) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToBackFace / halfWidthY;
            featherValue *= (featherY.y > 0.0f) ? smoothstep(0.0f, featherY.y, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherY.y > 0.0f) ? smoothstep(0.0f, featherY.y, distanceToBackFace) : 1.0f;
        }
    }

    // Feather Z Near
    if (outsideBottom) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToBottomFace / halfWidthZ;
            featherValue *= (featherZ.x > 0.0f) ? smoothstep(0.0f, featherZ.x, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherZ.x > 0.0f) ? smoothstep(0.0f, featherZ.x, distanceToBottomFace) : 1.0f;
        }
    }

    // Feather Z Far
    if (outsideTop) {
        featherValue = 0.0f;
    }
    else {
        if (featherNormalized) {
            float normalizedDistance = distanceToTopFace / halfWidthZ;
            featherValue *= (featherZ.y > 0.0f) ? smoothstep(0.0f, featherZ.y, normalizedDistance) : 1.0f;
        }
        else {
            featherValue *= (featherZ.y > 0.0f) ? smoothstep(0.0f, featherZ.y, distanceToTopFace) : 1.0f;
        }
    }

    // Falloff Z
    float normalizedDistance = distanceToTopFace / (halfWidthZ * 2.0f);
    float t = clamp(normalizedDistance, 0.0f, 1.0f);

    float intensity = featherValue;
    if (falloff > 0.0f) {
        intensity *= powf(t, falloff);
    }

    // Ambient

    float ambientColorIntensity = 1.0f;
    float3 ambientColor = { 0.0f, 0.0f, 0.0f };
    float3 ambient = { 0.0f, 0.0f, 0.0f };
    if (ambientToggle && (renderMode == 1 || renderMode == 2)) {
        if (ambientColorFalloff > 0.0f) {
            ambientColorIntensity *= powf(1.0f - t, ambientColorFalloff);
        }

        if (ambientColorFarToggle) {
            ambientColor = mixf3(ambientColorFar, ambientColorNear, ambientColorIntensity);
        }
        else {
            ambientColor = ambientColorNear;
        }

        ambientColor = rgbMultiplySaturation(ambientColor, ambientSaturation);

        ambient = mulf3(ambientColor, ambientIntensity * intensity);
    }

    // Diffuse

    float diffuseColorIntensity = 1.0f;
    float3 diffuseColor = { 0.0f, 0.0f, 0.0f };
    float3 diffuse = { 0.0f, 0.0f, 0.0f };
    if (diffuseToggle && (renderMode == 1 || renderMode == 3)) {
        if (diffuseColorFalloff > 0.0f) {
            diffuseColorIntensity *= powf(1.0f - t, diffuseColorFalloff);
        }

        if (diffuseColorFarToggle) {
            diffuseColor = mixf3(diffuseColorFar, diffuseColorNear, diffuseColorIntensity);
        }
        else {
            diffuseColor = diffuseColorNear;
        }

        diffuseColor = rgbMultiplySaturation(diffuseColor, diffuseSaturation);

        float diffuseStrength = fmaxf(dotf3(normal, lightDir), 0.0f);
        diffuse = mulf3(diffuseColor, diffuseStrength * diffuseIntensity * intensity);
    }

    // Specular 

    float specularColorIntensity = 1.0f;
    float3 specular = { 0.0f, 0.0f, 0.0f };
    if (specularToggle && (renderMode == 1 || renderMode == 4)) {
        if (specularColorFalloff > 0.0f) {
            specularColorIntensity *= powf(1.0f - t, specularColorFalloff);
        }

        float3 specularColor = { 0.0f, 0.0f, 0.0f };
        if (specularColorFarToggle) {
            specularColor = mixf3(specularColorFar, specularColorNear, specularColorIntensity);
        }
        else {
            specularColor = specularColorNear;
        }

        specularColor = rgbMultiplySaturation(specularColor, specularSaturation);

        float specularStrength = fmaxf(0.0f, dotf3(viewDir, reflectLightDir));
        specularStrength = powf(specularStrength, specularSize);
        specular = mulf3(specularColor, specularStrength * specularIntensity * intensity);
    }

    // Shadow

    float shadows = 1.0f;
    float3 shadowsColored = { 0.0f, 0.0f, 0.0f };
    if (shadowToggle && (renderMode == 1 || renderMode == 5)) {

        if (shadowClipToLightToggle && intensity > 0.0f) {
            shadows = getSoftShadows(
                shadowSoftness, shadowSamples,
                lightPos, shadowSampleStep, shadowImprovedSampleRadius, shadowMaxLength, shadowThresholdStart, shadowThresholdEnd,
                depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
                camVx, camVy, camVz, camPos, cameraZoom, cameraWidth, cameraHeight, downsample
            );
        }
        else if (!shadowClipToLightToggle) {
            shadows = getSoftShadows(
                shadowSoftness, shadowSamples,
                lightPos, shadowSampleStep, shadowImprovedSampleRadius, shadowMaxLength, shadowThresholdStart, shadowThresholdEnd,
                depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
                camVx, camVy, camVz, camPos, cameraZoom, cameraWidth, cameraHeight, downsample
            );
        }


        if (renderMode == 5) {
            return { shadows, shadows, shadows };
        }

        shadows = mix(shadows, 1.0f, 1.0f - shadowIntensity);
        shadowsColored = mulf3(shadowColor, (1.0f - shadows));
        shadows = clamp(shadows, 0.0f, 1.0f);

        if (!shadowIgnoreAmbientToggle) {
            ambient = mulf3(ambient, shadows);
        }
        if (!shadowIgnoreDiffuseToggle) {
            diffuse = mulf3(diffuse, shadows);
        }
        if (!shadowIgnoreSpecularToggle) {
            specular = mulf3(specular, shadows);
        }

    }

    // Result

    float3 color = addf3(addf3(addf3(ambient, diffuse), specular), shadowsColored);

    return color;

}

///////////////////////
// Ambient Occlusion //
///////////////////////

__MATH_UTILS_DECL__ float2 getRandom(float2 uv) {

    float x = fract(sin(dotf2(uv, { 127.1f, 311.7f })) * 43758.5453123f);
    float y = fract(sin(dotf2(uv, { 311.7f, 127.1f })) * 43758.5453123f);

    return { x, y }; // [0, 1]

}

__MATH_UTILS_DECL__ float2 generateKernel(int index, int samples) {

    float a = float(index) * 0.85443192f;
    float r = float(index) / float(samples);
    float2 dir = { cos(a), sin(a) };

    return mulf2(dir, r);

}

__MATH_UTILS_DECL__ float doAmbientOcclusion(
    // Calculated
    float2 sampleDir,

    // Ambient Occlusion Inputs
    float intensity, float threshold, float xyRadius, float zRadius, int samples,

    // Coords
    uint2 inXY,

    // Depth
    GF_PTR(float4 const) depthSrc, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depth16f, bool depthBlackIsNear, float depthFar,

    // Camera
    float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float downsample,

    // Normal Pass
    GF_PTR(float4 const) normalSrc, int normalPitch, unsigned int normalWidth, unsigned int normalHeight, int normal16f
) {

    int2 offset = { (int)(sampleDir.x * xyRadius / downsample), (int)(sampleDir.y * xyRadius / downsample) };
    int2 sampleXYint = { (int)inXY.x + offset.x, (int)inXY.y + offset.y };
    if (sampleXYint.x < 0 || sampleXYint.x >= depthWidth || sampleXYint.y < 0 || sampleXYint.y >= depthHeight) {
        return 0.0f;
    }
    uint2 sampleXY = { (unsigned int)sampleXYint.x, (unsigned int)sampleXYint.y };

    float3 p = getPosition(
        depthSrc, inXY, depthPitch, depthWidth, depthHeight, depth16f, depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );

    float3 pSample = getPosition(
        depthSrc, sampleXY, depthPitch, depthWidth, depthHeight, depth16f, depthBlackIsNear, depthFar,
        false, camVx, camVy, camVz, camPos, cameraZoom, downsample
    );

    float3 diff = subf3(pSample, p);
    float3 v = normalizef3(diff);
    float l = lengthf3(diff);

    float3 n = takeXYZf4(samplePixel(normalSrc, inXY, normalPitch, normalWidth, normalHeight, normal16f)); //normalized normal
    n = subf3(mulf3(n, 2.0f), { 1.0f, 1.0f, 1.0f }); //unnormalized normal

    if (dotf3(n, v) < 0.0f) {
        return 0.0f;
    }

    float ao = fmaxf(0.0f, dotf3(n, v) - threshold);

    ao *= 1.0f - fminf(l / zRadius, 1.0f);
    return ao;

}


__MATH_UTILS_DECL__ float ambientOcclusion(
    // Ambient Occlusion Inputs
    float intensity, float threshold, float xyRadius, float zRadius, int samples,

    // Coords
    uint2 inXY,

    // Depth
    GF_PTR(float4 const) depthSrc, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depth16f, bool depthBlackIsNear, float depthFar,

    // Camera
    float3 camVx, float3 camVy, float3 camVz, float3 camPos, float cameraZoom, float downsample,

    // Normal Pass
    GF_PTR(float4 const) normalSrc, int normalPitch, unsigned int normalWidth, unsigned int normalHeight, int normal16f
) {

    float2 uv = { (float)(inXY.x) / (float)(depthWidth), (float)(inXY.y) / (float)(depthHeight) };
    float2 rand = getRandom(uv);

    float ssao = 0.0f;

    // Rand angle [0, 360] degrees
    float angle = rand.x * 6.28318530718f;

    for (int i = 0; i < samples; i++) {
        float2 sampleDir = generateKernel(i, samples);

        sampleDir = { cos(angle) * sampleDir.x - sin(angle) * sampleDir.y,
                      sin(angle) * sampleDir.x + cos(angle) * sampleDir.y };

        ssao += doAmbientOcclusion(
            // Calculated
            sampleDir,

            // Ambient Occlusion Inputs
            intensity, threshold, xyRadius, zRadius, samples,

            // Coords
            inXY,

            // Depth
            depthSrc, depthPitch, depthWidth, depthHeight, depth16f, depthBlackIsNear, depthFar,

            // Camera
            camVx, camVy, camVz, camPos, cameraZoom, downsample,

            // Normal Pass
            normalSrc, normalPitch, normalWidth, normalHeight, normal16f
        );
    }

    ssao /= float(samples);
    ssao = clamp(1.0f - ssao * intensity, 0.0f, 1.0f);

    return ssao;

}



__MATH_UTILS_DECL__ float3 rimLight(float3 normal, float3 rimLightPosition, float3 rimLightLookAtPosition, float rimStart, float rimEnd, float rimIntensity, float rimSaturation, float3 rimColorNear, bool rimColorFarToggle, float3 rimColorFar, float rimColorFalloff) {

    // Case 1: = =
    if (rimStart == rimEnd) {
        return { 0.0f, 0.0f, 0.0f };
    }

    // Input

    float3 lightDir = normalizef3(subf3(rimLightLookAtPosition, rimLightPosition));

    // Linear mask

    float rimLightIntensity = dotf3(lightDir, normal); // -1 to 1

    // Case 2: + -
    if (rimStart > rimEnd) {
        float rimStartTemp = rimStart;
        float rimEndTemp = rimEnd;
        rimStart = rimEndTemp;
        rimEnd = rimStartTemp;
    }

    // Case 3: - +
    if (rimStart < 0.0f && rimEnd > 0.0f) {
        if (rimLightIntensity < 0.0f) {
            rimLightIntensity = smoothstep(0.0f, -rimStart, -rimLightIntensity);
        }
        else {
            rimLightIntensity = smoothstep(0.0f, rimEnd, rimLightIntensity);
        }
    }

    // Case 4: += +
    if (rimStart >= 0.0f && rimEnd > 0.0f) {
        rimLightIntensity = smoothstep(rimStart, rimEnd, rimLightIntensity);
    }

    // Case 5: - -=
    if (rimStart < 0.0f && rimEnd <= 0.0f) {
        rimLightIntensity = smoothstep(-rimEnd, -rimStart, -rimLightIntensity);
    }

    // Invert
    rimLightIntensity = 1.0f - rimLightIntensity;

    // Color
    float3 rimColor = { 0.0f, 0.0f, 0.0f };
    if (rimColorFarToggle) {
        float rimColorIntensity = powf(rimLightIntensity, rimColorFalloff);
        rimColor = mixf3(rimColorFar, rimColorNear, rimColorIntensity);
    }
    else {
        rimColor = { rimColorNear.x, rimColorNear.y, rimColorNear.z };
    }
    rimColor = rgbMultiplySaturation(rimColor, rimSaturation);

    // Result

    float3 rim = mulf3(rimColor, rimLightIntensity * rimIntensity);

    return rim;

}

__MATH_UTILS_DECL__ float3 dirLight(
    int renderMode, float3 camPos, float3 worldPos, float3 normal,

    float3 pos1, float3 pos2,

    bool ambientToggle, float ambientIntensity, float ambientSaturation, float3 ambientColor,
    bool diffuseToggle, float diffuseIntensity, float diffuseSaturation, float3 diffuseColor, 
    bool specularToggle, float specularSize, float specularIntensity, float specularSaturation, float3 specularColor,

    bool shadowToggle, bool shadowIgnoreAmbientToggle, bool shadowIgnoreDiffuseToggle, bool shadowIgnoreSpecularToggle,
    float shadowSoftness, int shadowSamples,
    float shadowSampleStep, float shadowImprovedSampleRadius, float shadowMaxLength, float shadowThresholdStart, float shadowThresholdEnd,
    float shadowIntensity, float3 shadowColor,
    GF_PTR(float4 const) depthSrc, uint2 inXY, int depthPitch, unsigned int depthWidth, unsigned int depthHeight, int depthIn16f, bool depthBlackIsNear, float depthFar,
    float3 camVx, float3 camVy, float3 camVz, float cameraZoom, float cameraWidth, float cameraHeight, float downsample

) {

    // Inputs
    float3 lightDir = normalizef3(subf3(pos1, pos2));
    float3 reflectLightDir = normalizef3(reflectf3(invertf3(lightDir), normal));
    //float3 worldPos = getPositionNoDepth(uv); 
    float3 viewDir = normalizef3(subf3(camPos, worldPos));

    // Ambient

    float ambientColorIntensity = 1.0f;
    float3 ambient = { 0.0f, 0.0f, 0.0f };
    if (ambientToggle && (renderMode == 1 || renderMode == 2)) {
        ambientColor = rgbMultiplySaturation(ambientColor, ambientSaturation);
        ambient = mulf3(ambientColor, ambientIntensity);
    }

    // Diffuse

    float diffuseColorIntensity = 1.0f;
    float3 diffuse = { 0.0f, 0.0f, 0.0f };
    if (diffuseToggle && (renderMode == 1 || renderMode == 3)) {
        float diffuseStrength = fmaxf(dotf3(normal, lightDir), 0.0f);
        diffuseColor = rgbMultiplySaturation(diffuseColor, diffuseSaturation);
        diffuse = mulf3(diffuseColor, diffuseStrength * diffuseIntensity);
    }

    // Specular 

    float specularColorIntensity = 1.0f;
    float3 specular = { 0.0f, 0.0f, 0.0f };
    if (specularToggle && (renderMode == 1 || renderMode == 4)) {
        float specularStrength = fmaxf(0.0f, dotf3(viewDir, reflectLightDir));
        specularStrength = powf(specularStrength, specularSize);
        specularColor = rgbMultiplySaturation(specularColor, specularSaturation);
        specular = mulf3(specularColor, specularStrength * specularIntensity);
    }

    // Shadow

    float shadows = 1.0f;
    float3 shadowsColored = { 0.0f, 0.0f, 0.0f };
    if (shadowToggle && (renderMode == 1 || renderMode == 5)) {

        shadows = getSoftShadowsDir(
            shadowSoftness, shadowSamples,
            pos1, pos2, shadowSampleStep, shadowImprovedSampleRadius, shadowMaxLength, shadowThresholdStart, shadowThresholdEnd,
            depthSrc, inXY, depthPitch, depthWidth, depthHeight, depthIn16f, depthBlackIsNear, depthFar,
            camVx, camVy, camVz, camPos, cameraZoom, cameraWidth, cameraHeight, downsample
        );

        if (renderMode == 5) {
            return { shadows, shadows, shadows };
        }

        shadows = mix(shadows, 1.0f, 1.0f - shadowIntensity);
        shadowsColored = mulf3(shadowColor, (1.0f - shadows));
        shadows = clamp(shadows, 0.0f, 1.0f);

        if (!shadowIgnoreAmbientToggle) {
            ambient = mulf3(ambient, shadows);
        }
        if (!shadowIgnoreDiffuseToggle) {
            diffuse = mulf3(diffuse, shadows);
        }
        if (!shadowIgnoreSpecularToggle) {
            specular = mulf3(specular, shadows);
        }

    }

    // Result

    float3 color = addf3(addf3(addf3(ambient, diffuse), specular), shadowsColored);

    return color;

}


__MATH_UTILS_DECL__ float3 normalRemap(float3 normal, bool inputIsNormalizedToggle, int x, int y, int z, bool normalizeOutputToggle) {

    if (inputIsNormalizedToggle) {
        normal = subf3(mulf3(normal, 2.0f), { 1.0f, 1.0f, 1.0f });
    }

    float3 remappedNormal = { 0.0f, 0.0f, 0.0f };

    if (x == 1) { remappedNormal.x = normal.x; }
    if (x == 2) { remappedNormal.x = normal.y; }
    if (x == 3) { remappedNormal.x = normal.z; }
    if (x == 4) { remappedNormal.x = -normal.x; }
    if (x == 5) { remappedNormal.x = -normal.y; }
    if (x == 6) { remappedNormal.x = -normal.z; }

    if (y == 1) { remappedNormal.y = normal.x; }
    if (y == 2) { remappedNormal.y = normal.y; }
    if (y == 3) { remappedNormal.y = normal.z; }
    if (y == 4) { remappedNormal.y = -normal.x; }
    if (y == 5) { remappedNormal.y = -normal.y; }
    if (y == 6) { remappedNormal.y = -normal.z; }

    if (z == 1) { remappedNormal.z = normal.x; }
    if (z == 2) { remappedNormal.z = normal.y; }
    if (z == 3) { remappedNormal.z = normal.z; }
    if (z == 4) { remappedNormal.z = -normal.x; }
    if (z == 5) { remappedNormal.z = -normal.y; }
    if (z == 6) { remappedNormal.z = -normal.z; }

    if (normalizeOutputToggle) { 
        remappedNormal = addf3(mulf3(remappedNormal, 0.5f), { 0.5f, 0.5f, 0.5f });
    }

    return remappedNormal;

}


#endif /*MATH_UTILS_H*/