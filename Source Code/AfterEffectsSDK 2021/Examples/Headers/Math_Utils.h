#pragma once
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#ifndef __MATH_UTILS_DECL__
#define __MATH_UTILS_DECL__ __inline__ __device__ __host__
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


__MATH_UTILS_DECL__ float4 writef4(float4 a) {
    return { a.x, a.y, a.z, a.w };
}

__MATH_UTILS_DECL__ float3 writef3(float3 a) {
    return { a.x, a.y, a.z };
}

__MATH_UTILS_DECL__ float2 writef2(float2 a) {
    return { a.x, a.y };
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

    return {
        a.x / w,
        a.y / w,
        a.z / w,
        a.w / w
    };

}

__MATH_UTILS_DECL__ float3 divf3(float3 a, float w) {

    return {
        a.x / w,
        a.y / w,
        a.z / w
    };

}

__MATH_UTILS_DECL__ float2 divf2(float2 a, float w) {

    return {
        a.x / w,
        a.y / w
    };

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

// Unmult

__MATH_UTILS_DECL__ float luminance(float3 color) {

    return dotf3(color, { 0.299f, 0.587f, 0.114f });
}

__MATH_UTILS_DECL__ float4 screen(float4 color) {

    float luma = luminance({ color.x, color.y, color.z });

    float alpha = clamp(luma, 0.0f, 1.0f);

    float4 screen = {
        color.x / alpha,
        color.y / alpha,
        color.z / alpha,
        alpha * color.w
    };

    return screen;

}

__MATH_UTILS_DECL__ float4 screenClamp(float4 color) {

    color.x = clamp(color.x * color.w, 0.0f, 1.0f);
    color.y = clamp(color.y * color.w, 0.0f, 1.0f);
    color.z = clamp(color.z * color.w, 0.0f, 1.0f);

    float maxChannel = fmaxf(fmaxf(color.x, color.y), color.z);

    float alpha = maxChannel;

    float4 screen = {
        color.x / alpha,
        color.y / alpha,
        color.z / alpha,
        alpha
    };

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

__MATH_UTILS_DECL__ float4 getIESpresetOLD(int preset, int index) {

    float4 colorsDistances[6];

    // default preset
    colorsDistances[0] = { 1.0f, 1.0f, 1.0f, 0.0f };
    colorsDistances[1] = { 1.0f, 1.0f, 1.0f, 0.2f };
    colorsDistances[2] = { 1.0f, 1.0f, 1.0f, 0.4f };
    colorsDistances[3] = { 1.0f, 1.0f, 1.0f, 0.6f };
    colorsDistances[4] = { 1.0f, 1.0f, 1.0f, 0.8f };
    colorsDistances[5] = { 1.0f, 1.0f, 1.0f, 1.0f };

    if (preset == 2) {
        colorsDistances[0] = { 1.0f, 1.0f, 1.0f, 0.0f };
        colorsDistances[1] = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorsDistances[2] = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorsDistances[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorsDistances[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorsDistances[5] = { 0.0f, 0.0f, 0.0f, 1.0f };
    }

    if (preset == 3) {
        colorsDistances[0] = { 0.66f, 0.66f, 0.66f, 0.0f };
        colorsDistances[1] = { 1.0f, 1.0f, 1.0f, 0.4f };
        colorsDistances[2] = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorsDistances[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorsDistances[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorsDistances[5] = { 0.0f, 0.0f, 0.0f, 1.0f };
    }

    if (preset == 4) {
        colorsDistances[0] = { 0.93f, 0.93f, 0.93f, 0.0f };
        colorsDistances[1] = { 0.56f, 0.56f, 0.56f, 0.3f };
        colorsDistances[2] = { 0.72f, 0.72f, 0.72f, 0.4f };
        colorsDistances[3] = { 0.384f, 0.384f, 0.384f, 0.6f };
        colorsDistances[4] = { 0.0f, 0.0f, 0.0f, 0.9f };
        colorsDistances[5] = { 0.0f, 0.0f, 0.0f, 1.0f };
    }

    return colorsDistances[index];

}

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

float3 rampOLD(float4* colsDist, float x) {
    x = clamp(x, 0.0f, 1.0f);

    for (int i = 0; i < 5; i++) {
        if (x <= colsDist[i + 1].w) {
            float denom = colsDist[i + 1].w - colsDist[i].w;
            if (denom == 0.0f) {
                return { colsDist[i].x, colsDist[i].y, colsDist[i].z };
            }

            float t = (x - colsDist[i].w) / denom;
            return mixf3({ colsDist[i].x, colsDist[i].y, colsDist[i].z }, { colsDist[i + 1].x, colsDist[i + 1].y, colsDist[i + 1].z }, smoothstep(0.0f, 1.0f, t));
        }
    }

    // if (x > other distances), return last color 
    return { colsDist[5].x, colsDist[5].y, colsDist[5].z };
}

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

__MATH_UTILS_DECL__ float4 samplePixel(uint2 coord) {

    if (coord.x < inWidth && coord.y < inHeight && coord.x >= 0)
    {
        return ReadFloat4(inSrc, coord.y * inSrcPitch + coord.x, !!in16f);
    }
    return { 0.0f, 0.0f, 0.0f, 0.0f }; // default

}

__MATH_UTILS_DECL__ float getDepth(float4 pixel, bool depthBlackIsNear, float depthFar) {

    float depth = pixel.x;

    if (depthBlackIsNear) {
        depth = depth * depthFar;
    }
    else {
        depth = (1.0f - depth) * depthFar;
    }

    return depth;

}

__MATH_UTILS_DECL__ float3 getPosition(float2 uv, bool screenSpace, float depth, float4 camVx, float4 camVy, float4 camVz, float4 camPos, float cameraZoom, float cameraWidth, float cameraHeight) {

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

    if (screenSpace) {
        return localPos;
    } else {
        //float4x4 camMatrix = make_float4x4(camVx, camVy, camVz, camPos);
        //float4 worldPos = mulMatrixVector(camMatrix, localPos);
        return {
	        camVx.x * localPos.x + camVy.x * localPos.y + camVz.x * localPos.z + camPos.x * 1.0f,
	        camVx.y * localPos.x + camVy.y * localPos.y + camVz.y * localPos.z + camPos.y * 1.0f,
	        camVx.z * localPos.x + camVy.z * localPos.y + camVz.z * localPos.z + camPos.z * 1.0f
        };
    }

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

#endif /*MATH_UTILS_H*/