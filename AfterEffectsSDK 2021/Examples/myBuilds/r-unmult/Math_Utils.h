#ifndef MATH_UTILS_H
#define MATH_UTILS_H


#ifndef __MATH_UTILS_DECL__
#define __MATH_UTILS_DECL__ __inline__ __device__ __host__
#endif

__MATH_UTILS_DECL__ float clampf(float x, float minVal, float maxVal) {
    return fminf(fmaxf(x, minVal), maxVal);
}

__MATH_UTILS_DECL__ float dotf(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__MATH_UTILS_DECL__ float distance(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

__MATH_UTILS_DECL__ float luminance(float3 color) {

    return dotf(color, { 0.299f, 0.587f, 0.114f });
}

__MATH_UTILS_DECL__ float4 screen(float4 color) {

    float luma = luminance({ color.x, color.y, color.z });

    float alpha = clampf(luma, 0.0f, 1.0f);

    float4 screen = {
        color.x / alpha,
        color.y / alpha,
        color.z / alpha,
        alpha * color.w
    };
        
    return screen;

}

__MATH_UTILS_DECL__ float4 screenClamp(float4 color) {

    color.x = clampf(color.x * color.w, 0.0f, 1.0f);
    color.y = clampf(color.y * color.w, 0.0f, 1.0f);
    color.z = clampf(color.z * color.w, 0.0f, 1.0f);

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

#endif /*MATH_UTILS_H*/