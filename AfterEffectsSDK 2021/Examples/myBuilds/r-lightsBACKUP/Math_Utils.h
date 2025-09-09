#ifndef MATH_UTILS_H
#define MATH_UTILS_H


#ifndef __MATH_UTILS_DECL__
#define __MATH_UTILS_DECL__ __inline__ __device__ __host__
#endif

__MATH_UTILS_DECL__ float clampf(float x, float minVal, float maxVal) {
    return fminf(fmaxf(x, minVal), maxVal);
}

__MATH_UTILS_DECL__ float distance(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

__MATH_UTILS_DECL__ float3 point(float3 worldPos, float3 lightPos, float radius, float fallOff) {
    
    float dist = distance(worldPos, lightPos);

    if (radius <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float intensity = 1.0f;
    if (fallOff > 0.0f) {
        float t = clampf(dist / radius, 0.0f, 1.0f);
        float falloffFactor = powf(1.0f - t, fallOff);
        intensity *= falloffFactor;
    }
    else {
        if (dist > radius) {
            intensity = 0.0f;
        }
    }

    float3 result = make_float3(intensity, intensity, intensity);

    return result;

}



#endif /*MATH_UTILS_H*/