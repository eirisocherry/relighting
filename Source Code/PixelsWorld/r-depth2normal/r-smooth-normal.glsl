float getDepth(vec2 uv) {

    return texture(_PixelsWorld_inLayer, uv).x;

}

vec3 getNormal(vec2 uv) {

    return texture(_PixelsWorld_outLayer, uv).xyz;

}

vec3 smoothNormal(vec2 uv, bool horizontalBlur, int radius, float normalThreshold, float depthWeight) {

    vec2 uvPixel = 1.0 / iResolution.xy;

    vec3 centerNormal = getNormal(uv) * 2.0 - 1.0; // [0..1] → [-1..1]
    float centerDepth = getDepth(uv);

    vec3 blurred = vec3(0.0);
    float weightSum = 0.0;

    for (int i = -radius; i <= radius; i++) {

        vec2 offset = vec2(0.0);
        if (horizontalBlur) {
            offset = vec2(i, 0) * uvPixel;
        } else {
            offset = vec2(0, i) * uvPixel;
        }

        vec3 sampleNormal = getNormal(uv + offset) * 2.0 - 1.0; // [0..1] → [-1..1];
        float sampleDepth = getDepth(uv + offset);

        float diffNormal = length(centerNormal - sampleNormal) * normalThreshold; // [0..2] * [0..1]
        float diffDepth = abs(centerDepth - sampleDepth) * depthWeight; // [0..1] * [0..100]
        float diff = diffNormal + diffDepth; // [0..102]

        float weight = max(0.0, 1.0 - diff); // [0..1] blur intensity
        blurred += sampleNormal * weight;
        weightSum += weight;
        
    }

    if (weightSum > 0.0001) {

        blurred /= weightSum;

    }

    blurred = blurred * 0.5 + 0.5; // [-1..1] -> [0..1]

    return blurred; 

}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {

    // Sliders
    int radius = int(clamp(_PixelsWorld_slider[1], 1.0, 100.0));

    float normalThreshold = _PixelsWorld_slider[2];
    normalThreshold = clamp(normalThreshold, 0.0, 1.0);

    float depthWeight = _PixelsWorld_slider[3];
    if (depthWeight < 0.0) {
        depthWeight = 0.0;
    }

    // Code

    bool horizontalBlur = true;

    vec2 uv = _PixelsWorld_uv;

    vec3 blur = smoothNormal(uv, horizontalBlur, radius, normalThreshold, depthWeight);

    fragColor = vec4(blur, 1.0);

}