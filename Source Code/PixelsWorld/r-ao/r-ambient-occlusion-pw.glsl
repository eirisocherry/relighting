#define intensity _PixelsWorld_slider[0]    // ao intensity
#define threshold _PixelsWorld_slider[1]    // angle threshold
#define xyRadius _PixelsWorld_slider[2]     // sample radius
#define zRadius _PixelsWorld_slider[3]      // sample radius
#define samples _PixelsWorld_slider[4]      // amount of samples

vec3 getPosition(vec2 uv) {

    return texture(iChannel0, uv).xyz;

}

vec3 getNormal(vec2 uv) {

   return texture(_PixelsWorld_inLayer, uv).xyz * 2.0 - 1.0;

}

vec2 getRandom(vec2 uv) {

    float x = fract(sin(dot(uv, vec2(127.1, 311.7))) * 43758.5453123);
    float y = fract(sin(dot(uv, vec2(311.7, 127.1))) * 43758.5453123);

    return vec2(x, y); // [0, 1]

}

vec2 generateKernel(int index) {

    float a = float(index) * 0.85443192;
    float r = float(index) / float(samples);
    vec2 dir = vec2(cos(a), sin(a));

    return dir * r;

}

float doAmbientOcclusion(vec2 uv, vec2 sampleDir) {

    vec2 offset = sampleDir * xyRadius;
    vec2 sampleUv = uv + offset / iResolution.xy;
    
    if (sampleUv.x < 0.0 || sampleUv.x > 1.0 || sampleUv.y < 0.0 || sampleUv.y > 1.0)
        return 0.0;

    vec3 p = getPosition(uv);
    vec3 pSample = getPosition(sampleUv);
    vec3 diff = pSample - p;
    vec3 v = normalize(diff);
    float l = length(diff);

    vec3 n = getNormal(uv);
    if (dot(n, v) < 0.0)
        return 0.0;

    float ao = max(0.0, dot(n, v) - threshold);
    ao *= 1.0 - min(l / zRadius, 1.0);
    return ao;

}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {

    vec2 uv = fragCoord.xy / iResolution.xy;
    vec2 rand = getRandom(uv);

    float ssao = 0.0;

    // Rand angle [0, 360] degrees
    float angle = rand.x * 6.28318530718;
    mat2 rot = mat2(cos(angle), -sin(angle),
                    sin(angle), cos(angle));

    for(int i = 0; i < samples; i++) {
        vec2 k = generateKernel(i);
        k = rot * k;
        ssao += doAmbientOcclusion(uv, k);
    }

    ssao /= float(samples);
    ssao = clamp(1.0 - ssao * intensity, 0.0, 1.0);

    fragColor = vec4(vec3(ssao), 1.0);

}