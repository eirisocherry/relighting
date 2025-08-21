// Get

vec3 getPositionNoDepth(vec2 uv) {

    vec2 fragCoord = uv * iResolution.xy;
    vec3 screenPos = vec3(
        fragCoord.x - 0.5 * iResolution.x,          // [-halfRes..halfRes]
        (fragCoord.y - 0.5 * iResolution.y) * -1,   // invert y -> [halfRes..-halfRes]
        _PixelsWorld_camera_info.z                  // focal length (zoom)
    );
    vec4 worldPos = _PixelsWorld_camera_matrix * vec4(screenPos, 1.0); // Multiplication order is important (Matrix x Vec4)
    return worldPos.xyz;

}

vec3 getNormal(vec2 uv) {

    return texture(_PixelsWorld_inLayer, uv).xyz * 2.0 - 1.0;

}

// Read

vec3 read3dPoint(int n) {

    return vec3(
        _PixelsWorld_point3d[n].x * iResolution.x,
        (1 - _PixelsWorld_point3d[n].y) * iResolution.y,
        _PixelsWorld_point3d[n].z
    );
    
}

vec2 readPoint(int n) {

    return vec2(
        _PixelsWorld_point[n].x * iResolution.x,
        (1 - _PixelsWorld_point[n].y) * iResolution.y
    );
    
}

// Color

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Main

vec3 normality(
    vec2 uv, vec3 lightPos, vec3 lookAt,
    bool diffuseToggle, float diffuseIntensity, vec3 diffuseColor,
    bool specularToggle, float specularSize, float specularIntensity, vec3 specularColor,
    bool rimToggle, float rimStart, float rimEnd, float rimIntensity, vec3 rimColor,
    float saturationMultiplier
) {
    vec3 normal = getNormal(uv);
    vec3 lightDir = -normalize(lookAt - lightPos);  // negative vector so it looks logically right when using dir light
    vec3 reflectLightDir = normalize(-reflect(lightDir, normal));
    vec3 camPos = _PixelsWorld_camera_matrix[3].xyz;
    vec3 worldPos = getPositionNoDepth(uv); // we dont care about depth map, because it won't affect viewDir vector anyways
    vec3 viewDir = normalize(camPos - worldPos);

    // Diffuse
    vec3 diffuse = vec3(0.0);
    if (diffuseToggle) {
        float diffuseStrength = max(0.0, dot(lightDir, normal)); 
        diffuse = diffuseColor * diffuseStrength * diffuseIntensity;
    }

    // Specular
    vec3 specular = vec3(0.0);
    if (specularToggle) {
        float specularStrength = max(0.0, dot(viewDir, reflectLightDir));
        specularStrength = pow(specularStrength, specularSize);
        specular = specularColor * specularStrength * specularIntensity;
    }

    // Rim
    vec3 rim = vec3(0.0);
    if (rimToggle) {
        // https://lettier.github.io/3d-game-shaders-for-beginners/rim-lighting.html

        float rimLightIntensity = dot(viewDir, normal);
        rimLightIntensity = 1.0 - rimLightIntensity;
        rimLightIntensity = max(0.0, rimLightIntensity);
        //rimLightIntensity = pow(rimLightIntensity, rimPower);
        rimLightIntensity = smoothstep(rimStart, rimEnd, rimLightIntensity); // 0.3 0.4

        rim = rimColor * rimLightIntensity * rimIntensity;
    }

    // Combine
    vec3 color = diffuse + specular + rim;

    // Saturation
    vec3 colorHSV = rgb2hsv(color);
    vec3 colorRGB = hsv2rgb(vec3(colorHSV.x, clamp(colorHSV.y * saturationMultiplier, 0.0, 1.0), colorHSV.z));
    vec3 colorFinal = colorRGB;

    return colorFinal;

}

void mainImage(out vec4 outColor, in vec2 fragCoord) {

    // Light

    vec3 dirPos = read3dPoint(0);
    vec3 dirLookAt = read3dPoint(1);

    // Global

    bool globalToggle = _PixelsWorld_checkbox[3];
    float intensityMultiplier = _PixelsWorld_slider[6];
    float saturationMultiplier = _PixelsWorld_slider[7];

    // Diffuse

    bool diffuseToggle = _PixelsWorld_checkbox[0];
    float diffuseIntensity = _PixelsWorld_slider[0] * intensityMultiplier;
    vec3 diffuseColor = _PixelsWorld_color[0].rgb;

    // Specular

    bool specularToggle = _PixelsWorld_checkbox[1];
    float specularSize = _PixelsWorld_slider[1];
    float specularIntensity = _PixelsWorld_slider[2] * intensityMultiplier;
    vec3 specularColor = _PixelsWorld_color[1].rgb;

    // Rim
    
    bool rimToggle = _PixelsWorld_checkbox[2];
    float rimStart = _PixelsWorld_slider[3];
    float rimEnd = _PixelsWorld_slider[4];
    float rimIntensity = _PixelsWorld_slider[5] * intensityMultiplier;
    vec3 rimColor = _PixelsWorld_color[2].rgb;

    // Code

    vec2 uv = _PixelsWorld_uv;

    vec3 draw = vec3(0.0);

    if (globalToggle) {
        draw = normality(
            uv, dirPos, dirLookAt,
            diffuseToggle, diffuseIntensity, diffuseColor,
            specularToggle, specularSize, specularIntensity, specularColor,
            rimToggle, rimStart, rimEnd, rimIntensity, rimColor,
            saturationMultiplier
        );
    }

    outColor = vec4(draw, 1.0);

}