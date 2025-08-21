vec3 camMatrixExtract() {

    mat4 camMatrix = _PixelsWorld_camera_matrix;

    vec3 xAxis = camMatrix[0].xyz;
    vec3 yAxis = camMatrix[1].xyz;
    vec3 zAxis = camMatrix[2].xyz;
    vec3 camPos = camMatrix[3].xyz;

    return vec3(camPos, 1.0);
    
}

vec3 read3dPoint(int n) {

    return vec3(
        _PixelsWorld_point3d[n].x * iResolution.x,
        (1 - _PixelsWorld_point3d[n].y) * iResolution.y,
        _PixelsWorld_point3d[n].z
    );
    
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{

    vec3 worldPos = read3dPoint(0);                 // Object position
    mat4 camMatrix = _PixelsWorld_camera_matrix;    // "Local Camera" to "World" coords
    mat4 invMatrix = inverse(camMatrix);            // "World" To "Local Camera" coords

    vec4 localPos = invMatrix * vec4(worldPos, 1.0); // Object position relative to the local camera axis
    vec4 worldPosNew = camMatrix * localPos;

    fragColor = vec4(worldPosNew);

}