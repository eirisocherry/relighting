#define ROT_XYZ 0
#define ROT_ZYX 1
#define ROT_YXZ 2
#define ROT_ZXY 3
#define ROT_YZX 4
#define ROT_XZY 5

const float PI = 3.1415926535897932384626433832795;

float getHorizontalFOV(float zoom) {
    vec2 vector = vec2(iResolution.x, 2 * zoom);
    vec2 normalized = normalize(vector);
    float hFOV = 2 * degrees(atan(normalized.x, normalized.y));
    return hFOV;
}

float getVerticalFOV(float zoom) {
    vec2 vector = vec2(iResolution.x, 2 * zoom);
    vec2 normalized = normalize(vector);
    float hFOV = 2 * degrees(atan(normalized.x, normalized.y));
    float hFOVhalf = tan(hFOV/2 * PI/180);
    float vFOV = 2 * atan(hFOVhalf * iResolution.y/iResolution.x) * 180/PI;
    return vFOV;
}

vec4 qMul(vec4 q1, vec4 q2) {
    float w1 = q1.x, x1 = q1.y, y1 = q1.z, z1 = q1.w;
    float w2 = q2.x, x2 = q2.y, y2 = q2.z, z2 = q2.w;

    return vec4(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, // w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, // x
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2, // y
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2  // z
    );
}

vec4 qFromEuler(vec3 eulerAngles, int order) {
    float phi = eulerAngles.x * 0.5;
    float theta = eulerAngles.y * 0.5;
    float psi = eulerAngles.z * 0.5;

    float cX = cos(phi);
    float cY = cos(theta);
    float cZ = cos(psi);
    float sX = sin(phi);
    float sY = sin(theta);
    float sZ = sin(psi);

    if (order == ROT_XYZ) {
        return vec4(
            cX * cY * cZ - sX * sY * sZ,
            sX * cY * cZ + sY * sZ * cX,
            sY * cX * cZ - sX * sZ * cY,
            sX * sY * cZ + sZ * cX * cY
        );
    } else if (order == ROT_ZYX) {
        return vec4(
            sX * sY * sZ + cX * cY * cZ,
            sZ * cX * cY - sX * sY * cZ,
            sX * sZ * cY + sY * cX * cZ,
            sX * cY * cZ - sY * sZ * cX
        );
    } else if (order == ROT_YXZ) {
        return vec4(
            sX * sY * sZ + cX * cY * cZ,
            sX * sZ * cY + sY * cX * cZ,
            sX * cY * cZ - sY * sZ * cX,
            sZ * cX * cY - sX * sY * cZ
        );
    } else if (order == ROT_ZXY) {
        return vec4(
            cX * cY * cZ - sX * sY * sZ,
            sY * cX * cZ - sX * sZ * cY,
            sX * sY * cZ + sZ * cX * cY,
            sX * cY * cZ + sY * sZ * cX
        );
    } else if (order == ROT_YZX) {
        return vec4(
            cX * cY * cZ - sX * sY * sZ,
            sX * sY * cZ + sZ * cX * cY,
            sX * cY * cZ + sY * sZ * cX,
            sY * cX * cZ - sX * sZ * cY
        );
    } else if (order == ROT_XZY) {
        return vec4(
            sX * sY * sZ + cX * cY * cZ,
            sX * cY * cZ - sY * sZ * cX,
            sZ * cX * cY - sX * sY * cZ,
            sX * sZ * cY + sY * cX * cZ
        );
    } else {
        // Error: unsupported order
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
}

vec4 qFromEulerLogical(float phi, float theta, float psi, int order) {
      return qFromEuler(vec3(psi, theta, phi), ROT_XYZ);
}

float asinFunc(float t) {
    return (t >= 1.0) ? 3.14159265359 / 2.0 : (t <= -1.0) ? -3.14159265359 / 2.0 : asin(t);
}

vec3 qToEuler(vec4 q, int order) {
    float w = q.x, x = q.y, y = q.z, z = q.w;

    float wx = w * x, wy = w * y, wz = w * z;
    float xx = x * x, xy = x * y, xz = x * z;
    float yy = y * y, yz = y * z, zz = z * z;

    if (order == ROT_XYZ) {
        return vec3(
            -atan(2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
            asinFunc(2.0 * (xz + wy)),
            -atan(2.0 * (xy - wz), 1.0 - 2.0 * (yy + zz))
        );
    } else if (order == ROT_ZYX) {
        return vec3(
            atan(2.0 * (xy + wz), 1.0 - 2.0 * (yy + zz)),
            -asinFunc(2.0 * (xz - wy)),
            atan(2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy))
        );
    } else if (order == ROT_YXZ) {
        return vec3(
            atan(2.0 * (xz + wy), 1.0 - 2.0 * (xx + zz)),
            -asinFunc(2.0 * (yz - wx)),
            atan(2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz))
        );
    } else if (order == ROT_ZXY) {
        return vec3(
            -atan(2.0 * (xz - wy), 1.0 - 2.0 * (xx + yy)),
            asinFunc(2.0 * (yz + wx)),
            -atan(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz))
        );
    } else if (order == ROT_YZX) {
        return vec3(
            atan(2.0 * (yz + wx), 1.0 - 2.0 * (xx + zz)),
            asinFunc(2.0 * (xy - wz)),
            atan(2.0 * (xz + wy), 1.0 - 2.0 * (yy + zz))
        );
    } else if (order == ROT_XZY) {
        return vec3(
            -atan(2.0 * (yz + wx), 1.0 - 2.0 * (xx + zz)),
            -asinFunc(2.0 * (xy - wz)),
            -atan(2.0 * (xz + wy), 1.0 - 2.0 * (yy + zz))
        );
    } else {
        // Error: unsupported order
        return vec3(0.0, 0.0, 0.0);
    }
}

vec3 getCord(float x, float y, float z, float rX, float rY, float distance) {
    float endX = x + distance * sin(rY);
    float endY = y - distance * sin(rX) * cos(rY);
    float endZ = z + distance * cos(rX) * cos(rY);

    return vec3(endX, endY, endZ);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  // Inputs
  vec2 uv = fragCoord / iResolution.xy;
  ivec2 texCoord = ivec2(fragCoord);
  vec4 textureInput = texelFetch(_PixelsWorld_inLayer, texCoord, 0);
  float xPos = float(_PixelsWorld_point3d[0].x * iResolution.x);
  float yPos = float((1 - _PixelsWorld_point3d[0].y) * iResolution.y);
  float zPos = float(_PixelsWorld_point3d[0].z);
  float xOr = float(_PixelsWorld_point3d[1].x * iResolution.x);
  float yOr = float((1 - _PixelsWorld_point3d[1].y) * iResolution.y);
  float zOr = float(_PixelsWorld_point3d[1].z);
  float xRot = float(_PixelsWorld_point3d[2].x * iResolution.x);
  float yRot = float((1 - _PixelsWorld_point3d[2].y) * iResolution.y);
  float zRot = float(_PixelsWorld_point3d[2].z);
  float zoom = float(_PixelsWorld_point3d[3].x * iResolution.x);
  bool black_is_near = bool(_PixelsWorld_checkbox[0]);
  float far = float(_PixelsWorld_slider[0]);

  // FOV
  float hFOV = getHorizontalFOV(zoom);
  float vFOV = getVerticalFOV(zoom);
  float focalLengthX = iResolution.x / (2.0 * tan(radians(hFOV) / 2.0));
  float focalLengthY = iResolution.y / (2.0 * tan(radians(vFOV) / 2.0));
  
  // Rotations
  float centerOffsetX = (uv.x - 0.5) * iResolution.x;
  float centerOffsetY = (uv.y - 0.5) * iResolution.y;
  float rXoffset = degrees(atan(centerOffsetY / zoom));
  float rYoffset = degrees(atan(centerOffsetX / zoom));

  // New orientation (converting local rotations to global using quaternions)
  vec4 qOr = qFromEulerLogical(radians(zOr), radians(yOr), radians(xOr), ROT_ZYX);
  vec4 qRot = qFromEulerLogical(radians(zRot), radians(yRot), radians(xRot), ROT_ZYX);
  vec4 qL = qFromEulerLogical(0, radians(rYoffset), radians(rXoffset), ROT_ZYX);
  vec4 q1 = qMul(qOr, qRot);
  vec4 q = qMul(q1, qL);
  vec3 finalOrientation = qToEuler(q, ROT_XYZ);

  // Distance
  float centerDistance;
  if (black_is_near) {
      centerDistance = textureInput.x * far;
  } else {
      centerDistance = (1 - textureInput.x) * far;
  }
  float realDistance = centerDistance / (cos(radians(rYoffset)) * cos(radians(rXoffset)));
  
  // Output
  vec3 coordinates = getCord(xPos, yPos, zPos, finalOrientation.x, finalOrientation.y, realDistance);
  fragColor = vec4(coordinates, 1);
}