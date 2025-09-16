function relighting(thisObj) {

  // Global variables
  var name = "Relighting";
  var version = "v2.0";

  var dropdownMenuSelection;
  var alertMessage = [];

  // Effect strings
  var mainLayerNameDefault = "Depth Projection";
  var projectedLayerNameDefault = "Projected Solid";
  var originName = "Origin - Dont Delete";
  var mainEffectName = "Depth Projection";
  var mainEffectProjectOnPoint = "Project On Point";
  var mainEffectAutoOrient = "Auto Orient";
  var mainEffectBlackIsNear = "Depth Black Is Near";
  var mainEffectFar = "Depth Far";
  var mainLabelColor = 11; // orange

  // UI
  function buildUI(thisObj) {

      // -------------------UI-------------------

      var myPanel = (thisObj instanceof Panel) ? thisObj : new Window("palette", name + " " + version, undefined, { resizeable: true });

      // UI elements
      res = "group\
          {\
            orientation:'column',  alignment:['fill','center'], alignChildren:['fill','fill'],\
            setupGroup: Group\
            {\
              orientation:'column', alignChildren:['fill','center'],\
              staticText: StaticText{text: '" + name + " " + version + "', alignment:['center','center']}\
              projectGroup: Group\
              {\
                orientation:'row', alignChildren:['fill','center'],\
                projectButton: Button{text: 'Project'},\
                typeDropdown: DropDownList{properties:{items:['Point Simple (CUDA)', 'Point Advanced (CUDA)', 'Spot Simple (CUDA)', 'Spot Advanced (CUDA)', 'Rect Simple (CUDA)', 'Rect Advanced (CUDA)', 'Dir Light (CUDA)', 'Rim Light (CUDA)', '-------------', 'Ambient Occlusion (CUDA)', 'Position Pass (CUDA)', 'Normal Pass (CUDA)', 'Normal Remap (CUDA)', 'Unmult (CUDA)', 'Clamp (CUDA)', '-------------', 'Point Light', 'Solid', '3D Null', '3D+2D Null']}},\
              },\
              settingsGroup: Group\
              {\
                orientation:'row', alignment:['right','center'],\
                helpButton: Button{text: '?', maximumSize:[25,25]},\
                setupButton: Button{text: '+', maximumSize:[25,25]}\
                deleteSetupButton: Button{text: 'x', maximumSize:[25,25]},\
              }\
            }\
          }";

      // Add UI elements to the panel
      myPanel.grp = myPanel.add(res);
      // Refresh the panel
      myPanel.layout.layout(true);
      // Set minimal panel size
      myPanel.grp.minimumSize = myPanel.grp.size;
      // Add panel resizing function 
      myPanel.layout.resize();
      myPanel.onResizing = myPanel.onResize = function () {
          this.layout.resize();
      }

      // -------------------Buttons-------------------

      myPanel.grp.setupGroup.projectGroup.projectButton.onClick = function () {
          projectButton();
      }

      myPanel.grp.setupGroup.projectGroup.typeDropdown.selection = 0;
      dropdownMenuSelection = myPanel.grp.setupGroup.projectGroup.typeDropdown.selection.text;
      myPanel.grp.setupGroup.projectGroup.typeDropdown.onChange = function () {
          dropdownMenuSelection = myPanel.grp.setupGroup.projectGroup.typeDropdown.selection.text;
      }

      myPanel.grp.setupGroup.settingsGroup.helpButton.onClick = function () {
          alertCopy(
              'Source: https://github.com/eirisocherry/relighting'
          );
      }

      myPanel.grp.setupGroup.settingsGroup.setupButton.onClick = function () {
          setupButton();
      }

      myPanel.grp.setupGroup.settingsGroup.deleteSetupButton.onClick = function () {
          deleteSetupButton();
      }

      return myPanel;
  }

  // -------------------Buttons-------------------
  
  function setupButton() {

    // Inputs
    var comp = app.project.activeItem;
    var mainLayer = comp.selectedLayers[0];

    // Checkers
    if (!(comp instanceof CompItem)) {
      alert("Open a composition first");
      return;
    }

    if (!comp.activeCamera) {
      alert("No active camera found");
      return;
    }

    if (comp.selectedLayers.length !== 1 || !mainLayer.hasVideo) {
      alert("Select a single depth map");
      return;
    }

    if (effectsExistance(mainLayer, mainEffectName)) {
      alert(
        "You've already made a setup for this layer\n" +
        "If you wanna delete a setup, press [x] button"
      );
      return;
    }

    // First touches
    app.project.save();
    app.project.expressionEngine = "javascript-1.0";
    app.project.bitsPerChannel = 32;
    app.beginUndoGroup("setupButton");

    // Unique name
    var index = getUniqueIndex(comp, [projectedLayerNameDefault, mainLayerNameDefault]);
    var projectedLayerName = projectedLayerNameDefault + " " + index;
    var mainLayerName = mainLayerNameDefault + " " + index;

    // Main layer
    if (applyPresetFile(mainLayer, "r-depth-projection") === 0) {
      return;
    }
    mainLayer.label = mainLabelColor;
    mainLayer.name = mainLayerName;

    // Origin (needed for orientation expression)
    var originLayer = comp.layers.byName(originName);
    if (!originLayer) {
      originLayer = comp.layers.addNull();
      originLayer.name = originName;
      originLayer.moveToEnd();
      originLayer.threeDLayer = true;
      originLayer.transform.position.setValue([0, 0, 0]);
      originLayer.enabled = false;
      originLayer.locked = true;
      originLayer.shy = true;
      originLayer.selected = false;
      originLayer.label = mainLabelColor;
    }
    comp.hideShyLayers = true;

    // Expressions
    var positionExpression;
    var orientationExpression;
    function expressions() {
      positionExpression =
        'function normalize(v) {\n' +
        '  var len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);\n' +
        '  return len !== 0 ? [v[0]/len, v[1]/len, v[2]/len] : [0, 0, 0];\n' +
        '}\n' +
        '\n' +
        'function bugFix(a) {\n' +
        '  return  (a != 0 ? a : 1);\n' +
        '}\n' +
        '\n' +
        'function getPosition(uv) {\n' +
        '\n' +
        '  // the following global variables required:\n' +
        '  // depthLayer, depthFar, depthBlackIsNear, cameraZoom, origin, bugfix()\n' +
        '\n' +
        '  var depth = (\n' +
        '    (depthBlackIsNear == 1)\n' +
        '    ? bugFix(depthLayer.sampleImage([uv[0], uv[1]], [0.5, 0.5], true, time)[0])\n' +
        '    : bugFix(1 - depthLayer.sampleImage([uv[0], uv[1]], [0.5, 0.5], true, time)[0])\n' +
        '  ) * depthFar;\n' +
        '  var screenSpaceCoordinate = [uv[0], uv[1], cameraZoom - cameraZoom / depth];\n' +
        '  var worldSpaceCoordinate  = origin.fromComp(screenSpaceCoordinate);\n' +
        '  return worldSpaceCoordinate;\n' +
        '\n' +
        '}\n' +
        '\n' +
        '// Global Variables\n' +
        'var depthLayer = thisComp.layer("' + mainLayer.name + '");\n' +
        'var point3D  = depthLayer.effect("' + mainEffectName + '")("' + mainEffectProjectOnPoint + '");\n' +
        'var depthFar = depthLayer.effect("' + mainEffectName + '")("' + mainEffectFar + '");\n' +
        'var depthBlackIsNear = depthLayer.effect("' + mainEffectName + '")("' + mainEffectBlackIsNear + '");\n' +
        'var cameraZoom = thisComp.activeCamera.zoom;\n' +
        'var origin = thisComp.layer("' + originName + '");\n' +
        '\n' +
        'function main() {\n' +
        '\n' +
        '  var worldPos = getPosition(point3D);\n' +
        '  var vZ = normalize(thisLayer.toWorldVec([0, 0, 1]));\n' +
        '\n' +
        '  var finalPos = [\n' +
        '    worldPos[0] + vZ[0] * point3D[2],\n' +
        '    worldPos[1] + vZ[1] * point3D[2],\n' +
        '    worldPos[2] + vZ[2] * point3D[2]\n' +
        '  ];\n' +
        '\n' +
        '  return finalPos;\n' +
        '\n' +
        '}\n' +
        '\n' +
        'var output = main();\n' +
        'output;';


      orientationExpression =
        '// Math functions adapted from three.js\n' +
        '\n' +
        '// Math Utils\n' +
        '\n' +
        'function degrees(angle) {\n' +
        '  return angle * 180 / Math.PI;\n' +
        '}\n' +
        '\n' +
        'function radians(angle) {\n' +
        '  return angle * Math.PI / 180;\n' +
        '}\n' +
        '\n' +
        'function clamp( value, min, max ) {\n' +
        '\n' +
        '	return Math.max( min, Math.min( max, value ) );\n' +
        '\n' +
        '}\n' +
        '\n' +
        '// Vector3 + Quaternion\n' +
        '\n' +
        'function lengthVec3(vec3) {\n' +
        '\n' +
        '  return Math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);\n' +
        '\n' +
        '}\n' +
        '\n' +
        'function lengthQ(q) {\n' +
        '\n' +
        '  return Math.sqrt(\n' +
        '    q[0] * q[0] +\n' +
        '    q[1] * q[1] +\n' +
        '    q[2] * q[2] +\n' +
        '    q[3] * q[3]\n' +
        '  );\n' +
        '\n' +
        '}\n' +
        '\n' +
        'function normalizeVec3(vec3) {\n' +
        '\n' +
        '  var len = lengthVec3(vec3);\n' +
        '\n' +
        '  return [\n' +
        '    vec3[0] / len || 0,\n' +
        '    vec3[1] / len || 0,\n' +
        '    vec3[2] / len || 0\n' +
        '  ];\n' +
        '\n' +
        '}\n' +
        '\n' +
        'function normalizeQ(q) {\n' +
        '\n' +
        '  const len = lengthQ(q);\n' +
        '\n' +
        '  return [\n' +
        '    q[0] / len || 0,\n' +
        '    q[1] / len || 0,\n' +
        '    q[2] / len || 0,\n' +
        '    q[3] / len || 1\n' +
        '  ];\n' +
        '\n' +
        '}\n' +
        '\n' +
        '// Vector3\n' +
        '\n' +
        'function dot(a, b) {\n' +
        '  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];\n' +
        '}\n' +
        '\n' +
        'function cross(a, b) {\n' +
        '  return [\n' +
        '    a[1]*b[2] - a[2]*b[1],\n' +
        '    a[2]*b[0] - a[0]*b[2],\n' +
        '    a[0]*b[1] - a[1]*b[0]\n' +
        '  ];\n' +
        '}\n' +
        '\n' +
        '// Quaternion\n' +
        '\n' +
        'function setFromUnitVectors(vFrom, vTo) {\n' +
        '\n' +
        '  let q = [];\n' +
        '\n' +
        '  // assumes direction vectors vFrom and vTo are normalized\n' +
        '\n' +
        '  let r = dot(vFrom, vTo) + 1;\n' +
        '\n' +
        '  if (r < Number.EPSILON) {\n' +
        '\n' +
        '    // vFrom and vTo point in opposite directions\n' +
        '\n' +
        '    r = 0;\n' +
        '\n' +
        '    if (Math.abs(vFrom[0]) > Math.abs(vFrom[2])) {\n' +
        '\n' +
        '      q = [\n' +
        '        -vFrom[1],   // x\n' +
        '        vFrom[0],    // y\n' +
        '        0,           // z\n' +
        '        r            // w\n' +
        '      ];\n' +
        '\n' +
        '    } else {\n' +
        '\n' +
        '      q = [\n' +
        '        0,           // x\n' +
        '        -vFrom[2],   // y\n' +
        '        vFrom[1],    // z\n' +
        '        r            // w\n' +
        '      ];\n' +
        '\n' +
        '    }\n' +
        '  } else {\n' +
        '\n' +
        '    const axis = cross(vFrom, vTo);\n' +
        '    q = [...axis, r];\n' +
        '\n' +
        '  }\n' +
        '\n' +
        '  return normalizeQ(q);\n' +
        '}\n' +
        '\n' +
        'function multiplyQuaternions(a, b) {\n' +
        '  const ax = a[0], ay = a[1], az = a[2], aw = a[3];\n' +
        '  const bx = b[0], by = b[1], bz = b[2], bw = b[3];\n' +
        '\n' +
        '  const x = ax * bw + aw * bx + ay * bz - az * by;\n' +
        '  const y = ay * bw + aw * by + az * bx - ax * bz;\n' +
        '  const z = az * bw + aw * bz + ax * by - ay * bx;\n' +
        '  const w = aw * bw - ax * bx - ay * by - az * bz;\n' +
        '\n' +
        '  return [x, y, z, w];\n' +
        '}\n' +
        '\n' +
        '// Euler\n' +
        '\n' +
        'function setFromQuaternion(q, order = "XYZ") {\n' +
        '  const matrix = makeRotationFromQuaternion(q);\n' +
        '  return setFromRotationMatrix(matrix, order);\n' +
        '}\n' +
        '\n' +
        'function setFromRotationMatrix(matrix, order = "XYZ") {\n' +
        '\n' +
        '  const m11 = matrix[0], m12 = matrix[4], m13 = matrix[8];\n' +
        '  const m21 = matrix[1], m22 = matrix[5], m23 = matrix[9];\n' +
        '  const m31 = matrix[2], m32 = matrix[6], m33 = matrix[10];\n' +
        '\n' +
        '  let x, y, z;\n' +
        '\n' +
        '  switch (order) {\n' +
        '    case "XYZ":\n' +
        '      y = Math.asin(clamp(m13, -1, 1));\n' +
        '      if (Math.abs(m13) < 0.9999999) {\n' +
        '        x = Math.atan2(-m23, m33);\n' +
        '        z = Math.atan2(-m12, m11);\n' +
        '      } else {\n' +
        '        x = Math.atan2(m32, m22);\n' +
        '        z = 0;\n' +
        '      }\n' +
        '      break;\n' +
        '\n' +
        '    case "YXZ":\n' +
        '      x = Math.asin(-clamp(m23, -1, 1));\n' +
        '      if (Math.abs(m23) < 0.9999999) {\n' +
        '        y = Math.atan2(m13, m33);\n' +
        '        z = Math.atan2(m21, m22);\n' +
        '      } else {\n' +
        '        y = Math.atan2(-m31, m11);\n' +
        '        z = 0;\n' +
        '      }\n' +
        '      break;\n' +
        '\n' +
        '    case "ZXY":\n' +
        '      x = Math.asin(clamp(m32, -1, 1));\n' +
        '      if (Math.abs(m32) < 0.9999999) {\n' +
        '        y = Math.atan2(-m31, m33);\n' +
        '        z = Math.atan2(-m12, m22);\n' +
        '      } else {\n' +
        '        y = 0;\n' +
        '        z = Math.atan2(m21, m11);\n' +
        '      }\n' +
        '      break;\n' +
        '\n' +
        '    case "ZYX":\n' +
        '      y = Math.asin(-clamp(m31, -1, 1));\n' +
        '      if (Math.abs(m31) < 0.9999999) {\n' +
        '        x = Math.atan2(m32, m33);\n' +
        '        z = Math.atan2(m21, m11);\n' +
        '      } else {\n' +
        '        x = 0;\n' +
        '        z = Math.atan2(-m12, m22);\n' +
        '      }\n' +
        '      break;\n' +
        '\n' +
        '    case "YZX":\n' +
        '      z = Math.asin(clamp(m21, -1, 1));\n' +
        '      if (Math.abs(m21) < 0.9999999) {\n' +
        '        x = Math.atan2(-m23, m22);\n' +
        '        y = Math.atan2(-m31, m11);\n' +
        '      } else {\n' +
        '        x = 0;\n' +
        '        y = Math.atan2(m13, m33);\n' +
        '      }\n' +
        '      break;\n' +
        '\n' +
        '    case "XZY":\n' +
        '      z = Math.asin(-clamp(m12, -1, 1));\n' +
        '      if (Math.abs(m12) < 0.9999999) {\n' +
        '        x = Math.atan2(m32, m22);\n' +
        '        y = Math.atan2(m13, m11);\n' +
        '      } else {\n' +
        '        x = Math.atan2(-m23, m33);\n' +
        '        y = 0;\n' +
        '      }\n' +
        '      break;\n' +
        '\n' +
        '    default:\n' +
        '      [x, y, z] = [0, 0, 0];\n' +
        '  }\n' +
        '\n' +
        '  return [x, y, z, order];\n' +
        '}\n' +
        '\n' +
        'function vec3ToEuler(vBasis, vDir, dotForward){\n' +
        '\n' +
        '  vBasis = normalizeVec3(vBasis);\n' +
        '  vDir = normalizeVec3(vDir);\n' +
        '\n' +
        '  let q = setFromUnitVectors(vBasis, vDir);\n' +
        '\n' +
        '  const qY180 = [0, 1, 0, 0];\n' +
        '  if (dotForward > 0) {\n' +
        '    q = multiplyQuaternions(q, qY180);\n' +
        '  }\n' +
        '\n' +
        '  return setFromQuaternion(q);\n' +
        '\n' +
        '}\n' +
        '\n' +
        '// Matrix4\n' +
        '\n' +
        'function makeRotationFromQuaternion(q) {\n' +
        '  return compose([0, 0, 0], q, [1, 1, 1]);\n' +
        '}\n' +
        '\n' +
        'function compose(position, quaternion, scale) {\n' +
        '  const te = new Array(16);\n' +
        '\n' +
        '  const [x, y, z, w] = quaternion;\n' +
        '  const x2 = x + x;\n' +
        '  const y2 = y + y;\n' +
        '  const z2 = z + z;\n' +
        '\n' +
        '  const xx = x * x2;\n' +
        '  const xy = x * y2;\n' +
        '  const xz = x * z2;\n' +
        '\n' +
        '  const yy = y * y2;\n' +
        '  const yz = y * z2;\n' +
        '  const zz = z * z2;\n' +
        '\n' +
        '  const wx = w * x2;\n' +
        '  const wy = w * y2;\n' +
        '  const wz = w * z2;\n' +
        '\n' +
        '  const [sx, sy, sz] = scale;\n' +
        '\n' +
        '  te[0] = (1 - (yy + zz)) * sx;\n' +
        '  te[1] = (xy + wz) * sx;\n' +
        '  te[2] = (xz - wy) * sx;\n' +
        '  te[3] = 0;\n' +
        '\n' +
        '  te[4] = (xy - wz) * sy;\n' +
        '  te[5] = (1 - (xx + zz)) * sy;\n' +
        '  te[6] = (yz + wx) * sy;\n' +
        '  te[7] = 0;\n' +
        '\n' +
        '  te[8] = (xz + wy) * sz;\n' +
        '  te[9] = (yz - wx) * sz;\n' +
        '  te[10] = (1 - (xx + yy)) * sz;\n' +
        '  te[11] = 0;\n' +
        '\n' +
        '  te[12] = position[0];\n' +
        '  te[13] = position[1];\n' +
        '  te[14] = position[2];\n' +
        '  te[15] = 1;\n' +
        '\n' +
        '  return te; // column-major\n' +
        '}\n' +
        '\n' +
        '// Get\n' +
        '\n' +
        'function bugFix(a) {\n' +
        '  return  (a != 0 ? a : 1);\n' +
        '}\n' +
        '\n' +
        'function getPosition(uv) {\n' +
        '\n' +
        '  // the following global variables required:\n' +
        '  // depthLayer, depthFar, depthBlackIsNear, cameraZoom, origin, bugfix()\n' +
        '\n' +
        '  var depth = (\n' +
        '    (depthBlackIsNear == 1)\n' +
        '    ? bugFix(depthLayer.sampleImage([uv[0], uv[1]], [0.5, 0.5], true, time)[0])\n' +
        '    : bugFix(1 - depthLayer.sampleImage([uv[0], uv[1]], [0.5, 0.5], true, time)[0])\n' +
        '  ) * depthFar;\n' +
        '  var screenSpaceCoordinate = [uv[0], uv[1], cameraZoom - cameraZoom / depth];\n' +
        '  var worldSpaceCoordinate  = origin.fromComp(screenSpaceCoordinate);\n' +
        '  return worldSpaceCoordinate;\n' +
        '\n' +
        '}\n' +
        '\n' +
        'function getNormal(uv) {\n' +
        '  var l1 = getPosition(uv + [-1, 0]);\n' +
        '  var r1 = getPosition(uv + [1, 0]);\n' +
        '  var t1 = getPosition(uv + [0, -1]);\n' +
        '  var b1 = getPosition(uv + [0, 1]);\n' +
        '\n' +
        '  var normal = normalizeVec3(cross(r1 - l1, t1 - b1)); \n' +
        '  return normal;\n' +
        '}\n' +
        '\n' +
        'function getNormalImproved(uv) {\n' +
        '  var c = getPosition(uv);\n' +
        '  var l1 = getPosition(uv + [-1, 0]);\n' +
        '  var r1 = getPosition(uv + [1, 0]);\n' +
        '  var t1 = getPosition(uv + [0, -1]);\n' +
        '  var b1 = getPosition(uv + [0, 1]);\n' +
        '\n' +
        '  var l2 = getPosition(uv + [-2, 0]);\n' +
        '  var r2 = getPosition(uv + [2, 0]);\n' +
        '  var t2 = getPosition(uv + [0, -2]);\n' +
        '  var b2 = getPosition(uv + [0, 2]);\n' +
        '\n' +
        '  var dl = Math.abs(l1[0] * l2[0] / (2.0 * l2[0] - l1[0]) - c[0]);\n' +
        '  var dr = Math.abs(r1[0] * r2[0] / (2.0 * r2[0] - r1[0]) - c[0]);\n' +
        '  var db = Math.abs(b1[1] * b2[1] / (2.0 * b2[1] - b1[1]) - c[1]);\n' +
        '  var dt = Math.abs(t1[1] * t2[1] / (2.0 * t2[1] - t1[1]) - c[1]);\n' +
        '\n' +
        '  var dpdx = (dl < dr)\n' +
        '              ? [c[0] - l1[0], c[1] - l1[1], c[2] - l1[2]]\n' +
        '              : [r1[0] - c[0], r1[1] - c[1], r1[2] - c[2]];\n' +
        '\n' +
        '  var dpdy = (db < dt)\n' +
        '              ? [c[0] - b1[0], c[1] - b1[1], c[2] - b1[2]]\n' +
        '              : [t1[0] - c[0], t1[1] - c[1], t1[2] - c[2]];\n' +
        '\n' +
        '  var normal = normalizeVec3(cross(dpdx, dpdy)); \n' +
        '  return normal;\n' +
        '\n' +
        '}\n' +
        '\n' +
        '// Main\n' +
        '\n' +
        'var depthLayer = thisComp.layer("' + mainLayer.name + '");\n' +
        'var point3D  = depthLayer.effect("' + mainEffectName + '")("' + mainEffectProjectOnPoint + '");\n' +
        'var depthFar = depthLayer.effect("' + mainEffectName + '")("' + mainEffectFar + '");\n' +
        'var depthBlackIsNear = depthLayer.effect("' + mainEffectName + '")("' + mainEffectBlackIsNear + '");\n' +
        'var cameraZoom = thisComp.activeCamera.zoom;\n' +
        'var origin = thisComp.layer("' + originName + '");\n' +
        'var autoOrient  = depthLayer.effect("' + mainEffectName + '")("' + mainEffectAutoOrient + '");\n' +
        '\n' +
        'function main() {\n' +
        '\n' +
        '  if (autoOrient == 1) {\n' +
        '\n' +
        '    // Points\n' +
        '    var c = getPosition(point3D);\n' +
        '    var normal = getNormalImproved(point3D);\n' +
        '\n' +
        '    // Basis\n' +
        '    var camPos = thisComp.activeCamera.toWorld([0, 0, 0]);\n' +
        '    var samplePos = [c[0], c[1], c[2]];\n' +
        '    var vView = normalizeVec3(sub(camPos, samplePos));\n' +
        '    var forward = [0, 0, 1];\n' +
        '    var backward = [0, 0, -1];\n' +
        '    var dotForward = dot(vView, forward);\n' +
        '\n' +
        '    if (dotForward > 0) {\n' +
        '      vBasis = forward;\n' +
        '    } else {\n' +
        '      vBasis = backward;\n' +
        '    }\n' +
        '\n' +
        '    var angles = vec3ToEuler(vBasis, normal, dotForward);\n' +
        '    return [degrees(angles[0]), degrees(angles[1]), degrees(angles[2])];\n' +
        '\n' +
        '  } else {\n' +
        '\n' +
        '    return value;\n' +
        '\n' +
        '  }\n' +
        '\n' +
        '}\n' +
        '\n' +
        'var output = main();\n' +
        'output;';

    }
    expressions();
 
    // Solid
    var projectedLayer = comp.layers.addSolid(
      [0.91, 0.57, 0.05],       // color: orange
      "Solid",                  // name
      100,                      // width
      100,                      // height
      1);                       // pixel aspect ratio
    projectedLayer.name = projectedLayerName;
    projectedLayer.threeDLayer = true;
    projectedLayer.transform.position.expression = positionExpression;
    projectedLayer.transform.orientation.expression = orientationExpression;
    projectedLayer.transform.anchorPoint.setValue([50, 50, 0]);
    projectedLayer.transform.scale.setValue([30, 30, 30]);
    projectedLayer.transform.opacity.setValue(50);
    if (comp.renderer == "ADBE Advanced 3d") {
      projectedLayer.property("ADBE Material Options Group").property("ADBE Accepts Lights").setValue(0);
    }
    projectedLayer.startTime = mainLayer.startTime;
    projectedLayer.inPoint = mainLayer.inPoint;
    projectedLayer.outPoint = mainLayer.outPoint;
    projectedLayer.label = mainLabelColor;

    // Final touches
    deselectAll(comp);
    mainLayer.property("ADBE Effect Parade").property(mainEffectName).selected = true;

    app.endUndoGroup();
  }

  function projectButton() {

    // First touches
    app.project.save();
    app.beginUndoGroup("projectButton");

    // Inputs
    var comp = app.project.activeItem;
    var mainLayer = comp.selectedLayers[0];

    // Checkers 1
    if (!(comp instanceof CompItem)) {
      alert("Open a composition first");
      return;
    }
    
    if (comp.selectedLayers.length !== 1 || !mainLayer.hasVideo) {
      alert("Select a single layer");
      return;
    }

    // Add simple effects
    switch (dropdownMenuSelection) {
      case 'Unmult (CUDA)':

        // Check the plugin
        if (!pluginExistance(mainLayer, "ADBE r-unmult")) {
          return;
        }

        // Add the effect
        var effect = mainLayer.property("ADBE Effect Parade").addProperty("ADBE r-unmult");
        effect.selected = true;

        return;

      break;

      case 'Clamp (CUDA)':

        // Check the plugin
        if (!pluginExistance(mainLayer, "ADBE r-clamp")) {
          return;
        }

        // Add the effect
        var effect = mainLayer.property("ADBE Effect Parade").addProperty("ADBE r-clamp");
        effect.selected = true;

        return;

      break;

      case 'Normal Remap (CUDA)':

        // Check the plugin
        if (!pluginExistance(mainLayer, "ADBE r-normal-remap")) {
          return;
        }

        // Add the effect
        var effect = mainLayer.property("ADBE Effect Parade").addProperty("ADBE r-normal-remap");
        effect.selected = true;

        return;

      break;
        
      // Debug

      case 'Unmult (PW)':

        // Check the plugin
        if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
          return;
        }

        // Add effect
        var effect = applyPresetFile(mainLayer, "r-unmult-pw");
        if (effect === 0) {
          return;
        }
        effect.selected = true;

        return;

      break;

      case 'Clamp (PW)':

        // Check the plugin
        if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
          return;
        }

        // Add effect
        var effect = applyPresetFile(mainLayer, "r-clamp-pw");
        if (effect === 0) {
          return;
        }
        effect.selected = true;

        return;

      break;

      case 'Normal Remap (PW)':

        // Check the plugin
        if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
          return;
        }

        // Add effect
        var effect = applyPresetFile(mainLayer, "r-normal-remap-pw");
        if (effect === 0) {
          return;
        }

        var mainEffect = mainLayer.property("ADBE Effect Parade").property("r-normal-remap-pw");
        var lightEffectName = "r-normal-remap-controller";

        // Input Is Normalized
        mainEffect.property("inputIsNormalized").expression =
        'var controllerProperty = effect("' +  lightEffectName + '")("Input Is Normalized");\n' +
        'var property1 = value;\n' +
        'if (controllerProperty) {\n' +
        '  property1 = controllerProperty;\n' +
        '}\n' +
        'property1;';

        // X
        mainEffect.property("x").expression =
        'var controllerProperty = effect("' +  lightEffectName + '")("X");\n' +
        'var property1 = value;\n' +
        'if (controllerProperty) {\n' +
        '  property1 = controllerProperty;\n' +
        '}\n' +
        'property1;';

        // Y
        mainEffect.property("y").expression =
        'var controllerProperty = effect("' +  lightEffectName + '")("Y");\n' +
        'var property1 = value;\n' +
        'if (controllerProperty) {\n' +
        '  property1 = controllerProperty;\n' +
        '}\n' +
        'property1;';

        // Z
        mainEffect.property("z").expression =
        'var controllerProperty = effect("' +  lightEffectName + '")("Z");\n' +
        'var property1 = value;\n' +
        'if (controllerProperty) {\n' +
        '  property1 = controllerProperty;\n' +
        '}\n' +
        'property1;';

        // Input Is Normalized
        mainEffect.property("normalizeOutput").expression =
        'var controllerProperty = effect("' +  lightEffectName + '")("Normalize Output");\n' +
        'var property1 = value;\n' +
        'if (controllerProperty) {\n' +
        '  property1 = controllerProperty;\n' +
        '}\n' +
        'property1;';
        
        mainEffect.selected = true;

        return;

      break;
        
      default:
      //continue
      break;
    }

    // Checkers 2
    if (!comp.activeCamera) {
      alert("No active camera found");
      return;
    }

    if (!effectsExistance(mainLayer, mainEffectName)) {
      alert("Select a layer with a " + mainEffectName + " effect");
      return;
    }

    // Parse values from main layer
    var blackIsNear = mainLayer.property("ADBE Effect Parade").property(mainEffectName).property(mainEffectBlackIsNear).value;
    var far = mainLayer.property("ADBE Effect Parade").property(mainEffectName).property(mainEffectFar).value;

    // Get projected solid
    var projectedLayer = getProjectedLayer(mainLayer);
    if (!projectedLayer) {
      return;
    }

    // Parse values from projected solid
    var projectedPosition = projectedLayer.transform.position.valueAtTime(comp.time, false);
    var projectedAnchorPoint = projectedLayer.transform.anchorPoint.valueAtTime(comp.time, false);
    var projectedOrientation = projectedLayer.transform.orientation.valueAtTime(comp.time, false);
    var projectedScale = projectedLayer.transform.scale.valueAtTime(comp.time, false);
    var projectedOpacity = projectedLayer.transform.opacity.valueAtTime(comp.time, false);
    var projectedWidth = projectedLayer.width;
    var projectedHeight = projectedLayer.height;
    var projectedAcceptsLights;
    if (comp.renderer == "ADBE Advanced 3d") {
      projectedAcceptsLights = projectedLayer.property("ADBE Material Options Group").property("ADBE Accepts Lights").value;
    }

    // Generate random color
    var labelColor;
    do {
      labelColor = Math.round(getRandomNumber(1, 16));
    } while (labelColor === mainLabelColor);
    var randomColorRGB = getRandomColorRGB(labelColor);
    var randomColorRGB2 = rgbToHSV(randomColorRGB);
    randomColorRGB2 = hsvToRGB([randomColorRGB2[0] - 60.0/360.0, randomColorRGB2[1], randomColorRGB2[2]]);

    // Project object
    switch (dropdownMenuSelection) {

      // Lights

      case 'Point Simple (CUDA)':

        function addExpressionsPointSimpleCUDA(effect, index) {

          var lightEffectName = "r-point-simple-light";

          // Light Toggle
          effect.property("Light Toggle" + " " + index).expression =
          '// Light Toggle\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightToggle = value;\n' +
          'if (lightLayer) {\n' +
          '  lightToggle = lightLayer.enabled;\n' +
          '}\n' +
          'lightToggle;';

          // Position
          effect.property("Light Position" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = thisComp.activeCamera.fromWorld(lightLayer.transform.position);\n' +
          '}\n' +
          'lightPosition;';

          // Invert
          effect.property("Invert Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Invert Toggle");\n' +
          '}\n' +
          'property1;';
          
          // Radius
          effect.property("Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Radius");\n' +
          '}\n' +
          'property1;';

          // Falloff
          effect.property("Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Falloff");\n' +
          '}\n' +
          'property1;';

          // Intensity
          effect.property("Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Intensity");\n' +
          '}\n' +
          'property1;';

          // Saturation
          effect.property("Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Saturation");\n' +
          '}\n' +
          'property1;';

          // Color Near
          effect.property("Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Near");\n' +
          '}\n' +
          'property1;';

          // Color Far Toggle
          effect.property("Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Color Far
          effect.property("Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Far");\n' +
          '}\n' +
          'property1;';

          // Color Falloff
          effect.property("Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Falloff");\n' +
          '}\n' +
          'property1;';

        }

        function addPointLightMatteCuda() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-point-simple")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-clamp")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-unmult")) {
            return;
          }

          // Index
          var index1 = getUniqueIndex(comp, ["Point Simple", "Clamp"]);
          var index2 = getUniqueIndex(comp, ["Point Light"]);
          var pointLightLayerName = "Point Light" + " " + index2;
          var baseName = "Point Simple" + " " + index1;
          var adjName = "Clamp" + " " + index1;

          // Adjustment Layer w Clamp effect
          var adjLayer = comp.layers.addSolid([1, 1, 1], adjName, comp.width, comp.height, 1);
          adjLayer.adjustmentLayer = true;
          adjLayer.name = adjName;
          adjLayer.startTime = mainLayer.startTime;
          adjLayer.inPoint = mainLayer.inPoint;
          adjLayer.outPoint = mainLayer.outPoint;
          adjLayer.label = labelColor;
          adjLayer.property("ADBE Effect Parade").addProperty("ADBE r-clamp");

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Point Light Layer
          var pointLightLayer = comp.layers.addNull(comp.width);
          pointLightLayer.name = pointLightLayerName;
          pointLightLayer.threeDLayer = true;
          pointLightLayer.transform.position.setValue(projectedPosition);
          pointLightLayer.transform.orientation.setValue(projectedOrientation);
          pointLightLayer.transform.scale.setValue(projectedScale);
          pointLightLayer.startTime = mainLayer.startTime;
          pointLightLayer.inPoint = mainLayer.inPoint;
          pointLightLayer.outPoint = mainLayer.outPoint;
          pointLightLayer.label = labelColor;

          var pointLightEffect = applyPresetFile(pointLightLayer, "r-point-simple-light");
          if (pointLightEffect === 0) {
            return;
          }
          pointLightEffect.property("Color Near").setValue(randomColorRGB);
          pointLightEffect.property("Color Far").setValue(randomColorRGB2);
 
          // Base effect settings
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-point-simple");
          baseEffect.property("Depth Far").setValue(far);
          baseEffect.property("Depth Black Is Near").setValue(blackIsNear);
          baseEffect.property("Light 1").setValue(pointLightLayer.index);

          // Expressions
          for (var i = 1; i <= 10; i++) {
            addExpressionsPointSimpleCUDA(baseEffect, i);
          }

          // Unmult effect
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-unmult");

        }
        addPointLightMatteCuda();
        break;

      case 'Point Advanced (CUDA)':

        function addExpressionsPointAdvancedCUDA(effect, index) {

          var lightEffectName = "r-point-advanced-light";

          // Light Toggle
          effect.property("Light Toggle" + " " + index).expression =
          '// Light Toggle\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightToggle = value;\n' +
          'if (lightLayer) {\n' +
          '  lightToggle = lightLayer.enabled;\n' +
          '}\n' +
          'lightToggle;';

          // Light Position
          effect.property("Light Position" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = lightLayer.toWorld(lightLayer.anchorPoint);\n' +
          '}\n' +
          'lightPosition;';
          
          // Radius
          effect.property("Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Radius");\n' +
          '}\n' +
          'property1;';


          // Ambient Toggle
          effect.property("Ambient Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Toggle");\n' +
          '}\n' +
          'property1;';

          // Ambient Falloff
          effect.property("Ambient Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Falloff");\n' +
          '}\n' +
          'property1;';

          // Ambient Intensity
          effect.property("Ambient Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Intensity");\n' +
          '}\n' +
          'property1;';

          // Ambient Saturation
          effect.property("Ambient Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Saturation");\n' +
          '}\n' +
          'property1;';

          // Ambient Near Color
          effect.property("Ambient Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Near");\n' +
          '}\n' +
          'property1;';

          // Ambient Color Far Toggle
          effect.property("Ambient Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Ambient Color Far
          effect.property("Ambient Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Far");\n' +
          '}\n' +
          'property1;';

          // Ambient Color Falloff
          effect.property("Ambient Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Falloff");\n' +
          '}\n' +
          'property1;';


          // Diffuse Toggle
          effect.property("Diffuse Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Toggle");\n' +
          '}\n' +
          'property1;';

          // Diffuse Falloff
          effect.property("Diffuse Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Falloff");\n' +
          '}\n' +
          'property1;';

          // Diffuse Intensity
          effect.property("Diffuse Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Intensity");\n' +
          '}\n' +
          'property1;';

          // Diffuse Saturation
          effect.property("Diffuse Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Saturation");\n' +
          '}\n' +
          'property1;';

          // Diffuse Near Color
          effect.property("Diffuse Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Near");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color Far Toggle
          effect.property("Diffuse Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color Far
          effect.property("Diffuse Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Far");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color Falloff
          effect.property("Diffuse Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Falloff");\n' +
          '}\n' +
          'property1;';


          // Specular Toggle
          effect.property("Specular Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Toggle");\n' +
          '}\n' +
          'property1;';

          // Specular Falloff
          effect.property("Specular Size" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Size");\n' +
          '}\n' +
          'property1;';

          // Specular Falloff
          effect.property("Specular Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Falloff");\n' +
          '}\n' +
          'property1;';

          // Specular Intensity
          effect.property("Specular Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Intensity");\n' +
          '}\n' +
          'property1;';

          // Specular Saturation
          effect.property("Specular Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Saturation");\n' +
          '}\n' +
          'property1;';

          // Specular Near Color
          effect.property("Specular Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Near");\n' +
          '}\n' +
          'property1;';

          // Specular Color Far Toggle
          effect.property("Specular Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Specular Color Far
          effect.property("Specular Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Far");\n' +
          '}\n' +
          'property1;';

          // Specular Color Falloff
          effect.property("Specular Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Falloff");\n' +
          '}\n' +
          'property1;';



          // Shadow Toggle
          effect.property("Shadow Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Ambient Toggle
          effect.property("Shadow Ignore Ambient Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Ambient Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Diffuse Toggle
          effect.property("Shadow Ignore Diffuse Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Diffuse Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Specular Toggle
          effect.property("Shadow Ignore Specular Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Specular Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Clip To Light Toggle
          effect.property("Shadow Clip To Light Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Clip To Light Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Sample Step
          effect.property("Shadow Sample Step" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Sample Step");\n' +
          '}\n' +
          'property1;';

          // Shadow Improved Sample Radius
          effect.property("Shadow Improved Sample Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Improved Sample Radius");\n' +
          '}\n' +
          'property1;';

          // Shadow Max Length
          effect.property("Shadow Max Length" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Max Length");\n' +
          '}\n' +
          'property1;';

          // Shadow Threshold Start
          effect.property("Shadow Threshold Start" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Threshold Start");\n' +
          '}\n' +
          'property1;';

          // Shadow Threshold End
          effect.property("Shadow Threshold End" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Threshold End");\n' +
          '}\n' +
          'property1;';

          // Shadow Softness Radius
          effect.property("Shadow Softness Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Softness Radius");\n' +
          '}\n' +
          'property1;';

          // Shadow Softness Samples
          effect.property("Shadow Softness Samples" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Softness Samples");\n' +
          '}\n' +
          'property1;';

          // Shadow Intensity
          effect.property("Shadow Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Intensity");\n' +
          '}\n' +
          'property1;';

          // Shadow Color
          effect.property("Shadow Color" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Color");\n' +
          '}\n' +
          'property1;';

        }

        function addPointAdvancedCUDA() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-depth2normal")) {
            return;
          }

          if (!pluginExistance(mainLayer, "ADBE r-point-advanced")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-clamp")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-unmult")) {
            return;
          }

          var normalPassLayer = comp.layers.byName("Normal Pass 1");
          if (!normalPassLayer) {
            var index = getUniqueIndex(comp, ["Normal Pass"]);
            var normalPassName = "Normal Pass" + " " + index;
            
            normalPassLayer = mainLayer.duplicate();
            normalPassLayer.name = normalPassName;
            normalPassLayer.enabled = false;
            normalPassLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
            normalPassLayer.label = 1; //red
            normalPassLayer.moveAfter(mainLayer);

            var normalPassEffect = normalPassLayer.property("ADBE Effect Parade").addProperty("ADBE r-depth2normal");
            normalPassEffect.property("Depth Far").setValue(far);
            normalPassEffect.property("Depth Black Is Near").setValue(blackIsNear);
          }

          // Index
          var index1 = getUniqueIndex(comp, ["Point Advanced", "Clamp"]);
          var index2 = getUniqueIndex(comp, ["Point Light"]);
          var pointLightLayerName = "Point Light" + " " + index2;
          var baseName = "Point Advanced" + " " + index1;
          var adjName = "Clamp" + " " + index1;

          // Adjustment Layer w Clamp effect
          var adjLayer = comp.layers.addSolid([1, 1, 1], adjName, comp.width, comp.height, 1);
          adjLayer.adjustmentLayer = true;
          adjLayer.name = adjName;
          adjLayer.startTime = mainLayer.startTime;
          adjLayer.inPoint = mainLayer.inPoint;
          adjLayer.outPoint = mainLayer.outPoint;
          adjLayer.label = labelColor;
          adjLayer.property("ADBE Effect Parade").addProperty("ADBE r-clamp");

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Point Light Layer
          var pointLightLayer = comp.layers.addNull(comp.width);
          pointLightLayer.name = pointLightLayerName;
          pointLightLayer.threeDLayer = true;
          pointLightLayer.transform.position.setValue(projectedPosition);
          pointLightLayer.transform.orientation.setValue(projectedOrientation);
          pointLightLayer.transform.scale.setValue(projectedScale);
          pointLightLayer.startTime = mainLayer.startTime;
          pointLightLayer.inPoint = mainLayer.inPoint;
          pointLightLayer.outPoint = mainLayer.outPoint;
          pointLightLayer.label = labelColor;

          var pointLightEffect = applyPresetFile(pointLightLayer, "r-point-advanced-light");
          if (pointLightEffect === 0) {
            return;
          }
          pointLightEffect.property("Diffuse Color Near").setValue(randomColorRGB);
          pointLightEffect.property("Diffuse Color Far").setValue(randomColorRGB2);
 
          // Base effect settings
          //var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-point-advanced");
          var baseEffect = applyPresetFile(baseLayer, "r-point-advanced-plugin");
          if (baseEffect === 0) {
            return;
          }
          baseEffect.property("Depth Far").setValue(far);
          baseEffect.property("Depth Black Is Near").setValue(blackIsNear);
          baseEffect.property("Normal Pass Normalized").setValue(normalPassLayer.index);
          baseEffect.property("Light 1").setValue(pointLightLayer.index);

          // Expressions
          for (var i = 1; i <= 10; i++) {
            addExpressionsPointAdvancedCUDA(baseEffect, i);
          }

          // Unmult effect
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-unmult");

        }
        addPointAdvancedCUDA();
        break;

      case 'Spot Simple (CUDA)':
        function addSpotSimpleExpressions(effect, index) {

          var lightEffectName = "r-spot-simple-light";

          // Toggle
          effect.property("Toggle" + " " + index).expression =
          '// Light Toggle\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightToggle = value;\n' +
          'if (lightLayer) {\n' +
          '  lightToggle = lightLayer.enabled;\n' +
          '}\n' +
          'lightToggle;';

          // Position
          effect.property("Position" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = lightLayer.toWorld(lightLayer.anchorPoint);\n' +
          '}\n' +
          'lightPosition;';

          // Vector X
          effect.property("Vector X" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = normalize(lightLayer.toWorldVec([1.0, 0.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
          '}\n' +
          'lightPosition;';

          // Vector Y
          effect.property("Vector Y" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = normalize(lightLayer.toWorldVec([0.0, -1.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
          '}\n' +
          'lightPosition;';

          // Vector Z
          effect.property("Vector Z" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = normalize(lightLayer.toWorldVec([0.0, 0.0, -1.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
          '}\n' +
          'lightPosition;';

          // Invert
          effect.property("Invert" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Invert");\n' +
          '}\n' +
          'property1;';

          // Length
          effect.property("Length" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Length");\n' +
          '}\n' +
          'property1;';

          // Angle X
          effect.property("Angle X" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Angle X");\n' +
          '}\n' +
          'property1;';

          // Angle Y
          effect.property("Angle Y" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Angle Y");\n' +
          '}\n' +
          'property1;';

          // Curvature
          effect.property("Curvature" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Curvature");\n' +
          '}\n' +
          'property1;';

          // Feather
          effect.property("Feather" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather");\n' +
          '}\n' +
          'property1;';

          // Falloff
          effect.property("Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Falloff");\n' +
          '}\n' +
          'property1;';

          // IES
          effect.property("IES" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("IES");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 1
          effect.property("Brightness / Distance 1" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 1");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 2
          effect.property("Brightness / Distance 2" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 2");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 3
          effect.property("Brightness / Distance 3" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 3");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 4
          effect.property("Brightness / Distance 4" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 4");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 5
          effect.property("Brightness / Distance 5" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 5");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 6
          effect.property("Brightness / Distance 6" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 6");\n' +
          '}\n' +
          'property1;';

          // Intensity
          effect.property("Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Intensity");\n' +
          '}\n' +
          'property1;';

          // Saturation
          effect.property("Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Saturation");\n' +
          '}\n' +
          'property1;';

          // Color Near
          effect.property("Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Near");\n' +
          '}\n' +
          'property1;';

          // Color Far Toggle
          effect.property("Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Color Far
          effect.property("Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Far");\n' +
          '}\n' +
          'property1;';

          // Color Falloff
          effect.property("Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Falloff");\n' +
          '}\n' +
          'property1;';

        }

        function addSpotSimpleCuda() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-spot-simple")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-clamp")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-unmult")) {
            return;
          }

          // Index
          var index1 = getUniqueIndex(comp, ["Spot Simple", "Clamp"]);
          var index2 = getUniqueIndex(comp, ["Spot Light", "Spot Look At"]);
          var spotLookAtLayerName = "Spot Look At" + " " + index2;
          var spotLightLayerName = "Spot Light" + " " + index2;
          var baseName = "Spot Simple" + " " + index1;
          var adjName = "Clamp" + " " + index1;

          // Adjustment Layer w Clamp effect
          var adjLayer = comp.layers.addSolid([1, 1, 1], adjName, comp.width, comp.height, 1);
          adjLayer.adjustmentLayer = true;
          adjLayer.name = adjName;
          adjLayer.startTime = mainLayer.startTime;
          adjLayer.inPoint = mainLayer.inPoint;
          adjLayer.outPoint = mainLayer.outPoint;
          adjLayer.label = labelColor;
          adjLayer.property("ADBE Effect Parade").addProperty("ADBE r-clamp");

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Spot Look At Layer
          var spotLookAtLayer = comp.layers.addNull(comp.width);
          spotLookAtLayer.name = spotLookAtLayerName;
          spotLookAtLayer.threeDLayer = true;
          spotLookAtLayer.transform.position.setValue(projectedPosition);
          spotLookAtLayer.transform.anchorPoint.setValue([50.0, 50.0, 0.0]);
          spotLookAtLayer.transform.orientation.setValue(projectedOrientation);
          spotLookAtLayer.transform.scale.setValue(projectedScale);
          spotLookAtLayer.source.width = 100.0;
          spotLookAtLayer.source.height = 100.0;
          spotLookAtLayer.startTime = mainLayer.startTime;
          spotLookAtLayer.inPoint = mainLayer.inPoint;
          spotLookAtLayer.outPoint = mainLayer.outPoint;
          spotLookAtLayer.label = labelColor;

          // Spot Light Layer
          var spotLightLayer = comp.layers.addNull(comp.width);
          spotLightLayer.name = spotLightLayerName;
          spotLightLayer.threeDLayer = true;
          spotLightLayer.transform.position.setValue(projectedPosition);
          spotLightLayer.transform.anchorPoint.setValue([50.0, 50.0, 0.0]);
          spotLightLayer.transform.orientation.setValue(projectedOrientation);
          spotLightLayer.transform.orientation.expression =
            'var lookAtLayer = effect("r-spot-simple-light")("Look At");\n' +
            'var property1 = value;\n' +
            'if (lookAtLayer) {\n' +
            '  property1 = lookAt(lookAtLayer.toWorld([0.0, 0.0, 0.0]), thisLayer.toWorld([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'property1;';
          spotLightLayer.transform.scale.setValue(projectedScale);
          spotLightLayer.source.width = 100.0;
          spotLightLayer.source.height = 100.0;
          spotLightLayer.startTime = mainLayer.startTime;
          spotLightLayer.inPoint = mainLayer.inPoint;
          spotLightLayer.outPoint = mainLayer.outPoint;
          spotLightLayer.label = labelColor;
          spotLightLayer.moveAfter(spotLookAtLayer);

          // Spot Light Layer Effect
          var spotLightEffect = applyPresetFile(spotLightLayer, "r-spot-simple-light");
          if (spotLightEffect === 0) {
            return;
          }
          var lightEffectName = "r-spot-simple-light";
          spotLightEffect.property("Look At").setValue(spotLookAtLayer.index);
          spotLightEffect.property("Length").expression =
            'var lookAtLayer = effect("' +  lightEffectName + '")("Look At");\n' +
            'var lengthToggle = effect("' +  lightEffectName + '")("Length Is Distance To Look At");\n' +
            'var property1 = value;\n' +
            'if (lookAtLayer && lengthToggle == true) {\n' +
            '  property1 = length(lookAtLayer.toWorld([0.0, 0.0, 0.0]), thisLayer.toWorld([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'property1;';
          spotLightEffect.property("Angle X").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Angle Uniform Toggle");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  property1 = effect("' +  lightEffectName + '")("Angle Uniform");\n' +
            '}\n' +
            'property1;';
          spotLightEffect.property("Angle Y").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Angle Uniform Toggle");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  property1 = effect("' +  lightEffectName + '")("Angle Uniform");\n' +
            '}\n' +
            'property1;';
          spotLightEffect.property("Color Near").setValue(randomColorRGB);
          spotLightEffect.property("Color Far").setValue(randomColorRGB2);
 
          // Base effect settings
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-spot-simple");
          baseEffect.property("Depth Far").setValue(far);
          baseEffect.property("Depth Black Is Near").setValue(blackIsNear);
          baseEffect.property("Light 1").setValue(spotLightLayer.index);

          // Expressions
          for (var i = 1; i <= 10; i++) {
            addSpotSimpleExpressions(baseEffect, i);
          }

          // Unmult effect
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-unmult");

        }
        addSpotSimpleCuda();
        break;
      
      case 'Spot Advanced (CUDA)':

        function addExpressionsSpotAdvancedCUDA(effect, index) {

          var lightEffectName = "r-spot-advanced-light";

          // Light Toggle
          effect.property("Light Toggle" + " " + index).expression =
          '// Light Toggle\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightToggle = value;\n' +
          'if (lightLayer) {\n' +
          '  lightToggle = lightLayer.enabled;\n' +
          '}\n' +
          'lightToggle;';

          // Light Position
          effect.property("Light Position" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = lightLayer.toWorld(lightLayer.anchorPoint);\n' +
          '}\n' +
          'lightPosition;';

          // Light Vector X
          effect.property("Light Vector X" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = normalize(lightLayer.toWorldVec([1.0, 0.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
          '}\n' +
          'lightPosition;';

          // Light Vector Y
          effect.property("Light Vector Y" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = normalize(lightLayer.toWorldVec([0.0, -1.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
          '}\n' +
          'lightPosition;';

          // Light Vector Z
          effect.property("Light Vector Z" + " " + index).expression =
          '// Light Position Local\n' +
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var lightPosition = value;\n' +
          'if (lightLayer) {\n' +
          '    lightPosition = normalize(lightLayer.toWorldVec([0.0, 0.0, -1.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
          '}\n' +
          'lightPosition;';

          // Invert Toggle
          effect.property("Invert Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Invert Toggle");\n' +
          '}\n' +
          'property1;';

          // Length
          effect.property("Length" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Length");\n' +
          '}\n' +
          'property1;';

          // Angle X
          effect.property("Angle X" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Angle X");\n' +
          '}\n' +
          'property1;';

          // Angle Y
          effect.property("Angle Y" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Angle Y");\n' +
          '}\n' +
          'property1;';

          // Curvature
          effect.property("Curvature" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Curvature");\n' +
          '}\n' +
          'property1;';

          // Feather
          effect.property("Feather" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather");\n' +
          '}\n' +
          'property1;';

          // Falloff
          effect.property("Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Falloff");\n' +
          '}\n' +
          'property1;';

          // IES
          effect.property("IES" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("IES");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 1
          effect.property("Brightness / Distance 1" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 1");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 2
          effect.property("Brightness / Distance 2" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 2");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 3
          effect.property("Brightness / Distance 3" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 3");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 4
          effect.property("Brightness / Distance 4" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 4");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 5
          effect.property("Brightness / Distance 5" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 5");\n' +
          '}\n' +
          'property1;';

          // Brightness / Distance 6
          effect.property("Brightness / Distance 6" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Brightness / Distance 6");\n' +
          '}\n' +
          'property1;';


          



          // Ambient Toggle
          effect.property("Ambient Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Toggle");\n' +
          '}\n' +
          'property1;';

          // Ambient Intensity
          effect.property("Ambient Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Intensity");\n' +
          '}\n' +
          'property1;';

          // Ambient Saturation
          effect.property("Ambient Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Saturation");\n' +
          '}\n' +
          'property1;';

          // Ambient Near Color
          effect.property("Ambient Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Near");\n' +
          '}\n' +
          'property1;';

          // Ambient Color Far Toggle
          effect.property("Ambient Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Ambient Color Far
          effect.property("Ambient Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Far");\n' +
          '}\n' +
          'property1;';

          // Ambient Color Falloff
          effect.property("Ambient Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Falloff");\n' +
          '}\n' +
          'property1;';




          // Diffuse Toggle
          effect.property("Diffuse Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Toggle");\n' +
          '}\n' +
          'property1;';

          // Diffuse Intensity
          effect.property("Diffuse Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Intensity");\n' +
          '}\n' +
          'property1;';

          // Diffuse Saturation
          effect.property("Diffuse Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Saturation");\n' +
          '}\n' +
          'property1;';

          // Diffuse Near Color
          effect.property("Diffuse Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Near");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color Far Toggle
          effect.property("Diffuse Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color Far
          effect.property("Diffuse Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Far");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color Falloff
          effect.property("Diffuse Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Falloff");\n' +
          '}\n' +
          'property1;';


          // Specular Toggle
          effect.property("Specular Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Toggle");\n' +
          '}\n' +
          'property1;';

          // Specular Size
          effect.property("Specular Size" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Size");\n' +
          '}\n' +
          'property1;';

          // Specular Intensity
          effect.property("Specular Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Intensity");\n' +
          '}\n' +
          'property1;';

          // Specular Saturation
          effect.property("Specular Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Saturation");\n' +
          '}\n' +
          'property1;';

          // Specular Near Color
          effect.property("Specular Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Near");\n' +
          '}\n' +
          'property1;';

          // Specular Color Far Toggle
          effect.property("Specular Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Specular Color Far
          effect.property("Specular Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Far");\n' +
          '}\n' +
          'property1;';

          // Specular Color Falloff
          effect.property("Specular Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Falloff");\n' +
          '}\n' +
          'property1;';



          // Shadow Toggle
          effect.property("Shadow Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Ambient Toggle
          effect.property("Shadow Ignore Ambient Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Ambient Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Diffuse Toggle
          effect.property("Shadow Ignore Diffuse Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Diffuse Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Specular Toggle
          effect.property("Shadow Ignore Specular Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Specular Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Clip To Light Toggle
          effect.property("Shadow Clip To Light Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Clip To Light Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Sample Step
          effect.property("Shadow Sample Step" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Sample Step");\n' +
          '}\n' +
          'property1;';

          // Shadow Improved Sample Radius
          effect.property("Shadow Improved Sample Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Improved Sample Radius");\n' +
          '}\n' +
          'property1;';

          // Shadow Max Length
          effect.property("Shadow Max Length" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Max Length");\n' +
          '}\n' +
          'property1;';

          // Shadow Threshold Start
          effect.property("Shadow Threshold Start" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Threshold Start");\n' +
          '}\n' +
          'property1;';

          // Shadow Threshold End
          effect.property("Shadow Threshold End" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Threshold End");\n' +
          '}\n' +
          'property1;';

          // Shadow Softness Radius
          effect.property("Shadow Softness Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Softness Radius");\n' +
          '}\n' +
          'property1;';

          // Shadow Softness Samples
          effect.property("Shadow Softness Samples" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Softness Samples");\n' +
          '}\n' +
          'property1;';

          // Shadow Intensity
          effect.property("Shadow Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Intensity");\n' +
          '}\n' +
          'property1;';

          // Shadow Color
          effect.property("Shadow Color" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Color");\n' +
          '}\n' +
          'property1;';

        }

        function addSpotAdvancedCUDA() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-depth2normal")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-spot-advanced")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-clamp")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-unmult")) {
            return;
          }

          // Normal Pass
          var normalPassLayer = comp.layers.byName("Normal Pass 1");
          if (!normalPassLayer) {
            var index = getUniqueIndex(comp, ["Normal Pass"]);
            var normalPassName = "Normal Pass" + " " + index;
            
            normalPassLayer = mainLayer.duplicate();
            normalPassLayer.name = normalPassName;
            normalPassLayer.enabled = false;
            normalPassLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
            normalPassLayer.label = 1; //red
            normalPassLayer.moveAfter(mainLayer);

            var normalPassEffect = normalPassLayer.property("ADBE Effect Parade").addProperty("ADBE r-depth2normal");
            normalPassEffect.property("Depth Far").setValue(far);
            normalPassEffect.property("Depth Black Is Near").setValue(blackIsNear);
          }

          // Index
          var index1 = getUniqueIndex(comp, ["Spot Advanced", "Clamp"]);
          var index2 = getUniqueIndex(comp, ["Spot Light", "Spot Look At"]);
          var spotLookAtLayerName = "Spot Look At" + " " + index2;
          var spotLightLayerName = "Spot Light" + " " + index2;
          var baseName = "Spot Advanced" + " " + index1;
          var adjName = "Clamp" + " " + index1;

          // Adjustment Layer w Clamp effect
          var adjLayer = comp.layers.addSolid([1, 1, 1], adjName, comp.width, comp.height, 1);
          adjLayer.adjustmentLayer = true;
          adjLayer.name = adjName;
          adjLayer.startTime = mainLayer.startTime;
          adjLayer.inPoint = mainLayer.inPoint;
          adjLayer.outPoint = mainLayer.outPoint;
          adjLayer.label = labelColor;
          adjLayer.property("ADBE Effect Parade").addProperty("ADBE r-clamp");

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Spot Look At Layer
          var spotLookAtLayer = comp.layers.addNull(comp.width);
          spotLookAtLayer.name = spotLookAtLayerName;
          spotLookAtLayer.threeDLayer = true;
          spotLookAtLayer.transform.position.setValue(projectedPosition);
          spotLookAtLayer.transform.anchorPoint.setValue([50.0, 50.0, 0.0]);
          spotLookAtLayer.transform.orientation.setValue(projectedOrientation);
          spotLookAtLayer.transform.scale.setValue(projectedScale);
          spotLookAtLayer.source.width = 100.0;
          spotLookAtLayer.source.height = 100.0;
          spotLookAtLayer.startTime = mainLayer.startTime;
          spotLookAtLayer.inPoint = mainLayer.inPoint;
          spotLookAtLayer.outPoint = mainLayer.outPoint;
          spotLookAtLayer.label = labelColor;

          // Spot Light Layer
          var spotLightLayer = comp.layers.addNull(comp.width);
          spotLightLayer.name = spotLightLayerName;
          spotLightLayer.threeDLayer = true;
          spotLightLayer.transform.position.setValue(projectedPosition);
          spotLightLayer.transform.anchorPoint.setValue([50.0, 50.0, 0.0]);
          spotLightLayer.transform.orientation.setValue(projectedOrientation);
          spotLightLayer.transform.orientation.expression =
            'var lookAtLayer = effect("r-spot-advanced-light")("Look At");\n' +
            'var property1 = value;\n' +
            'if (lookAtLayer) {\n' +
            '  property1 = lookAt(lookAtLayer.toWorld([0.0, 0.0, 0.0]), thisLayer.toWorld([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'property1;';
          spotLightLayer.transform.scale.setValue(projectedScale);
          spotLightLayer.source.width = 100.0;
          spotLightLayer.source.height = 100.0;
          spotLightLayer.startTime = mainLayer.startTime;
          spotLightLayer.inPoint = mainLayer.inPoint;
          spotLightLayer.outPoint = mainLayer.outPoint;
          spotLightLayer.label = labelColor;
          spotLightLayer.moveAfter(spotLookAtLayer);

          // Spot Light Layer Effect
          var spotLightEffect = applyPresetFile(spotLightLayer, "r-spot-advanced-light");
          if (spotLightEffect === 0) {
            return;
          }
          var lightEffectName = "r-spot-advanced-light";
          spotLightEffect.property("Look At").setValue(spotLookAtLayer.index);
          spotLightEffect.property("Length").expression =
            'var lookAtLayer = effect("' +  lightEffectName + '")("Look At");\n' +
            'var lengthToggle = effect("' +  lightEffectName + '")("Length Is Distance To Look At");\n' +
            'var property1 = value;\n' +
            'if (lookAtLayer && lengthToggle == true) {\n' +
            '  property1 = length(lookAtLayer.toWorld([0.0, 0.0, 0.0]), thisLayer.toWorld([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'property1;';
          spotLightEffect.property("Angle X").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Angle Uniform Toggle");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  property1 = effect("' +  lightEffectName + '")("Angle Uniform");\n' +
            '}\n' +
            'property1;';
          spotLightEffect.property("Angle Y").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Angle Uniform Toggle");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  property1 = effect("' +  lightEffectName + '")("Angle Uniform");\n' +
            '}\n' +
            'property1;';
          spotLightEffect.property("Diffuse Color Near").setValue(randomColorRGB);
          spotLightEffect.property("Diffuse Color Far").setValue(randomColorRGB2);
 
          // Base Effect
          //var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-spot-advanced");
          var baseEffect = applyPresetFile(baseLayer, "r-spot-advanced-plugin");
          if (baseEffect === 0) {
            return;
          }
          baseEffect.property("Depth Far").setValue(far);
          baseEffect.property("Depth Black Is Near").setValue(blackIsNear);
          baseEffect.property("Normal Pass Normalized").setValue(normalPassLayer.index);
          baseEffect.property("Light 1").setValue(spotLightLayer.index);

          // Expressions
          for (var i = 1; i <= 10; i++) {
            addExpressionsSpotAdvancedCUDA(baseEffect, i);
          }

          // Unmult effect
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-unmult");

        }
        addSpotAdvancedCUDA();
        break;

      case 'Rect Simple (CUDA)':
        function addRectSimpleExpressions(effect, index) {

          var lightEffectName = "r-rect-simple-light";

          // Toggle
          effect.property("Light Toggle" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightToggle = value;\n' +
            'if (lightLayer) {\n' +
            '  lightToggle = lightLayer.enabled;\n' +
            '}\n' +
            'lightToggle;';

          // Position 1
          effect.property("Light Position 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';

          // Vector X 1
          effect.property("Light Vector X 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([1.0, 0.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Vector Y 1
          effect.property("Light Vector Y 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([0.0, -1.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Vector Z 1
          effect.property("Light Vector Z 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([0.0, 0.0, -1.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Resolution 1
          effect.property("Light Resolution 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = [lightLayer.width, lightLayer.height, 100.0];\n' +
            '}\n' +
            'property1;';

          // Scale 1
          effect.property("Light Scale 1" + " " + index).expression =
            'var lightLayerStart = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayerStart) {  \n' +
            '  if(lightLayerStart.hasParent) {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    var scaleValueEnd = lightLayerStart.parent.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueEnd[0] * scaleValueStart[0], scaleValueEnd[1] * scaleValueStart[1], scaleValueEnd[2] * scaleValueStart[2]];\n' +
            '  } else {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueStart[0], scaleValueStart[1], scaleValueStart[2]];\n' +
            '  }\n' +
            '}\n' +
            'property1;';

          

          // Position 2
          effect.property("Light Position 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';

          // Vector X 2
          effect.property("Light Vector X 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([1.0, 0.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Vector Y 2
          effect.property("Light Vector Y 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([0.0, -1.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Vector Z 2
          effect.property("Light Vector Z 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([0.0, 0.0, -1.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Resolution 2
          effect.property("Light Resolution 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = [lightLayer.width, lightLayer.height, 100.0];\n' +
            '}\n' +
            'property1;';

          // Scale 2
          effect.property("Light Scale 2" + " " + index).expression =
            'var lightLayerStart = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayerStart) {  \n' +
            '  if(lightLayerStart.hasParent) {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    var scaleValueEnd = lightLayerStart.parent.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueEnd[0] * scaleValueStart[0], scaleValueEnd[1] * scaleValueStart[1], scaleValueEnd[2] * scaleValueStart[2]];\n' +
            '  } else {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueStart[0], scaleValueStart[1], scaleValueStart[2]];\n' +
            '  }\n' +
            '}\n' +
            'property1;';


          // Invert
          effect.property("Invert Toggle" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Invert Toggle");\n' +
            '}\n' +
            'property1;';

          // Feather Normalize
          effect.property("Feather Normalize" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather Normalize");\n' +
            '}\n' +
            'property1;';

          // Feather X
          effect.property("Feather X" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather X");\n' +
            '}\n' +
            'property1;';

          // Feather Y
          effect.property("Feather Y" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather Y");\n' +
            '}\n' +
            'property1;';

          // Feather Z
          effect.property("Feather Z" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather Z");\n' +
            '}\n' +
            'property1;';

          // Falloff
          effect.property("Falloff" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Falloff");\n' +
            '}\n' +
            'property1;';

          // Intensity
          effect.property("Intensity" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Intensity");\n' +
            '}\n' +
            'property1;';

          // Saturation
          effect.property("Saturation" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Saturation");\n' +
            '}\n' +
            'property1;';

          // Color Near
          effect.property("Color Near" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Near");\n' +
            '}\n' +
            'property1;';

          // Color Far Toggle
          effect.property("Color Far Toggle" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Far Toggle");\n' +
            '}\n' +
            'property1;';

          // Color Far
          effect.property("Color Far" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Far");\n' +
            '}\n' +
            'property1;';

          // Color Falloff
          effect.property("Color Falloff" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Color Falloff");\n' +
            '}\n' +
            'property1;';
          
        }

        function addRectSimpleCuda() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-rect-simple")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-clamp")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-unmult")) {
            return;
          }

          // Index
          var index1 = getUniqueIndex(comp, ["Rect Simple", "Clamp"]);
          var index2 = getUniqueIndex(comp, ["Rect Start", "Rect End"]);
          var rectLightStartLayerName = "Rect Start" + " " + index2;
          var rectLightEndLayerName = "Rect End" + " " + index2;
          var baseName = "Rect Simple" + " " + index1;
          var adjName = "Clamp" + " " + index1;

          // Adjustment Layer w Clamp effect
          var adjLayer = comp.layers.addSolid([1, 1, 1], adjName, comp.width, comp.height, 1);
          adjLayer.adjustmentLayer = true;
          adjLayer.name = adjName;
          adjLayer.startTime = mainLayer.startTime;
          adjLayer.inPoint = mainLayer.inPoint;
          adjLayer.outPoint = mainLayer.outPoint;
          adjLayer.label = labelColor;
          adjLayer.property("ADBE Effect Parade").addProperty("ADBE r-clamp");

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Rect Start Layer
          var rectLightStartLayer = comp.layers.addNull(comp.width);
          rectLightStartLayer.name = rectLightStartLayerName;
          rectLightStartLayer.threeDLayer = true;
          rectLightStartLayer.transform.position.setValue(projectedPosition);
          rectLightStartLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          rectLightStartLayer.transform.orientation.setValue(projectedOrientation);
          rectLightStartLayer.transform.scale.setValue(projectedScale);
          rectLightStartLayer.source.width = 100.0;
          rectLightStartLayer.source.height = 100.0;
          rectLightStartLayer.startTime = mainLayer.startTime;
          rectLightStartLayer.inPoint = mainLayer.inPoint;
          rectLightStartLayer.outPoint = mainLayer.outPoint;
          rectLightStartLayer.label = labelColor;

          // Rect End Layer
          var rectLightEndLayer = comp.layers.addNull(comp.width);
          rectLightEndLayer.name = rectLightEndLayerName;
          rectLightEndLayer.threeDLayer = true;
          rectLightEndLayer.transform.position.setValue(projectedPosition);
          rectLightEndLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          rectLightEndLayer.transform.orientation.setValue(projectedOrientation);
          rectLightEndLayer.transform.scale.setValue(projectedScale);
          rectLightEndLayer.source.width = 100.0;
          rectLightEndLayer.source.height = 100.0;
          rectLightEndLayer.startTime = mainLayer.startTime;
          rectLightEndLayer.inPoint = mainLayer.inPoint;
          rectLightEndLayer.outPoint = mainLayer.outPoint;
          rectLightEndLayer.label = labelColor;

          // Rect Light Start Effect
          var rectLightEffect = applyPresetFile(rectLightStartLayer, "r-rect-simple-light");
          if (rectLightEffect === 0) {
            return;
          }
          var lightEffectName = "r-rect-simple-light";
          rectLightEffect.property("Feather X").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Feather Uniform");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  var tempProperty1 = effect("' +  lightEffectName + '")("Feather");\n' +
            '  property1 = [tempProperty1, tempProperty1];\n' +
            '}\n' +
            'property1;';
          rectLightEffect.property("Feather Y").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Feather Uniform");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  var tempProperty1 = effect("' +  lightEffectName + '")("Feather");\n' +
            '  property1 = [tempProperty1, tempProperty1];\n' +
            '}\n' +
            'property1;';
          rectLightEffect.property("Feather Z").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Feather Uniform");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  var tempProperty1 = effect("' +  lightEffectName + '")("Feather");\n' +
            '  property1 = [tempProperty1, tempProperty1];\n' +
            '}\n' +
            'property1;';
          rectLightEffect.property("Color Near").setValue(randomColorRGB);
          rectLightEffect.property("Color Far").setValue(randomColorRGB2);

          // Base effect settings
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-rect-simple");
          baseEffect.property("Depth Far").setValue(far);
          baseEffect.property("Depth Black Is Near").setValue(blackIsNear);
          baseEffect.property("Light Start 1").setValue(rectLightStartLayer.index);
          baseEffect.property("Light End 1").setValue(rectLightEndLayer.index);

          // Expressions
          for (var i = 1; i <= 10; i++) {
            addRectSimpleExpressions(baseEffect, i);
          }

          // Unmult effect
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-unmult");

        }
        addRectSimpleCuda();
      break;
      
      case 'Rect Advanced (CUDA)':

        function addExpressionsRectAdvancedCUDA(effect, index) {

          var lightEffectName = "r-rect-advanced-light";

          // Light Toggle
          effect.property("Light Toggle" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightToggle = value;\n' +
            'if (lightLayer) {\n' +
            '  lightToggle = lightLayer.enabled;\n' +
            '}\n' +
            'lightToggle;';

          // Light Position 1
          effect.property("Light Position 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';

          // Light Vector X 1
          effect.property("Light Vector X 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([1.0, 0.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Light Vector Y 1
          effect.property("Light Vector Y 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([0.0, -1.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Light Vector Z 1
          effect.property("Light Vector Z 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([0.0, 0.0, -1.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Light Resolution 1
          effect.property("Light Resolution 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = [lightLayer.width, lightLayer.height, 100.0];\n' +
            '}\n' +
            'property1;';

          // Light Scale 1
          effect.property("Light Scale 1" + " " + index).expression =
            'var lightLayerStart = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayerStart) {  \n' +
            '  if(lightLayerStart.hasParent) {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    var scaleValueEnd = lightLayerStart.parent.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueEnd[0] * scaleValueStart[0], scaleValueEnd[1] * scaleValueStart[1], scaleValueEnd[2] * scaleValueStart[2]];\n' +
            '  } else {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueStart[0], scaleValueStart[1], scaleValueStart[2]];\n' +
            '  }\n' +
            '}\n' +
            'property1;';

          

          // Light Position 2
          effect.property("Light Position 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';

          // Light Vector X 2
          effect.property("Light Vector X 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([1.0, 0.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Light Vector Y 2
          effect.property("Light Vector Y 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([0.0, -1.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Light Vector Z 2
          effect.property("Light Vector Z 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = normalize(lightLayer.toWorldVec([0.0, 0.0, -1.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';

          // Light Resolution 2
          effect.property("Light Resolution 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = [lightLayer.width, lightLayer.height, 100.0];\n' +
            '}\n' +
            'property1;';

          // Light Scale 2
          effect.property("Light Scale 2" + " " + index).expression =
            'var lightLayerStart = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayerStart) {  \n' +
            '  if(lightLayerStart.hasParent) {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    var scaleValueEnd = lightLayerStart.parent.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueEnd[0] * scaleValueStart[0], scaleValueEnd[1] * scaleValueStart[1], scaleValueEnd[2] * scaleValueStart[2]];\n' +
            '  } else {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueStart[0], scaleValueStart[1], scaleValueStart[2]];\n' +
            '  }\n' +
            '}\n' +
            'property1;';

          // Feather Normalize
          effect.property("Feather Normalize" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather Normalize");\n' +
            '}\n' +
            'property1;';

          // Feather X
          effect.property("Feather X" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather X");\n' +
            '}\n' +
            'property1;';

          // Feather Y
          effect.property("Feather Y" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather Y");\n' +
            '}\n' +
            'property1;';

          // Feather Z
          effect.property("Feather Z" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Feather Z");\n' +
            '}\n' +
            'property1;';

          // Falloff
          effect.property("Falloff" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = lightLayer.effect("' +  lightEffectName + '")("Falloff");\n' +
            '}\n' +
            'property1;';

          

          
          // Ambient Toggle
          effect.property("Ambient Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Toggle");\n' +
          '}\n' +
          'property1;';

          // Ambient Intensity
          effect.property("Ambient Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Intensity");\n' +
          '}\n' +
          'property1;';

          // Ambient Saturation
          effect.property("Ambient Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Saturation");\n' +
          '}\n' +
          'property1;';

          // Ambient Near Color
          effect.property("Ambient Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Near");\n' +
          '}\n' +
          'property1;';

          // Ambient Color Far Toggle
          effect.property("Ambient Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Ambient Color Far
          effect.property("Ambient Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Far");\n' +
          '}\n' +
          'property1;';

          // Ambient Color Falloff
          effect.property("Ambient Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color Falloff");\n' +
          '}\n' +
          'property1;';




          // Diffuse Toggle
          effect.property("Diffuse Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Toggle");\n' +
          '}\n' +
          'property1;';

          // Diffuse Intensity
          effect.property("Diffuse Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Intensity");\n' +
          '}\n' +
          'property1;';

          // Diffuse Saturation
          effect.property("Diffuse Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Saturation");\n' +
          '}\n' +
          'property1;';

          // Diffuse Near Color
          effect.property("Diffuse Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Near");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color Far Toggle
          effect.property("Diffuse Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color Far
          effect.property("Diffuse Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Far");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color Falloff
          effect.property("Diffuse Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color Falloff");\n' +
          '}\n' +
          'property1;';


          // Specular Toggle
          effect.property("Specular Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Toggle");\n' +
          '}\n' +
          'property1;';

          // Specular Size
          effect.property("Specular Size" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Size");\n' +
          '}\n' +
          'property1;';

          // Specular Intensity
          effect.property("Specular Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Intensity");\n' +
          '}\n' +
          'property1;';

          // Specular Saturation
          effect.property("Specular Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Saturation");\n' +
          '}\n' +
          'property1;';

          // Specular Near Color
          effect.property("Specular Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Near");\n' +
          '}\n' +
          'property1;';

          // Specular Color Far Toggle
          effect.property("Specular Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Specular Color Far
          effect.property("Specular Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Far");\n' +
          '}\n' +
          'property1;';

          // Specular Color Falloff
          effect.property("Specular Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color Falloff");\n' +
          '}\n' +
          'property1;';



          // Shadow Toggle
          effect.property("Shadow Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Ambient Toggle
          effect.property("Shadow Ignore Ambient Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Ambient Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Diffuse Toggle
          effect.property("Shadow Ignore Diffuse Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Diffuse Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Specular Toggle
          effect.property("Shadow Ignore Specular Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Specular Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Clip To Light Toggle
          effect.property("Shadow Clip To Light Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Clip To Light Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Sample Step
          effect.property("Shadow Sample Step" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Sample Step");\n' +
          '}\n' +
          'property1;';

          // Shadow Improved Sample Radius
          effect.property("Shadow Improved Sample Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Improved Sample Radius");\n' +
          '}\n' +
          'property1;';

          // Shadow Max Length
          effect.property("Shadow Max Length" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Max Length");\n' +
          '}\n' +
          'property1;';

          // Shadow Threshold Start
          effect.property("Shadow Threshold Start" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Threshold Start");\n' +
          '}\n' +
          'property1;';

          // Shadow Threshold End
          effect.property("Shadow Threshold End" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Threshold End");\n' +
          '}\n' +
          'property1;';

          // Shadow Softness Radius
          effect.property("Shadow Softness Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Softness Radius");\n' +
          '}\n' +
          'property1;';

          // Shadow Softness Samples
          effect.property("Shadow Softness Samples" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Softness Samples");\n' +
          '}\n' +
          'property1;';

          // Shadow Intensity
          effect.property("Shadow Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Intensity");\n' +
          '}\n' +
          'property1;';

          // Shadow Color
          effect.property("Shadow Color" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Color");\n' +
          '}\n' +
          'property1;';
          
        }

        function addRectAdvancedCUDA() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-depth2normal")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-rect-advanced")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-clamp")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-unmult")) {
            return;
          }

          // Normal Pass
          var normalPassLayer = comp.layers.byName("Normal Pass 1");
          if (!normalPassLayer) {
            var index = getUniqueIndex(comp, ["Normal Pass"]);
            var normalPassName = "Normal Pass" + " " + index;
            
            normalPassLayer = mainLayer.duplicate();
            normalPassLayer.name = normalPassName;
            normalPassLayer.enabled = false;
            normalPassLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
            normalPassLayer.label = 1; //red
            normalPassLayer.moveAfter(mainLayer);

            var normalPassEffect = normalPassLayer.property("ADBE Effect Parade").addProperty("ADBE r-depth2normal");
            normalPassEffect.property("Depth Far").setValue(far);
            normalPassEffect.property("Depth Black Is Near").setValue(blackIsNear);
          }

          // Index
          var index1 = getUniqueIndex(comp, ["Rect Advanced", "Clamp"]);
          var index2 = getUniqueIndex(comp, ["Rect Start", "Rect End"]);
          var rectLightStartLayerName = "Rect Start" + " " + index2;
          var rectLightEndLayerName = "Rect End" + " " + index2;
          var baseName = "Rect Advanced" + " " + index1;
          var adjName = "Clamp" + " " + index1;

          // Adjustment Layer w Clamp effect
          var adjLayer = comp.layers.addSolid([1, 1, 1], adjName, comp.width, comp.height, 1);
          adjLayer.adjustmentLayer = true;
          adjLayer.name = adjName;
          adjLayer.startTime = mainLayer.startTime;
          adjLayer.inPoint = mainLayer.inPoint;
          adjLayer.outPoint = mainLayer.outPoint;
          adjLayer.label = labelColor;
          adjLayer.property("ADBE Effect Parade").addProperty("ADBE r-clamp");

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Rect Start Layer
          var rectLightStartLayer = comp.layers.addNull(comp.width);
          rectLightStartLayer.name = rectLightStartLayerName;
          rectLightStartLayer.threeDLayer = true;
          rectLightStartLayer.transform.position.setValue(projectedPosition);
          rectLightStartLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          rectLightStartLayer.transform.orientation.setValue(projectedOrientation);
          rectLightStartLayer.transform.scale.setValue(projectedScale);
          rectLightStartLayer.source.width = 100.0;
          rectLightStartLayer.source.height = 100.0;
          rectLightStartLayer.startTime = mainLayer.startTime;
          rectLightStartLayer.inPoint = mainLayer.inPoint;
          rectLightStartLayer.outPoint = mainLayer.outPoint;
          rectLightStartLayer.label = labelColor;

          // Rect End Layer
          var rectLightEndLayer = comp.layers.addNull(comp.width);
          rectLightEndLayer.name = rectLightEndLayerName;
          rectLightEndLayer.threeDLayer = true;
          rectLightEndLayer.transform.position.setValue(projectedPosition);
          rectLightEndLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          rectLightEndLayer.transform.orientation.setValue(projectedOrientation);
          rectLightEndLayer.transform.scale.setValue(projectedScale);
          rectLightEndLayer.source.width = 100.0;
          rectLightEndLayer.source.height = 100.0;
          rectLightEndLayer.startTime = mainLayer.startTime;
          rectLightEndLayer.inPoint = mainLayer.inPoint;
          rectLightEndLayer.outPoint = mainLayer.outPoint;
          rectLightEndLayer.label = labelColor;


          // Rect Light Start Effect
          var rectLightEffect = applyPresetFile(rectLightStartLayer, "r-rect-advanced-light");
          if (rectLightEffect === 0) {
            return;
          }
          var lightEffectName = "r-rect-advanced-light";
          rectLightEffect.property("Feather X").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Feather Uniform");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  var tempProperty1 = effect("' +  lightEffectName + '")("Feather");\n' +
            '  property1 = [tempProperty1, tempProperty1];\n' +
            '}\n' +
            'property1;';
          rectLightEffect.property("Feather Y").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Feather Uniform");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  var tempProperty1 = effect("' +  lightEffectName + '")("Feather");\n' +
            '  property1 = [tempProperty1, tempProperty1];\n' +
            '}\n' +
            'property1;';
          rectLightEffect.property("Feather Z").expression =
            'var uniformToggle = effect("' +  lightEffectName + '")("Feather Uniform");\n' +
            'var property1 = value;\n' +
            'if (uniformToggle == true) {\n' +
            '  var tempProperty1 = effect("' +  lightEffectName + '")("Feather");\n' +
            '  property1 = [tempProperty1, tempProperty1];\n' +
            '}\n' +
            'property1;';
          rectLightEffect.property("Ambient Color Near").setValue(randomColorRGB);
          rectLightEffect.property("Ambient Color Far").setValue(randomColorRGB2);
          rectLightEffect.property("Diffuse Color Near").setValue(randomColorRGB);
          rectLightEffect.property("Diffuse Color Far").setValue(randomColorRGB2);
          rectLightEffect.property("Specular Color Near").setValue(randomColorRGB);
          rectLightEffect.property("Specular Color Far").setValue(randomColorRGB2);

          // Base effect settings
          //var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-rect-advanced");
          var baseEffect = applyPresetFile(baseLayer, "r-rect-advanced-plugin");
          if (baseEffect === 0) {
            return;
          }
          baseEffect.property("Depth Far").setValue(far);
          baseEffect.property("Depth Black Is Near").setValue(blackIsNear);
          baseEffect.property("Normal Pass Normalized").setValue(normalPassLayer.index);
          baseEffect.property("Light Start 1").setValue(rectLightStartLayer.index);
          baseEffect.property("Light End 1").setValue(rectLightEndLayer.index);

          // Expressions
          for (var i = 1; i <= 10; i++) {
            addExpressionsRectAdvancedCUDA(baseEffect, i);
          }

          // Unmult effect
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-unmult");

        }
        addRectAdvancedCUDA();
        break;

      case 'Dir Light (CUDA)':

        function addExpressionsDirLightCUDA(effect, index) {

          var lightEffectName = "r-dir-light";

          // Light Toggle
          effect.property("Light Toggle" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightToggle = value;\n' +
            'if (lightLayer) {\n' +
            '  lightToggle = lightLayer.enabled;\n' +
            '}\n' +
            'lightToggle;';

          // Light Position 1
          effect.property("Light Position 1" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';

          // Light Position 2
          effect.property("Light Position 2" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light End ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';
          
          // Ambient Toggle
          effect.property("Ambient Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Toggle");\n' +
          '}\n' +
          'property1;';

          // Ambient Intensity
          effect.property("Ambient Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Intensity");\n' +
          '}\n' +
          'property1;';

          // Ambient Saturation
          effect.property("Ambient Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Saturation");\n' +
          '}\n' +
          'property1;';

          // Ambient Color
          effect.property("Ambient Color" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Ambient Color");\n' +
          '}\n' +
          'property1;';



          // Diffuse Toggle
          effect.property("Diffuse Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Toggle");\n' +
          '}\n' +
          'property1;';

          // Diffuse Intensity
          effect.property("Diffuse Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Intensity");\n' +
          '}\n' +
          'property1;';

          // Diffuse Saturation
          effect.property("Diffuse Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Saturation");\n' +
          '}\n' +
          'property1;';

          // Diffuse Color
          effect.property("Diffuse Color" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Diffuse Color");\n' +
          '}\n' +
          'property1;';



          // Specular Toggle
          effect.property("Specular Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Toggle");\n' +
          '}\n' +
          'property1;';

          // Specular Size
          effect.property("Specular Size" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Size");\n' +
          '}\n' +
          'property1;';

          // Specular Intensity
          effect.property("Specular Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Intensity");\n' +
          '}\n' +
          'property1;';

          // Specular Saturation
          effect.property("Specular Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Saturation");\n' +
          '}\n' +
          'property1;';

          // Specular Color
          effect.property("Specular Color" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Specular Color");\n' +
          '}\n' +
          'property1;';



          // Shadow Toggle
          effect.property("Shadow Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Ambient Toggle
          effect.property("Shadow Ignore Ambient Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Ambient Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Diffuse Toggle
          effect.property("Shadow Ignore Diffuse Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Diffuse Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Ignore Specular Toggle
          effect.property("Shadow Ignore Specular Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Ignore Specular Toggle");\n' +
          '}\n' +
          'property1;';

          // Shadow Sample Step
          effect.property("Shadow Sample Step" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Sample Step");\n' +
          '}\n' +
          'property1;';

          // Shadow Improved Sample Radius
          effect.property("Shadow Improved Sample Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Improved Sample Radius");\n' +
          '}\n' +
          'property1;';

          // Shadow Max Length
          effect.property("Shadow Max Length" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Max Length");\n' +
          '}\n' +
          'property1;';

          // Shadow Threshold Start
          effect.property("Shadow Threshold Start" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Threshold Start");\n' +
          '}\n' +
          'property1;';

          // Shadow Threshold End
          effect.property("Shadow Threshold End" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Threshold End");\n' +
          '}\n' +
          'property1;';

          // Shadow Softness Radius
          effect.property("Shadow Softness Radius" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Softness Radius");\n' +
          '}\n' +
          'property1;';

          // Shadow Softness Samples
          effect.property("Shadow Softness Samples" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Softness Samples");\n' +
          '}\n' +
          'property1;';

          // Shadow Intensity
          effect.property("Shadow Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Intensity");\n' +
          '}\n' +
          'property1;';

          // Shadow Color
          effect.property("Shadow Color" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light Start ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Shadow Color");\n' +
          '}\n' +
          'property1;';
          
        }

        function addDirLightCUDA() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-depth2normal")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-dir")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-clamp")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-unmult")) {
            return;
          }

          // Normal Pass
          var normalPassLayer = comp.layers.byName("Normal Pass 1");
          if (!normalPassLayer) {
            var index = getUniqueIndex(comp, ["Normal Pass"]);
            var normalPassName = "Normal Pass" + " " + index;
            
            normalPassLayer = mainLayer.duplicate();
            normalPassLayer.name = normalPassName;
            normalPassLayer.enabled = false;
            normalPassLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
            normalPassLayer.label = 1; //red
            normalPassLayer.moveAfter(mainLayer);

            var normalPassEffect = normalPassLayer.property("ADBE Effect Parade").addProperty("ADBE r-depth2normal");
            normalPassEffect.property("Depth Far").setValue(far);
            normalPassEffect.property("Depth Black Is Near").setValue(blackIsNear);
          }

          // Index
          var index1 = getUniqueIndex(comp, ["Dir", "Clamp"]);
          var index2 = getUniqueIndex(comp, ["Dir Start", "Dir End"]);
          var dirLightStartLayerName = "Dir Start" + " " + index2;
          var dirLightEndLayerName = "Dir End" + " " + index2;
          var baseName = "Dir" + " " + index1;
          var adjName = "Clamp" + " " + index1;

          // Adjustment Layer w Clamp effect
          var adjLayer = comp.layers.addSolid([1, 1, 1], adjName, comp.width, comp.height, 1);
          adjLayer.adjustmentLayer = true;
          adjLayer.name = adjName;
          adjLayer.startTime = mainLayer.startTime;
          adjLayer.inPoint = mainLayer.inPoint;
          adjLayer.outPoint = mainLayer.outPoint;
          adjLayer.label = labelColor;
          adjLayer.property("ADBE Effect Parade").addProperty("ADBE r-clamp");

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Dir Start Layer
          var dirLightStartLayer = comp.layers.addNull(comp.width);
          dirLightStartLayer.name = dirLightStartLayerName;
          dirLightStartLayer.threeDLayer = true;
          dirLightStartLayer.transform.position.setValue(projectedPosition);
          dirLightStartLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          dirLightStartLayer.transform.orientation.setValue(projectedOrientation);
          dirLightStartLayer.transform.scale.setValue(projectedScale);
          dirLightStartLayer.source.width = 100.0;
          dirLightStartLayer.source.height = 100.0;
          dirLightStartLayer.startTime = mainLayer.startTime;
          dirLightStartLayer.inPoint = mainLayer.inPoint;
          dirLightStartLayer.outPoint = mainLayer.outPoint;
          dirLightStartLayer.label = labelColor;

          // Dir End Layer
          var dirLightEndLayer = comp.layers.addNull(comp.width);
          dirLightEndLayer.name = dirLightEndLayerName;
          dirLightEndLayer.threeDLayer = true;
          dirLightEndLayer.transform.position.setValue(projectedPosition);
          dirLightEndLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          dirLightEndLayer.transform.orientation.setValue(projectedOrientation);
          dirLightEndLayer.transform.scale.setValue(projectedScale);
          dirLightEndLayer.source.width = 100.0;
          dirLightEndLayer.source.height = 100.0;
          dirLightEndLayer.startTime = mainLayer.startTime;
          dirLightEndLayer.inPoint = mainLayer.inPoint;
          dirLightEndLayer.outPoint = mainLayer.outPoint;
          dirLightEndLayer.label = labelColor;


          // Rect Light Start Effect
          var dirLightEffect = applyPresetFile(dirLightStartLayer, "r-dir-light");
          if (dirLightEffect === 0) {
            return;
          }
          dirLightEffect.property("Ambient Color").setValue(randomColorRGB);
          dirLightEffect.property("Diffuse Color").setValue(randomColorRGB);
          dirLightEffect.property("Specular Color").setValue(randomColorRGB);

          // Base effect settings
          //var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-dir");
          var baseEffect = applyPresetFile(baseLayer, "r-dir-plugin");
          if (baseEffect === 0) {
            return;
          }
          baseEffect.property("Depth Far").setValue(far);
          baseEffect.property("Depth Black Is Near").setValue(blackIsNear);
          baseEffect.property("Normal Pass Normalized").setValue(normalPassLayer.index);
          baseEffect.property("Light Start 1").setValue(dirLightStartLayer.index);
          baseEffect.property("Light End 1").setValue(dirLightEndLayer.index);

          // Expressions
          for (var i = 1; i <= 5; i++) {
            addExpressionsDirLightCUDA(baseEffect, i);
          }

          // Unmult effect
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-unmult");

        }
        addDirLightCUDA();
        break;

      case 'Rim Light (CUDA)':

        function addExpressionsRimCUDA(effect, index) {

          var lightEffectName = "r-rim-light";

          // Rim Toggle
          effect.property("Rim Toggle" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
            'var lightToggle = value;\n' +
            'if (lightLayer) {\n' +
            '  lightToggle = lightLayer.enabled;\n' +
            '}\n' +
            'lightToggle;';

          // Rim Light Position
          effect.property("Rim Light Position" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';

          // Rim Light Look At Position
          effect.property("Rim Light Look At Position" + " " + index).expression =
            'var lightLayer = effect("' +  effect.name + '")("Light Look At ' + index + '");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';

          // Rim Start
          effect.property("Rim Start" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Rim Start");\n' +
          '}\n' +
          'property1;';

          // Rim End
          effect.property("Rim End" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Rim End");\n' +
          '}\n' +
          'property1;';

          // Rim Intensity
          effect.property("Rim Intensity" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Rim Intensity");\n' +
          '}\n' +
          'property1;';

          // Rim Saturation
          effect.property("Rim Saturation" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Rim Saturation");\n' +
          '}\n' +
          'property1;';

          // Rim Near Color
          effect.property("Rim Color Near" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Rim Color Near");\n' +
          '}\n' +
          'property1;';

          // Rim Color Far Toggle
          effect.property("Rim Color Far Toggle" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Rim Color Far Toggle");\n' +
          '}\n' +
          'property1;';

          // Rim Color Far
          effect.property("Rim Color Far" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Rim Color Far");\n' +
          '}\n' +
          'property1;';

          // Rim Color Falloff
          effect.property("Rim Color Falloff" + " " + index).expression =
          'var lightLayer = effect("' +  effect.name + '")("Light ' + index + '");\n' +
          'var property1 = value;\n' +
          'if (lightLayer) {\n' +
          '  property1 = lightLayer.effect("' +  lightEffectName + '")("Rim Color Falloff");\n' +
          '}\n' +
          'property1;';
          
        }

        function addRimCUDA() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-depth2normal")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-rim")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-clamp")) {
            return;
          }
          if (!pluginExistance(mainLayer, "ADBE r-unmult")) {
            return;
          }

          // Index
          var index1 = getUniqueIndex(comp, ["Rim", "Clamp"]);
          var index2 = getUniqueIndex(comp, ["Rim Light", "Rim Look At"]);
          var rimLightLayerName = "Rim Light" + " " + index2;
          var rimLookAtLayerName = "Rim Look At" + " " + index2;
          var baseName = "Rim" + " " + index1;
          var adjName = "Clamp" + " " + index1;

          // Adjustment Layer w Clamp effect
          var adjLayer = comp.layers.addSolid([1, 1, 1], adjName, comp.width, comp.height, 1);
          adjLayer.adjustmentLayer = true;
          adjLayer.name = adjName;
          adjLayer.startTime = mainLayer.startTime;
          adjLayer.inPoint = mainLayer.inPoint;
          adjLayer.outPoint = mainLayer.outPoint;
          adjLayer.label = labelColor;
          adjLayer.property("ADBE Effect Parade").addProperty("ADBE r-clamp");

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Rim Light
          var rimLightLayer = comp.layers.addNull(comp.width);
          rimLightLayer.name = rimLightLayerName;
          rimLightLayer.threeDLayer = true;
          rimLightLayer.transform.position.setValue(projectedPosition);
          rimLightLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          rimLightLayer.transform.orientation.setValue(projectedOrientation);
          rimLightLayer.transform.scale.setValue(projectedScale);
          rimLightLayer.source.width = 100.0;
          rimLightLayer.source.height = 100.0;
          rimLightLayer.startTime = mainLayer.startTime;
          rimLightLayer.inPoint = mainLayer.inPoint;
          rimLightLayer.outPoint = mainLayer.outPoint;
          rimLightLayer.label = labelColor;

          // Rim look At
          var rimLookAtLayer = comp.layers.addNull(comp.width);
          rimLookAtLayer.name = rimLookAtLayerName;
          rimLookAtLayer.threeDLayer = true;
          rimLookAtLayer.transform.position.setValue(projectedPosition);
          rimLookAtLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          rimLookAtLayer.transform.orientation.setValue(projectedOrientation);
          rimLookAtLayer.transform.scale.setValue(projectedScale);
          rimLookAtLayer.source.width = 100.0;
          rimLookAtLayer.source.height = 100.0;
          rimLookAtLayer.startTime = mainLayer.startTime;
          rimLookAtLayer.inPoint = mainLayer.inPoint;
          rimLookAtLayer.outPoint = mainLayer.outPoint;
          rimLookAtLayer.label = labelColor;


          // Rect Light Start Effect
          var rimLightEffect = applyPresetFile(rimLightLayer, "r-rim-light");
          if (rimLightEffect === 0) {
            return;
          }
          rimLightEffect.property("Rim Color Near").setValue(randomColorRGB);
          rimLightEffect.property("Rim Color Far").setValue(randomColorRGB2);

          // Base effect settings
          var baseEffectNormal = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-depth2normal");
          baseEffectNormal.property("Depth Far").setValue(far);
          baseEffectNormal.property("Depth Black Is Near").setValue(blackIsNear);
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-rim");
          baseEffect.property("Light 1").setValue(rimLightLayer.index);
          baseEffect.property("Light Look At 1").setValue(rimLookAtLayer.index);

          // Expressions
          for (var i = 1; i <= 5; i++) {
            addExpressionsRimCUDA(baseEffect, i);
          }

          // Unmult effect
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-unmult");

        }
        addRimCUDA();
        break;

      // Unique

      case 'Ambient Occlusion (CUDA)':
        function addAOcuda() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-ambient-occlusion")) {
            return;
          }

          // Index
          var index = getUniqueIndex(comp, ["Ambient Occlusion"]);
          var aoLayerName = "Ambient Occlusion" + " " + index;
          
          // Normal Pass
          var normalPassLayer = comp.layers.byName("Normal Pass 1");
          if (!normalPassLayer) {
            var index = getUniqueIndex(comp, ["Normal Pass"]);
            var normalPassName = "Normal Pass" + " " + index;
            
            normalPassLayer = mainLayer.duplicate();
            normalPassLayer.name = normalPassName;
            normalPassLayer.enabled = false;
            normalPassLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
            normalPassLayer.label = 1; //red
            normalPassLayer.moveAfter(mainLayer);

            var normalPassEffect = normalPassLayer.property("ADBE Effect Parade").addProperty("ADBE r-depth2normal");
            normalPassEffect.property("Depth Far").setValue(far);
            normalPassEffect.property("Depth Black Is Near").setValue(blackIsNear);
          }

          // Base Layer
          var aoLayer = mainLayer.duplicate();
          aoLayer.name = aoLayerName;
          aoLayer.moveToBeginning();
          aoLayer.enabled = true;
          aoLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          aoLayer.label = 0;
          aoLayer.blendingMode = BlendingMode.MULTIPLY;

          // Base effects settings
          //var aoEffect = aoLayer.property("ADBE Effect Parade").addProperty("ADBE r-ambient-occlusion");
          var aoEffect = applyPresetFile(aoLayer, "r-ambient-occlusion-plugin");
          if (aoEffect === 0) {
            return;
          }
          aoEffect.property("Normal Pass Normalized").setValue(normalPassLayer.index);
          aoEffect.property("Depth Far").setValue(far);
          aoEffect.property("Depth Black Is Near").setValue(blackIsNear);

        }
        addAOcuda();
      break;

      case 'Position Pass (CUDA)':
        function addPositionPassCUDA() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-depth2position")) {
            return;
          }

          // Index
          var index = getUniqueIndex(comp, ["Position Pass"]);
          var baseName = "Position Pass" + " " + index;

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = 1; // red
          baseLayer.enabled = false;
          baseLayer.moveAfter(mainLayer);

          // Base effect settings
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-depth2position");
          baseEffect.property("Depth Far").setValue(far);
          baseEffect.property("Depth Black Is Near").setValue(blackIsNear);

        }
        addPositionPassCUDA();
      break;

      case 'Normal Pass (CUDA)':
        function addNormalPassCUDA() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "ADBE r-depth2normal")) {
            return;
          }

          // Index
          var index = getUniqueIndex(comp, ["Normal Pass"]);
          var baseName = "Normal Pass" + " " + index;

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = 1; // red
          baseLayer.enabled = false;
          baseLayer.moveAfter(mainLayer);

          // Base effect settings
          var baseEffect = baseLayer.property("ADBE Effect Parade").addProperty("ADBE r-depth2normal");
          baseEffect.property("Depth Far").setValue(far);
          baseEffect.property("Depth Black Is Near").setValue(blackIsNear);

        }
        addNormalPassCUDA();
      break;

      // Basic

      case 'Point Light':
        function addPointLight() {
          var index = getUniqueIndex(comp, ["Point Light"]);
          var lightName = "Point Light" + " " + index;
          var lightLayer = comp.layers.addLight(lightName, [comp.width / 2, comp.height / 2]);
          lightLayer.lightType = LightType.POINT;
          lightLayer.lightOption.color.setValue(randomColorRGB);
          lightLayer.transform.position.setValue(projectedPosition);
          lightLayer.startTime = mainLayer.startTime;
          lightLayer.inPoint = mainLayer.inPoint;
          lightLayer.outPoint = mainLayer.outPoint;
        }
        addPointLight();
        break;

      case 'Solid':
        function addSolid() {
          var index = getUniqueIndex(comp, ["Solid"]);
          var solidName = "Solid" + " " + index;
          var solid = comp.layers.addSolid(
            randomColorRGB,
            "Solid",
            projectedWidth,
            projectedHeight,
            1
          );
          solid.name = solidName;
          solid.threeDLayer = true;
          solid.transform.position.setValue(projectedPosition);
          solid.transform.anchorPoint.setValue(projectedAnchorPoint);
          solid.transform.orientation.setValue(projectedOrientation);
          solid.transform.scale.setValue(projectedScale);
          solid.transform.opacity.setValue(projectedOpacity);
          solid.source.width = projectedWidth;
          solid.source.height = projectedHeight;
          if (comp.renderer == "ADBE Advanced 3d") {
            solid.property("ADBE Material Options Group").property("ADBE Accepts Lights").setValue(projectedAcceptsLights);
          }
          solid.label = labelColor;
          solid.startTime = mainLayer.startTime;
          solid.inPoint = mainLayer.inPoint;
          solid.outPoint = mainLayer.outPoint;
        }
        addSolid();
        break;

      case '3D Null':
        function add3dNull() {
          var index = getUniqueIndex(comp, ["3D Null"]);
          var null3dName = "3D Null" + " " + index;
          var null3d = comp.layers.addNull(comp.width);
          null3d.name = null3dName;
          null3d.threeDLayer = true;
          null3d.transform.position.setValue(projectedPosition);
          null3d.transform.anchorPoint.setValue(projectedAnchorPoint);
          null3d.transform.orientation.setValue(projectedOrientation);
          null3d.transform.scale.setValue(projectedScale);
          null3d.source.width = projectedWidth;
          null3d.source.height = projectedHeight;
          null3d.startTime = mainLayer.startTime;
          null3d.inPoint = mainLayer.inPoint;
          null3d.outPoint = mainLayer.outPoint;
        }
        add3dNull();
        break;

      case '3D+2D Null':
        function add3d2dNull() {
          // Index
          var index = getUniqueIndex(comp, ["3D Null (Parent)", "2D Null (Child)"]);
          var null3dName = "3D Null (Parent)" + " " + index;
          var null2dName = "2D Null (Child)" + " " + index;

          //3D Null
          var null3d = comp.layers.addNull(comp.width);
          null3d.name = null3dName;
          null3d.threeDLayer = true;
          null3d.transform.position.setValue(projectedPosition);
          null3d.transform.orientation.setValue(projectedOrientation);
          null3d.transform.scale.setValue([20, 20, 20]);
          null3d.transform.scale.expression =
            '// 3d scale imitation for 2d layers\n' +
            'var camera = thisComp.activeCamera;\n' +
            'var cameraPos = camera.position;\n' +
            'var layerPos = thisLayer.position;\n' +
            'var distance = length(cameraPos, layerPos);\n' +
            'var baseScale = value[0];\n' +
            'var cameraZoom = camera.zoom;\n' +
            'var scaleFactor = cameraZoom / distance;\n' +
            'var newScale = baseScale * scaleFactor;\n' +
            '[newScale, newScale, newScale];';
          null3d.startTime = mainLayer.startTime;
          null3d.inPoint = mainLayer.inPoint;
          null3d.outPoint = mainLayer.outPoint;

          //2D Null
          var null2d = comp.layers.addNull();
          null2d.name = null2dName;
          null2d.threeDLayer = false;
          null2d.startTime = mainLayer.startTime;
          null2d.inPoint = mainLayer.inPoint;
          null2d.outPoint = mainLayer.outPoint;
          null2d.position.expression =
            '// 3d to 2d coordinates converter\n' +
            'null3d = thisComp.layer("' + null3dName + '");\n' +
            'null3d.toComp(null3d.transform.anchorPoint);';
          null2d.scale.expression =
            '// 3d to 2d scale\n' +
            'scale3D = thisComp.layer("' + null3dName + '").transform.scale;\n' +
            '[scale3D[0], scale3D[1]];';
        }
        add3d2dNull();
        break;

      // Debug

      case 'Point Simple (PW)':
        function addPointLightMattePW() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
            return;
          }

          // Index
          var index = getUniqueIndex(comp, ["Point Light PW", "Point Light Matte PW"]);
          var pointLightLayerName = "Point Light PW" + " " + index;
          var baseName = "Point Light Matte PW" + " " + index;

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.enabled = false;
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Point Light Layer
          var pointLightLayer = comp.layers.addNull(comp.width);
          pointLightLayer.name = pointLightLayerName;
          pointLightLayer.threeDLayer = true;
          pointLightLayer.transform.position.setValue(projectedPosition);
          pointLightLayer.transform.anchorPoint.setValue([50.0, 50.0, 0.0]);
          pointLightLayer.transform.orientation.setValue(projectedOrientation);
          pointLightLayer.transform.scale.setValue([20, 20, 20]);
          pointLightLayer.source.width = 100.0;
          pointLightLayer.source.height = 100.0;
          pointLightLayer.startTime = mainLayer.startTime;
          pointLightLayer.inPoint = mainLayer.inPoint;
          pointLightLayer.outPoint = mainLayer.outPoint;
          pointLightLayer.label = labelColor;

          // Base effects settings
          var baseLayerEffect = applyPresetFile(baseLayer, "r-depth2point-pw");
          if (baseLayerEffect === 0) {
            return;
          }
          baseLayer.property("ADBE Effect Parade").property("r-depth2point-matte-pw").property("lightPos").expression =
            'var lightLayer = effect("r-depth2point-light-pw")("Light Layer");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '    lightPosition = thisComp.activeCamera.fromWorld(lightLayer.transform.position);\n' +
            '}\n' +
            'lightPosition;'
          baseLayer.property("ADBE Effect Parade").property("r-depth2point-matte-pw").property("depthFar").setValue(far);
          baseLayer.property("ADBE Effect Parade").property("r-depth2point-matte-pw").property("depthBlackIsNear").setValue(blackIsNear);
          baseLayer.property("ADBE Effect Parade").property("r-depth2point-matte-pw").property("colorNear").setValue(randomColorRGB);
          baseLayer.property("ADBE Effect Parade").property("r-depth2point-matte-pw").property("colorFar").setValue(randomColorRGB2);
          baseLayer.property("ADBE Effect Parade").property("r-depth2point-light-pw").property("Light Layer").setValue(pointLightLayer.index);

        }
        addPointLightMattePW();
        break;
      
      case 'Spot Simple (PW)':
        function addSpotLightMattePW() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
            return;
          }

          // Index
          var index = getUniqueIndex(comp, ["Spot Look At PW", "Spot Light PW", "Spot Light Matte PW"]);
          var spotLookAtLayerName = "Spot Look At PW" + " " + index;
          var spotLightLayerName = "Spot Light PW" + " " + index;
          var baseName = "Spot Light Matte PW" + " " + index;

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.enabled = false;
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Spot Look At Layer
          var spotLookAtLayer = comp.layers.addNull(comp.width);
          spotLookAtLayer.name = spotLookAtLayerName;
          spotLookAtLayer.threeDLayer = true;
          spotLookAtLayer.transform.position.setValue(projectedPosition);
          spotLookAtLayer.transform.anchorPoint.setValue([50.0, 50.0, 0.0]);
          spotLookAtLayer.transform.orientation.setValue(projectedOrientation);
          spotLookAtLayer.transform.scale.setValue(projectedScale);
          spotLookAtLayer.source.width = 100.0;
          spotLookAtLayer.source.height = 100.0;
          spotLookAtLayer.startTime = mainLayer.startTime;
          spotLookAtLayer.inPoint = mainLayer.inPoint;
          spotLookAtLayer.outPoint = mainLayer.outPoint;
          spotLookAtLayer.label = labelColor;

          // Spot Light Layer
          var spotLightLayer = comp.layers.addNull(comp.width);
          spotLightLayer.name = spotLightLayerName;
          spotLightLayer.threeDLayer = true;
          spotLightLayer.transform.position.setValue(projectedPosition);
          spotLightLayer.transform.anchorPoint.setValue([50.0, 50.0, 0.0]);
          spotLightLayer.transform.orientation.setValue(projectedOrientation);
          spotLightLayer.transform.orientation.expression =
            'var lookAtLayer = thisComp.layer("' + spotLookAtLayerName + '");\n' +
            'var property1 = value;\n' +
            'if (lookAtLayer) {\n' +
            '  property1 = lookAt(lookAtLayer.toWorld([0.0, 0.0, 0.0]), thisLayer.toWorld([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'property1;';
          spotLightLayer.transform.scale.setValue(projectedScale);
          spotLightLayer.source.width = 100.0;
          spotLightLayer.source.height = 100.0;
          spotLightLayer.startTime = mainLayer.startTime;
          spotLightLayer.inPoint = mainLayer.inPoint;
          spotLightLayer.outPoint = mainLayer.outPoint;
          spotLightLayer.label = labelColor;
          spotLightLayer.moveAfter(spotLookAtLayer);

          // Base effects settings
          var baseLayerEffect = applyPresetFile(baseLayer, "r-depth2spot-pw");
          if (baseLayerEffect === 0) {
            return;
          }
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-matte-pw").property("iesPreset").expression =
            '// IES Preset\n' +
            'var preset = effect("r-depth2spot-light-pw")("IES Preset");\n' +
            'var property1 = value;\n' +
            'if (preset) {\n' +
            '  property1 = [preset, preset];\n' +
            '}\n' +
            'property1;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-matte-pw").property("lightPos").expression =
            '// Position\n' +
            'var lightLayer = effect("r-depth2spot-light-pw")("Light Layer");\n' +
            'var lightPosition = [0.0, 0.0, 0.0];\n' +
            'if (lightLayer) {\n' +
            '  lightPosition = lightLayer.toWorld(lightLayer.anchorPoint);\n' +
            '}\n' +
            'lightPosition;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-matte-pw").property("lightVx").expression =
            '// Vector X\n' +
            'var lightLayer = effect("r-depth2spot-light-pw")("Light Layer");\n' +
            'var lightPosition = [0.0, 0.0, 0.0];\n' +
            'if (lightLayer) {\n' +
            '  lightPosition = normalize(lightLayer.toWorldVec([1.0, 0.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-matte-pw").property("lightVy").expression =
            '// Vector Y\n' +
            'var lightLayer = effect("r-depth2spot-light-pw")("Light Layer");\n' +
            'var lightPosition = [0.0, 0.0, 0.0];\n' +
            'if (lightLayer) {\n' +
            '  lightPosition = normalize(lightLayer.toWorldVec([0.0, -1.0, 0.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-matte-pw").property("lightVz").expression =
            '// Vector Z\n' +
            'var lightLayer = effect("r-depth2spot-light-pw")("Light Layer");\n' +
            'var lightPosition = [0.0, 0.0, 0.0];\n' +
            'if (lightLayer) {\n' +
            '  lightPosition = normalize(lightLayer.toWorldVec([0.0, 0.0, -1.0]) - lightLayer.toWorldVec([0.0, 0.0, 0.0]));\n' +
            '}\n' +
            'lightPosition;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-matte-pw").property("depthFar").setValue(far);
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-matte-pw").property("depthBlackIsNear").setValue(blackIsNear);
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-matte-pw").property("colorNear").setValue(randomColorRGB);
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-matte-pw").property("colorFar").setValue(randomColorRGB2);
          baseLayer.property("ADBE Effect Parade").property("r-depth2spot-light-pw").property("Light Layer").setValue(spotLightLayer.index);

        }
        addSpotLightMattePW();
        break;

      case 'Rect Simple (PW)':
        function addRectLightMattePW() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
            return;
          }

          // Index
          var index = getUniqueIndex(comp, ["Rect Light Start PW", "Rect Light End PW", "Rect Light Matte PW"]);
          var rectLightStartLayerName = "Rect Light Start PW" + " " + index;
          var rectLightEndLayerName = "Rect Light End PW" + " " + index;
          var baseName = "Rect Light Matte PW" + " " + index;

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.enabled = false;
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Rect Start Layer
          var rectLightStartLayer = comp.layers.addNull(comp.width);
          rectLightStartLayer.name = rectLightStartLayerName;
          rectLightStartLayer.threeDLayer = true;
          rectLightStartLayer.transform.position.setValue(projectedPosition);
          rectLightStartLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          rectLightStartLayer.transform.orientation.setValue(projectedOrientation);
          rectLightStartLayer.transform.scale.setValue(projectedScale);
          rectLightStartLayer.source.width = 100.0;
          rectLightStartLayer.source.height = 100.0;
          rectLightStartLayer.startTime = mainLayer.startTime;
          rectLightStartLayer.inPoint = mainLayer.inPoint;
          rectLightStartLayer.outPoint = mainLayer.outPoint;
          rectLightStartLayer.label = labelColor;

          // Rect End Layer
          var rectLightEndLayer = comp.layers.addNull(comp.width);
          rectLightEndLayer.name = rectLightEndLayerName;
          rectLightEndLayer.threeDLayer = true;
          rectLightEndLayer.transform.position.setValue(projectedPosition);
          rectLightEndLayer.transform.anchorPoint.setValue(projectedAnchorPoint);
          rectLightEndLayer.transform.orientation.setValue(projectedOrientation);
          rectLightEndLayer.transform.scale.setValue(projectedScale);
          rectLightEndLayer.source.width = 100.0;
          rectLightEndLayer.source.height = 100.0;
          rectLightEndLayer.startTime = mainLayer.startTime;
          rectLightEndLayer.inPoint = mainLayer.inPoint;
          rectLightEndLayer.outPoint = mainLayer.outPoint;
          rectLightEndLayer.label = labelColor;

          // Base effects settings
          var baseLayerEffect = applyPresetFile(baseLayer, "r-depth2rect-pw");
          if (baseLayerEffect === 0) {
            return;
          }
          // Start Layer Expressions
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("pos1").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light Start Layer");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '  lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("vX1").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light Start Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = normalize(lightLayer.toWorldVec([1,0,0]) - lightLayer.toWorldVec([0,0,0]));\n' +
            '}\n' +
            'property1;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("vY1").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light Start Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = normalize(lightLayer.toWorldVec([0,-1,0]) - lightLayer.toWorldVec([0,0,0]));\n' +
            '}\n' +
            'property1;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("vZ1").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light Start Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = normalize(lightLayer.toWorldVec([0,0,-1]) - lightLayer.toWorldVec([0,0,0]));\n' +
            '}\n' +
            'property1;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("res1").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light Start Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = [lightLayer.width, lightLayer.height, 100.0];\n' +
            '}\n' +
            'property1;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("scale1").expression =
            'var lightLayerStart = effect("r-depth2rect-light-pw")("Light Start Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayerStart) {  \n' +
            '  if(lightLayerStart.hasParent) {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    var scaleValueEnd = lightLayerStart.parent.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueEnd[0] * scaleValueStart[0], scaleValueEnd[1] * scaleValueStart[1], scaleValueEnd[2] * scaleValueStart[2]];\n' +
            '  } else {\n' +
            '    var scaleValueStart = lightLayerStart.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueStart[0], scaleValueStart[1], scaleValueStart[2]];\n' +
            '  }\n' +
            '}\n' +
            'property1;';
          // End Layer Expressions
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("pos2").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light End Layer");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '  lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("vX2").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light End Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = normalize(lightLayer.toWorldVec([1,0,0]) - lightLayer.toWorldVec([0,0,0]));\n' +
            '}\n' +
            'property1;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("vY2").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light End Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = normalize(lightLayer.toWorldVec([0,-1,0]) - lightLayer.toWorldVec([0,0,0]));\n' +
            '}\n' +
            'property1;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("vZ2").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light End Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = normalize(lightLayer.toWorldVec([0,0,-1]) - lightLayer.toWorldVec([0,0,0]));\n' +
            '}\n' +
            'property1;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("res2").expression =
            'var lightLayer = effect("r-depth2rect-light-pw")("Light End Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayer) {\n' +
            '  property1 = [lightLayer.width, lightLayer.height, 100.0, 1.0];\n' +
            '}\n' +
            'property1;';
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("scale2").expression =
            'var lightLayerEnd = effect("r-depth2rect-light-pw")("Light End Layer");\n' +
            'var property1 = value;\n' +
            'if (lightLayerEnd) {  \n' +
            '  if(lightLayerEnd.hasParent) {\n' +
            '    var scaleValueEnd = lightLayerEnd.transform.scale * 0.01;\n' +
            '    var scaleValueStart = lightLayerEnd.parent.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueEnd[0] * scaleValueStart[0], scaleValueEnd[1] * scaleValueStart[1], scaleValueEnd[2] * scaleValueStart[2], 1.0];\n' +
            '  } else {\n' +
            '    var scaleValueEnd = lightLayerEnd.transform.scale * 0.01;\n' +
            '    property1 = [scaleValueEnd[0], scaleValueEnd[1], scaleValueEnd[2], 1.0];\n' +
            '  }\n' +
            '}\n' +
            'property1;';

          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("depthFar").setValue(far);
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("depthBlackIsNear").setValue(blackIsNear);
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("colorNear").setValue(randomColorRGB);
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-matte-pw").property("colorFar").setValue(randomColorRGB2);
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-light-pw").property("Light Start Layer").setValue(rectLightStartLayer.index);
          baseLayer.property("ADBE Effect Parade").property("r-depth2rect-light-pw").property("Light End Layer").setValue(rectLightEndLayer.index);

        }
        addRectLightMattePW();
        break;

      case 'Position Pass (PW)':
        function addPositionPass() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
            return;
          }

          // Index
          var index = getUniqueIndex(comp, ["Position Pass"]);
          var baseName = "Position Pass" + " " + index;
          
          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.enabled = false;
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = 1; // red
          baseLayer.moveAfter(mainLayer);

          // Base effects settings
          var baseEffect = applyPresetFile(baseLayer, "r-depth2position-pw");
          if (baseEffect === 0) {
            return;
          }
          baseLayer.property("ADBE Effect Parade").property("r-depth2position-pw").property("depthFar").setValue(far);
          baseLayer.property("ADBE Effect Parade").property("r-depth2position-pw").property("depthBlackIsNear").setValue(blackIsNear);

        }
        addPositionPass();
        break;

      case 'Normal Pass (PW)':
        function addNormalPass() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
            return;
          }

          // Index
          var index = getUniqueIndex(comp, ["Normal Pass"]);
          var baseName = "Normal Pass" + " " + index;
          
          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.enabled = false;
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = 9; //green
          baseLayer.moveAfter(mainLayer);

          // Base effects settings
          var baseEffect = applyPresetFile(baseLayer, "r-depth2normal-pw");
          if (baseEffect === 0) {
            return;
          }
          baseLayer.property("ADBE Effect Parade").property("r-depth2normal-pw").property("depthFar").setValue(far);
          baseLayer.property("ADBE Effect Parade").property("r-depth2normal-pw").property("depthBlackIsNear").setValue(blackIsNear);

        }
        addNormalPass();
        break;

      case 'Ambient Occlusion (PW)':
        function addAOpw() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
            return;
          }

          // Index
          var posPassLayer = comp.layers.byName("Position Pass 1");
          if (!posPassLayer) {
            var index = getUniqueIndex(comp, ["Position Pass"]);
            var posPassName = "Position Pass" + " " + index;
            
            // Base Layer
            posPassLayer = mainLayer.duplicate();
            posPassLayer.name = posPassName;
            posPassLayer.enabled = false;
            posPassLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
            posPassLayer.label = 1; //red
            posPassLayer.moveAfter(mainLayer);

            // Base effects settings
            var posPassEffect = applyPresetFile(posPassLayer, "r-depth2position-pw");
            if (posPassEffect === 0) {
              return;
            }
            posPassLayer.property("ADBE Effect Parade").property("r-depth2position-pw").property("depthFar").setValue(far);
            posPassLayer.property("ADBE Effect Parade").property("r-depth2position-pw").property("depthBlackIsNear").setValue(blackIsNear);
          }


          // Index
          var index = getUniqueIndex(comp, ["AO PW"]);
          var aoLayerName = "AO PW" + " " + index;
          
          // Base Layer
          var aoLayer = mainLayer.duplicate();
          aoLayer.name = aoLayerName;
          aoLayer.moveToBeginning();
          aoLayer.enabled = true;
          aoLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          aoLayer.label = labelColor;
          aoLayer.blendingMode = BlendingMode.MULTIPLY;

          // Base effects settings
          var aoEffect = applyPresetFile(aoLayer, "r-ambient-occlusion-pw");
          if (aoEffect === 0) {
            return;
          }
          aoLayer.property("ADBE Effect Parade").property("r-ambient-occlusion-pw").property("positionPass").setValue(posPassLayer.index);

        }
        addAOpw();

      break;

      case 'Normality (PW)':
        function addNormalityPW() {

          // Check the plugin
          if (!pluginExistance(mainLayer, "MiLai PixelsWorld")) {
            return;
          }

          // Index
          var index = getUniqueIndex(comp, ["Dir Look At PW", "Dir Light PW", "Normality PW"]);
          var dirLookAtLayerName = "Dir Look At PW" + " " + index;
          var dirLightLayerName = "Dir Light PW" + " " + index;
          var baseName = "Normality PW" + " " + index;

          // Base Layer
          var baseLayer = mainLayer.duplicate();
          baseLayer.name = baseName;
          baseLayer.moveToBeginning();
          baseLayer.enabled = false;
          baseLayer.property("ADBE Effect Parade").property(mainEffectName).remove();
          baseLayer.label = labelColor;
          baseLayer.enabled = true;
          baseLayer.blendingMode = BlendingMode.OVERLAY;

          // Dir Light Layer
          var dirLightLayer = comp.layers.addNull(comp.width);
          dirLightLayer.name = dirLightLayerName;
          dirLightLayer.threeDLayer = true;
          dirLightLayer.transform.position.setValue(projectedPosition);
          dirLightLayer.transform.anchorPoint.setValue([50.0, 50.0, 0.0]);
          dirLightLayer.transform.orientation.setValue(projectedOrientation);
          dirLightLayer.transform.scale.setValue(projectedScale);
          dirLightLayer.source.width = 100.0;
          dirLightLayer.source.height = 100.0;
          dirLightLayer.startTime = mainLayer.startTime;
          dirLightLayer.inPoint = mainLayer.inPoint;
          dirLightLayer.outPoint = mainLayer.outPoint;
          dirLightLayer.label = labelColor;

          // Dir Look At Layer
          var dirLookAtLayer = comp.layers.addNull(comp.width);
          dirLookAtLayer.parent = dirLightLayer;
          dirLookAtLayer.name = dirLookAtLayerName;
          dirLookAtLayer.threeDLayer = true;
          dirLookAtLayer.transform.position.setValue([0.0, 0.0, 0.0]);
          dirLookAtLayer.transform.anchorPoint.setValue([50.0, 50.0, 0.0]);
          dirLookAtLayer.transform.orientation.setValue([0.0, 0.0, 0.0]);
          dirLookAtLayer.transform.scale.setValue(projectedScale);
          dirLookAtLayer.source.width = 100.0;
          dirLookAtLayer.source.height = 100.0;
          dirLookAtLayer.startTime = mainLayer.startTime;
          dirLookAtLayer.inPoint = mainLayer.inPoint;
          dirLookAtLayer.outPoint = mainLayer.outPoint;
          dirLookAtLayer.label = labelColor;
          
          // Add Base Effects
          var baseLayerEffects = applyPresetFile(baseLayer, "r-normality-pw");
          if (baseLayerEffects === 0) {
            return;
          }

          // Base Effects Links
          var normalMapEffect = baseLayer.property("ADBE Effect Parade").property("r-depth2normal-pw");
          var normalityEffect = baseLayer.property("ADBE Effect Parade").property("r-normality-pw");
          var controllerEffect = baseLayer.property("ADBE Effect Parade").property("r-normality-pw-controller");
          var controllerEffectName = controllerEffect.name;

          // Normal Map
          normalMapEffect.property("depthFar").setValue(far);
          normalMapEffect.property("depthBlackIsNear").setValue(blackIsNear);

          // Dir Light
          normalityEffect.property("dirPos").expression =
            'var lightLayer = effect("' + controllerEffectName + '")("Directional Light");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '  lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';
          normalityEffect.property("dirLookAt").expression =
            'var lightLayer = effect("' + controllerEffectName + '")("Directional Look At");\n' +
            'var lightPosition = value;\n' +
            'if (lightLayer) {\n' +
            '  lightPosition = lightLayer.toWorld([0.0, 0.0, 0.0]);\n' +
            '}\n' +
            'lightPosition;';
          // Global
          normalityEffect.property("globalToggle").expression =
            'var lightLayer = effect("' + controllerEffectName + '")("Directional Light");\n' +
            'var lightToggle = value;\n' +
            'if (lightLayer) {\n' +
            '  lightToggle = lightLayer.enabled;\n' +
            '}\n' +
            'lightToggle;';
          normalityEffect.property("intensityMultiplier").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Intensity Multiplier");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("saturationMultiplier").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Saturation Multiplier");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          // Diffuse
          normalityEffect.property("diffuseToggle").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Diffuse Toggle");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("diffuseIntensity").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Diffuse Intensity");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("diffuseColor").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Diffuse Color");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          // Specular
          normalityEffect.property("specularToggle").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Specular Toggle");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("specularSize").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Specular Size");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("specularIntensity").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Specular Intensity");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("specularColor").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Specular Color");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          // Rim
          normalityEffect.property("rimToggle").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Rim Toggle");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("rimStart").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Rim Start");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("rimEnd").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Rim End");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("rimIntensity").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Rim Intensity");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';
          normalityEffect.property("rimColor").expression =
            'var controllerValue = effect("' + controllerEffectName + '")("Rim Color");\n' +
            'var curValue = value;\n' +
            'if (controllerValue) {\n' +
            '  curValue = controllerValue;\n' +
            '}\n' +
            'curValue;';

          // Controller
          var colorExpression =
            'var globalColorToggle = effect("' + controllerEffectName + '")("Use Global Color");\n' +
            'var globalColor = effect("' + controllerEffectName + '")("Global Color");\n' +
            'var curValue = value;\n' +
            'if (globalColorToggle == true) {\n' +
            '  curValue = globalColor;\n' +
            '}\n' +
            'curValue;';

          controllerEffect.property("Directional Light").setValue(dirLightLayer.index);
          controllerEffect.property("Directional Look At").setValue(dirLookAtLayer.index);
          controllerEffect.property("Global Color").setValue(randomColorRGB);
          controllerEffect.property("Diffuse Color").setValue(randomColorRGB);
          controllerEffect.property("Diffuse Color").expression = colorExpression;
          controllerEffect.property("Specular Color").setValue(randomColorRGB);
          controllerEffect.property("Specular Color").expression = colorExpression;
          controllerEffect.property("Rim Color").setValue(randomColorRGB);
          controllerEffect.property("Rim Color").expression = colorExpression;

        }
        addNormalityPW();
      break;

      default:
        alert("Select what you want to project");
        break;
    }

    // Final touches
    deselectAll(comp);
    projectedLayer.moveToBeginning();
    mainLayer.property("ADBE Effect Parade").property(mainEffectName).selected = true;
    app.endUndoGroup();

  }

  function deleteSetupButton() {

    // Inputs
    var comp = app.project.activeItem;
    var mainLayer = comp.selectedLayers[0];

    // Checkers
    if (!(comp instanceof CompItem)) {
      alert("Open a composition first");
      return;
    }

    if (comp.selectedLayers.length !== 1 || !mainLayer.hasVideo) {
      alert("Select a single depth map you wish to delete setup on");
      return;
    }

    if (!effectsExistance(mainLayer, mainEffectName)) {
      alert("Nothing to delete");
      return;
    }

    var confirmDelete = confirm("Are you sure you want to delete the setup?");
    if (!confirmDelete) {
        return;
    }

    // First touches
    app.project.save();
    app.beginUndoGroup("deleteSetupButton");

    // Delete main effect
    mainLayer.property("ADBE Effect Parade").property(mainEffectName).remove();

    // Delete projected solid
    var projectedLayer = getProjectedLayer(mainLayer);
    if (projectedLayer) {
      projectedLayer.remove();
    }

    // Сlear name
    mainLayer.name = '';
    mainLayer.label = 15; // Color: sandstone

    // Final touches
    deselectAll(comp);
    mainLayer.selected = true;
    app.endUndoGroup();
  }

  // -------------------Functions-------------------

  function rgbToHSV(rgb) {

    //R, G and B input range = 0 ÷ 1
    //H, S and V output range = 0 ÷ 1.0

    var r = rgb[0]; //red
    var g = rgb[1]; //green
    var b = rgb[2]; //blue
    // var a = rgba[3]; //alpha

    var var_Min = Math.min(r, g, b); //Min. value of RGB
    var var_Max = Math.max(r, g, b); //Max. value of RGB
    var del_Max = var_Max - var_Min; //Delta RGB value

    V = var_Max;

    //This is a gray, no chroma
    if (del_Max == 0) {
      H = 0;
      S = 0;
    }
    //Chromatic data
    else {
      S = del_Max / var_Max;

      del_R = (((var_Max - r) / 6) + (del_Max / 2)) / del_Max;
      del_G = (((var_Max - g) / 6) + (del_Max / 2)) / del_Max;
      del_B = (((var_Max - b) / 6) + (del_Max / 2)) / del_Max;

      if (r == var_Max) {
          H = del_B - del_G;
      }
      else if (g == var_Max) {
          H = (1 / 3) + del_R - del_B;
      }
      else if (b == var_Max) {
          H = (2 / 3) + del_G - del_R;
      }

      if (H < 0) H += 1;
      if (H > 1) H -= 1;
    };

    return [H, S, V];

  }

  function hsvToRGB(hsv) {

    //H, S and V input range = 0 ÷ 1.0
    //R, G and B output range = 0 ÷ 1

    var H = hsv[0];
    var S = hsv[1];
    var V = hsv[2];
    // var A = HSVA[3];

    if (S == 0) {
      var R = V * 1;
      var G = V * 1;
      var B = V * 1;
    }
    else {
      var_h = H * 6;

      if (var_h == 6) var_h = 0; //H must be < 1

      var_i = Math.floor(var_h); //Or ... var_i = floor(var_h)
      var_1 = V * (1 - S);
      var_2 = V * (1 - S * (var_h - var_i));
      var_3 = V * (1 - S * (1 - (var_h - var_i)));

      if (var_i == 0) {
          var_r = V;
          var_g = var_3;
          var_b = var_1;
      }
      else if (var_i == 1) {
          var_r = var_2;
          var_g = V;
          var_b = var_1;
      }
      else if (var_i == 2) {
          var_r = var_1;
          var_g = V;
          var_b = var_3;
      }
      else if (var_i == 3) {
          var_r = var_1;
          var_g = var_2;
          var_b = V;
      }
      else if (var_i == 4) {
          var_r = var_3;
          var_g = var_1;
          var_b = V;
      }
      else {
          var_r = V;
          var_g = var_1;
          var_b = var_2;
      }

      R = var_r * 1;
      G = var_g * 1;
      B = var_b * 1;
    };

    return [R, G, B];

  }

  // Get

  function getUniqueIndex(comp, baseNames) {

    // input ex: ["Depth Projection", "Projected Solid"]
    // output: int index;

    var isArray = baseNames && typeof baseNames.length === 'number' && typeof baseNames !== 'string';
    if (!isArray) {
      baseNames = [baseNames];
    }

    var index = 1;
    var nameExists;

    do {

      nameExists = false;

      for (var i = 0; i < baseNames.length; i++) {
        var currentName = baseNames[i] + " " + index; // Name template
        if (comp.layers.byName(currentName)) {
          nameExists = true;
          index++;
          break;
        }
      }

    } while (nameExists);

    return index;

  }

  function parseIndex(layer) {
    var layerName = layer.name;
    var match = layerName.match(/\d+$/);  // "Name 123" -> 123 (numbers at the end of the name)

    return match ? match[0] : null; 
  }

  function getProjectedLayer(mainLayer) {

    var index = parseIndex(mainLayer);
    var projectedLayerName = projectedLayerNameDefault + " " + index;

    var comp = mainLayer.containingComp;
    var projectedLayer = comp.layer(projectedLayerName);

    if (!projectedLayer) {
      alert(projectedLayerName + " not found");
      return false;
    } else {
      return projectedLayer;
    }

  }

  function getRandomNumber(min, max) {
    return Math.random() * (max - min) + min;
  }

  function getRandomColorRGB(labelColor) {
    var randomColorRGB;

    switch(labelColor) {
      case 1:
        randomColorRGB = hsvToRGB([0, 1, 1]); // Red
        break;
      case 2:
        randomColorRGB = hsvToRGB([55/360, 1, 1]); // Yellow
        break;
      case 3:
        randomColorRGB = hsvToRGB([173/360, 1, 1]); // Aqua
        break;
      case 4:
        randomColorRGB = hsvToRGB([341/360, 1, 1]); // Pink
        break;
      case 5:
        randomColorRGB = hsvToRGB([240/360, 1, 1]); // Lavender
        break;
      case 6:
        randomColorRGB = hsvToRGB([29/360, 1, 1]); // Peach
        break;
      case 7:
        randomColorRGB = hsvToRGB([120/360, 1, 1]); // Sea Foam
        break;
      case 8:
        randomColorRGB = hsvToRGB([229/360, 1, 1]); // Blue
        break;
      case 9:
        randomColorRGB = hsvToRGB([121/360, 1, 1]); // Green
        break;
      case 10:
        randomColorRGB = hsvToRGB([293/360, 1, 1]); // Purple
        break;
      case 11:
        randomColorRGB = hsvToRGB([36/360, 1, 1]); // Orange
        break;
      case 12:
        randomColorRGB = hsvToRGB([19/360, 1, 1]); // Brown
        break;
      case 13:
        randomColorRGB = hsvToRGB([313/360, 1, 1]); // Fuchsia
        break;
      case 14:
        randomColorRGB = hsvToRGB([182/360, 1, 1]); // Cyan
        break;
      case 15:
        randomColorRGB = hsvToRGB([38/360, 1, 1]); // Sandstone
        break;
      case 16:
        randomColorRGB = hsvToRGB([120/360, 1, 1]); // Dark Green
        break;
      default:
        randomColorRGB = [1, 1, 1];
        break;
    }

    return randomColorRGB;
  }

  // Bool

  function effectsExistance(layer, effectsToCheck) {

    var isArray = effectsToCheck && typeof effectsToCheck.length === 'number' && typeof effectsToCheck !== 'string';
    if (!isArray) {
      effectsToCheck = [effectsToCheck];
    }

    var effects = layer.property("ADBE Effect Parade");
    var effectsExist = false;

    if (!layer.hasVideo) {
      return effectsExist;
    }

    for (var i = 0; i < effectsToCheck.length; i++) {
      for (var j = 1; j <= effects.numProperties; j++) {
          if (effects.property(j).name.indexOf(effectsToCheck[i]) !== -1) {
              effectsExist = true;
              break;
          }
      }
      if (effectsExist) {
        break;
      }
    }

    return effectsExist;

  }

  function pluginExistance(layer, pluginName) {
    try {
      var plugin = layer.property("ADBE Effect Parade").addProperty(pluginName);
      plugin.remove();
      return true;
    } catch (e) {
      alert("'" + pluginName + "' plugin is not installed");
      return false;
    }
  }

  // Add

  function applyPresetFile(layer, presetName) {

    // Check if preset exists
    var appFolderPath = Folder.appPackage.parent.fsName; // Path to AE folder 
    var ffxFile = new File(appFolderPath + '/Support Files/Scripts/ScriptUI Panels/r-relighting/' + presetName + '.ffx');
    if (!ffxFile.exists) {
      alert(
        presetName + ".ffx not found\n" +
        "Please ensure the script is installed correctly"
      );

      return 0;
    }

    // Make a right selection
    var comp = layer.containingComp;
    deselectAll(comp);
    layer.selected = true;

    // Apply preset and return effect
    var effects = layer.property("ADBE Effect Parade");
    var numEffectsBefore = effects.numProperties;
    layer.applyPreset(ffxFile);
    var numEffectsAfter = effects.numProperties;
    for (var i = numEffectsBefore + 1; i <= numEffectsAfter; i++) {
      var effect = effects.property(i);
      if (effect) {
        return effect;
      }
    }

    return -1;

  }

  function addPhotoFilter(layer, filter, color, density, preserveLuminosity) {
    var photoFilter = layer.property("ADBE Effect Parade").addProperty("ADBE PhotoFilterPS");
    photoFilter.property("ADBE PhotoFilterPS-0001").setValue(filter);             // Filter: custom
    photoFilter.property("ADBE PhotoFilterPS-0002").setValue(color);              // Color
    photoFilter.property("ADBE PhotoFilterPS-0003").setValue(density);            // Density
    photoFilter.property("ADBE PhotoFilterPS-0004").setValue(preserveLuminosity); // Preserve Luminosity

    return photoFilter;
  }
  
  function addExposure(layer, exposureValue) {
    var exposure = layer.property("ADBE Effect Parade").addProperty("ADBE Exposure");
    exposure.property("ADBE Exposure-0003").setValue(exposureValue); // Exposure

    return exposure;
  }

  // Do

  function deselectAll(comp) {
      var selectedLayers = comp.selectedLayers;
      for (var i = selectedLayers.length - 1; i >= 0; i--) {
          selectedLayers[i].selected = false;
      }
  }

  // Debug

  function alertPush(message) {
    alertMessage.push(message);
  }

  function alertShow(message) {

    alertMessage.push(message);

    if (alertMessage.length === 0) {
        return;
    }

    var allMessages = alertMessage.join("\n\n")

    var dialog = new Window("dialog", "Debug");
    var textGroup = dialog.add("group");
    textGroup.orientation = "column";
    textGroup.alignment = ["fill", "top"];

    var text = textGroup.add("edittext", undefined, allMessages, { multiline: true, readonly: true });
    text.alignment = ["fill", "fill"];
    text.preferredSize.width = 300;
    text.preferredSize.height = 300;

    var closeButton = textGroup.add("button", undefined, "Close");
    closeButton.onClick = function () {
        dialog.close();
    };

    dialog.show();

    alertMessage = [];

  }

  function alertCopy(message) {

    if (message === undefined || message === "") {
        return;
    }

    var dialog = new Window("dialog", "Information");
    var textGroup = dialog.add("group");
    textGroup.orientation = "column";
    textGroup.alignment = ["fill", "top"];

    var text = textGroup.add("edittext", undefined, message, { multiline: true, readonly: true });
    text.alignment = ["fill", "fill"];
    text.preferredSize.width = 300;
    text.preferredSize.height = 150;

    var closeButton = textGroup.add("button", undefined, "Close");
    closeButton.onClick = function () {
        dialog.close();
    };

    dialog.show();

    alertMessage = [];

  }

  // -------------------Show UI-------------------

  var myScriptPal = buildUI(thisObj);
  if ((myScriptPal != null) && (myScriptPal instanceof Window)) {
    myScriptPal.center();
    myScriptPal.show();
  }
  if (this instanceof Panel) {
    myScriptPal.show();
  }

}
relighting(this);