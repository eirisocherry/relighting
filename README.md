# Relighting
An After Effects tool that allows you to quickly project 3d lights  

Author: https://www.youtube.com/@shy_rikki (cuda plugins, glsl shaders, script and expressions)  

## Features

- Point Light Matte
- Spot Light Matte
- Rect Light Matte
- Position Pass
- Normal Pass
- Normality (directional light)
- Ambient Occlusion (edge shadows)
- Clamp (clamps values to 0-1 range)
- Unmult (removes black)

## Requirements
- Depth Map and Camera Data  
- Project must be set to `32 bit`:  
<img width="361" height="145" alt="image" src="https://github.com/user-attachments/assets/b4909133-5e18-44e9-83cd-00cd61788b07" />

- Expressions engine must be set to `JavaScript`:  
<img width="527" height="124" alt="image" src="https://github.com/user-attachments/assets/6144f19f-5f15-4d8d-8003-1e6f7685d653" />

- CUDA effects work only on nvidia cards that support CUDA (you must use `Mercury GPU Acceleration (CUDA)` ):  
<img width="269" height="112" alt="image" src="https://github.com/user-attachments/assets/193df02c-9453-4a81-9cbe-ecf60ed0278e" />

- PW effects work only with PixelsWorld plugin installed:  
<img width="431" height="166" alt="image" src="https://github.com/user-attachments/assets/1d75dd79-2634-435a-be9a-c8fd567c70ec" />  

## Installation
1. Download the tool: https://github.com/eirisocherry/relighting/releases  
2. Move `r-relighting` folder and `r-relighting.jsx` script to:  
`C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Scripts\ScriptUI Panels`  
3. Move `rikki` folder that contains plugins to:  
   `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore`  
4. Install PixelsWorld plugin: https://aescripts.com/pixelsworld/  (https://t.me/ancient_storage/162)  
5. Restart After Effects  

## Usage

1. Launch After Effects, go to `Window`, scroll down and open the `r-relighting.jsx` script  
2. Import camera data  
CS2 & CSGO: https://www.youtube.com/watch?v=FWEqkaiXNM0  
COD4: https://www.youtube.com/watch?v=ZKsAgvfdi4I  
BO2: https://www.youtube.com/watch?v=6pkkpgb8VYY  
3. Import a depth map, select it and create a setup by pressing `[+]` button  
It will also automatically set project to `32 bit` and expressions engine to `JavaScript`  
If it didn't, do it manually, otherwise nothing will work  
4. Adjust `Depth Settings`:    
`Depth Black Is Near` whether a black color is near on your depth map or not  
`Depth Far` the farthest depth point value:  
CS2 & CSGO (if you use my cfgs): normal `4096`, exr `25000`  
COD4: `4080`  
EXR Depth: set the same value you set in `EXtractoR` effect  
5. Select something from the dropdowm menu
6. Select `Depth Projection` depth layer and use `Project On Point` cursor to select where you want to project an object  
7. Press `Project` and wait a bit... Done!  

- `Auto Orient` works only with EXR depth maps  
- The script is heavy and may crash your After Effects, but don't worry!  
It automatically saves your project before doing any actions, so even if your AE will crash, you will not lose any of your progress  
