# Relighting
An open source After Effects tool that allows you to quickly relight a scene  
Author: https://www.youtube.com/@shy_rikki  
Report a bug: https://discord.gg/AAJxThhbBf  
If you wish to donate ❤️: https://boosty.to/shy_rikki  

<img width="960" height="540" alt="Screenshot_1" src="https://github.com/user-attachments/assets/1341d857-e10f-43ac-ad97-b7234c545be3" />  

## Features
- Point Light
- Spot Light
- Rect Light
- Directional Light  
- Rim Light  
- Ambient Occlusion (edge shadows)  
- Position Pass  
- Normal Pass  
- Normal Remap  
- Clamp (clamps values to 0-1 range)  
- Unmult (removes black)  

## Requirements
- NVIDIA GPU that support CUDA  
- EXR Depth Map  
- Camera Data  
- Normal Pass (Optional)  

What can meet the requirements?  
**Games**: CS2, CSGO, COD4 (partially, cuz bad depth), BO2 (partially, cuz bad depth)  
**3D Software**: almost any (Blender, Cinema 4D, Unreal Engine 5 etc)  

## Installation
1. Download the tool: https://github.com/eirisocherry/relighting/releases  
2. Move `r-relighting` folder and `r-relighting.jsx` script to:  
`C:\Program Files\Adobe\Adobe After Effects <version>\Support Files\Scripts\ScriptUI Panels`  
3. Move `Rikki` folder that contains plugins to:  
   `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore`  
4. Restart After Effects  

## Usage
1. Launch After Effects, go to `Window`, scroll down and open the `r-relighting.jsx` script  
2. Set Rendering Engine to `Mercury GPU Acceleration (CUDA)`  
<img width="335" height="118" alt="image" src="https://github.com/user-attachments/assets/b4d03007-479e-466a-be39-2e66043c8519" />
  
3. Import camera data  
CS2 & CSGO: https://www.youtube.com/watch?v=FWEqkaiXNM0  
COD4: https://www.youtube.com/watch?v=ZKsAgvfdi4I  
BO2: https://www.youtube.com/watch?v=6pkkpgb8VYY  
4. Import an exr depth map, select it and create a setup by pressing `[+]` button  
It will also automatically set project to `32 bit` and expressions engine to `JavaScript`  
If it didn't, do it manually, otherwise nothing will work  
5. Adjust `Depth Settings`:    
`Depth Black Is Near` whether a black color is near on your depth map or not  
`Depth Far` the farthest depth point value:  
CS2 & CSGO (if you use my cfgs): normal `4096`, exr `25000`  
COD4: `4080`  
EXR Depth: set the same value you set in `EXtractoR` effect  
6. Select something from the dropdowm menu  
7. Select `Depth Projection` depth layer and use `Project On Point` cursor to select where you want to project an object  
8. Press `Project` and wait a bit... Done!  


## Important Notes
- The script is heavy and may crash your After Effects, but don't worry!  
It automatically saves your project before doing any actions, so even if your AE will crash, you will not lose any of your progress  
- During the process AE may freeze time by time, don't panic, just wait. It happens because light effects have hundreads of parameters and expressions, AE needs some time to process them.  
