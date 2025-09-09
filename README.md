# Relighting
An After Effects tool that allows you to quickly relight a scene  
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
- NVIDIA GPU that supports CUDA  
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

<details>
<summary> First Steps </summary>
<br>

1. Launch After Effects, go to `Window`, scroll down and open the `r-relighting.jsx` script
<img width="641" height="840" alt="image" src="https://github.com/user-attachments/assets/d9c96967-3840-493e-860f-cee18a0d444b" />  

2. Set Rendering Engine to `Mercury GPU Acceleration (CUDA)`  
<img width="335" height="118" alt="image" src="https://github.com/user-attachments/assets/b4d03007-479e-466a-be39-2e66043c8519" />

3. Set project to `32 bit`
<img width="296" height="68" alt="image" src="https://github.com/user-attachments/assets/125d2801-12e4-48ed-be26-3540c6546c3e" />  
  
4. Set expressions engine to `JavaScript`  
<img width="497" height="94" alt="image" src="https://github.com/user-attachments/assets/113f8f1e-2245-4d46-9283-593e6517a340" />  
  
<br>
</details>

<details>
<summary> CS2 & CSGO </summary>
<br>

1. Record the required layers: game, exr depth, camera  
CS2 Tutorials: https://github.com/eirisocherry/cs2-editing/wiki  
CSGO Tutorials: https://www.youtube.com/watch?v=PtdO_I-fBRo&list=PLiyMyFJsq2_VbQNn3nL4sYAXbaIQRQsZH  
EXR Depth Guide (10:49): https://youtu.be/NE5nAPHn_P4?list=PLiyMyFJsq2_VbQNn3nL4sYAXbaIQRQsZH&t=649  
2. Import layers, camera data and setup an exr depth: https://www.youtube.com/watch?v=FWEqkaiXNM0  
3. Select a depth map and create a setup by pressing `[+]` button  

<img width="624" height="379" alt="image" src="https://github.com/user-attachments/assets/33a08da5-0bc1-4c15-8377-b5ffbe99de25" />  

4. Adjust `Depth Settings`:  
`Depth Black Is Near` whether a black color is near on your depth map or not: `unchecked`  
`Depth Far` the farthest depth point value: `25000` (the same value you set in `EXtractoR` effect)  

<img width="352" height="254" alt="Screenshot_5" src="https://github.com/user-attachments/assets/d0c7bbd9-0ee5-41a8-91ee-500354944968" />  
  
<img width="336" height="152" alt="Screenshot_4" src="https://github.com/user-attachments/assets/619dc92a-11f8-4010-b684-510ed7beab39" />  
  
5.  
a) Select `Depth Projection` depth layer and use `Project On Point` cursor to select where you want to project an object  
b) Select something from the dropdowm menu, ex: `Point Advanced`  
c) Press `Project` and wait a bit... Done!  
  
<img width="1284" height="611" alt="Screenshot_2" src="https://github.com/user-attachments/assets/3ad52698-e915-4db4-8b0c-57c14c66a5e5" />  
<img width="1261" height="524" alt="Screenshot_3" src="https://github.com/user-attachments/assets/a7206384-f213-4173-b5b8-50919014d5f8" />  

<br>
</details>



<details>
<summary> COD4 </summary>
<br>

1. Record the required layers  

<img width="552" height="326" alt="Screenshot_1" src="https://github.com/user-attachments/assets/8060be0e-6acc-4fc8-b1de-dc582c08892c" />  
  
```
mvm_output_directory "S:\Screens"      // output directory
mvm_avidemo_fps 0                      // disables default avi recording
mvm_streams_fps 125                    // fps to record layers in
mvm_streams_passes mvm_w mvm_wd mvm_wn // layers: game, depth, normal
mvm_streams_depthFormat 2              // depth format: 2 - rainbow (more precise)
mvm_export_format avi                  // output format must be set to avi, otherwise depth format will not work
mvm_streams_aeExport 1                 // export camera
mvm_streams_aeExport_sun 1             // export sun
```
2. Convert the layers using my `ftool-converter.bat`  
Download: https://github.com/eirisocherry/ftools/blob/main/ftool-converter.bat  
Guide: https://github.com/eirisocherry/ftools/tree/main  
3. Import the layers, camera data and convert a depth: https://github.com/gmzorz/MVMAETools/blob/main/Support%20Files/Scripts/ScriptUI%20Panels/MVMTools.jsx  
4.  
a) Rename Normal Map to `Normal Pass 1`  
b) Apply `Normal Remap` effect to the `Normal Pass 1` layer  
c) Copy the settings  
```
Input Is Normalized: `Checked`
X -> -X
Y -> -Z
Z -> -Y
Normalize Output: `Checked`
```  

<img width="1500" height="792" alt="Screenshot_7" src="https://github.com/user-attachments/assets/e9aef0bc-276e-45d0-8298-5df4402ea581" />  
  
5. Select a depth map and create a setup by pressing `[+]` button  
<img width="619" height="394" alt="Screenshot_8" src="https://github.com/user-attachments/assets/06709fe2-1a31-4250-b06d-f62e2d20a999" />  
  
6. Adjust `Depth Settings`:  
`Depth Black Is Near` whether a black color is near on your depth map or not: `unchecked`  
`Depth Far` the farthest depth point value: `4080`  
  
<img width="337" height="129" alt="image" src="https://github.com/user-attachments/assets/9f6993fd-d248-4986-8323-1dffca0aad65" />  
  
7.  
a) Select `Depth Projection` depth layer and use `Project On Point` cursor to select where you want to project an object  
b) Select something from the dropdowm menu, ex: `Point Advanced`  
c) Press `Project` and wait a bit... Done!  
  
<img width="1494" height="775" alt="image" src="https://github.com/user-attachments/assets/a2cea40f-fc40-4387-b86a-c2ce2fb6d22a" />  
  
<img width="1478" height="699" alt="image" src="https://github.com/user-attachments/assets/56f94c7e-5e44-4088-bacf-5b65f2ecdcf0" />  
  
<br>
</details>



## Important Notes
- The script is heavy and may crash your After Effects, but don't worry!  
It automatically saves your project before doing any actions, so even if your AE will crash, you will not lose any of your progress  
- During the process AE may freeze time by time, don't panic, just wait. It happens because light effects have hundreads of parameters and expressions, AE needs some time to process them.  
