# DSP_final_project
Applications of the GrabCut Algorithm in Background Removal and Image Blending
-------------------------------------------------------------------------------------------------
**How to use the code?**
1. Open up and run "implementation.py".
2. After running "implemetation.py", you will get the segmented picture automatically stored in ./result
3. Open up and run "New_background_function.py" to put the picture on new background you choose.
-------------------------------------------------------------------------------------------------
**Meaning of each folder and files**  

Feathering_function.py &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; ==>	Edge Smoothing with Gaussian Filter  
Feathering_function_version2.py ==>	Edge Smoothing with Gaussian Filter (better one)  
Grabcut_handmade.py  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;           ==> handmade Grabcut function  
Implementation.py      &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;  &ensp;&ensp;&ensp;       ==> main function that performs grabcut, magic wand, and edge smoothing  
Magic_wand_function.py    &ensp;&ensp;&ensp;&ensp;   &ensp;&ensp;  ==> magic wand function that help improve the performance of grabcut result  
New_background_function.py  &ensp;&ensp;   ==> second main function that performs putting segmented picture on new background  

results&ensp;&ensp;&ensp;&ensp;    &ensp; ==>  save all the results picture inside this folder  
raw_picture&ensp; ==> some example input picture to be used  
Backgrounds ==> background picture library   

Edited by Chia-Sung Chang 12/20/2024
