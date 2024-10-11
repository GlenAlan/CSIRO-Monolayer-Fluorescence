# CSIRO-Monolayer-Fluorescence
An algorithm to scan and identify monolayers by fluorescence microscopy using Thorlabs camera motorised stage  

If running on a new device firstly run:
`conda env create -f environment.yml`  

And if using a NVIDIA GPU:
`conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia`

### Instructions for General Use
Hello! Are you tired of manually searching for monolayers, spending hours on those spinny wheels just to find a speck of fluorescing hair strand? Look no further, we have the solution for you!

Welcome to the Automatic Monolayer Finder (AMF). This software scans across your desired area of detection, taking photos and stitching them together to create a map of all your monolayers. The main functionality is contained within the *Main Control* tab. Please follow the instructions below to use the AMF:

1. Go to the Calibration tab and set the magnification of your lens (making sure to press Enter on your keyboard). Place the material on the stage and click 'Auto Expose' if the exposure needs further adjusting. After that, locate a section of the sample and click 'Auto Focus'. The checkbox 'Auto Focus Enabled' enables or disables the automatic focus during a scan. Other parameters such as the image save directory, monolayer colour and camera rotation can be altered and further parameter adjustments can be made in 'Advanced Parameters'.
2. Go to the Main Control tab and move to the top bottom right corner, either with the navigation buttons, by clicking the live view or with the thorlabs stage control. Then click 'Bottom Right' at the top of the screen to define the corner. This is where the scan will start so ensure there is material within view and it is within focus.
3. Do the same for the ending corner in the top left and click 'Top Left' at the top of the screen.
4. Click 'Begin Search'. A live view of the progress will be shown.
5. Wait for scan to complete (coffee time!)
6. You're done!

The images will be automatically saved in the specified directory with a timestamp.
You can head to the Results & Analysis tab to find a the final image. This can be zoomed in/out and panned around by dragging the mouse on the image. Double clicking on the image will magically move you to the associated position under the microscope. Monolayers should be highlighted and numbered. Additionally, the monolayers will be displayed in a table along with a list of useful features such as area. Selecting one of these will magically move the microscope to that monolayer.

The live view image of the microscope can be navigated in three ways:
- The *Move Controls* found on the Main Control tab
- Entering the position in the boxes on the Main Control tab
- Clicking on the live view image to move to the position you clicked

Thank you for using the AMF and happy monolayer finding!