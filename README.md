# CSIRO-Monolayer-Fluorescence
An algorithm to scan and identify monolayers by fluorescence microscopy using Thorlabs camera motorised stage  

`conda env create -f environment.yml`  

`conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia`

### Instructions for GUI
Hello! Are you tired of manually searching for monolayers, spending hours on those spinny wheels just to find a speck of fluorescing hair strand? Look no further, we have the solution for you!

Welcome to the Automatic Monolayer Finder (AMF). This software scans across your desired area of detection, taking photos and stitching them together to create a map of all your monolayers. The main function is found in the Main Control tab. Please follow the instructions below to use the AMF:

1. Go to the Calibration tab and set the focus magnitude of your lens (making sure to press Enter on your keyboard) then click AutoExpose. An option for AutoFocus is also available if required. Further parameter adjustments can be made in 'Advanced Parameters'.
2. Go to the Main Control tab and move to your starting corner in the top left either with the navigation buttons or manually and click 'Top Left' at the top of the screen.
3. Do the same for the ending corner in the bottom right and click 'Bottom Right' at the top of the screen.
4. Click 'Begin Search'. A live view of the progress will be shown.
5. You're done!

Once this is done you can head to the Results & Analysis tab where you will find the entire stitched image. This can be zoomed in on and panned around by dragging the mouse on the image. Double clicking on the image will magically move you to the position of the image under the microscope. You will also find a list of useful features of your monolayers such as area. Each of these monolayers can also be clicked and will magically display under the microscope.

The live view image of the microscope can be navigated in three ways:
- through the buttons found on the Main Control tab
- by entering the position in the boxes on the Main Control tab, or
- by clicking on the live view to the position you desire it to move to

Thank you for using the AMF and happy monolayer finding!