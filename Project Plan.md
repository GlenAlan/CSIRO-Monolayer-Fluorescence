Goal: Automatically find fluorescent monolayers through fluorescence microscopy by utilizing Python and a Thorlabs camera and motorized stage.

The basic idea is to use Python to control the stage and use it to scan over the material sample, taking pictures at regular intervals with the camera. These images will then be stitched together to form a large picture of the entire sample. Using this we can use image processing such as openCV or machine learning to identify the location of monolayers (as they fluoresce under our microscope setup). This will all be accessible through a simple GUI (tkinter) which will allow for ease of use and calibration. If time permits we will extend ourselves by implementing additional features such as autofocus, autocalibration, autoexposure, machine learning, monolayer area and size detection, listing the dimensions of all monolayers, automatic travel to specific monolayer sites, etc.
  
We will use git and github in conjunction additional documentation and code comments to ensure our implementation is accessible and repeatable within and outside of CSIRO.

- Create a coordinate system
- Take images at regular intervals then either:
  - Stitch image together and find monolayers by coordinates/position.
  - Locate monolayers in each photo and find coordinates based on image location and monolayer location within image.

Stretch goal: Create an algorithm to identify location and size of monolayers automatically.
- Use algorithm to identify location of monolayers based off image system.
- Use algorithm to calculate area.

GUI:
- Calibration
- Start + end
- Image live view
- Results
- Focus

