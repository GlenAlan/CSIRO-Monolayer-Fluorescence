Goal: Automatically find fluorescent monolayers through fluorescence microscopy by utilizing Python and a Thorlabs camera and motorized stage.

The basic idea is to use Python to control the stage and use it to scan over the material sample, taking pictures at regular intervals with the camera. These images will then be stitched together to form a large picture of the entire sample. Using this we can use image processing such as openCV or machine learning to identify the location of monolayers (as they fluoresce under our microscope setup). This will all be accessible through a simple GUI (tkinter) which will allow for ease of use and calibration. If time permits we will extend ourselves by implementing additional features such as autofocus, autocalibration, autoexposure, machine learning, monolayer area and size detection, listing the dimensions of all monolayers, automatic travel to specific monolayer sites, etc.
  
We will use git and github in conjunction additional documentation and code comments to ensure our implementation is accessible and repeatable within and outside of CSIRO.
  
Our idea for the implementation of this is 
- Read the camera live feed with Python
- Control the stage movements wth Python
- Get stage coordinates
- Simplify the system controls with basic functions
- Identify the sample location on the stage
- Create basic scanning movement
- Take photos during the scan 
- Convert stage coordinates to camera pixels
- Stitch the images together
- Blend the image boundaries
- **Add camera exposure control (and other params)**
- **Automate for consistent exposure**
- Image post processing
  - Greyscale (with bias)
  - Bright/Contrast
  - Blur
  - Threshold
- Find monolayers with OpenCV
- Draw and list all monolayers on sample
- Find location and area of monolayers
- Sort by largest
- **Add a goto command (move to a specific monolayer)**
- **Add a consistent start point (or some method of location reproducibility)**
- GUI (ensure all necessary steps are accessible in the GUI)
  - Image live view
  - Image live stitched view
  - Manual controls for defining start/end
  - Manual focus controls
  - Position live view (nm & px)
  - Progress bar (or progress visualisation)
  - Calibration tab
    - Focus
    - Camera Rotation
  - Results tab?
    - Visualise largest monolayers
    - Get monolayer statistics 
      - Number
      - Area
      - Location
      - Distribution
- Automatic focus
- Automatic calibration
- Machine learning identification
- Bilayers???????????????
  
  
## Timeline 
Week 1 (02/08):
- Obtain project outline
- Plan basic goals
- Set up good file management practices

Week 2 (09/08):
- Base functions defined
- Movement and Camera work
- Simple GUI interface

Week 3 (16/08):
- Basic implementation of algorithm
- Display stitched image
- GUI tab for calibration
- GUI movement buttons
- GUI view position

Week 4 (23/08):
- Postprocessing
- Basic monolayer identification
- GUI camera view
- GUI basic use flow
- GUI start/end controls
- GUI calibration

Week 5 (30/08):
- GUI image live stitched view
- GUI progress bar
- Autofocus
- Crystal Stats
- Table output
- Data visualisation