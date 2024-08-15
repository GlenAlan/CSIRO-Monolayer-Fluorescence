Format: some projects are well suited to keeping a lab book, either physical or electronic, which records all your daily tasks, recorded values, results, plots, thoughts, useful numbers, etc. For other projects, this documentation may come in a different form, e.g. organised notes or minutes from project meetings, GitHub commit records, or progress reports.

Whatever form you and your supervisor think is appropriate, your project documentation should:
- Include a project plan formed during the first few weeks of the project. This should include key milestones and dates.
- Be legible and clearly laid out, with dates against every entry/contribution.
- Include regular entries or updates. Most projects will make weekly progress, and so should have weekly updates to the documentation collection.

Lachlan Catto and Glen Pearce's Lab Record

# 2/8/24

**We went through the Risk Assessment**  
This included emergency exits, lab hazards and risk control strategies

Today we familiarised ourselves with the microscope apparatus and SDK software. We setup a GitHub folder and organised the folders to suit the project. We commenced a project plan for how we are going to approach the project. We created a python file and created our own code from the ground by modifying the code that Matt had setup for us. A python file was created for each task, namely camera viewing and stage movement. Within the stage movement file, we were able to move the stage in all 3 directions the desired step amount and make it go to the home positions.

We started considering how the monolayer locating algorithm was going to work. Some ideas included sweeping across and moving down once it detects an edge through colour or contrast recognition; taking photos as it does so including overlap and stitching the photos together at the end.

### Next week
- [X] Check FOV
- [x] Start scanning algorithm
- [x] Take photos
- [x] Save Photos
- [x] Get accurate location + coord system
- [ ] Formalise project plan and steps

# 9/8/24

We worked on the project plan together, brainstorming our ideas on the whiteboard. We established the key steps that need to be done in the project including creating a GUI, calibrating the origin, creating a conversion between nm and pixels, and a scanning algorithm.

### Glen
I commenced

### Lachlan
I started designing and coding the GUI for viewing the image and controlling the stage. I spent today familiarising myself more with git including how commits work as well as Tkinter which is used for creating a GUI. I learned how to produce frames and tabs in a GUI and display images, text and buttons. I then implemented what I learnt to display a test image of a monolayer in the GUI as well as a few buttons which don't do anything yet.

Next week I will continue working on the GUI, creating buttons used for navigation on the image and text displaying the position that the image will be in based off its locationg from ThorLabs. I will also create a calibration tab that is used for manually calibrating the start and end points for the sweeping algorithm.

### Next week
- [ ] Formalise Project Plan
- [ ] Create GUI

# 16/8/24