## Future suggested improvements:
- [ ] Find the hidden cat button and add more cat images
- [ ] Add a stop button (requires termination of threads begun in the `run_sequence` function)
- [ ] Add a *Browse* button to allow the user the select the image save location. (Edit the image paths or the save function to use a path stored as a new variable in `CONFIG.py`, make the button change the path variable)
- [ ] Give the autofocus function a bias towards the monolayer colour (stored in `CONFIG.py`) such that the autofocus prefers to focus on fluorescing objects more than other material.
- [ ] Refactor the code to use custom tkinter styles for GUI elements instead of specifying the colours every time and GUI element is added. (Unnecessary for functionality but would clean up the code)
- [ ] Refactor to avoid threading issues (much of the logic isn't strictly Thread Safe even though it may still work...)
