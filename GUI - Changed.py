import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import threading
import queue
import math
import cv2  # OpenCV library for accessing the webcam
import random
import time

from MCM301_COMMAND_LIB import *

# Constants and Configuration
CONFIRMATION_BITS = (2147484928, 2147484930, 2147483904)
CAMERA_DIMS = [2448, 2048]  # Dynamically updated later
CAMERA_PROPERTIES = {"gain": 255, "exposure": 150000}
NM_PER_PX = 171.6
IMAGE_OVERLAP = 0.05
DIST = int(min(CAMERA_DIMS) * NM_PER_PX * (1 - IMAGE_OVERLAP))
MAX_QUEUE_SIZE = 10
IMAGE_DISPLAY_SIZE = (640, 480)  # Display size optimized for performance
UPDATE_DELAY = 30  # Faster GUI updates (30ms)
POSITION_UPDATE_INTERVAL = 500  # Update interval for live position (ms)
THEME_COLOR = "#1E1E1E"
BUTTON_COLOR = "#2D2D2D"
TEXT_COLOR = "#FFFFFF"
HIGHLIGHT_COLOR = "#FF8C00"
FONT_FAMILY = "Arial"
FONT_SIZE = 10
HEADING_FONT = (FONT_FAMILY, 12, "bold")
LABEL_FONT = (FONT_FAMILY, 10)
BUTTON_FONT = (FONT_FAMILY, 10, "bold")

class LiveViewCanvas(tk.Canvas):
    def __init__(self, parent, image_queue):
        super().__init__(parent, bg=THEME_COLOR, highlightthickness=0)
        self.image_queue = image_queue
        self._image_width = 0
        self._image_height = 0
        self.is_active = False  # Initialize attribute to control image updates
        self.bind("<Configure>", self.on_resize)  # Bind resizing event
        self._display_image()

    def _display_image(self):
        if not self.is_active:
            self.after(UPDATE_DELAY, self._display_image)
            return

        try:
            image = self.image_queue.get_nowait()
            
            # Resize image to match canvas size while maintaining aspect ratio
            self.update_image_display(image)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error updating image: {e}")

        self.after(UPDATE_DELAY, self._display_image)  # Continue updating the image

    def update_image_display(self, image):
        """Resize image based on canvas size while maintaining aspect ratio."""
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        aspect = image.size[0] / image.size[1]
        
        if canvas_width / aspect < canvas_height:
            new_width = canvas_width
            new_height = int(canvas_width / aspect)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * aspect)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        self._image = ImageTk.PhotoImage(master=self, image=image)
        self.create_image(0, 0, image=self._image, anchor='nw')

    def on_resize(self, event):
        """Handle canvas resizing."""
        self._image_width = event.width
        self._image_height = event.height
        self._display_image()  # Update display to fit new canvas size

    def set_active(self, active):
        """Enable or disable image updating based on tab visibility."""
        self.is_active = active



def stage_setup():
    """Initializes and sets up the stage for movement."""
    mcm301obj = MCM301()
    return mcm301obj


def move(mcm301obj, pos, stages=(4, 5), wait=True):
    """
    Moves the stage to a specified position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (tuple): The desired position to move to, given as a tuple of coordinates in nanometers.
        stages (tuple): The stages to move, represented by integers between 4 and 6 (e.g., 4 for X-axis and 5 for Y-axis).
        wait (bool): Whether to wait for the movement to complete before returning.
   
    The function converts the given nanometer position into encoder units that the stage controller can use,
    then commands the stage to move to those positions. It continues to check the status of the stage
    until it confirms that the movement is complete.
    """
    print(f"Moving to {', '.join(str(p) for p in pos)}")

    for i, stage in enumerate(stages):
        coord = [0]

        # Convert the positions from nanometers to encoder units
        mcm301obj.convert_nm_to_encoder(stage, pos[i], coord)

        # Move the stages to the required encoder position
        mcm301obj.move_absolute(stage, coord[0])

    if wait:
        time.sleep(0.1)  # Small delay to ensure the stage starts moving


def get_pos(mcm301obj, stages=(4, 5, 6)):
    """Retrieves the current position of the specified stages."""
    pos = []
    for stage in stages:
        encoder_val, nm = [0], [0]
        mcm301obj.get_mot_status(stage, encoder_val, [0])
        mcm301obj.convert_encoder_to_nm(stage, encoder_val[0], nm)
        pos.append(nm[0])
    pos = [random.randint(0, 1000000) for _ in stages]  # Dummy values for testing
    return pos


def move_relative(mcm301obj, pos=[0, 0], stages=(4, 5), wait=True):
    """
    Moves the stage to a specified position relative to the current position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (list): The desired relative position to move to, given as a tuple of coordinates in nanometers.
        stages (tuple): The stages to move, represented by integers between 4 and 6 (e.g., 4 for X-axis and 5 for Y-axis).
        wait (bool): Whether to wait for the movement to complete before returning.
   
    The function retrieves the current position of the specified stage, adds the relative position to it,
    and then moves the stage to the new position. It continues to check the status of the stage
    until it confirms that the movement is complete.
    """
    pos = [p + c for p, c in zip(pos, get_pos(mcm301obj, stages))] 
    move(mcm301obj, pos, stages, wait)


class ImageAcquisitionThread(threading.Thread):
    def __init__(self, image_queue):
        super().__init__()
        self.image_queue = image_queue
        self.capture = cv2.VideoCapture(0)  # Open the first available camera
        self._is_running = True

    def run(self):
        while self._is_running:
            try:
                ret, frame = self.capture.read()
                if ret:
                    # Reduce frame size for performance
                    frame = cv2.resize(frame, IMAGE_DISPLAY_SIZE)

                    # Convert the frame to PIL image
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)

                    # Push the image to the queue if not full
                    if not self.image_queue.full():
                        self.image_queue.put(image)
                else:
                    print("Failed to grab frame")
                cv2.waitKey(1)  # Small delay for smoother performance
            except Exception as e:
                print(f"Error in image acquisition thread: {e}")

    def stop(self):
        self._is_running = False
        self.capture.release()


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Camera Control Interface')
        self.root.configure(bg=THEME_COLOR)

        # Initialize image queue before use
        self.image_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.image_acquisition_thread = ImageAcquisitionThread(self.image_queue)
        self.image_acquisition_thread.start()

        # Notebook and Tabs Setup
        self.setup_notebook()

        # Flags to control image updates
        self.update_active_main = False
        self.update_active_calib = False

        # Initializes MCM301 object
        self.mcm301obj = stage_setup()

        # Create UI Elements
        self.create_control_buttons()
        self.create_sliders()
        self.create_360_wheel()

        # Initialize Live Position Labels
        self.init_position_labels()

        # Start Live Position Updates
        self.update_positions()

        # Initialize Progress Bar and Status Label
        self.init_progress_bar()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_progress_bar(self):
        """Initialize the progress bar and status label in the main tab."""
        self.progress_value = tk.DoubleVar()
        self.progress_status = tk.StringVar()
        self.progress_status.set("Idle...")

        # Progress bar widget
        self.progress_bar = ttk.Progressbar(self.main_frame_text, orient="horizontal", length=200, mode="determinate", variable=self.progress_value)
        self.progress_bar.grid(row=6, column=0, pady=10, sticky='ew', columnspan=2)

        # Progress status label
        self.progress_label = tk.Label(self.main_frame_text, textvariable=self.progress_status, bg=THEME_COLOR, fg=TEXT_COLOR, font=LABEL_FONT)
        self.progress_label.grid(row=7, column=0, pady=5, sticky='w', columnspan=2)

    def update_progress(self, value, status_text):
        """Update the progress bar value and the status label."""
        self.progress_value.set(value)
        self.progress_status.set(status_text)

    def reset_progress(self):
        """Reset the progress bar and status label to initial values."""
        self.progress_value.set(0)
        self.progress_status.set("Idle...")

    def setup_notebook(self):
        """Setup Notebook and tabs for the GUI."""
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill="both")

        self.tabs = {
            "main": ttk.Frame(notebook, style='TFrame'),
            "calibration": ttk.Frame(notebook, style='TFrame'),
            "results": ttk.Frame(notebook, style='TFrame'),
            "devices": ttk.Frame(notebook, style='TFrame')
        }

        notebook.add(self.tabs["main"], text="Main Control")
        notebook.add(self.tabs["calibration"], text="Calibration")
        notebook.add(self.tabs["results"], text="Results & Analysis")
        notebook.add(self.tabs["devices"], text="Devices")

        # Apply styles
        style = ttk.Style()
        style.theme_use('default')

        # Configure styles for the notebook
        style.configure('TFrame', background=THEME_COLOR)
        style.configure('TNotebook', background=THEME_COLOR, foreground=TEXT_COLOR)
        style.configure('TNotebook.Tab', 
                        background=BUTTON_COLOR, 
                        foreground=TEXT_COLOR, 
                        font=LABEL_FONT, 
                        padding=(5, 1))  # Add padding for better aesthetics
        style.map('TNotebook.Tab', 
                background=[('selected', HIGHLIGHT_COLOR)], 
                foreground=[('selected', TEXT_COLOR)],
                expand=[('selected', [1, 1, 1, 0])])  # Makes the selected tab standout

        # Bind tab change event
        notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # Main Control Tab Layout
        self.main_frame_image = LiveViewCanvas(self.tabs["main"], self.image_queue)
        self.main_frame_image.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        self.main_frame_text = tk.Frame(self.tabs["main"], bg=THEME_COLOR)
        self.main_frame_text.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        # Calibration Tab Layout
        self.calib_frame_image = LiveViewCanvas(self.tabs["calibration"], self.image_queue)
        self.calib_frame_image.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        self.calib_frame_text = tk.Frame(self.tabs["calibration"], bg=THEME_COLOR)
        self.calib_frame_text.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        # Responsive Layout Configuration
        for tab in self.tabs.values():
            tab.grid_columnconfigure(0, weight=3)
            tab.grid_columnconfigure(1, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        self.main_frame_image.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.calib_frame_image.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')


    def init_position_labels(self):
        """Initialize position labels for live updates."""
        self.pos_names = ["Pos X", "Pos Y", "Focus Z"]
        self.position_labels_main = []
        self.position_labels_calib = []

        for i, name in enumerate(self.pos_names):
            label_main = tk.Label(self.main_frame_text, text=name, padx=10, pady=5, bg=THEME_COLOR, fg=TEXT_COLOR, font=LABEL_FONT)
            label_main.grid(row=i, column=0, sticky='w')
            pos_label_main = tk.Label(self.main_frame_text, text="0.00 nm", padx=5, bg=BUTTON_COLOR, fg=TEXT_COLOR, width=15, font=LABEL_FONT)
            pos_label_main.grid(row=i, column=1, sticky='w')
            self.position_labels_main.append(pos_label_main)

            label_calib = tk.Label(self.calib_frame_text, text=name, padx=10, pady=5, bg=THEME_COLOR, fg=TEXT_COLOR, font=LABEL_FONT)
            label_calib.grid(row=i, column=0, sticky='w', padx=(10, 5), pady=(5, 5))
            pos_label_calib = tk.Label(self.calib_frame_text, text="0.00 nm", padx=5, bg=BUTTON_COLOR, fg=TEXT_COLOR, width=15, font=LABEL_FONT)
            pos_label_calib.grid(row=i, column=1, sticky='w', padx=(5, 10), pady=(5, 5))
            self.position_labels_calib.append(pos_label_calib)

    def update_positions(self):
        """Update the positions displayed in the GUI."""
        positions = get_pos(self.mcm301obj, stages=(4, 5, 6))
        for i, pos_label in enumerate(self.position_labels_main):
            pos_label.config(text=f"{positions[i]:.2e} nm")

        for i, pos_label in enumerate(self.position_labels_calib):
            pos_label.config(text=f"{positions[i]:.2e} nm")

        # Schedule the next update
        self.root.after(POSITION_UPDATE_INTERVAL, self.update_positions)

    def on_tab_change(self, event):
        selected_tab = event.widget.index("current")
        if selected_tab == 0:  # Main Tab
            self.update_active_main = True
            self.update_active_calib = False
            self.main_frame_image.set_active(True)
            self.calib_frame_image.set_active(False)
        elif selected_tab == 1:  # Calibration Tab
            self.update_active_main = False
            self.update_active_calib = True
            self.main_frame_image.set_active(False)
            self.calib_frame_image.set_active(True)
        else:
            self.update_active_main = False
            self.update_active_calib = False
            self.stop_image_updates()

    def stop_image_updates(self):
        """Stop image updates when not on the main or calibration tabs."""
        self.main_frame_image.set_active(False)
        self.calib_frame_image.set_active(False)

    def create_control_buttons(self):
        self.enter_pos = tk.Label(self.main_frame_text, text="Enter Positions (nm):", bg=THEME_COLOR, fg=TEXT_COLOR, font=HEADING_FONT)
        self.enter_pos.grid(row=3, column=0, pady=10, sticky='w')
        self.enter_focus = tk.Label(self.calib_frame_text, text="Focus Control:", bg=THEME_COLOR, fg=TEXT_COLOR, font=HEADING_FONT)
        self.enter_focus.grid(row=5, column=0, pady=10, sticky='w', columnspan=2)

        self.create_position_entries()
        self.create_main_frame_buttons()
        self.create_calibration_controls()

    def create_position_entries(self):
        self.pos_entry_x = tk.Entry(self.main_frame_text, font=LABEL_FONT)
        self.pos_entry_x.grid(row=4, column=0, pady=10, sticky='w')
        self.pos_entry_y = tk.Entry(self.main_frame_text, font=LABEL_FONT)
        self.pos_entry_y.grid(row=4, column=1, pady=10, sticky='w')
        self.pos_entry_z = tk.Entry(self.calib_frame_text, font=LABEL_FONT)
        self.pos_entry_z.grid(row=4, column=1, pady=10, sticky='w')

        self.pos_entry_x.bind('<Return>', lambda event, type="XY": self.submit_entries(type))
        self.pos_entry_y.bind('<Return>', lambda event, type="XY": self.submit_entries(type))
        self.pos_entry_z.bind('<Return>', lambda event, type="Z": self.submit_entries(type))

    def create_main_frame_buttons(self):
        self.main_frame_text_pos = tk.Frame(self.main_frame_text, bg=THEME_COLOR, padx=25)
        self.main_frame_text_pos.grid(row=5, sticky='w', pady=30)

    def create_calibration_controls(self):
        calib_button_focus_label = tk.Label(self.calib_frame_text, text="Focus Slider", bg=THEME_COLOR, fg=TEXT_COLOR, font=LABEL_FONT)
        calib_button_focus_label.grid(row=6, column=0, pady=10, columnspan=2, sticky='w')

        # Create navigation buttons for calibration tab
        calib_controls = [
            ("Up", (7, 1), lambda: move_relative(self.mcm301obj, pos=[int(DIST / 2)], stages=(5,), wait=False)),
            ("Left", (8, 0), lambda: move_relative(self.mcm301obj, pos=[int(-DIST / 2)], stages=(4,), wait=False)),
            ("Right", (8, 2), lambda: move_relative(self.mcm301obj, pos=[int(DIST / 2)], stages=(4,), wait=False)),
            ("Down", (9, 1), lambda: move_relative(self.mcm301obj, pos=[int(-DIST / 2)], stages=(5,), wait=False)),
            ("Zoom in", (7, 3), lambda: move_relative(self.mcm301obj, pos=[10000], stages=(6,), wait=False)),
            ("Zoom out", (9, 3), lambda: move_relative(self.mcm301obj, pos=[-10000], stages=(6,), wait=False))
        ]

        for text, (row, col), cmd in calib_controls:
            button = tk.Button(self.calib_frame_text, text=text, command=cmd, bg=BUTTON_COLOR, fg=TEXT_COLOR, font=BUTTON_FONT)
            button.grid(row=row, column=col, padx=10, pady=5, sticky='nsew')

    def create_sliders(self):
        slider_frame = tk.Frame(self.calib_frame_text, bg=THEME_COLOR)
        slider_frame.grid(row=10, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

        slider_focus_label = tk.Label(slider_frame, text="Focus Slider:", bg=THEME_COLOR, fg=TEXT_COLOR, font=LABEL_FONT)
        slider_focus_label.grid(row=0, column=0, pady=10, sticky='w')

        slider_focus = tk.Scale(slider_frame, from_=100000, to=1000000, orient='vertical',
                                command=lambda event: self.on_focus_slider_change(event), bg=THEME_COLOR, fg=TEXT_COLOR)
        slider_focus.grid(row=1, column=0, padx=20, pady=10, sticky='nsew')

    def create_360_wheel(self):
        wheel_frame = tk.Frame(self.calib_frame_text, bg=THEME_COLOR)
        wheel_frame.grid(row=11, column=0, columnspan=2, padx=20, pady=20, sticky='nsew')

        camera_wheel_label = tk.Label(self.calib_frame_text, text="360 Degree Wheel:", background=THEME_COLOR, foreground=TEXT_COLOR, font=LABEL_FONT)
        camera_wheel_label.grid(row=12, column=0, pady=10, columnspan=2)

        self.canvas = tk.Canvas(wheel_frame, width=200, height=200, bg=THEME_COLOR, highlightthickness=0)
        self.canvas.pack()

        self.radius = 80
        self.center_x = 100
        self.center_y = 100

        self.image = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.draw_anti_aliased_wheel()

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.pointer_line = self.canvas.create_line(self.center_x, self.center_y,
                                                    self.center_x + self.radius, self.center_y,
                                                    width=2, fill=HIGHLIGHT_COLOR)
        self.canvas.bind("<B1-Motion>", self.update_wheel)

        self.angle_entry = tk.Entry(wheel_frame, width=5, font=LABEL_FONT)
        self.angle_entry.pack(pady=10)
        self.angle_entry.insert(0, "0")
        self.angle_entry.bind("<Return>", self.set_angle_from_entry)


    def on_focus_slider_change(self, event):
        focus_value = int(event)
        print(f"Focus Slider moved to: {focus_value}")


    def draw_anti_aliased_wheel(self):
        self.draw.ellipse(
            [self.center_x - self.radius, self.center_y - self.radius,
             self.center_x + self.radius, self.center_y + self.radius],
            outline=HIGHLIGHT_COLOR, width=2
        )

    def update_wheel(self, event):
        dx = event.x - self.center_x
        dy = event.y - self.center_y
        angle = int(math.degrees(math.atan2(dy, dx))) % 360

        end_x = self.center_x + self.radius * math.cos(math.radians(angle))
        end_y = self.center_y + self.radius * math.sin(math.radians(angle))
        self.canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)

        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, f"{angle:.0f}")

        print(f"Camera rotation updated to: {angle:.0f} degrees")

    def submit_entries(self, type="XY"):
        if type == "XY":
            enter_x = self.pos_entry_x.get().strip()
            enter_y = self.pos_entry_y.get().strip()
            if self.validate_entries(enter_x, enter_y):
                threading.Thread(target=self.move_and_update_progress, args=([int(enter_x) * 1000000, int(enter_y) * 1000000],), daemon=True).start()
        elif type == "Z":
            enter_z = self.pos_entry_z.get().strip()
            if self.validate_entries(enter_z):
                threading.Thread(target=self.move_and_update_progress, args=([int(enter_z) * 1000000],), daemon=True).start()



    ############################## TEST ONLY, REPLACE WITH ACTUAL UPDATES ##############################
    def move_and_update_progress(self, pos, stages=(4, 5)):
        """Move the stage to a position and update the progress bar during the operation."""
        self.update_progress(0, "Moving stage...")
        total_steps = 100
        for i in range(total_steps):
            move(self.mcm301obj, pos, stages)  # Simulating movement
            progress = (i + 1) / total_steps * 100
            self.update_progress(progress, f"Progress: {int(progress)}%")
            self.root.update_idletasks()  # Ensure UI gets updated during the loop

        self.update_progress(100, "Movement complete.")
    #####################################################################################################

    def validate_entries(self, *entries):
        for entry in entries:
            if not entry.isdigit():
                messagebox.showerror("Input Error", "All fields must be integers!")
                return False
        return True

    def set_angle_from_entry(self, event):
        try:
            angle = float(self.angle_entry.get()) % 360
        except ValueError:
            angle = 0

        end_x = self.center_x + self.radius * math.cos(math.radians(angle))
        end_y = self.center_y + self.radius * math.sin(math.radians(angle))
        self.canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)

        print(f"Camera rotation set to: {angle:.2f} degrees")

    def on_closing(self):
        """Handle the cleanup on closing the application."""
        try:
            self.image_acquisition_thread.stop()
            self.image_acquisition_thread.join()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        self.root.destroy()


if __name__ == "__main__":
    print("App initializing...")
    root = tk.Tk()
    GUI_main = GUI(root)

    print("App starting")
    root.mainloop()

    print("Waiting for image acquisition thread to finish...")
    GUI_main.image_acquisition_thread.stop()
    GUI_main.image_acquisition_thread.join()

    print("Closing resources...")
    print("App terminated. Goodbye!")
