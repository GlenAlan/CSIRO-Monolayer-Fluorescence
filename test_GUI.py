import tkinter as tk
from tkinter import ttk, filedialog, colorchooser
from tkinter import font as tkFont
import random
import math
import warnings
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import platform

OS = platform.system()

class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

class CanvasImage:
    """ Display and zoom image """
    def __init__(self, placeholder, path):
        """ Initialize the ImageFrame """
        # To do - change the image scale so that it fits inside rectangle every time. Delete black space.
        
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Image.LANCZOS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.path = path  # path to the image, should be public for outer classes
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')

        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set,
                                width=800, height=800)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        """ Scale the image to fit within the canvas initially and center it. """
        # canvas_width = self.canvas.winfo_width()
        # canvas_height = self.canvas.winfo_height()

        # ############################################ Inputted code, may not work
        # # Load the image using PIL
        # self.__image = Image.open(image_path)
        # self.imwidth, self.imheight = self.__image.size  # Original image dimensions
        # self.imscale = 1.0  # Initial scale factor
        # if canvas_width > 1 and canvas_height > 1:
        #     # Calculate the scale factor based on canvas size and image size
        #     scale_factor = min(canvas_width / self.imwidth, canvas_height / self.imheight)

        #     # Set the initial scaling factor to make the image fully visible
        #     self.imscale = scale_factor

        #     # Resize the image based on the scale factor
        #     new_width = int(self.imwidth * self.imscale)
        #     new_height = int(self.imheight * self.imscale)
        #     self.tkimage = self.__image.crop((0,0,new_width, new_height))
        #     self.image_resized = self.tkimage.resize((new_width, new_height), Image.LANCZOS)
        # ############################################

        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)

        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>',     self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.__wheel)  # zoom for Linux, wheel scroll up
        self.canvas.bind('<Double-Button-1>', self.__double_click)  # handle double-click for coordinates
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))

        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning for big image
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)  # open image, but down't load it into RAM
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
           self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side

        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        (w, h), m, j = self.__pyramid[-1].size, 512, 0
        n = math.ceil(math.log(min(w, h) / m, self.__reduction)) + 1  # image pyramid length
        while w > m and h > m:  # top pyramid image is around 512 pixels in size
            j += 1
            print('\rCreating image pyramid: {j} from {n}'.format(j=j, n=n), end='')
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        print('\r' + (40 * ' ') + '\r', end='')  # hide printed string
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 0, math.ceil(self.imheight / self.__band_width)
        while i < self.imheight:
            j += 1
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k)+1), self.__filter), (0, int(i * k)))
            i += band
        print('\r' + (40 * ' ') + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if  box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0]  = box_img_int[0]
            box_scroll[2]  = box_img_int[2]
        # Vertical part of the image is in the visible area
        if  box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1]  = box_img_int[1]
            box_scroll[3]  = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image, which does not fit in RAM
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                                    (int(x1 / self.__scale), int(y1 / self.__scale),
                                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def __double_click(self, event):
        """ Get the original pixel coordinates of the image on double-click """
        # Get the canvas coordinates (including any panning/scrolling)
        x = self.canvas.canvasx(event.x) - self.canvas.coords(self.container)[0]
        y = self.canvas.canvasy(event.y) - self.canvas.coords(self.container)[1]

        # Scale back to the original image size by dividing by the current zoom (imscale)
        x_original = x / self.imscale
        y_original = y / self.imscale

        # Ensure the coordinates are within the bounds of the original image size
        x_original = max(0, min(self.imwidth, x_original))
        y_original = max(0, min(self.imheight, y_original))

        print(f"Original image coordinates: ({int(x_original)}, {int(y_original)})")

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        if OS == 'Darwin':
            if event.delta<0:  # scroll down, zoom out, smaller
                if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
                self.imscale /= self.__delta
                scale        /= self.__delta
            if event.delta>0:  # scroll up, zoom in, bigger
                i = float(min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1)
                if i < self.imscale: return  # 1 pixel is bigger than the visible area
                self.imscale *= self.__delta
                scale        *= self.__delta
        else:
            # Respond to Linux (event.num) or Windows (event.delta) wheel event
            if event.num == 5 or event.delta == -120:  # scroll down, zoom out, smaller
                if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
                self.imscale /= self.__delta
                scale        /= self.__delta
            if event.num == 4 or event.delta == 120:  # scroll up, zoom in, bigger
                i = float(min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1)
                if i < self.imscale: return  # 1 pixel is bigger than the visible area
                self.imscale *= self.__delta
                scale        *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right, keys 'd' or 'Right'
                self.__scroll_x('scroll',  1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left, keys 'a' or 'Left'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up, keys 'w' or 'Up'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down, keys 's' or 'Down'
                self.__scroll_y('scroll',  1, 'unit', event=event)

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        if self.__huge:  # image is huge and not totally in RAM
            band = bbox[3] - bbox[1]  # width of the tile band
            self.__tile[1][3] = band  # set the tile height
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3  # set offset of the band
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # image is totally in RAM
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()
        self.__imframe.destroy()

class MainWindow(ttk.Frame):
    """ Main window class """
    def __init__(self, mainframe, path):
        """ Initialize the main Frame """
        ttk.Frame.__init__(self, master=mainframe)
        # self.master.title('Advanced Zoom v3.0')
        # self.master.geometry('800x600')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)
        canvas = CanvasImage(self.master, path)  # create widget
        canvas.grid(row=0, column=0)  # show widget

'''
Class for running tkinter root
'''
class ImageDisplay:
    '''
    Initialising for creating tabs, labels and initial images.
    '''
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("Image display and rolling a die")
        
        # This/these are not needed anymore -->
        # self.root.call('tk', 'scaling', 3)  # Adjust this scaling factor as needed

        # Set a global font for the entire app
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=12)  # Set font size here

        # Create a Notebook (tab container)
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill="both")

        # Create the first tab
        self.tab1 = ttk.Frame(notebook)
        notebook.add(self.tab1, text="Tab 1")

        # Add content to the first tab
        label1 = tk.Label(self.tab1, text="This is Tab 1", font=("Arial", 16))
        label1.pack(pady=20)

        # Create the second tab
        self.tab2 = ttk.Frame(notebook)
        notebook.add(self.tab2, text="Tab 2")

        # Create the third tab
        self.tab3 = ttk.Frame(notebook)
        notebook.add(self.tab3, text="Tab 3")

        # Create the fourth tab
        self.tab4 = ttk.Frame(notebook)
        notebook.add(self.tab4, text="Tab 4")
        # Create the fourth tab
        self.tab5 = ttk.Frame(notebook)
        notebook.add(self.tab5, text="Tab 5")

        # Add content to the second tab
        label2 = tk.Label(self.tab2, text="This is Tab 2", font=("Arial", 16))
        label2.pack(pady=20)

        # Frame configs
        self.root.columnconfigure(0, minsize=150)
        self.root.rowconfigure([0, 1], minsize=50)

        # Buttons for the second tab
        self.btn_roll = tk.Button(self.tab3, text = "Roll!", command = self.randNum)
        self.btn_roll.grid(row=2, column=0)
        self.lbl_roll = tk.Label(self.tab3)
        self.lbl_roll.grid(row=3, column=0)
        self.cat_button = tk.Button(self.tab3, text="~ Magic button ~", command=self.show_cat_image)
        self.cat_button.grid(row=4, column=0, pady=20)

        # Test zoomable image in second tab
        self.zoom_frame = tk.Frame(self.tab2, width=1000, height=1000)
        self.zoom_frame.pack()

        # Initial zoom factor and region for zoomed view
        self.zoom_factor = 2
        self.region_size = 200  # Size of the region to zoom into
        self.region = [0,0,600,600]
        self.fixed_resize = (600, 600)

        # Create an entry widget for the zoom factor
        self.zoom_factor_entry = tk.Entry(self.tab1)
        self.zoom_factor_entry.pack(pady=10)
        self.zoom_factor_entry.insert(0, str(self.zoom_factor))  # Set default zoom factor
        self.zoom_factor_entry.bind('<Return>', self.update_zoom_factor)

        # Create an entry widget
        self.entry = tk.Entry(self.root)
        self.entry.pack(pady=10)

        # Bind the Enter key to the entry widget
        self.entry.bind('<Return>', self.get_entry_value)

        # Dummy variables
        self.dummy_var1 = tk.DoubleVar(value=0)
        self.dummy_var2 = tk.DoubleVar(value=0)

        # Colour picker frame in first tab
        color_frame = tk.Frame(self.tab1)
        color_frame.pack(pady=10)
        
        # Set the default color
        self.color_code = "#FF4500"  # Default color (reddish-orange)

        # Add a text label to the left
        self.text_label = tk.Label(color_frame, text="Fluorescence colour picker:")
        self.text_label.grid(row=0, column=0, padx=10)

        # Create a smaller square label that acts as the color picker with the default color
        self.color_label = tk.Label(color_frame, bg=self.color_code, width=5, height=2)  # Smaller square
        self.color_label.grid(row=0, column=1)

        # Bind the label click event to open the color picker
        self.color_label.bind("<Button-1>", self.pick_color)

        # Frame configs
        self.root.columnconfigure(0, minsize=150)
        self.root.rowconfigure([0, 1], minsize=50)

        # Image handling
        self.image_path = "Images/highresphoto.jpg"  # Replace with your image path
        self.original_image = Image.open(self.image_path)  # Get the image from file
        self.imwidth, self.imheight = self.original_image.size  # Original image dimensions

        # Canvas for displaying images
        self.image_canvas = tk.Canvas(self.tab1, width=1400, height=600)
        self.image_canvas.pack()
        self.image_canvas_move = tk.Canvas(self.tab5, width=600, height=600)
        self.image_canvas_move.pack()

        # Display the full image on the right and bind mouse motion
        self.tk_resized_image = self.original_image.resize(
            (int((self.region[2] - self.region[0])),
             int((self.region[3] - self.region[1]))),
            Image.LANCZOS)
        self.tk_full_image = ImageTk.PhotoImage(self.tk_resized_image)
        self.image_canvas.create_image(650, 0, anchor=tk.NW, image=self.tk_full_image)

        # Bind mouse movement on the full image area to trigger zooming
        self.image_canvas.bind("<Motion>", self.update_zoom_on_mouse_move)

        # Moving around in zoom image. Distance moved depending on distance of click from center of canvas
        # Initial zoom (1x in this case)
        self.imscale = 1.0  
        self.image_resized = self.original_image.resize((int(self.imwidth * self.imscale),
                                                int(self.imheight * self.imscale)), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.image_resized)

        # Add image to the canvas
        self.imageid = self.image_canvas_move.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.image_canvas_move.config(scrollregion=self.image_canvas_move.bbox(self.imageid))

        # Bind the mouse click event
        self.image_canvas_move.bind("<Button-1>", self._on_click)

        tick_width = 500
        tick_height = 300
        tick_scale = tk.Canvas(self.tab1, width=tick_width, height=tick_height)
        tick_scale.pack()

        # Draw a rectangle as an example of an image placeholder
        tick_scale.create_rectangle(50, 50, tick_width-50, tick_height-100, outline='blue', fill='lightblue')

        # Add a measurement scale below the image
        self.draw_scale(tick_scale, tick_width, tick_height - 20)

        # self.plot_histograms(self.tab4)

        self.create_control_buttons()

        self.create_sliders()

        self.create_360_wheel()

    def draw_scale(canvas, width, height, nm_per_tick=10):
        # Parameters for the scale
        tick_length = 10  # Length of the major ticks
        scale_start = 0   # Starting point in nm (0nm)
        scale_end = 100   # End point in nm (100nm)
        
        # Number of ticks based on the canvas width
        total_ticks = int((scale_end - scale_start) / nm_per_tick)

        # Space between each tick mark
        tick_spacing = width / total_ticks
        
        # Draw scale
        for i in range(total_ticks + 1):
            x = i * tick_spacing
            canvas.create_line(x, height, x, height - tick_length, fill='black')
            canvas.create_text(x, height - tick_length - 10, text=str(i * nm_per_tick) + 'nm', anchor='s', font=('Arial', 8))

    def _on_click(self, event):
        """ Handle a mouse click to move the image """
        # Get the position of the click
        click_x, click_y = event.x, event.y

        # Calculate the center of the canvas
        canvas_center_x = self.image_canvas_move.winfo_width() / 2
        canvas_center_y = self.image_canvas_move.winfo_height() / 2

        # Calculate the distance and direction from the center of the canvas
        move_x = click_x - canvas_center_x
        move_y = click_y - canvas_center_y

        # Move the image to the new position, respecting the boundaries
        ##### replace with move function for liveview camera with move_x and move_y parameters
        self.image_canvas_move.move(self.imageid, -move_x, -move_y)
            
    def update_zoom_on_mouse_move(self, event):
        mouse_x = event.x
        mouse_y = event.y

        # Adjust the mouse position to be relative to the large image (on the right)
        image_x = mouse_x - 650  # Since the large image starts at x=650
        image_y = mouse_y

        # Calculate the adjusted region size based on the zoom factor
        adjusted_region_size = self.region_size / self.zoom_factor  # Region size shrinks as zoom factor increases

        # Make sure the mouse is within the bounds of the image
        if 0 <= image_x < self.tk_resized_image.width and 0 <= image_y < self.tk_resized_image.height:
            # Calculate the region to zoom into, centered around the mouse position
            left = max(0, image_x - adjusted_region_size // 2)
            top = max(0, image_y - adjusted_region_size // 2)
            right = min(self.tk_resized_image.width, image_x + adjusted_region_size // 2)
            bottom = min(self.tk_resized_image.height, image_y + adjusted_region_size // 2)

            # If the mouse is too close to the right/bottom edges, adjust the left/top to fit the region
            if right - left < adjusted_region_size:
                if left == 0:
                    right = adjusted_region_size
                else:
                    left = right - adjusted_region_size
            if bottom - top < adjusted_region_size:
                if top == 0:
                    bottom = adjusted_region_size
                else:
                    top = bottom - adjusted_region_size

            # Update the region to reflect the zoomed area
            self.region = [left, top, right, bottom]

            # Update the zoomed image
            self.update_zoomed_image()


    def update_zoomed_image(self):
        # Crop the image to the selected region
        cropped_image = self.tk_resized_image.crop(self.region)
        
        # Resize the cropped image based on the new size
        resized_image = cropped_image.resize(self.fixed_resize, Image.LANCZOS)

        # If the resized image exceeds (600, 600), resize to (600, 600)
        if resized_image.size[0] > 600 or resized_image.size[1] > 600:
            resized_image = resized_image.resize(self.fixed_resize, Image.LANCZOS)

        # Display the zoomed-in image on the left side of the canvas
        self.tk_zoomed_image = ImageTk.PhotoImage(resized_image)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_zoomed_image)


    def update_zoom_factor(self, event):
        try:
            new_zoom_factor = float(self.zoom_factor_entry.get())
            if new_zoom_factor > 0:  # Ensure the zoom factor is positive
                self.zoom_factor = new_zoom_factor
                self.update_zoomed_image()  # Update the zoomed image with the new zoom factor
                print(self.zoom_factor)
        except ValueError:
            print("Invalid zoom factor. Please enter a valid number.")
    
    '''
    Creates the buttons
    '''
    def create_control_buttons(self):
        button_frame = tk.Frame(self.tab1)
        button_frame.pack(pady=10)

        btn_up = tk.Button(button_frame, text="Up", command = lambda: self.move(dy=-1))
        btn_up.grid(row=0, column=1)

        btn_left = tk.Button(button_frame, text="Left", command = lambda: self.move(dx=-1))
        btn_left.grid(row=1, column=0)

        btn_right = tk.Button(button_frame, text="Right", command = lambda: self.move(dx=1))
        btn_right.grid(row=1, column=2)

        btn_down = tk.Button(button_frame, text="Down", command = lambda: self.move(dy=1))
        btn_down.grid(row=2, column=1)

    def create_sliders(self):
        # Slider 1 for Dummy Variable 1
        slider1_label = ttk.Label(self.tab3, text="Dummy Variable 1:")
        slider1_label.grid(row=0, column=0, pady=10)

        # Slider displays value from variable and does command whenever you slide it.
        # Could use command to update z position to change focus and/or have position as the variable.
        # Could also just not have variable displayed and have it as a label somewhere else.
        slider1 = tk.Scale(self.tab3, from_=0, to=100, orient='vertical')
        slider1.grid(row=0, column=1, padx=20, pady=10)

        # Slider 2 for Dummy Variable 2
        slider2_label = ttk.Label(self.tab3, text="Dummy Variable 2:")
        slider2_label.grid(row=1, column=0, pady=10)

        slider2 = tk.Scale(self.tab3, from_=0, to=100, orient='vertical',
                           variable=self.dummy_var2, command=self.update_var2)
        slider2.grid(row=1, column=1, padx=20, pady=10)

    def create_360_wheel(self):
        wheel_frame = ttk.Frame(self.tab3)
        wheel_frame.grid(padx=20, pady=20)

        wheel_label = ttk.Label(wheel_frame, text="360 Degree Wheel (Dummy Variable 2):")
        wheel_label.pack()

        # Create a canvas for the 360-degree wheel
        self.wheel_canvas = tk.Canvas(wheel_frame, width=200, height=200, bg="white")
        self.wheel_canvas.pack()

        # Set up image parameters
        self.radius = 80  # Radius of the wheel
        self.center_x = 100  # Center X of the wheel
        self.center_y = 100  # Center Y of the wheel

        # Create a blank image with transparency
        self.wheel_image = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
        self.draw = ImageDraw.Draw(self.wheel_image)

        # Draw the anti-aliased wheel circle
        self.draw_anti_aliased_wheel()

        # Convert the image to Tkinter-compatible format
        self.tk_wheel_image = ImageTk.PhotoImage(self.wheel_image)

        # Display the wheel on the canvas
        self.wheel_circle = self.wheel_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_wheel_image)

        # Draw the initial pointer
        self.pointer_line = self.wheel_canvas.create_line(self.center_x, self.center_y,
                                                    self.center_x + self.radius, self.center_y,
                                                    width=2, fill="red")

        # Bind mouse events to drag the wheel
        self.wheel_canvas.bind("<B1-Motion>", self.update_wheel)

        # Entry box for manually setting the angle
        self.angle_entry = tk.Entry(wheel_frame, width=5)
        self.angle_entry.pack(pady=10)
        self.angle_entry.insert(0, "0")
        self.angle_entry.bind("<Return>", self.set_angle_from_entry)

    def plot_histograms(self, parent_frame):
        # Data for histogram
        x = np.random.randn(1000)
        y = 3 + np.random.randn(500)
        z = np.random.poisson(5, 1000)

        hist_frame = ttk.Frame(parent_frame)
        hist_frame.grid(row=0, column=0, sticky=tk.NSEW)

        # Create a figure and axis for the histogram
        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        fig.tight_layout(pad=2.5)

        # Embed the figure into the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=hist_frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
        canvas.draw()

        # Add labels and title
        fig.suptitle('Distributions of monolayer statistics', fontsize=18)
        plt.subplots_adjust(top=0.9)
        plt.tick_params(axis='both', which='major', labelsize=12)
        x_labels = ['Area (um^2)', 'Entropy', 'TV Norm', 'Local Intensity Variation', 'CNR', 'Skewness']
        data = [[x], [y], [z], [x], [y], [z]]
        for i, ax in enumerate(axes.flat):
            # Plot the histogram for data in x - change to Area (1), Entropy (2), TV Norm (3), Local Internsity Variation (4), CNR (5), Skewness (6)
            ax.hist(data[i], bins=30, color='blue', edgecolor='black')
            ax.set_xlabel(x_labels[i])

        # Create a toolbar for interactivity (zoom, pan, save, etc.)
        toolbar_frame = ttk.Frame(parent_frame)  # Create a separate frame for the toolbar
        toolbar_frame.grid(row=1, column=0, sticky=tk.W)  # Align toolbar to the left
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def draw_anti_aliased_wheel(self):
        # Draw the circle for the wheel with anti-aliasing
        self.draw.ellipse(
            [self.center_x - self.radius, self.center_y - self.radius,
             self.center_x + self.radius, self.center_y + self.radius],
            outline="black", width=2
        )

    '''
    Updates the pointer on the wheel and calls the move_and_wait function to 
    '''
    def update_wheel(self, event):
        # Calculate angle from center to current mouse position
        dx = event.x - self.center_x
        dy = event.y - self.center_y
        angle = math.degrees(math.atan2(dy, dx)) % 360

        # Update the pointer line position
        end_x = self.center_x + self.radius * math.cos(math.radians(angle))
        end_y = self.center_y + self.radius * math.sin(math.radians(angle))
        self.wheel_canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)

        # Update the dummy variable 2
        self.dummy_var2.set(angle)
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, f"{angle:.2f}")
        print(f"Dummy Variable 2 updated to: {angle:.2f} degrees")
    
    def set_angle_from_entry(self, event):
        # Get the angle from the entry box
        try:
            angle = float(self.angle_entry.get()) % 360
        except ValueError:
            angle = 0

        # Update the pointer line based on the entered angle
        end_x = self.center_x + self.radius * math.cos(math.radians(angle))
        end_y = self.center_y + self.radius * math.sin(math.radians(angle))
        self.wheel_canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)

        # Update the dummy variable 2
        self.dummy_var2.set(angle)
        print(f"Dummy Variable 2 set to: {angle:.2f} degrees")
    
    def update_var1(self, value):
        print(f"Dummy Variable 1 updated to: {value}")

    def update_var2(self, value):
        print(f"Dummy Variable 2 updated to: {value}")
    
    '''
    Random number generator for die.
    '''
    def randNum(self):
        self.lbl_roll['text'] = random.randint(1,6)

    # Create a function to get the entry value
    def get_entry_value(self, event=None):
        entered_text = self.entry.get()
        print(f"You entered: {entered_text}")

    def show_cat_image(self):
        # Load the cat image
        cat_image = Image.open("Images/test_image1.jpg")  # Replace with your cat image path
        cat_image = cat_image.resize((100, 100))  # Resize the image if needed
        cat_photo = ImageTk.PhotoImage(cat_image)

        self.cat_canvas = tk.Canvas(self.tab3, width=100, height=100)
        self.cat_canvas.grid(row=4, column=0, pady=20)

        self.cat_button.destroy()

        self.cat_canvas.delete("all")
        self.cat_canvas.create_image(0, 0, anchor=tk.NW, image=cat_photo)
        self.cat_canvas.image = cat_photo  # Keep a reference to avoid garbage collection

    def pick_color(self, event):
        # Open the color picker dialog
        color_code = colorchooser.askcolor(initialcolor=self.color_code, title="Choose a color")

        # If a color is selected, update the color_code and change the background of the label
        if color_code[1]:  # Check if a valid color was selected
            self.color_code = color_code[1]  # Update the color_code with the selected color
            self.color_label.config(bg=self.color_code)




# Start the Tkinter event loop
if __name__ == "__main__":
    image_path = 'Images/highresphoto.jpg'  # place path to your image here
    root = tk.Tk()
    poo = ImageDisplay(root)
    # app = Zoom_Advanced(poo.zoom_frame, path=image_path)
    app = MainWindow(poo.zoom_frame, path=image_path)
    root.mainloop()