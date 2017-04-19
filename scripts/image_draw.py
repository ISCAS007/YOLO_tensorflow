# -*- coding: utf-8 -*-

import cv2
import numpy as np

# ============================================================================



# ============================================================================

class PolygonDrawer(object):
    def __init__(self, file_name):
        self.window_name = file_name # Name for our window

        self.done = False # Flag signalling we're done
        self.start = False # Flag start to draw

        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.area = [] # unregular area

        self.CANVAS_SIZE = (600,800)

        self.FINAL_LINE_COLOR = (255, 255, 255)
        self.WORKING_LINE_COLOR = (127, 127, 127)
        self.LINE_SIZE=3

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            if self.start:
                self.current = (x, y)
                print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
                self.points.append((x, y))
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.start=True
        elif event == cv2.EVENT_LBUTTONUP:
            self.start=False
            if self.points.__len__() > 2:
                self.area.append(self.points)
                self.points=[]

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.CV_WINDOW_AUTOSIZE)
        # picture only
        img = cv2.imread(self.window_name)
        height, width, channels = img.shape
        self.CANVAS_SIZE = (height,width,channels)
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)
        cv2.cv.SetMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            # canvas = np.zeros(self.CANVAS_SIZE, np.uint8)
            canvas = img.copy()
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, self.FINAL_LINE_COLOR, self.LINE_SIZE)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, self.WORKING_LINE_COLOR)

            for area in self.area:
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([area]), False, self.FINAL_LINE_COLOR, self.LINE_SIZE)
                # And  also show what the current segment would look like
                cv2.line(canvas, area[-1], area[0], self.WORKING_LINE_COLOR)

            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) & 0xFF == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        # of a filled polygon
        canvas = img.copy()
        for area in self.area:
            cv2.fillPoly(canvas, np.array([area]), self.FINAL_LINE_COLOR)

        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return canvas

# ============================================================================

if __name__ == "__main__":
    pd = PolygonDrawer("/home/yzbx/Pictures/car.png")
    image = pd.run()
    cv2.imwrite("polygon.png", image)
    print("Polygon = %s" % pd.points)