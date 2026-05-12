"""
HIT137 - Group Assignment 3
Spot the Difference Game
Authors: Sooraj, Fariha, Daniyal, Hassan
"""

import cv2
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTks

#  BASE ALTERATION CLASS  (Abstract Parent)

class BaseAlteration:
    """
    Abstract base class for all image alterations.
    Demonstrates: encapsulation, constructor, inheritance base.
    """

    def __init__(self, x, y, width, height):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self.found = False
        self.name = "Base"

    def apply(self, image):
        """Apply the alteration to the image. Must be overridden."""
        raise NotImplementedError("Subclasses must implement apply()")

    def get_region(self):
        """Return (x, y, width, height) of this alteration."""
        return (self._x, self._y, self._width, self._height)

    def get_center(self):
        """Return (cx, cy) centre pixel of this region."""
        return (self._x + self._width // 2, self._y + self._height // 2)

    def contains_point(self, px, py, tolerance=30):
        """Return True if (px, py) is within the region (+tolerance pixels)."""
        cx, cy = self.get_center()
        return (abs(px - cx) <= self._width  // 2 + tolerance and
                abs(py - cy) <= self._height // 2 + tolerance)

    def __repr__(self):
        return f"{self.name}(x={self._x}, y={self._y}, w={self._width}, h={self._height})"

#  COLOUR SHIFT ALTERATION
class ColourShiftAlteration(BaseAlteration):
    """Shifts the hue of a rectangular region in HSV colour space."""

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.name = "Colour Shift"
        self._shift = random.randint(40, 90)

    def apply(self, image):
        region = image[self._y:self._y+self._height, self._x:self._x+self._width]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self._shift) % 180
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        image[self._y:self._y+self._height, self._x:self._x+self._width] = (
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        )
        return image  


#  BRIGHTNESS ALTERATION
class BrightnessAlteration(BaseAlteration):
    """Darkens or brightens a rectangular region."""

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.name = "Brightness"
        self._factor = random.choice([-55, -45, 45, 55])

    def apply(self, image):
        region = image[self._y:self._y+self._height,
                       self._x:self._x+self._width].astype(np.int32)
        region = np.clip(region + self._factor, 0, 255).astype(np.uint8)
        image[self._y:self._y+self._height, self._x:self._x+self._width] = region
        return image   


#  BLUR ALTERATION
class BlurAlteration(BaseAlteration):
    """Applies Gaussian blur to a rectangular region."""

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.name = "Blur"
        self._kernel = random.choice([13, 17, 21])

    def apply(self, image):
        region = image[self._y:self._y+self._height, self._x:self._x+self._width]
        blurred = cv2.GaussianBlur(region, (self._kernel, self._kernel), 0)
        image[self._y:self._y+self._height, self._x:self._x+self._width] = blurred
        return image        
    

#  COLOUR PATCH ALTERATION
class ColourPatchAlteration(BaseAlteration):
    """Overlays a semi-transparent colour patch on a region."""

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.name = "Colour Patch"
        self._colour = (
            random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255),
        )

    def apply(self, image):
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (self._x, self._y),
            (self._x + self._width, self._y + self._height),
            self._colour, -1
        )
        cv2.addWeighted(overlay, 0.30, image, 0.70, 0, image)
        return image    