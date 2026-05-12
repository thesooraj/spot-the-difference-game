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
    

#  IMAGE PROCESSOR
class ImageProcessor:
    """
    Loads an image, clones it, and programmatically introduces
    exactly 5 non-overlapping random differences using OpenCV.
    """

    NUM_DIFFERENCES = 5
    MIN_REGION      = 40
    MAX_REGION      = 80
    MAX_DIM         = 500

    def __init__(self):
        self.original_image  = None
        self.modified_image  = None
        self.alterations     = []
        self._alteration_types = [
            ColourShiftAlteration,
            BrightnessAlteration,
            BlurAlteration,
            ColourPatchAlteration,
        ]    

def load_image(self, filepath: str):
        """Load & resize image, clone it, apply 5 differences."""
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Cannot open image: {filepath}")
        img = self._resize(img)
        self.original_image = img.copy()
        self.modified_image = img.copy()
        self.alterations    = []
        self._generate_differences()

    def cv2_to_pil(self, cv2_img) -> Image.Image:
        """Convert BGR cv2 array to RGB PIL Image."""
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    def draw_circle(self, image, center: tuple, colour: tuple):
        """Return a copy of image with a circle drawn at center."""
        out = image.copy()
        cv2.circle(out, center, 35, colour, 3)
        return out

    def _resize(self, img):
        h, w = img.shape[:2]
        if w > self.MAX_DIM or h > self.MAX_DIM:
            scale = min(self.MAX_DIM / w, self.MAX_DIM / h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img    

    def _random_region(self, img_h, img_w, existing):
        """Try up to 300 times to find a non-overlapping region."""
        for _ in range(300):
            rw = random.randint(self.MIN_REGION, self.MAX_REGION)
            rh = random.randint(self.MIN_REGION, self.MAX_REGION)
            x  = random.randint(10, img_w - rw - 10)
            y  = random.randint(10, img_h - rh - 10)
            if not any(
                not (x + rw < ex or x > ex + ew or y + rh < ey or y > ey + eh)
                for (ex, ey, ew, eh) in existing
            ):
                return x, y, rw, rh
        return None

    def _generate_differences(self):
        h, w  = self.original_image.shape[:2]
        used  = []
        # Ensure all 4 types are represented (one type appears twice)
        types = self._alteration_types.copy()
        types.append(random.choice(self._alteration_types))
        random.shuffle(types)

        for i in range(self.NUM_DIFFERENCES):
            region = self._random_region(h, w, used)
            if region is None:
                continue
            x, y, rw, rh = region
            used.append(region)
            alt = types[i](x, y, rw, rh)
            alt.apply(self.modified_image)
            self.alterations.append(alt)    