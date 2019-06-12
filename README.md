# CAM16.py

CAM16.py is a Python implementation of the CIECAM16 color model.
It can translate from, and to, sRGB and XYZ color spaces.

## Requirements
- Python 3.7.x
- Numpy
- Pandas

## Todo
- Add documentation, and explain how to use the class and methods.

## Ideas
- Add HSV, HSL, CIELab transforms.
- Add multiple ways to create a CAM16 color (Jab, JCh, QMh, etc).
- Add color interpolation.

## Code notes
- Underscore-prefixed methods are internal, and use Numpy arrays as input and output.
- Non-prefixed methods use data types as input, and return data types or tuples.

## Science notes
- _J_ can be calculated with _Q_, and vice versa.
Also, _C_ can be calculated with _M_, and vice versa, too.
Therefore, only one chromatic and achromatic parameter are needed.
- _s_ can replace any chromatic or achromatic parameter.
- _h_ can be calculated with _H_, ad vice versa.
- _M_ can be calculated with _a_ and _b_.
