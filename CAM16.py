from dataclasses import dataclass
import numpy as np
import pandas as pd


CAT16_MATRIX = np.array([
    [0.401288, 0.650173, -0.051461],
    [-0.250268, 1.204414, 0.045854],
    [-0.002079, 0.048952, 0.953127]
])

CAT16_INVERSE = np.array([
    [1.8620679, -1.011255, 0.149187],
    [0.387526, 0.621447, -0.008974],
    [-0.015842, -0.034123, 1.049964]
])

SRGB_TO_XYZ_MATRIX = np.array([
    [0.412456, 0.357576, 0.180438],
    [0.212673, 0.715152, 0.072175],
    [0.019334, 0.119192, 0.950304]
])

XYZ_TO_SRGB_MATRIX = np.array([
    [3.240454, -1.537138, -0.498531],
    [-0.969266, 1.876011, 0.041556],
    [0.055643, -0.204026, 1.057225]
])

AB_TO_RGB_MATRIX = np.array([
    [460, 451, 228],
    [460, -891, -261],
    [460, -220, -6300]
])

HUE_DATA = pd.DataFrame(np.array([
    [20.14, 0.8, 0, 'Red'],
    [90.00, 0.7, 100, 'Yellow'],
    [164.25, 1.0, 200, 'Green'],
    [237.53, 1.2, 300, 'Blue'],
    [380.14, 0.8, 400, 'Red']
]), columns=['h', 'e', 'H', 'hue'])
HUE_DATA.h = pd.to_numeric(HUE_DATA.h)
HUE_DATA.e = pd.to_numeric(HUE_DATA.e)
HUE_DATA.H = pd.to_numeric(HUE_DATA.H)

SURROUND_PARAMETERS = pd.DataFrame(np.array([
    [1.0, 0.690, 1.0],
    [0.9, 0.590, 0.9],
    [0.8, 0.525, 0.8]
]), index=['average', 'dim', 'dark'], columns=['F', 'c', 'Nc'])


ILLUMINANT_D65 = pd.DataFrame(np.array([
    [95.047, 100.0, 108.883],
    [94.811, 100.0, 107.304]
]), index=['2°', '10°'])


@dataclass
class VC:

    # Using Illuminant D65/2°
    # XYZ tristimulus values, normalizing for relative luminance
    # Source: Illuminant D65, Wikipedia.
    XYZw = ILLUMINANT_D65.loc['2°']

    # Using Average surround
    # Source: Table A1, Comprehensive color solutions. DOI: 10.1002/col.22131
    S = SURROUND_PARAMETERS.loc['average']

    # Using a sRGB luminance of 64 lux, and surround reflectance of 20%
    # Source: sRGB, Wikipedia.
    Ew = 64
    Lw = Ew/np.pi
    Yb = 20
    LA = (Lw * Yb)/XYZw[1]

    RGBw = SRGB_TO_XYZ_MATRIX @ XYZw

    D = S.F * (1 - (1/3.6) * np.exp((-LA - 42)/92))
    D = np.clip(D, 0, 1)

    DRGB = D * XYZw[1]/RGBw + 1 - D

    k = 1/(5*LA + 1)

    FL = 0.2*k**4 * 5*LA + 0.1*(1 - k**4)**2 * (5*LA)**(1/3)

    n = Yb/XYZw[1]

    z = 1.48 + n**(1/2)

    Nbb = 0.725 * (1/n)**(0.2)
    Ncb = Nbb

    RGBwc = DRGB * RGBw

    RGBaw = 400 * ((FL*RGBwc/100)**0.42)/((FL*RGBwc/100)**0.42 + 27.13) + 0.1

    Aw = (np.array([2, 1, 1/20]) @ RGBaw - 0.305) * Nbb


# mode='hex'|'hexadecimal', mode='dec'|'decimal', mode='frac'|'fractional'
def CAM16_to_sRGB(J, C, h):

    t = (C / ((J/100)**(1/2) * (1.64 - 0.29**VC.n)**0.73))**(1/0.9)
    e = (1/4) * (np.cos(h*(np.pi/180) + 2) + 3.8)
    A = VC.Aw * (J/100)**(1/(VC.S.c * VC.z))

    p1 = ((50000/13) * VC.S.Nc * VC.Ncb) * e * (1/t)
    p2 = A/VC.Nbb + 0.305
    p3 = 21/20

    h = np.radians(h)

    if t == 0:
        a = 0
        b = 0
    elif np.abs(np.sin(h)) >= np.abs(np.cos(h)):
        p4 = p1/np.sin(h)

        b = (p2 * (2+p3) * (460/1403))/(p4 + (2+p3) * (220/1403) * (1/np.tan(h)) - (27/1403) + p3 * (6300/1403))
        a = b / np.tan(h)
    else:
        p5 = p1/np.cos(h)

        a = (p2 * (2+p3) * (460/1403))/(p5 + (2+p3) * (220/1403) - ((27/1403) - p3 * (6300/1403)) * np.tan(h))
        b = a * np.tan(h)

    RGBa = (AB_TO_RGB_MATRIX @ np.array([p2, a, b]))/1403
    RGBc = np.sign(RGBa - 0.1) * (100/VC.FL) * ((27.13 * np.abs(RGBa-0.1))/(400 - np.abs(RGBa-0.1)))**(1/0.42)
    RGB = RGBc / VC.DRGB

    XYZ = CAT16_INVERSE @ RGB

    sRGB = XYZ_TO_SRGB_MATRIX @ (XYZ/100)

    # def companding(v):
    #     if v <= 0.0031308:
    #         V = 12.92*v
    #     else:
    #         V = 1.055 * v**(1/2.4) - 0.055
    #     return V

    # sRGB = np.vectorize(companding)(sRGB)
    # # sRGB = np.clip(sRGB, 0, 1)

    sRGB = np.where(
        sRGB <= 0.0031308,
        12.92*sRGB,
        1.055*sRGB**(1/2.4) - 0.055
    )

    return np.round(255*sRGB).astype(int)


def sRGB_to_CAM16(R, G, B):

    # Companded sRGB
    sRGB = np.array([R, G, B])

    # Decompanded sRGB
    sRGB = np.where(
        sRGB <= 0.04045,
        sRGB/12.92,
        ((sRGB + 0.055)/1.055)**2.4
    )

    # Tristimulus values
    XYZ = 100*(SRGB_TO_XYZ_MATRIX @ sRGB)

    # Cone response
    RGB = CAT16_MATRIX @ XYZ
    RGB = VC.DRGB * RGB
    RGB = np.where(
        RGB < 0,
        400 * (VC.FL*RGB/100)**0.42/((VC.FL*RGB/100)**0.42 + 27.13) + 0.1,
        -400 * (-VC.FL*RGB/100)**0.42/((-VC.FL*RGB/100)**0.42 + 27.13) + 0.1
    )

    # Red-green and yellow-blue components
    a = np.array([1, -12/11, 1/11]) @ RGB
    b = np.array([1/9, 1/9, -2/9]) @ RGB

    # Hue angle
    h = (180/np.pi)*np.arctan2(b, a)
    if h > 360:
        h -= 360
    elif h < 0:
        h += 360

    hp = h + 360 if h < HUE_DATA.at[0, 'h'] else h

    # Eccentricity
    e = (1/4)*(np.cos(hp*np.pi/180 + 2) + 3.8)

    # Hue quadrature composition
    if HUE_DATA.at[0, 'h'] < h < HUE_DATA.at[1, 'h']:
        i = 1
    elif HUE_DATA.at[1, 'h'] < h < HUE_DATA.at[2, 'h']:
        i = 2
    elif HUE_DATA.at[2, 'h'] < h < HUE_DATA.at[3, 'h']:
        i = 3
    elif HUE_DATA.at[3, 'h'] < h < HUE_DATA.at[4, 'h']:
        i = 4
    pr = (h - HUE_DATA.at[i-1, 'h'])/HUE_DATA.at[i-1, 'e']
    pl = (HUE_DATA.at[i, 'h'] - h)/HUE_DATA.at[i, 'e']
    H = HUE_DATA.at[i-1, 'H'] + (100*pr)/(pr + pl)

    # Hue composition
    if HUE_DATA.at[0, 'H'] < H < HUE_DATA.at[1, 'H']:
            i = 1
    elif HUE_DATA.at[1, 'H'] < H < HUE_DATA.at[2, 'H']:
        i = 2
    elif HUE_DATA.at[2, 'H'] < H < HUE_DATA.at[3, 'H']:
        i = 3
    elif HUE_DATA.at[3, 'H'] < H < HUE_DATA.at[4, 'H']:
        i = 4
    PL = HUE_DATA.at[i, 'H'] - H
    PR = H - HUE_DATA.at[i-1, 'H']
    Hc = {HUE_DATA.at[i-1, 'hue']: PL, HUE_DATA.at[i, 'hue']: PR}

    # Achromatic response
    A = (np.array([2, 1, 1/20]) @ RGB - 0.305)*VC.Nbb

    # Lightness
    J = 100*(A/VC.Aw)**(VC.S.c*VC.z)

    # Brightness
    Q = (4/VC.S.c) * (J/100)**0.5 * (VC.Aw + 4) * VC.FL**0.25

    # Chroma
    tl = (50000/13) * VC.S.Nc * VC.Ncb * e * (a**2 + b**2)**(1/2)
    tr = np.array([1, 1, 21/20]) @ RGB
    C = (tl/tr)**0.9 * (J/100)**0.5 * (1.64 - 0.29**VC.n)**0.73

    # Colorfulness
    M = C * VC.FL**0.25

    # Saturation
    s = 100 * (M/Q)**0.5

    return (J, C, h)
