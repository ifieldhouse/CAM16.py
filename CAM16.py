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

ACES2065_TO_XYZ_MATRIX = np.array([
    [0.95255_23959, 0.00000_00000, 0.00009_36786],
    [0.34396_64498, 0.72816_60966, -0.07213_25464],
    [0.00000_00000, 0.00000_00000, 1.00882_51844]
])

XYZ_TO_ACES2065_MATRIX = np.array([
    [1.04981_10175, 0.00000_00000, -0.00009_74845],
    [-0.49590_30231, 1.37331_30458, 0.09824_00361],
    [0.00000_00000, 0.00000_00000, 0.99125_20182]
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


@dataclass
class Illuminant:

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    @property
    def X(self):
        return self.Y * self.x/self.y
    
    @property
    def Y(self):
        return 100

    @property    
    def Z(self):
        return self.Y * (1-self.x-self.y)/self.y


@dataclass
class VC:
    # Using Illuminant D65/2°.
    # XYZ tristimulus values, normalizing for relative luminance.
    # Source: Illuminant D65, Wikipedia.
    D65 = Illuminant(x=0.31270, y=0.32900)
    XYZ_w = np.array([D65.X, D65.Y, D65.Z])

    # Using dim surround (most likely you will use a screen display).
    # Source: Table A1, Comprehensive color solutions. DOI: 10.1002/col.22131.
    S = SURROUND_PARAMETERS.loc['dim']

    # Using a sRGB luminance of 64 lux, and surround reflectance of 20%.
    # Source: sRGB, Wikipedia.
    # Illuminance of the reference white
    E_w = 64
    # Absolute luminance of the reference white
    L_w = E_w/np.pi
    # Relative luminance of the adapting field
    Y_b = 20
    # Absolute luminance of the adapting field
    L_A = (L_w * Y_b)/XYZ_w[1]

    # Cone response
    RGB_w = CAT16_MATRIX @ XYZ_w

    # Degree of adaptation
    D = S.F * (1 - (1/3.6) * np.exp((-L_A - 42)/92))
    D = np.clip(D, 0, 1)
    D_RGB = D * XYZ_w[1]/RGB_w + 1 - D

    # Factor of luminance level adaptation
    k = 1/(5*L_A + 1)
    F_L = 0.2*k**4 * 5*L_A + 0.1*(1 - k**4)**2 * (5*L_A)**(1/3)

    n = Y_b/XYZ_w[1]
    z = 1.48 + n**(1/2)
    N_bb = 0.725 * (1/n)**(0.2)
    N_cb = N_bb

    RGB_wc = D_RGB * RGB_w
    RGB_aw = 400 * ((F_L*RGB_wc/100)**0.42)/((F_L*RGB_wc/100)**0.42 + 27.13) + 0.1

    # Achromatic response
    A_w = (np.array([2, 1, 1/20]) @ RGB_aw - 0.305) * N_bb


class CAM16:

    def __init__(self, J, C, h):
        self.J = J
        self.C = C
        self.h = h

        # set_M(), set_s(), set_Q(), set_H(), set_a(), set_b()

    @property
    def H(self):
        H1 = HUE_DATA[HUE_DATA['h'] < self.h].iloc[-1]
        H2 = HUE_DATA[HUE_DATA['h'] >= self.h].iloc[0]

        p1 = (self.h - H1.h) / H1.e
        p2 = (H2.h - self.h) / H2.e

        H = H1.H + (100*p1)/(p1 + p2)

        p3 = H2.H - H
        p4 = H - H1.H
        
        return {H1.hue: p3, H2.hue: p4}

    @property
    def Q(self):
        return (4/VC.S.c) * (self.J/100)**0.5 * (VC.A_w + 4) * VC.F_L**0.25
    
    @property
    def M(self):
        return self.C * VC.F_L**0.25
    
    @property
    def s(self):
        return 100 * (self.M/self.Q)**0.5

    @classmethod
    def from_CAM16UCS(cls, Jp, Cp, hp):
        J = Jp/(1.7-0.007*Jp)

        Mp = Cp*VC.F_L**0.25
        M = (np.exp(0.0228*Mp) - 1)/0.0228
        C = M/VC.F_L**0.25

        h = hp

        return cls(J, C, h)

    def as_CAM16UCS(self):
        Jp = 1.7*self.J/(1+0.007*self.J)

        Mp = np.log(1+0.0228*self.M)/0.0228

        Cp = Mp/VC.F_L**0.25
        
        hp = self.h

        return (Jp, Cp, hp)

    @classmethod
    def from_XYZ(cls, X, Y, Z):
        XYZ = np.array([X, Y, Z])

        return cls._from_XYZ(XYZ)

    @classmethod
    def _from_XYZ(cls, XYZ):
        # Cone response
        RGB = CAT16_MATRIX @ XYZ
        RGB = VC.D_RGB * RGB
        RGB = np.where(
            RGB < 0,
            400 * (VC.F_L*RGB/100)**0.42/((VC.F_L*RGB/100)**0.42 + 27.13) + 0.1,
            -400 * (-VC.F_L*RGB/100)**0.42/((-VC.F_L*RGB/100)**0.42 + 27.13) + 0.1
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
        e = cls._eccentricity(hp)

        # Achromatic response
        A = (np.array([2, 1, 1/20]) @ RGB - 0.305)*VC.N_bb

        # Lightness
        J = 100*(A/VC.A_w)**(VC.S.c*VC.z)

        # Chroma
        p1 = (50000/13) * VC.S.Nc * VC.N_cb * e * (a**2 + b**2)**(1/2)
        p2 = np.array([1, 1, 21/20]) @ RGB

        C = (p1/p2)**0.9 * (J/100)**0.5 * (1.64 - 0.29**VC.n)**0.73

        return cls(J, C, h)

    def as_XYZ(self):
        X, Y, Z = self._as_XYZ()

        return X, Y, Z

    def _as_XYZ(self):
        t = (self.C / ((self.J/100)**(1/2) * (1.64 - 0.29**VC.n)**0.73))**(1/0.9)
        e = (1/4) * (np.cos(self.h*(np.pi/180) + 2) + 3.8)
        A = VC.A_w * (self.J/100)**(1/(VC.S.c * VC.z))

        p1 = ((50000/13) * VC.S.Nc * VC.N_cb) * e * (1/t)
        p2 = A/VC.N_bb + 0.305
        p3 = 21/20

        h = np.radians(self.h)

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

        RGB_a = (AB_TO_RGB_MATRIX @ np.array([p2, a, b]))/1403
        RGB_c = np.sign(RGB_a - 0.1) * (100/VC.F_L) * ((27.13 * np.abs(RGB_a-0.1))/(400 - np.abs(RGB_a-0.1)))**(1/0.42)
        RGB = RGB_c / VC.D_RGB

        XYZ = CAT16_INVERSE @ RGB

        return XYZ

    @classmethod
    def from_sRGB(cls, R, G, B):
        sRGB = np.array([R, G, B])

        return cls._from_sRGB(sRGB)
    
    @classmethod
    def _from_sRGB(cls, sRGB):
        # Decompanded sRGB
        sRGB = np.where(
            sRGB <= 0.04045,
            sRGB/12.92,
            ((sRGB + 0.055)/1.055)**2.4
        )

        # Tristimulus values
        XYZ = 100*(SRGB_TO_XYZ_MATRIX @ sRGB)

        return cls._from_XYZ(XYZ)

    def as_sRGB(self, mode='decimal'):
        """
        Mode:
        - dec | decimal
        - hex | hexadecimal
        - fractional
        - percent
        """

        sRGB = self._as_sRGB()

        # Check if the value is inside [0, 1]

        if mode in ('dec', 'decimal'):
            R, G, B = (255*sRGB).astype(int)
            return R, G, B
        elif mode in ('hex', 'hexadecimal'):
            R, G, B = (255*sRGB).astype(int)
            return f'#{R:02x}{G:02x}{B:02x}'
        elif mode == 'fractional':
            R, G, B = np.round(sRGB, decimals=3)
            return R, G, B
        elif mode == 'percent':
            R, G, B = (100*sRGB).astype(int)
            return R, G, B

    def _as_sRGB(self):
        # Tristimulus values
        XYZ = self._as_XYZ()

        # Decompanded sRGB
        sRGB = XYZ_TO_SRGB_MATRIX @ (XYZ/100)

        # Companded sRGB
        sRGB = np.where(
            sRGB <= 0.0031308,
            12.92*sRGB,
            1.055*sRGB**(1/2.4) - 0.055
        )

        return sRGB

    @classmethod
    def from_ACES2065(cls, R, G, B):
        ACES = np.array([R, G, B])

        return cls._from_ACES2065(ACES)
    
    @classmethod
    def _from_ACES2065(cls, ACES):
        # Tristimulus values
        XYZ = 100*(ACES2065_TO_XYZ_MATRIX @ ACES)

        return cls._from_XYZ(XYZ)

    def as_ACES2065(self):

        ACES = self._as_ACES2065()

        R, G, B = ACES

        return R, G, B

    def _as_ACES2065(self):
        # Tristimulus values
        XYZ = self._as_XYZ()

        ACES = XYZ_TO_ACES2065_MATRIX @ (XYZ/100)

        return ACES

    @staticmethod
    def _eccentricity(h):
        return (1/4)*(np.cos(h*np.pi/180 + 2) + 3.8)
