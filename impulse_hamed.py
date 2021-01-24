from __future__ import print_function
# we need to clean up code
# Copyright (c) 2017, Simon Brodeur
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the NECOTIS research group nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.

import os
import logging
import scipy.signal
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d

#import math (for math.pi)
import math as m

from scipy import signal

from evert import Room, Polygon, Vector3, Source, Listener, PathSolution, Viewer

from gn_range import pos_range

logger = logging.getLogger(__name__)

CDIR = os.path.dirname(os.path.realpath(__file__))

class MaterialAbsorptionTable(object):
    
    # From: Auralization : fundamentals of acoustics, modelling, simulation, algorithms and acoustic virtual reality
    
    categories = ['hard surfaces', 'linings', 'glazing', 'wood', 
                  'floor coverings', 'curtains']
    frequencies = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    
    materials = [[  # Massive constructions and hard surfaces 
                    "average",                                  # Walls, hard surfaces average (brick walls, plaster, hard floors, etc.) 
                    "walls rendered brickwork",                 # Walls, rendered brickwork
                    "rough concrete",                           # Rough concrete
                    "smooth unpainted concrete",                # Smooth unpainted concrete 
                    "rough lime wash",                          # Rough lime wash 
                    "smooth brickwork with flush pointing, painted", # Smooth brickwork with flush pointing, painted 
                    "smooth brickwork, 10 mm deep pointing, pit sand mortar", # Smooth brickwork, 10 mm deep pointing, pit sand mortar 
                    "brick wall, stuccoed with a rough finish", # Brick wall, stuccoed with a rough finish 
                    "ceramic tiles with a smooth surface",      # Ceramic tiles with a smooth surface 
                    "limestone walls",                          # Limestone walls 
                    "reverberation chamber walls",              # Reverberation chamber walls 
                    "concrete",                                 # Concrete floor 
                    "marble floor",                             # Marble floor 
                ],
                [   # Lightweight constructions and linings
                    "plasterboard on steel frame",              # 2 * 13 mm plasterboard on steel frame, 50 mm mineral wool in cavity, surface painted 
                    "wooden lining",                            # Wooden lining, 12 mm fixed on frame 
                ],
                [   # Glazing
                    "single pane of glass",                     # Single pane of glass, 3 mm                                                  
                    "glass window",                             # Glass window, 0.68 kg/m^2
                    "lead glazing",                             # Lead glazing
                    "double glazing, 30 mm gap",                # Double glazing, 2-3 mm glass,  > 30 mm gap 
                    "double glazing, 10 mm gap ",               # Double glazing, 2-3 mm glass,  10 mm gap 
                    "double glazing, lead on the inside",       # Double glazing, lead on the inside
                ],
                [   # Wood
                    "wood, 1.6 cm thick",                       # Wood, 1.6 cm thick,  on 4 cm wooden planks 
                    "thin plywood panelling",                   # Thin plywood panelling
                    "16 mm wood on 40 mm studs",                # 16 mm wood on 40 mm studs 
                    "audience floor",                           # Audience floor, 2 layers,  33 mm on sleepers over concrete 
                    "stage floor",                              # Wood, stage floor, 2 layers, 27 mm over airspace 
                    "solid wooden door",                        # Solid wooden door 
                ],
                [   # Floor coverings
                    "linoleum, asphalt, rubber, or cork tile on concrete", # Linoleum, asphalt, rubber, or cork tile on concrete 
                    "cotton carpet",                            # Cotton carpet 
                    "loop pile tufted carpet",                  # Loop pile tufted carpet, 1.4 kg/m^2, 9.5 mm pile height: On hair pad, 3.0kg/m^2
                    "thin carpet",                              # Thin carpet, cemented to concrete
                    "pile carpet bonded to closed-cell foam underlay", # 6 mm pile carpet bonded to closed-cell foam underlay 
                    "pile carpet bonded to open-cell foam underlay", # 6 mm pile carpet bonded to open-cell foam underlay 
                    "tufted pile carpet",                       # 9 mm tufted pile carpet on felt underlay
                    "needle felt",                              # Needle felt 5 mm stuck to concrete 
                    "soft carpet",                              # 10 mm soft carpet on concrete
                    "hairy carpet",                             # Hairy carpet on 3 mm felt 
                    "rubber carpet",                            # 5 mm rubber carpet on concrete 
                    "carpet on hair felt or foam rubber",       # Carpet 1.35 kg/m^2, on hair felt or foam rubber 
                    "cocos fibre roll felt",                    # Cocos fibre roll felt, 29 mm thick (unstressed), reverse side clad  with paper, 2.2kg/m^2, 2 Rayl 
                ],
                [   # Curtains
                    "cotton curtains",                          # Cotton curtains (0.5 kg/m^2) draped to 3/4 area approx. 130 mm from wall
                    "curtains",                                 # Curtains (0.2 kg/m^2) hung 90 mm from wall 
                    "cotton cloth",                             # Cotton cloth (0.33 kg/m^2) folded to 7/8 area 
                    "densely woven window curtains",            # Densely woven window curtains 90 mm from wall 
                    "vertical blinds, half opened",             # Vertical blinds, 15 cm from wall,   half opened (45 deg) 
                    "vertical blinds, open",                    # Vertical blinds, 15 cm from wall,   open (90 deg) 
                    "tight velvet curtains",                    # Tight velvet curtains 
                    "curtain fabric",                           # Curtain fabric, 15 cm from wall 
                    "curtain fabric, folded",                   # Curtain fabric, folded, 15 cm from wall
                    "curtain of close-woven glass mat",         # Curtains of close-woven glass mat   hung 50 mm from wall 
                    "studio curtain",                           # Studio curtains, 22 cm from wall 
                ],
            ]
    
    # Tables of random-incidence absorption coefficients
    table = [   [   # Massive constructions and hard surfaces 
                    [0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05], # Walls, hard surfaces average (brick walls, plaster, hard floors, etc.) 
                    [0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04], # Walls, rendered brickwork
                    [0.02, 0.03, 0.03, 0.03, 0.04, 0.07, 0.07], # Rough concrete
                    [0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05], # Smooth unpainted concrete 
                    [0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.02], # Rough lime wash 
                    [0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02], # Smooth brickwork with flush pointing, painted 
                    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24, 0.24], # Smooth brickwork, 10 mm deep pointing, pit sand mortar 
                    [0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.07], # Brick wall, stuccoed with a rough finish 
                    [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02], # Ceramic tiles with a smooth surface 
                    [0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05], # Limestone walls 
                    [0.01, 0.01, 0.01, 0.02, 0.02, 0.04, 0.04], # Reverberation chamber walls 
                    [0.01, 0.03, 0.05, 0.02, 0.02, 0.02, 0.02], # Concrete floor 
                    [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02], # Marble floor 
                ],
                [   # Lightweight constructions and linings
                    [0.15, 0.10, 0.06, 0.04, 0.04, 0.05, 0.05], # 2 * 13 mm plasterboard on steel frame, 50 mm mineral wool in cavity, surface painted 
                    [0.27, 0.23, 0.22, 0.15, 0.10, 0.07, 0.06], # Wooden lining, 12 mm fixed on frame 
                ],
                [   # Glazing
                    [0.08, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02], # Single pane of glass, 3 mm                                                  
                    [0.10, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03], # Glass window,, 0.68 kg/m^2
                    [0.30, 0.20, 0.14, 0.10, 0.05, 0.05, 0.05], # Lead glazing
                    [0.15, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02], # Double glazing, 2-3 mm glass,  > 30 mm gap 
                    [0.10, 0.07, 0.05, 0.03, 0.02, 0.02, 0.02], # Double glazing, 2-3 mm glass,  10 mm gap 
                    [0.15, 0.30, 0.18, 0.10, 0.05, 0.05, 0.05], # Double glazing, lead on the inside
                ],
                [   # Wood
                    [0.18, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07], # Wood, 1.6 cm thick,  on 4 cm wooden planks 
                    [0.42, 0.21, 0.10, 0.08, 0.06, 0.06, 0.06], # Thin plywood panelling
                    [0.18, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07], # 16 mm wood on 40 mm studs 
                    [0.09, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04], # Audience floor, 2 layers,  33 mm on sleepers over concrete 
                    [0.10, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06], # Wood, stage floor, 2 layers, 27 mm over airspace 
                    [0.14, 0.10, 0.06, 0.08, 0.10, 0.10, 0.10], # Solid wooden door 
                ],
                [   # Floor coverings
                    [0.02, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02], # Linoleum, asphalt, rubber, or cork tile on concrete 
                    [0.07, 0.31, 0.49, 0.81, 0.66, 0.54, 0.48], # Cotton carpet 
                    [0.10, 0.40, 0.62, 0.70, 0.63, 0.88, 0.88], # Loop pile tufted carpet, 1.4 kg/m^2, 9.5 mm pile height: On hair pad, 3.0kg/m^2
                    [0.02, 0.04, 0.08, 0.20, 0.35, 0.40, 0.40], # Thin carpet, cemented to concrete
                    [0.03, 0.09, 0.25, 0.31, 0.33, 0.44, 0.44], # 6 mm pile carpet bonded to closed-cell foam underlay 
                    [0.03, 0.09, 0.20, 0.54, 0.70, 0.72, 0.72], # 6 mm pile carpet bonded to open-cell foam underlay 
                    [0.08, 0.08, 0.30, 0.60, 0.75, 0.80, 0.80], # 9 mm tufted pile carpet on felt underlay
                    [0.02, 0.02, 0.05, 0.15, 0.30, 0.40, 0.40], # Needle felt 5 mm stuck to concrete 
                    [0.09, 0.08, 0.21, 0.26, 0.27, 0.37, 0.37], # 10 mm soft carpet on concrete
                    [0.11, 0.14, 0.37, 0.43, 0.27, 0.25, 0.25], # Hairy carpet on 3 mm felt 
                    [0.04, 0.04, 0.08, 0.12, 0.10, 0.10, 0.10], # 5 mm rubber carpet on concrete 
                    [0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.73], # Carpet 1.35 kg/m^2, on hair felt or foam rubber 
                    [0.10, 0.13, 0.22, 0.35, 0.47, 0.57, 0.57], # Cocos fibre roll felt, 29 mm thick (unstressed), reverse side clad  with paper, 2.2kg/m^2, 2 Rayl 
                ],
                [   # Curtains
                    [0.30, 0.45, 0.65, 0.56, 0.59, 0.71, 0.71], # Cotton curtains (0.5 kg/m^2) draped to 3/4 area approx. 130 mm from wall
                    [0.05, 0.06, 0.39, 0.63, 0.70, 0.73, 0.73], # Curtains (0.2 kg/m^2) hung 90 mm from wall 
                    [0.03, 0.12, 0.15, 0.27, 0.37, 0.42, 0.42], # Cotton cloth (0.33 kg/m^2) folded to 7/8 area 
                    [0.06, 0.10, 0.38, 0.63, 0.70, 0.73, 0.73], # Densely woven window curtains 90 mm from wall 
                    [0.03, 0.09, 0.24, 0.46, 0.79, 0.76, 0.76], # Vertical blinds, 15 cm from wall,   half opened (45 deg) 
                    [0.03, 0.06, 0.13, 0.28, 0.49, 0.56, 0.56], # Vertical blinds, 15 cm from wall,   open (90 deg) 
                    [0.05, 0.12, 0.35, 0.45, 0.38, 0.36, 0.36], # Tight velvet curtains 
                    [0.10, 0.38, 0.63, 0.52, 0.55, 0.65, 0.65], # Curtain fabric, 15 cm from wall 
                    [0.12, 0.60, 0.98, 1.00, 1.00, 1.00, 1.00], # Curtain fabric, folded, 15 cm from wall
                    [0.03, 0.03, 0.15, 0.40, 0.50, 0.50, 0.50], # Curtains of close-woven glass mat   hung 50 mm from wall 
                    [0.36, 0.26, 0.51, 0.45, 0.62, 0.76, 0.76], # Studio curtains, 22 cm from wall 
                ],
            ]
    
    @staticmethod
    def getAbsorptionCoefficients(category, material):
        
        category = category.lower().strip()
        if category not in MaterialAbsorptionTable.categories:
            raise Exception('Unknown category for material absorption table: %s' % (category))
        categoryIdx = MaterialAbsorptionTable.categories.index(category)
        
        material = material.lower().strip()
        if material not in MaterialAbsorptionTable.materials[categoryIdx]:
            raise Exception('Unknown material for category %s in material absorption table: %s' % (category, material))
        materialIdx = MaterialAbsorptionTable.materials[categoryIdx].index(material)
        
        coefficients = np.array(MaterialAbsorptionTable.table[categoryIdx][materialIdx])
        frequencies = np.array(AirAttenuationTable.frequencies)
        
        eps = np.finfo(np.float).eps
        coefficientsDb = 20.0 * np.log10(1.0 - coefficients + eps)
        
        return coefficientsDb, frequencies
    
class AirAttenuationTable(object):
    
    # From: Auralization : fundamentals of acoustics, modelling, simulation, algorithms and acoustic virtual reality
    
    temperatures = [10.0, 20.0]
    relativeHumidities = [40.0, 60.0, 80.0]
    frequencies = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    
    # Air attenuation coefficient, in 10^-3 / m 
    table = [   [ # 10 deg C
                    [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0],  # 30-50% hum
                    [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1],  # 50-70% hum
                    [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8],  # 70-90% hum
                ],
                [ # 20 deg C
                    [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3],  # 30-50% hum
                    [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5],  # 50-70% hum
                    [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6],  # 70-90% hum
                ]
            ]
    
    @staticmethod
    def getAttenuations(distance, temperature, relativeHumidity):
        closestTemperatureIdx = np.argmin(np.sqrt((np.array(AirAttenuationTable.temperatures) - temperature)**2))
        closestHumidityIdx = np.argmin(np.sqrt((np.array(AirAttenuationTable.relativeHumidities) - relativeHumidity)**2))
        
        attenuations = np.array(AirAttenuationTable.table[closestTemperatureIdx][closestHumidityIdx])
        frequencies = np.array(AirAttenuationTable.frequencies)
        
        eps = np.finfo(np.float).eps
        attenuations = np.clip(distance * 1e-3 * attenuations, 0.0, 1.0 - eps)
        attenuationsDb = 20.0 * np.log10(1.0 - attenuations)
        
        return attenuationsDb, frequencies
    
class FilterBank(object):
    
    def __init__(self, n, centerFrequencies, samplingRate):
        self.n = n
        
        if n % 2 == 0:
            self.n = n + 1
            logger.warn('Length of the FIR filter adjusted to the next odd number to ensure symmetry: %d' % (self.n))
        else:
            self.n = n
            
        self.centerFrequencies = centerFrequencies
        self.samplingRate = samplingRate
    
        centerFrequencies = np.array(centerFrequencies, dtype=np.float)
        centerNormFreqs = centerFrequencies/(self.samplingRate/2.0)
        cutoffs = centerNormFreqs[:-1] + np.diff(centerNormFreqs)/2
        
        filters = []
        for i in range(len(centerFrequencies)):
            if i == 0:
                # Low-pass filter
                b = scipy.signal.firwin(self.n, cutoff=cutoffs[0], window='hamming')
            elif i == len(centerFrequencies) - 1:
                # High-pass filter
                b = scipy.signal.firwin(self.n, cutoff=cutoffs[-1], window = 'hamming', pass_zero=False)
            else:
                # Band-pass filter
                b = scipy.signal.firwin(self.n, [cutoffs[i-1], cutoffs[i]], pass_zero=False)
                
            filters.append(b)
        self.filters = np.array(filters)
    
    def getScaledImpulseResponse(self, scales=1):
        if not isinstance(scales, (list, tuple)):
            scales = scales * np.ones(len(self.filters))
        return np.sum(self.filters * scales[:, np.newaxis], axis=0)
        
    def display(self, scales=1, merged=False):
        # Adapted from: http://mpastell.com/2010/01/18/fir-with-scipy/
        
        if merged:
            b = self.getScaledImpulseResponse(scales)
            filters = [b]
        else:
            filters = np.copy(self.filters)
            if not isinstance(scales, (list, tuple)):
                scales = scales * np.ones(len(filters))
            filters *= scales[:,np.newaxis]
        
        fig = plt.figure(figsize=(8,6), facecolor='white', frameon=True)
        for b in filters:
            w,h = signal.freqz(b,1)
            h_dB = 20 * np.log10(abs(h))
            plt.subplot(211)
            plt.plot(w/max(w),h_dB)
            plt.ylim(-150, 5)
            plt.ylabel('Magnitude (db)')
            plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
            plt.title(r'Frequency response')
            plt.subplot(212)
            h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
            plt.plot(w/max(w),h_Phase)
            plt.ylabel('Phase (radians)')
            plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
            plt.title(r'Phase response')
            plt.subplots_adjust(hspace=0.5)
        return fig
    
class EvertAcousticRoom(Room):

    materials = [   # category,         # material name          # index
                    ['hard surfaces',   'average'           ],   #   0
                    ['hard surfaces',   'concrete'          ],   #   1
                    ['glazing',         'glass window'      ],   #   2
                    ['wood',            'wood, 1.6 cm thick'],   #   3
                    ['floor coverings', 'linoleum'          ],   #   4
                    ['floor coverings', 'soft carpet'       ],   #   5
                    ['curtains',        'cotton curtains'   ],   #   6
                ]

    def __init__(self, samplingRate=16000, maximumOrder=3, 
                 materialAbsorption=True, frequencyDependent=True):

        super(EvertAcousticRoom,self).__init__()

        self.samplingRate = samplingRate
        self.maximumOrder = maximumOrder
        self.materialAbsorption = materialAbsorption
        self.frequencyDependent = frequencyDependent
        
        self.filterbank = FilterBank(n=513, 
                                     centerFrequencies=MaterialAbsorptionTable.frequencies,
                                     samplingRate=samplingRate)
        self.setAirConditions()
        
    def getMaterialIdByName(self, name):
        idx = None
        for i, (_, materialName) in enumerate(EvertAcousticRoom.materials):
            if materialName == name:
                idx = i
                break
        if idx is None:
            raise Exception('Unknown material %s' % (name))
        return idx
        
    def setAirConditions(self, temperature=20.0, relativeHumidity=65.0):
        self.temperature = temperature
        self.relativeHumidity = relativeHumidity

    def _calculateSoundSpeed(self):
        # Approximate speed of sound in dry (0% humidity) air, in meters per second, at temperatures near 0 deg C
        return 331.3*np.sqrt(1+self.temperature/273.15)
    
    def _calculateDelayAndAttenuation(self, path):
        
        # Calculate path length and corresponding delay
        pathLength = 0.0
        lastPt = path.m_points[0]
        for pt in path.m_points[1:]:
            pathLength += np.sqrt((lastPt.x - pt.x)**2 + 
                                  (lastPt.y - pt.y)**2 + 
                                  (lastPt.z - pt.z)**2)
            lastPt = pt
        pathLength = pathLength / 1000.0 # mm to m
        delay = pathLength/self._calculateSoundSpeed()
        #if len(path.m_points) == 2:
        	#print('Path Length: ', pathLength)

        # Calculate air attenuation coefficient (dB)
        airAttenuations, frequencies = AirAttenuationTable.getAttenuations(pathLength, self.temperature, self.relativeHumidity)

        # Calculate spherical geometric spreading attenuation (dB)
        distanceAttenuations = 20.0 * np.log10(1.0/pathLength)
        
        # Calculat material attenuation (dB)
        materialAttenuations = np.zeros((len(MaterialAbsorptionTable.frequencies),))
        if self.materialAbsorption:
            for polygon in path.m_polygons:
                materialId = polygon.getMaterialId()
                category, material = EvertAcousticRoom.materials[materialId]
                materialAbsoption, _ = MaterialAbsorptionTable.getAbsorptionCoefficients(category, material)
                materialAttenuations += materialAbsoption
        
        # Total attenuation (dB)
        #attenuation = 0  #Change variable 'LinearGain'
        attenuation = airAttenuations + distanceAttenuations + materialAttenuations #Change variable 'LinearGain'
         
        return delay, attenuation, frequencies
    
    def calculateImpulseResponse(self, solution, maxImpulseLength=1.0, threshold=120.0, pathFilter=None):
    
        impulse = np.zeros((int(maxImpulseLength * self.samplingRate),))
        realImpulseLength = 0.0
        for i in range(solution.numPaths()):
            path = solution.getPath(i)
            delay, attenuationsDb, _ = self._calculateDelayAndAttenuation(path)
            
            if pathFilter is not None and not pathFilter(path): continue
            
            # Add path impulse to global impulse
            delaySamples = int(delay * self.samplingRate)
            #if len(path.m_points) == 2: 
                #print('Delay Samples: ', delaySamples)
            # Skip paths that are below attenuation threshold (dB)
            if np.any(abs(attenuationsDb) < threshold):
                
                #if self.frequencyDependent:
                if False:
                    # Skip paths that would have their impulse responses truncated at the end
                    if delaySamples + self.filterbank.n < len(impulse):
                    
                        linearGains = 1.0/np.exp(-attenuationsDb/20.0 * np.log(10.0))
                        pathImpulse = self.filterbank.getScaledImpulseResponse(linearGains)
                            
                        # Random phase inversion
                        if np.random.random() > 0.5:
                            pathImpulse *= -1
                            
                        startIdx = delaySamples - int(self.filterbank.n/2)
                        endIdx = startIdx + self.filterbank.n - 1
                        if startIdx < 0:
                            trimStartIdx = -startIdx
                            startIdx = 0
                        else:
                            trimStartIdx = 0
                        
                        impulse[startIdx:endIdx+1] += pathImpulse[trimStartIdx:]
                    
                        if endIdx+1 > realImpulseLength:
                            realImpulseLength = endIdx+1
                else:
                    # Use attenuation at 1000 Hz
                    linearGain = 1.0/np.exp(-attenuationsDb[3]/20.0 * np.log(10.0)) # FOR ANTTENUATION ~= 0
                    #linearGain = 1.0/np.exp(-attenuationsDb/20.0 * np.log(10.0)) #FOR ATTENUATION = 0
                    # Random phase inversion
                    sign = 1.0
                    if np.random.random() > 0.5:
                        sign *= -1
                        
                    impulse[delaySamples] += linearGain * sign
                    if delaySamples+1 > realImpulseLength:
                        realImpulseLength = delaySamples+1
            else:
                print('delay: ', delay)
          
        return impulse[:realImpulseLength]

    def getSolutions(self):
        
        self.constructBSP()
        
        solutions = []
        for s in range(self.numSources()):
            for l in range(self.numListeners()):
                src = self.getSource(s)
                lst = self.getListener(l)
                solution = PathSolution(self, src, lst, self.maximumOrder)
                solution.update()
                solutions.append(solution)
        
        return solutions


if __name__ == '__main__':

    fs = 192000
    # Create acoustic environment
    room = EvertAcousticRoom(samplingRate=fs, maximumOrder=2, materialAbsorption=True, frequencyDependent=True)

    # Define a simple rectangular (length x width x height) as room geometry, with average hard-surface walls
    length = 5000  # mm
    width =  5000  # mm
    height = 3000   # mm
    face1poly = Polygon([Vector3(0,0,0), Vector3(0,width,0), Vector3(length,width,0), Vector3(length,0,0)])
    face2poly = Polygon([Vector3(0,0,0), Vector3(0,width,0), Vector3(0,width,height), Vector3(0,0,height)])
    face3poly = Polygon([Vector3(0,0,0), Vector3(length,0,0), Vector3(length,0,height), Vector3(0,0,height)])
    face4poly = Polygon([Vector3(0,0,height), Vector3(0,width,height), Vector3(length,width,height), Vector3(length,0,height)])
    face5poly = Polygon([Vector3(0,width,height), Vector3(0,width,0), Vector3(length,width,0), Vector3(length,width,height)])
    face6poly = Polygon([Vector3(length,0,height), Vector3(length,width,height), Vector3(length,width,0), Vector3(length,0,0)])
    roomPolygons = [face1poly, face2poly, face3poly, face4poly, face5poly, face6poly]
    for polygon in roomPolygons:
        polygon.setMaterialId(room.getMaterialIdByName('cotton curtains'))
        room.addPolygon(polygon, color=Vector3(0.5,0.5,0.5))
    
    # Print some room information
    center = room.getCenter()
    print('Room maximum length: ', room.getMaxLength())
    print('Room center: x=%f, y=%f, z=%f' % (center.x, center.y, center.z))
    print('Number of elements: ', room.numElements())
    print('Number of convex elements: ', room.numConvexElements())
    
    # Display room layout
    #ax = plt3d.Axes3D(plt.figure())
    #for polygon in roomPolygons:
        #vtx = []
        #for i in range(polygon.numPoints()):
            #vtx.append([polygon[i].x, polygon[i].y, polygon[i].z])
        #vtx = np.array(vtx)
        #tri = plt3d.art3d.Poly3DCollection([vtx])
        #tri.set_color([0.5, 0.5, 0.5, 0.25])
        #tri.set_edgecolor('k')
        #ax.add_collection3d(tri)
        #ax.set_xlim(0, length)
        #ax.set_ylim(0, width)
        #ax.set_zlim(0, height) 
        #ax.set_xlabel("x axis [mm]")
        #ax.set_ylabel("y axis [mm]")
        #ax.set_zlabel("z axis [mm]")
        #ax.invert_xaxis()
        #ax.invert_yaxis()

    #2000 x 1000 able at the center of the room
    #table surface
    tl = 2000 #length of object in x-axis
    tw = 2000 #width of object in y-axis
    th = 1000 #hight of object (from floor to upper surface) in z-axis
    tp = 50   #depth of object (from uper surface to lower surface) in z-axis
    tface1 = Polygon([Vector3((length-tl)/2,(width-tw)/2,th), Vector3((length-tl)/2,(width+tw)/2,th), Vector3((length+tl)/2,(width+tw)/2,th), Vector3((length+tl)/2,(width-tw)/2,th)])
    tface2 = Polygon([Vector3((length-tl)/2,(width-tw)/2,th-tp), Vector3((length-tl)/2,(width+tw)/2,th-tp), Vector3((length+tl)/2,(width+tw)/2,th-tp), Vector3((length+tl)/2,(width-tw)/2,th-tp)])
    tface3 = Polygon([Vector3((length-tl)/2,(width-tw)/2,th), Vector3((length-tl)/2,(width-tw)/2,th-tp), Vector3((length-tl)/2,(width+tw)/2,th-tp), Vector3((length-tl)/2,(width+tw)/2,th)])
    tface4 = Polygon([Vector3((length+tl)/2,(width-tw)/2,th), Vector3((length+tl)/2,(width-tw)/2,th-tp), Vector3((length+tl)/2,(width+tw)/2,th-tp), Vector3((length+tl)/2,(width+tw)/2,th)])
    tface5 = Polygon([Vector3((length-tl)/2,(width-tw)/2,th), Vector3((length-tl)/2,(width-tw)/2,th-tp), Vector3((length+tl)/2,(width-tw)/2,th-tp), Vector3((length+tl)/2,(width-tw)/2,th)])      
    tface6 = Polygon([Vector3((length-tl)/2,(width+tw)/2,th), Vector3((length-tl)/2,(width+tw)/2,th-tp), Vector3((length+tl)/2,(width+tw)/2,th-tp), Vector3((length+tl)/2,(width+tw)/2,th)])  
    #table base
    tl2 = 800 #length of object in x-axis
    tw2 = 800 #width of object in y-axis
    th2 = 50  #hight of object (from floor to upper surface) in z-axis
    tp2 = 50  #depth of object (from uper surface to lower surface) in z-axis
    tface7 = Polygon([Vector3((length-tl2)/2,(width-tw2)/2,th2), Vector3((length-tl2)/2,(width+tw2)/2,th2), Vector3((length+tl2)/2,(width+tw2)/2,th2), Vector3((length+tl2)/2,(width-tw2)/2,th2)])
    tface8 = Polygon([Vector3((length-tl2)/2,(width-tw2)/2,th2-tp2), Vector3((length-tl2)/2,(width+tw2)/2,th2-tp2), Vector3((length+tl2)/2,(width+tw2)/2,th2-tp2), Vector3((length+tl2)/2,(width-tw2)/2,th2-tp2)])
    tface9 = Polygon([Vector3((length-tl2)/2,(width-tw2)/2,th2), Vector3((length-tl2)/2,(width-tw2)/2,th2-tp2), Vector3((length-tl2)/2,(width+tw2)/2,th2-tp2), Vector3((length-tl2)/2,(width+tw2)/2,th2)])
    tface10 = Polygon([Vector3((length+tl2)/2,(width-tw2)/2,th2), Vector3((length+tl2)/2,(width-tw2)/2,th2-tp2), Vector3((length+tl2)/2,(width+tw2)/2,th2-tp2), Vector3((length+tl2)/2,(width+tw2)/2,th2)])
    tface11 = Polygon([Vector3((length-tl2)/2,(width-tw2)/2,th2), Vector3((length-tl2)/2,(width-tw2)/2,th2-tp2), Vector3((length+tl2)/2,(width-tw2)/2,th2-tp2), Vector3((length+tl2)/2,(width-tw2)/2,th2)])      
    tface12 = Polygon([Vector3((length-tl2)/2,(width+tw2)/2,th2), Vector3((length-tl2)/2,(width+tw2)/2,th2-tp2), Vector3((length+tl2)/2,(width+tw2)/2,th2-tp2), Vector3((length+tl2)/2,(width+tw2)/2,th2)])  
    #table bar
    tl3 = 200 #length of object in x-axis
    tw3 = 200 #width of object in y-axis
    th3 = 950 #hight of object (from floor to upper surface) in z-axis
    tp3 = 900 #depth of object (from uper surface to lower surface) in z-axis   
    tface13 = Polygon([Vector3((length-tl3)/2,(width-tw3)/2,th3), Vector3((length-tl3)/2,(width+tw3)/2,th3), Vector3((length+tl3)/2,(width+tw3)/2,th3), Vector3((length+tl3)/2,(width-tw3)/2,th3)])
    tface14 = Polygon([Vector3((length-tl3)/2,(width-tw3)/2,th3-tp3), Vector3((length-tl3)/2,(width+tw3)/2,th3-tp3), Vector3((length+tl3)/2,(width+tw3)/2,th3-tp3), Vector3((length+tl3)/2,(width-tw3)/2,th3-tp3)])
    tface15 = Polygon([Vector3((length-tl3)/2,(width-tw3)/2,th3), Vector3((length-tl3)/2,(width-tw3)/2,th3-tp3), Vector3((length-tl3)/2,(width+tw3)/2,th3-tp3), Vector3((length-tl3)/2,(width+tw3)/2,th3)])
    tface16 = Polygon([Vector3((length+tl3)/2,(width-tw3)/2,th3), Vector3((length+tl3)/2,(width-tw3)/2,th3-tp3), Vector3((length+tl3)/2,(width+tw3)/2,th3-tp3), Vector3((length+tl3)/2,(width+tw3)/2,th3)])
    tface17 = Polygon([Vector3((length-tl3)/2,(width-tw3)/2,th3), Vector3((length-tl3)/2,(width-tw3)/2,th3-tp3), Vector3((length+tl3)/2,(width-tw3)/2,th3-tp3), Vector3((length+tl3)/2,(width-tw3)/2,th3)])      
    tface18 = Polygon([Vector3((length-tl3)/2,(width+tw3)/2,th3), Vector3((length-tl3)/2,(width+tw3)/2,th3-tp3), Vector3((length+tl3)/2,(width+tw3)/2,th3-tp3), Vector3((length+tl3)/2,(width+tw3)/2,th3)])  
      
    objectPolygons = [tface1, tface2, tface3, tface4, tface5, tface6, tface7, tface8, tface9, tface10, tface11, tface12, tface13, tface14, tface15, tface16, tface17, tface18]
    
    for polygon in objectPolygons:
        polygon.setMaterialId(room.getMaterialIdByName('concrete'))
        room.addPolygon(polygon, color=Vector3(0.8,0,0))
    
    # Display object layout
    #for polygon in objectPolygons:
        #vtx = []
        #for i in range(polygon.numPoints()):
            #vtx.append([polygon[i].x, polygon[i].y, polygon[i].z])
        #vtx = np.array(vtx)
        #tri = plt3d.art3d.Poly3DCollection([vtx])
        #tri.set_color([1.0, 0.5, 0.5, 0.25])
        #tri.set_edgecolor('r')
        #ax.add_collection3d(tri)  
          
    #M_NV= 3, 5, 6, 9, 10, 12 
    #M_V = 1, 2, 4, 7, 8, 11, 13, 14
    #Create the transmitted signals
    N = 15
    M1 = 2
    M2 = 7
    M3 = 11
    M4 = 14
    
    k = np.arange(0,N)
    
    ak1 = m.pi*(M1/N)*np.multiply(k,k+1)
    ak2 = m.pi*(M2/N)*np.multiply(k,k+1)
    ak3 = m.pi*(M3/N)*np.multiply(k,k+1)
    ak4 = m.pi*(M4/N)*np.multiply(k,k+1)
    
    #Zadoff-Chu Seq.
    sk1 = np.exp(1j*ak1)
    sk2 = np.exp(1j*ak2)
    sk3 = np.exp(1j*ak3)
    sk4 = np.exp(1j*ak4)
    
    #repeating
    re = 2
    skr1 = np.tile(sk1,(1,re))
    skr2 = np.tile(sk2,(1,re))
    skr3 = np.tile(sk3,(1,re))
    skr4 = np.tile(sk4,(1,re))
    
    tsym = 0.3125
    p1 = (tsym*N)*fs*re
    q1 = tsym*fs
    
    #upsampling
    skr1up = np.zeros(int(p1))+1j*np.zeros(int(p1))
    skr1up[0::int(q1)] = skr1
    skr2up = np.zeros(int(p1))+1j*np.zeros(int(p1))
    skr2up[0::int(q1)] = skr2
    skr3up = np.zeros(int(p1))+1j*np.zeros(int(p1))
    skr3up[0::int(q1)] = skr3
    skr4up = np.zeros(int(p1))+1j*np.zeros(int(p1))
    skr4up[0::int(q1)] = skr4
    
    #sources signals (Modulated)
    fc = 20000
    tc = np.linspace(0,tsym*N,len(skr1up))
    
    signal1 = np.multiply(skr1up.real, np.cos(2*m.pi*fc*tc)) - np.multiply(skr1up.imag, np.sin(2*m.pi*fc*tc))
    signal2 = np.multiply(skr2up.real, np.cos(2*m.pi*fc*tc)) - np.multiply(skr2up.imag, np.sin(2*m.pi*fc*tc))
    signal3 = np.multiply(skr3up.real, np.cos(2*m.pi*fc*tc)) - np.multiply(skr3up.imag, np.sin(2*m.pi*fc*tc))
    signal4 = np.multiply(skr4up.real, np.cos(2*m.pi*fc*tc)) - np.multiply(skr4up.imag, np.sin(2*m.pi*fc*tc)) 
    #signal1 = skr1up   
    #signal2 = skr2up
    #signal3 = skr3up
    #signal4 = skr4up   
    t = np.arange(len(signal1), dtype=np.float)/fs
 
    #plot sources signals
    #fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,sharey=True)
    #ax1.plot(t, signal1.real)
    #ax1.set_title('Source signal 1')
    #ax2.plot(t, signal2.real)
    #ax2.set_title('Source signal 2')
    #ax3.plot(t, signal3.real)
    #ax3.set_title('Source signal 3')
    #ax4.plot(t, signal4.real)
    #ax4.set_title('Source signal 4')
    
    #fig.text(0.5, 0.04, 'Time [sec]', ha='center')
    #fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

    # Create four sources localized between in the corners
    clerance = 50      #distance in mm from walls
    #source 1
    src1 = Source()
    src1.setPosition(Vector3(xxxx,xxxx,height-xxxx))
    src1.setName('Src1')
    room.addSource(src1)
    #source 2
    src2 = Source()
    src2.setPosition(Vector3(xxxx,width-xxxx,height-xxxx))
    src2.setName('Src2')
    room.addSource(src2)
    #source 3
    src3 = Source()
    src3.setPosition(Vector3(length-xxxx,width-xxxx,height-xxxx))
    src3.setName('Src3')
    room.addSource(src3)
    #source 4
    src4 = Source()
    src4.setPosition(Vector3(length-xxxx,xxxx,height-xxxx))
    src4.setName('Src4')
    room.addSource(src4)
    
    #making the ideal path
    fp = 256
    delta_tp = 1/fp

    wp = 5
    wp = wp*m.pi/180

    hp = 1*1000    #from meters to milis
    rp = 2*1000    #from meters to milis

    tp_end = 2*m.pi/wp
    tp = np.arange(0,tp_end*fp,dtype=np.float)*delta_tp
    thetap = wp*tp

    xp = rp*np.cos(thetap)
    yp = rp*np.sin(thetap)
    zp = hp*np.ones(len(tp))

    p = np.array([xp,yp,zp])
    p = p.transpose()  
    

    #Create listener moving on the ideal path in room(length x width x height)
    points = 4               #points we want to calculate the positions at
    i = 0
    indexp = 0
    p_est = np.zeros([points,3])
    
    list1 = Listener()
    list1.setName('List1') 
    room.addListener(list1)   
        
    for indexp in range(points):

        room.getListener(0).setPosition(Vector3(p[i,0]+length/2,p[i,1]+width/2,p[i,2]))   #shift them to the center of room (0,0) = corner ==> (length/2,width/2) center
   

        # Display listeners and source layout
        #ax.plot([list1.getPosition().x], [list1.getPosition().y], [list1.getPosition().z], color='k', marker='o')
        #ax.text(list1.getPosition().x, list1.getPosition().y, list1.getPosition().z, "Listener", color='k')
        #ax.plot([src1.getPosition().x], [src1.getPosition().y], [src1.getPosition().z], color='r', marker='o')
        #ax.text(src1.getPosition().x, src1.getPosition().y, src1.getPosition().z, "Source 1", color='r')
        #ax.plot([src2.getPosition().x], [src2.getPosition().y], [src2.getPosition().z], color='r', marker='o')
        #ax.text(src2.getPosition().x, src2.getPosition().y, src2.getPosition().z, "Source 2", color='r')    
        #ax.plot([src3.getPosition().x], [src3.getPosition().y], [src3.getPosition().z], color='r', marker='o')
        #ax.text(src3.getPosition().x, src3.getPosition().y, src3.getPosition().z, "Source 3", color='r')    
        #ax.plot([src4.getPosition().x], [src4.getPosition().y], [src4.getPosition().z], color='r', marker='o')    
        #ax.text(src4.getPosition().x, src4.getPosition().y, src4.getPosition().z, "Source 4", color='r')
            
        # Only consider reverberation paths that actually hit the object
        def mustHitObjectFilter(path):
            hitObject = False
            for polygon in path.m_polygons:
                materialId = polygon.getMaterialId()
                if materialId == room.getMaterialIdByName('concrete'):
                    hitObject = True
                    break
            return hitObject
    
        # Only consider non-direct reverberation paths
        def noDirectFilter(path):
            return len(path.m_polygons) > 0
        
        # Compute the reverberation paths
        solutions = room.getSolutions()
        
        print(len(solutions))
        
        h1 = room.calculateImpulseResponse(solutions[0], maxImpulseLength=1.0, threshold=120.0, pathFilter=None)
        h2 = room.calculateImpulseResponse(solutions[1], maxImpulseLength=1.0, threshold=120.0, pathFilter=None)
        h3 = room.calculateImpulseResponse(solutions[2], maxImpulseLength=1.0, threshold=120.0, pathFilter=None)    
        h4 = room.calculateImpulseResponse(solutions[3], maxImpulseLength=1.0, threshold=120.0, pathFilter=None)
        
        maxImpulseLength = max(len(h1), len(h2), len(h3), len(h4))
        h1 = np.pad(h1, (0, max(0, maxImpulseLength - len(h1))), mode='constant')
        h2 = np.pad(h2, (0, max(0, maxImpulseLength - len(h2))), mode='constant')
        h3 = np.pad(h3, (0, max(0, maxImpulseLength - len(h3))), mode='constant')
        h4 = np.pad(h4, (0, max(0, maxImpulseLength - len(h4))), mode='constant') 
       
        # Display impulse response
        th = np.arange(len(h1), dtype=np.float)/fs
        #fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,sharey=True)
        #ax1.plot(th, h1,'b')
        #ax1.set_title('Estimated impulse 1')
        #ax2.plot(th, h2,'g')
        #ax2.set_title('Estimated impulse 2')
        #ax3.plot(th, h3,'r')
        #ax3.set_title('Estimated impulse 3')
        #ax4.plot(th, h4,'k')
        #ax4.set_title('Estimated impulse 4')
    
        #fig.text(0.5, 0.04, 'Time [sec]', ha='center')
        #fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
    
    
        #Demodulation
        signal1dd = np.multiply(signal1, np.cos(2*m.pi*fc*tc)) - 1j*np.multiply(signal1, np.sin(2*m.pi*fc*tc))
        signal2dd = np.multiply(signal2, np.cos(2*m.pi*fc*tc)) - 1j*np.multiply(signal2, np.sin(2*m.pi*fc*tc))
        signal3dd = np.multiply(signal3, np.cos(2*m.pi*fc*tc)) - 1j*np.multiply(signal3, np.sin(2*m.pi*fc*tc))
        signal4dd = np.multiply(signal4, np.cos(2*m.pi*fc*tc)) - 1j*np.multiply(signal4, np.sin(2*m.pi*fc*tc))    
    
        #downsampling before conv.
        #signal1d = signal1dd[0::int(q1)]
        #signal2d = signal2dd[0::int(q1)]
        #signal3d = signal3dd[0::int(q1)]
        #signal4d = signal4dd[0::int(q1)]
        signal1d = signal1[0::int(q1)]
        signal2d = signal2[0::int(q1)]
        signal3d = signal3[0::int(q1)]
        signal4d = signal4[0::int(q1)]

        # Apply impulse responses to click signal
        o1 = np.convolve(signal1d, h1, mode='full')
        o2 = np.convolve(signal2d, h2, mode='full')
        o3 = np.convolve(signal3d, h3, mode='full')
        o4 = np.convolve(signal4d, h4, mode='full')
    
        #To calculate lags due to conv
        to_l = len(h1)+re*len(sk1)-1
        too = np.zeros(int(to_l))
        too[0:re*len(sk1)-1]= np.arange(-re*len(sk1),-1, dtype=np.float)/fs
        too[re*len(sk1):to_l] = np.arange(0,len(h1)-1, dtype=np.float)/fs
        top = too[re*len(sk1):to_l]
    
         #received signal
        #plt.figure()
        #plt.title('Total received signal from all sources')
        signalr = o1+o2+o3+o4
        #plt.plot(too, abs(signalr), color='b')
        #plt.xlabel('Time [sec]')
        #plt.ylabel('Amplitude')
    
    
        #plt.figure()
        #plt.title('Individual received signals from each source')
        #plt.plot(too, abs(o1), color='b', label='o1')
        #plt.plot(too, abs(o2), color='g', label='o2')
        #plt.plot(too, abs(o3), color='r', label='o3')
        #plt.plot(too, abs(o4), color='k', label='o4')
        #plt.legend()
        #plt.xlabel('Time [sec]')
        #plt.ylabel('Amplitude')
    
        #Cross-Correlation
        sd1 = np.correlate(signalr,sk1, 'full')
        sd2 = np.correlate(signalr,sk2, 'full')
        sd3 = np.correlate(signalr,sk3, 'full')
        sd4 = np.correlate(signalr,sk4, 'full')
 
        #To calculate lags due to conv and corr
        #tc_l = len(o1)+len(sk1)-1
        tc_l = (len(h1)+re*len(sk1)-1)+len(sk1)-1
        tcc = np.zeros(int(tc_l))
        tcc[0:(re+1)*len(sk1)-2]= np.arange(-(re+1)*len(sk1)+1,-1, dtype=np.float)/fs
        tcc[(re+1)*len(sk1)-1:tc_l] = np.arange(0,len(h1)-1, dtype=np.float)/fs
        tch = tcc[(re+1)*len(sk1)-1:tc_l]
    
    
        # Output signals
        #fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True,sharey=True)
        #ax1.plot(tcc, abs(sd1))
        #ax1.set_title('Recived signal from source 1')
        #ax2.plot(tcc, abs(sd2))
        #ax2.set_title('Recived signal from source 2')
        #ax3.plot(tcc, abs(sd3))
        #ax3.set_title('Recived signal from source 3')
        #ax4.plot(tcc, abs(sd4))
        #ax4.set_title('Recived signal from source 4')
    
        #fig.text(0.5, 0.04, 'Time [sec]', ha='center')
        #fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
    
        #Distances (actual vs. estimated)
        threshold = 0.6
    
        #for source 1
        d1_act = m.sqrt((room.getListener(0).getPosition().x-src1.getPosition().x)**2 + (room.getListener(0).getPosition().y-src1.getPosition().y)**2 +(room.getListener(0).getPosition().z-src1.getPosition().z)**2 )
        index1=0
        h1_max = max(h1)
        for x in h1:
            if abs(x) >= threshold*h1_max:
                d1_est = tch[index1]*room._calculateSoundSpeed()
                break
            index1 += 1		
        #for source 2
        d2_act = m.sqrt((room.getListener(0).getPosition().x-src2.getPosition().x)**2 + (room.getListener(0).getPosition().y-src2.getPosition().y)**2 +(room.getListener(0).getPosition().z-src2.getPosition().z)**2 )
        index2=0
        h2_max = max(h2)
        for x in h2:
            if abs(x) >= threshold*h2_max:
                d2_est = tch[index2]*room._calculateSoundSpeed()
                break
            index2 += 1	
        #for source 3
        d3_act = m.sqrt((room.getListener(0).getPosition().x-src3.getPosition().x)**2 + (room.getListener(0).getPosition().y-src3.getPosition().y)**2 +(room.getListener(0).getPosition().z-src3.getPosition().z)**2 )
        index3=0
        h3_max = max(h3)
        for x in h3:
            if abs(x) >= threshold*h3_max:
                d3_est = tch[index3]*room._calculateSoundSpeed()
                break
            index3 += 1
        #for source 4
        d4_act = m.sqrt((room.getListener(0).getPosition().x-src4.getPosition().x)**2 + (room.getListener(0).getPosition().y-src4.getPosition().y)**2 +(room.getListener(0).getPosition().z-src4.getPosition().z)**2 )
        index4=0
        h4_max = max(h4)
        for x in h4:
            if abs(x) >= threshold*h4_max:
                d4_est = tch[index4]*room._calculateSoundSpeed()
                break
            index4 += 1
        #display distances
        print('Actual Distance from Source 1: ', (d1_act),'mm', ', Estimated Distance from Source 1: ', (d1_est*1000),'mm')
        print('Actual Distance from Source 2: ', (d2_act),'mm', ', Estimated Distance from Source 2: ', (d2_est*1000),'mm')
        print('Actual Distance from Source 3: ', (d3_act),'mm', ', Estimated Distance from Source 3: ', (d3_est*1000),'mm')
        print('Actual Distance from Source 4: ', (d4_act),'mm', ', Estimated Distance from Source 4: ', (d4_est*1000),'mm')
    
        #print('Index1: ',index1)
        #print('Index2: ',index2)
        #print('Index3: ',index3)
        #print('Index4: ',index4)
    
        #display time 
        #print('Actual Delay from Source 1: ',(d1_act/1000)/room._calculateSoundSpeed(),'s',', Estimated Delay from Source 1: ',tch[index1],'s')
        #print('Actual Delay from Source 2: ',(d2_act/1000)/room._calculateSoundSpeed(),'s',', Estimated Delay from Source 2: ',tch[index2],'s')
        #print('Actual Delay from Source 3: ',(d3_act/1000)/room._calculateSoundSpeed(),'s',', Estimated Delay from Source 3: ',tch[index3],'s')
        #print('Actual Delay from Source 4: ',(d4_act/1000)/room._calculateSoundSpeed(),'s',', Estimated Delay from Source 4: ',tch[index4],'s')

        #general info.
        #print('Signal Speed: ', room._calculateSoundSpeed(), 'm/s')
        #print('Maximum Accepted Error: ', room._calculateSoundSpeed()*(1/fs)*1000, 'mm')
    
        #calculate the position from ranging (distances)
        tx_pos = np.array([[src1.getPosition().x,src1.getPosition().y,src1.getPosition().z],[src2.getPosition().x,src2.getPosition().y,src2.getPosition().z],[src3.getPosition().x,src3.getPosition().y,src3.getPosition().z],[src4.getPosition().x,src4.getPosition().y,src4.getPosition().z]])
        dis_tx_rx = np.array([[d1_est*1000],[d2_est*1000],[d3_est*1000],[d4_est*1000]])
    
        rx_pos_est = pos_range(tx_pos,dis_tx_rx)
        rx_pos_act = np.array([[room.getListener(0).getPosition().x,room.getListener(0).getPosition().y,room.getListener(0).getPosition().z]])
        p_est[int(indexp),:] = rx_pos_est
        print(' ')
        print(' ')
        print('Estimated Object Position [x y z]: ',rx_pos_est)
        print('Actual Object Position [ x y z]:   ',rx_pos_act)
        print(' ')
        print(' ')
        i += int(np.ceil(len(p)/points))
        
    
    print(p_est)
    
    
    #animation of moving body (p_est and q_est)
    ca = np.arange(0,2*m.pi/0.01,dtype=np.float)*0.01
    cr = rp
    cx = cr*np.cos(ca)
    cy = cr*np.sin(ca)

    plt.figure()
    plt.plot(cx,cy,'b')
    plt.xlim(-rp-1, rp+1)
    plt.ylim(-rp-1, rp+1)
    plt.axis('scaled')

    i = 0
    for x in range(points):
        plt.plot(p_est[i,0]-length/2,p_est[i,1]-width/2,'ro')        # plot a point at p(i,2) and p(i,2)
        plt.pause(0.5)
        i += int(np.ceil(len(p_est)/points))
    
  
    #to see the room and the traces form sources to receiver with reflictions of order MP
    #MP = 2
    #viewer = Viewer(room, MP)
    #viewer.show()
    
    # Wait until all figures are closed
    plt.show()
