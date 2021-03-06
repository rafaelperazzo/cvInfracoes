# -*- coding: utf-8 -*-

import logging
import math

import cv2
import numpy as np

# ============================================================================

CAR_COLOURS = [ (0,0,255), (0,106,255), (0,216,255), (0,255,182), (0,255,76)
    , (144,255,0), (255,255,0), (255,148,0), (255,0,178), (220,0,255) ]

# ============================================================================

MIN_MOTO_W = 21
MIN_CAR_W = 2
MIN_BIG_H = 90

class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.frames_since_seen = 0
        self.counted = False
        self.type = 4 #4 - carro; 2 - Moto
        self.direcao = 2 #0 descenco e 1 subindo

    @property
    def last_position(self):
        return self.positions[-1]

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0

    def draw(self, output_image):
        car_colour = CAR_COLOURS[self.id % len(CAR_COLOURS)]
        for point in self.positions:
            cv2.circle(output_image, point, 2, car_colour, -1)
            cv2.polylines(output_image, [np.int32(self.positions)]
                , False, car_colour, 1)
        cv2.putText(output_image, str(self.id), self.positions[-1], cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

    def setDirecao(self,sentido):
        self.direcao = sentido

# ============================================================================

class VehicleCounter(object):
    def __init__(self, shape, divider):
        self.log = logging.getLogger("vehicle_counter")

        self.height, self.width = shape
        self.divider = divider

        self.vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_count = 0
        self.motocycle_count = 0
        self.bigvehicle_count = 0
        self.car_count = 0
        self.max_unseen_frames = 7


    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values increase in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        #print(dy)
        distance = math.sqrt(dx**2 + dy**2)
        #print(distance,a[1],b[1],b[1]-a[1])
        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
            else:
                angle = 180.0
        #print(angle)
        return distance, angle


    @staticmethod
    def is_valid_vector(a):
        distance, angle = a
        threshold_distance = max(10.0, -0.008 * angle**2 + 0.4 * angle + 25.0)
        return (distance <= threshold_distance)


    def update_vehicle(self, vehicle, matches):
        # Find if any of the matches fits this vehicle

        for i, match in enumerate(matches):
            contour, centroid = match

            vector = self.get_vector(vehicle.last_position, centroid)

            if self.is_valid_vector(vector):
                #print(vehicle.id, vehicle.last_position[1],centroid[1],centroid[1]-vehicle.last_position[1])
                if (centroid[1]-vehicle.last_position[1]>=0): #ALTERADO: Mudei de 0 para -10
                    self.log.info("Veiculo %i esta DESCENDO - posicao:(%i)" % (vehicle.id,centroid[1]))
                    vehicle.setDirecao(0)
                elif (centroid[1]-vehicle.last_position[1]<0): #ALTERADO : Mudei de 0 para -10
                    self.log.info("Veiculo %i esta SUBINDO - posicao:(%i)" % (vehicle.id,centroid[1]))
                    vehicle.setDirecao(1)
                self.log.debug("Veiculo #%d encontrado na posicao (%d). distancia=(%0.2f). Posicao anterior: (%d). Diferenca: (%d). ", vehicle.id, centroid[1], vector[0],vehicle.last_position[1],centroid[1]-vehicle.last_position[1])
                vehicle.add_position(centroid)
                #self.log.info("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)", centroid[0], centroid[1], vehicle.id, vector[0], vector[1])

                return i

        # No matches fit...
        vehicle.frames_since_seen += 1
        self.log.debug("No match for vehicle #%d. frames_since_seen=%d", vehicle.id, vehicle.frames_since_seen)

        return None


    def update_count(self, matches, output_image = None):
        self.log.debug("Updating count using %d matches...", len(matches))

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            contour, centroid = match
            #RAFAEL - CRIANDO UM NOVO VEICULO
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            #RAFAEL - CLASSIFICANDO
            #print (contour[2]*contour[3]) #Imprimindo altura vs largura
            if contour[2]<=MIN_MOTO_W:
                new_vehicle.type = 2
            else:
                new_vehicle.type = 4
            # ----------------------
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            self.log.debug("Created new vehicle #%d from match (%d, %d)."
                , new_vehicle.id, centroid[0], centroid[1])

        # Count any uncounted vehicles that are past the divider
        for vehicle in self.vehicles:
            if not vehicle.counted and (vehicle.last_position[1] > self.divider) and (vehicle.direcao==0):
                self.vehicle_count += 1
                vehicle.counted = True
                #RAFAEL - CLASSIFICAÇÃO
                if vehicle.type==2:
                    self.motocycle_count += 1
                else:
                    self.car_count += 1
                self.log.debug("Counted vehicle #%d (total count=%d)."
                    , vehicle.id, self.vehicle_count)


        # Optionally draw the vehicles on an image
        if output_image is not None:
            for vehicle in self.vehicles:
                vehicle.draw(output_image)

            #cv2.putText(output_image, ("V: %02d" % self.motocycle_count), (100, 20), cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)
            #IMPRIMINDO CONTADOR
            cv2.putText(output_image, ("%02d" % self.vehicle_count), (80, 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)
            #cv2.putText(output_image, ("%02d" % self), (80, 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)

        # Remove vehicles that have not been seen long enough
        removed = [ v.id for v in self.vehicles
            if v.frames_since_seen >= self.max_unseen_frames ]
        self.vehicles[:] = [ v for v in self.vehicles
            if not v.frames_since_seen >= self.max_unseen_frames ]
        for id in removed:
            self.log.debug("Removed vehicle #%d.", id)

        self.log.debug("Count updated, tracking %d vehicles.", len(self.vehicles))
