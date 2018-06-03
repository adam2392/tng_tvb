import numpy as np 
from enum import Enum

from tvbsim.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, \
                                                                                    labels_to_inds, monopolar_to_bipolar
from tvbsim.base.computations.math_utils import compute_gain_matrix
from tvbsim.base.utils.data_structures_utils import split_string_text_numbers

# SDE model inversion constants
class SensorTypes(Enum):
    TYPE_EEG = 'EEG'
    TYPE_SEEG = "SEEG"
    TYPE_ECOG = "ECOG"

class SensorsH5Field(object):
    GAIN_MATRIX = "gain_matrix"
    LABELS = "labels"
    LOCATIONS = "locations"
    NEEDLES = "needles"

class Sensors(object):
    TYPE_EEG = SensorTypes.TYPE_EEG.value
    TYPE_SEEG = SensorTypes.TYPE_SEEG.value
    TYPE_ECOG = SensorTypes.TYPE_ECOG.value

    number_of_sensors = None
    labels = np.array([])       # label of each sensor
    locations = np.array([])    # xyz location of each sensor
    needles = np.array([]) 
    orientations = np.array([]) # orientation of the sensor
    gain_matrix = np.array([])  # gain matrix with respect to parcellated regions
    s_type = TYPE_SEEG          # what is the sensor type
    
    def __init__(self, labels, locations, 
                gain_matrix=np.array([]),
                needles=np.array([]), 
                orientations=np.array([]), 
                s_type=TYPE_SEEG):
        self.labels = labels
        self.locations = locations 
        self.gain_matrix = gain_matrix
        self.channel_labels = np.array([]) # channel label
        self.orientations = orientations
        self.needles = needles
        self.s_type = s_type
        self.elec_labels = np.array([])    # all the electrode names
        self.elec_inds = np.array([])      # the indices of the electrodes
        if len(self.labels) > 1:
            self.elec_labels, self.elec_inds = self.group_sensors_to_electrodes()
            if self.needles.size == self.number_of_sensors:
                self.channel_labels, self.channel_inds = self.get_inds_labels_from_needles()
            else:
                self.channel_labels, self.channel_inds = self.group_sensors_to_electrodes()
                self.get_needles_from_inds_labels()

    @property
    def number_of_sensors(self):
        return self.locations.shape[0]


    def __repr__(self):
        d = {"1. sensors' type": self.s_type,
             "2. number of sensors": self.number_of_sensors,
             "3. labels": reg_dict(self.labels),
             "4. locations": reg_dict(self.locations, self.labels),
             "5. orientations": reg_dict(self.orientations, self.labels),
             "6. gain_matrix": self.gain_matrix}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def get_sensors_inds_by_sensors_labels(self, lbls):
        """
        Return the indices of 'lbls' inside our labels
        """
        return labels_to_inds(self.labels, lbls)

    def get_elecs_inds_by_elecs_labels(self, lbls):
        """
        Return the indices of 'lbls' inside our electrode labels
        """
        return labels_to_inds(self.elec_labels, lbls)

    def get_sensors_inds_by_elec_labels(self, lbls):
        """
        Get the indices of sensors from our electrode labels
        """
        elec_inds = self.get_elecs_inds_by_elecs_labels(lbls)
        sensors_inds = []
        for ind in elec_inds:
            sensors_inds += self.elec_inds[ind]
        return np.unique(sensors_inds)

    def compute_gain_matrix(self, connectivity):
        """
        Compute gain matrix given our locations and
        connectivity object

        Can use inverse-square, or dipole model.
        """
        return compute_gain_matrix(self.locations, connectivity.centres, normalize=95, ceil=1.0)

    def get_inds_labels_from_needles(self):
        channel_inds = []
        channel_labels = []
        for id in np.unique(self.needles):
            inds = np.where(self.needles == id)[0]
            channel_inds.append(inds)
            label = split_string_text_numbers(self.labels[inds[0]])[0][0]
            channel_labels.append(label)
        return channel_labels, channel_inds

    def get_needles_from_inds_labels(self):
        self.needles = np.zeros((self.number_of_sensors,), dtype="i")
        for idx, inds in enumerate(self.channel_inds):
            self.needles[inds] = idx

    def group_sensors_to_electrodes(self, labels=None):
        """
        Given labels, group sensors into unique electrodes

        For ex: A'1, A'3, B'1, B2 = A', B', B
        """
        if labels is None:
            labels = self.labels
        # split each label into string and number
        sensor_names = np.array(split_string_text_numbers(labels))
        # get the unique labels
        elec_labels = np.unique(sensor_names[:, 0])
        elec_inds = []
        # go through each unique label
        for chlbl in elec_labels:
            elec_inds.append(np.where(sensor_names[:, 0] == chlbl)[0])
        return elec_labels, elec_inds

    def get_bipolar_sensors(self, sensors_inds=None):
        if sensors_inds is None:
            sensors_inds = range(self.number_of_sensors)
        return monopolar_to_bipolar(self.labels, sensors_inds)

    def get_bipolar_elecs(self, elecs):
        try:
            bipolar_sensors_lbls = []
            bipolar_sensors_inds = []
            for elec_ind in elecs:
                curr_inds, curr_lbls = self.get_bipolar_sensors(sensors_inds=self.elec_inds[elec_ind])
                bipolar_sensors_inds.append(curr_inds)
                bipolar_sensors_lbls.append(curr_lbls)
        except:
            elecs_inds = self.get_elecs_inds_by_elecs_labels(elecs)
            bipolar_sensors_inds, bipolar_sensors_lbls = self.get_bipolar_elecs(elecs_inds)
        return bipolar_sensors_inds, bipolar_sensors_lbls


    # TODO: verify this try and change message
    # def sensor_label_to_index(self, labels):
    #     indexes = []
    #     for label in labels:
    #         try:
    #             indexes.append(np.where([np.array(lbl) == np.array(label) for lbl in self.labels])[0][0])
    #         except:
    #             print("WTF")
    #     if len(indexes) == 1:
    #         return indexes[0]
    #     else:
    #         return indexes



