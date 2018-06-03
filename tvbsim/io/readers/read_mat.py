import scipy.io
import json

class MatReader():
    '''
    Object to read mat files into a nested dictionary if need be.
    Helps keep strucutre from matlab similar to what is used in python.
    '''
    def __init__(self, filename=None):
        self.filename = filename

    def loadmat(self, filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        data = scipy.io.loadmat(
            filename,
            struct_as_record=False,
            squeeze_me=True)
        return self._check_keys(data)

    def _check_keys(self, dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
                dict[key] = self._todict(dict[key])
        return dict

    def _todict(self, matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                dict[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                dict[strg] = _tolist(elem)
            else:
                dict[strg] = elem
        return dict

    def _tolist(self, ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(self._todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(self._tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    def convertMatToJSON(self, matData, fileName):
        for key in matData.keys():
            if (type(matData[key])) is np.ndarray:
                serializedData = pickle.dumps(
                    matData[key], protocol=0)  # protocol 0 is printable ASCII
                jsonData[key] = serializedData
            else:
                jsonData[key] = matData[key]

        with contextlib.closing(bz2.BZ2File(fileName, 'wb')) as f:
            json.dump(jsonData, f)
