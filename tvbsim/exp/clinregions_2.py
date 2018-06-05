'''
For the TNG pipeline datasets

'''


def outsideclininds(patient):
    ''' THE REAL CLINICALLY ANNOTATED AREAS '''
    # 001
    if 'id001' in patient:
        ezinds = [26, 27, 32, 37, 54, 70, 78, 82]
    if 'id002' in patient:
        # temporal region and the right hemisphere too
        ezinds = [3, 34, 38, 55, 56, 73]
    if 'id003' in patient:
        # occipital region
        ezinds = [7, 12, 24, 28, 35]
    if 'id004' in patient:
        # occipital region
        ezinds = [7, 12, 24, 28, 35]
    if 'id005' in patient:
        #
        ezinds = [3, 34, 17, 29]
    if 'id006' in patient:
        ezinds = []
    if 'id008' in patient:
        # in the frontal region
        ezinds = [74, 75, 80, 82, 77]
    if 'id009' in patient:
        ezinds = [1, 6, 7, 30, 35]
    if 'id010' in patient:
        # from the left hemisphere
        ezinds = [3, 11, 26, 32, 37]
    if 'id011' in patient:
        ezinds = [62, 67, 74, 75, 80]
    if 'id012' in patient:
        ezinds = [3, 34, 37, 39, 41]
    # 013
    if 'id013' in patient:
        # in the frontal region with rostral middlefrontal
        ezinds = [74, 75, 80, 82, 77]
    # 014
    if 'id014' in patient:
        # both hemispheres in frontal region
        ezinds = [13, 25, 74, 60, 62]
    if 'id015' in patient:
        ezinds = [43, 54, 70, 72, 78, 82]
    return ezinds


def clinregions(patient):
    ''' THE REAL CLINICALLY ANNOTATED AREAS '''
    # 001
    if 'id001' in patient:
        ezregions = ['ctx-rh-lateralorbitofrontal', 'ctx-rh-temporalpole']
        pzregions = [
            'ctx-rh-superiorfrontal',
            'ctx-rh-rostralmiddlefrontal',
            'ctx-lh-lateralorbitofrontal']
    if 'id002' in patient:
        ezregions = ['ctx-lh-lateraloccipital']
        pzregions = ['ctx-lh-inferiorparietal', 'ctx-lh-superiorparietal']
    if 'id003' in patient:
        ezregions = ['ctx-lh-insula']
        pzregions = ['Left-Putamen', 'ctx-lh-postcentral']
    if 'id004' in patient:
        ''' '''
        ezregions = [
            'ctx-lh-posteriorcingulate',
            'ctx-lh-caudalmiddlefrontal',
            'ctx-lh-superiorfrontal']
        pzregions = ['ctx-lh-precentral', 'ctx-lh-postcentral']
    if 'id005' in patient:
        ''' '''
        ezregions = ['ctx-lh-posteriorcingulate', 'ctx-lh-precuneus']
        pzregions = ['ctx-lh-postcentral', 'ctx-lh-superiorparietal']
    if 'id006' in patient:
        ''' '''
        ezregions = ['ctx-rh-precentral']
        pzregions = ['ctx-rh-postcentral', 'ctx-rh-superiorparietal']
    if 'id007' in patient:
        ''' '''
        ezregions = [
            'Right-Amygdala',
            'ctx-rh-temporalpole',
            'ctx-rh-lateralorbitofrontal']
        pzregions = ['Right-Hippocampus', 'ctx-rh-entorhinal', 'ctx-rh-medialorbitofrontal',
                     'ctx-rh-inferiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-lateralorbitofrontal']    # 008
    if 'id008' in patient:
        ezregions = ['Right-Amygdala', 'Right-Hippocampus']
        pzregions = [
            'ctx-rh-superiortemporal',
            'ctx-rh-temporalpole',
            'ctx-rh-inferiortemporal',
            'ctx-rh-medialorbitofrontal',
            'ctx-rh-lateralorbitofrontal']
    if 'id009' in patient:
        ezregions = ['ctx-rh-lingual', 'ctx-rh-parahippocampal']
        pzregions = [
            'ctx-rh-lateraloccipital',
            'ctx-rh-fusiform',
            'ctx-rh-inferiorparietal']  # rlocc, rfug, ripc
    if 'id010' in patient:

        ezregions = [
            'ctx-rh-medialorbitofrontal',
            'ctx-rh-frontalpole',
            'ctx-rh-rostralmiddlefrontal',
            'ctx-rh-parsorbitalis']  # rmofc, rfp, rrmfg, rpor
        pzregions = ['ctx-rh-lateralorbitofrontal', 'ctx-rh-rostralmiddlefrontal',
                     'ctx-rh-superiorfrontal', 'ctx-rh-caudalmiddlefrontal']  # rlofc, rrmfc, rsfc, rcmfg
    if 'id011' in patient:
        ezregions = ['Right-Hippocampus', 'Right-Amygdala']  # rhi, ramg
        pzregions = ['Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
                     'ctx-rh-insula', 'ctx-rh-entorhinal', 'ctx-rh-temporalpole']  # rth, rcd, rpu, rins, rentc, rtmp
    if 'id012' in patient:
        ezregions = [
            'Right-Hippocampus',
            'ctx-rh-fusiform',
            'ctx-rh-entorhinal',
            'ctx-rh-temporalpole']  # rhi, rfug, rentc, rtmp
        pzregions = ['ctx-lh-fusiform', 'ctx-rh-inferiorparietal', 'ctx-rh-inferiortemporal',
                     'ctx-rh-lateraloccipital', 'ctx-rh-parahippocampal', 'ctx-rh-precuneus', 'ctx-rh-supramarginal']  # lfug, ripc, ritg, rloc, rphig, rpcunc, rsmg
    # 013
    if 'id013' in patient:
        ezregions = ['ctx-rh-fusiform']
        pzregions = ['ctx-rh-inferiortemporal', 'Right-Hippocampus', 'Right-Amygdala',
                     'ctx-rh-middletemporal', 'ctx-rh-entorhinal']
    # 014
    if 'id014' in patient:
        ezregions = ['Left-Amygdala', 'Left-Hippocampus', 'ctx-lh-entorhinal', 'ctx-lh-fusiform',
                     'ctx-lh-temporalpole', 'ctx-rh-entorhinal']
        pzregions = ['ctx-lh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-lh-inferiortemporal',
                     'ctx-lh-insula', 'ctx-lh-parahippocampal']
    if 'id015' in patient:
        ezregions = ['ctx-rh-lingual', 'ctx-rh-lateraloccipital', 'ctx-rh-cuneus',
                     'ctx-rh-parahippocampal', 'ctx-rh-superiorparietal', 'ctx-rh-fusiform', 'ctx-rh-pericalcarine']  # rlgg, rloc, rcun, rphig, rspc, rfug, rpc
        pzregions = [
            'ctx-rh-parahippocampal',
            'ctx-rh-superiorparietal',
            'ctx-rh-fusiform']  # rphig, rspc, rfug
    return ezregions, pzregions
