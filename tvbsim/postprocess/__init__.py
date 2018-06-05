import sys
if int(sys.version_info[0]) < 3:
    import filters
    import seegrecording
    import peakdetect
    # import noise
    from postprocess import *
    import detectonsetoffset
