iso core branch idea:
    Alternate approach to generating stream zone data for extract_mpsurface
    script. It involves the following steps:

    1. generate iso surface at r=1Re
    2. seed stream lines on iso surface object using builtin tecplot fnc
    3. create single "stream zone" with all data from stream lines
    4. dump data to pandas as before, maybe with single command
       (Array.to_numpy_array)
