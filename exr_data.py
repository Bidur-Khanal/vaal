import numpy as np
import OpenEXR as exr
import Imath
import PIL.Image as Image
def readEXR(filename):
    """Read depth data from EXR image file.
    
    Parameters
    ----------
    filename : str
        File path.
        
    Returns
    -------
    
    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
    """
    
    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    
    
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    channelData = dict()
    
    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)
        
        channelData[c] = C

     
    
    Z = None if 'Y' not in header['channels'] else channelData['Y']
    Z = Image.fromarray(Z)

    return Z