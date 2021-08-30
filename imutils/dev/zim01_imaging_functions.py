import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt


def calculate_signal(rfp_filepath,rfp_mask_filepath,gcamp_filepath,background):
    """
    returns a list of background corrected gcamp and rfp signal from a roi specified via the mask
    Parameters:
    rfp_filepath: btf file, rfp signal
    rfp_mask: btf file,
    gcamp_filepath: gfp signal
    background: background intensity
    """
    gcamp_signal=[]
    rfp_signal=[]
    with tiff.TiffFile(rfp_mask_filepath, multifile=True) as tif_mask,\
        tiff.TiffFile(gcamp_filepath, multifile=False) as tif_gcamp,\
        tiff.TiffFile(rfp_filepath, multifile=False) as tif_rfp:
        for i, page in enumerate(tif_mask.pages):
            if i>len(tif_gcamp.pages)-1:break
            mask=page.asarray()
            gcamp=tif_gcamp.pages[i].asarray()
            rfp=tif_rfp.pages[i].asarray()
        
            #find where the mask is
            roi=np.where(mask==255)
        
            gcamp_roi_signal=np.mean(gcamp[roi])
            rfp_roi_signal=np.mean(rfp[roi])
                   
            #append
            gcamp_signal.append(gcamp_roi_signal)
            rfp_signal.append(rfp_roi_signal)  
        gcamp_signal=np.array(gcamp_signal)
        rfp_signal=np.array(rfp_signal)
        gcamp_signal=gcamp_signal-background
        rfp_signal=rfp_signal-background
    return gcamp_signal,rfp_signal 


def plot_gcamp_rfp_ratio(gcamp_signal,rfp_signal,low,high):
    fig, axes=plt.subplots(figsize=(20,5),nrows=3, dpi=150)
    axes[0].plot(gcamp_signal)
    axes[0].set_ylabel('GCamp')
    axes[1].plot(rfp_signal)
    axes[1].set_ylabel('RFP')
    axes[2].plot(gcamp_signal/rfp_signal)
    axes[2].set_ylabel('GCamp/RFP Ratio')

    for i,ax in enumerate(axes):
        ax.set_xlim([low, high])
        #ax.set_xlim([0, len(gcamp_signal)])
        if i<2:
            ax.set_xticks([])

        if i==2:
            #ax.set_xticks(np.arange(0, len(gcamp_signal),100))
            ax.tick_params(axis='x', labelrotation = 60)
            #ax.set_xlim([0, len(gcamp_signal)])
            ax.set_xlim([low, high])

    axes[2].set_ylim([0, 0.3])
    axes[2].set_xlabel('Time (acquisition volumes)')