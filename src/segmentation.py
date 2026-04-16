import numpy as np
from scipy import ndimage
from skimage.morphology import opening
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.measure import label
from aicssegmentation.core.utils import topology_preserving_thinning, size_filter
from infer_subc.core.img import (select_channel_from_raw, 
                                 scale_and_smooth, 
                                 masked_object_thresh, 
                                 dot_filter_3, 
                                 fill_and_filter_linear_size)

import time
from typing import Union, List
from pathlib import Path
from infer_subc.core.file_io import (list_image_files, read_czi_image, export_inferred_organelle)

from bioio import BioImage

import napari
from napari.settings import get_settings
settings = get_settings()
settings.application.ipy_interactive = True


############################
## FUNCTIONS FOR DECLUMPING
############################
def highpass_filter(in_img: np.ndarray, sigma:float=0.0,
                     iterations:int=1, open:bool=False) -> np.ndarray:
    """
    Applies a highpass filter to the input image.

    Parameters:
    ---------

    in_img:np.ndarray:
        The input image to apply the filter to
    sigma:float
        The sigma value used for the lowpass filter
    iterations:int
        The number of times to apply the highpass filter
    open:bool
        A true/false statement of whether to apply an opening filter or not
    """
    highpass = in_img.copy().astype(np.float32)
    for _ in range(iterations):
        lowpass = ndimage.gaussian_filter(highpass, sigma)
        highpass -= lowpass
        np.clip(highpass, 0, None, out=highpass)
    if open:
        highpass=opening(highpass)
    return highpass

def otsu_size_filter(in_img: np.ndarray, thresh_adj:float=1, min_size:int=0) -> np.ndarray:
    """
    Both thresholds the image and ensures that all objects are above specified size

    Parameters:
    ---------
    in_img:np.ndarray
        The input image to apply the filter to
    thresh_adj:float
        A scalar value to adjust the threshold value by
    min_size:int
        The minimum value for the volume of the objects
    
    Returns:
    ----------
    A thresholded np.ndarray
    """
    threshold = threshold_otsu(in_img)
    ots = (in_img >= (threshold*thresh_adj))
    return size_filter(img=ots, min_size=min_size, method='3D')

def watershed_declumping(raw_img:np.ndarray, seg_img:np.ndarray, declump:bool, 
                         sigma:float, iterations:int=1, open:bool=False, 
                         thresh_adj:float=1, min_size:int=0) -> np.ndarray:
    """
    Declumps the input organelle using a highpass and threshold to develop seeds for watershedding

    Parameters:
    ---------
    raw_img:np.ndarray
        The raw image to determine the peak points of intensity gradient
    seg_img:np.ndarray
        The segmentation image to determine the area the organelle is located
    declump:bool
        A true/false statement of whether to declump the organelle or not
    sigma:float
        The sigma value used in the gaussian blur lowpass for the highpass filter
    iterations:int
        The number of times the highpass filter is applied
    open:bool
        A true/false statement of whether to apply an opening filter in the highpass filter
    thresh_adj:float
        A scalar for the otsu threshold
    min_size:int
        A number used to ensure the minimum size of the "seeds" for watershedding
    
    Returns:
    ---------
    A labeled np.ndarray of individual organelles
    """
    seg_img = label(seg_img)
    if declump and iterations>=1:
        highpass = highpass_filter(in_img=raw_img, sigma=sigma, open=open, iterations=iterations)
        ots = otsu_size_filter(in_img=highpass, thresh_adj=thresh_adj, min_size=min_size)
        
        return label((seg_img) + watershed(image=(np.max(raw_img)-raw_img), 
                                                      markers=label(ots), 
                                                      mask=seg_img,
                                                      connectivity=np.ones((3, 3, 3), bool)))
    else:
        return seg_img

##########################
#  infer_golgi
##########################
def infer_golgi(
            in_img: np.ndarray,
            golgi_ch: int,
            median_sz: int,
            gauss_sig: float,
            mo_method: str,
            mo_adjust: float,
            mo_cutoff_size: int,
            min_thickness: int,
            thin_dist: int,
            dot_scale_1: float,
            dot_cut_1: float,
            dot_scale_2: float,
            dot_cut_2: float,
            dot_scale_3: float,
            dot_cut_3: float,
            dot_method: str,
            min_hole_w: int,
            max_hole_w: int,
            small_obj_w: int,
            fill_filter_method: str,
                declump: bool,
                dec_sig: float,
                dec_iter: int,
                dec_open: bool,
                dec_adj: float,
                dec_min_size: int
        ) -> np.ndarray:

    """
    Procedure to infer golgi from linearly unmixed input.

   Parameters
    ------------
    in_img: 
        a 3d image containing all the channels
    median_sz: 
        width of median filter for signal
    mo_method: 
         which method to use for calculating global threshold. Options include:
         "triangle" (or "tri"), "median" (or "med"), and "ave_tri_med" (or "ave").
         "ave" refers the average of "triangle" threshold and "mean" threshold.
    mo_adjust: 
        Masked Object threshold `local_adjust`
    mo_cutoff_size: 
        Masked Object threshold `size_min`
    min_thinkness: 
        Half of the minimum width you want to keep from being thinned.
        For example, when the object width is smaller than 4, you don't
        want to make this part even thinner (may break the thin object
        and alter the topology), you can set this value as 2.
    thin_dist: 
        the amount to thin (has to be an positive integer). The number of
         pixels to be removed from outter boundary towards center.
    dot_scale: 
        scales (log_sigma) for dot filter (1,2, and 3)
    dot_cut: 
        threshold for dot filter thresholds (1,2,and 3)
    small_obj_w: 
        minimu object size cutoff for nuclei post-processing
    
    Returns
    -------------
    golgi_object
        mask defined extent of golgi object
    """

    ###################
    # EXTRACT
    ###################    
    golgi = select_channel_from_raw(in_img, golgi_ch)

    ###################
    # PRE_PROCESSING
    ###################    
    golgi_smooth =  scale_and_smooth(golgi,
                              median_size = median_sz, 
                              gauss_sigma = gauss_sig)
    ###################
    # CORE_PROCESSING
    ###################
    bw = masked_object_thresh(golgi_smooth, global_method=mo_method, cutoff_size=mo_cutoff_size, local_adjust=mo_adjust)

    bw_thin = topology_preserving_thinning(bw, min_thickness, thin_dist)

    bw_extra = dot_filter_3(golgi_smooth, dot_scale_1, dot_cut_1, dot_scale_2, dot_cut_2, dot_scale_3, dot_cut_3, dot_method)

    bw = np.logical_or(bw_extra, bw_thin)
    ###################
    # POST_PROCESSING
    ###################
    struct_obj = fill_and_filter_linear_size(bw, 
                                             hole_min=min_hole_w, 
                                             hole_max=max_hole_w, 
                                             min_size=small_obj_w,
                                             method=fill_filter_method)

    ###################
    # LABELING
    ###################
    struct_obj1 = watershed_declumping(raw_img = golgi,
                                       seg_img = struct_obj,
                                       declump = declump,
                                       sigma = dec_sig,
                                       iterations = dec_iter,
                                       open = dec_open,
                                       thresh_adj = dec_adj,
                                       min_size = dec_min_size)

    return struct_obj1

############################
## BATCH PROCESSING FUNCTION
############################
def batch_process_segmentation(raw_path: Union[Path,str],
                               raw_file_type: str,
                               seg_path: Union[Path, str],
                               name_suffix: Union[str, None],
                               masks_settings: Union[List, None],
                               golgi_settings: Union[List, None]):
    """
    This function batch processes the segmentation workflows for multiple organelles and masks across multiple images.

    Parameters:
    ----------
    raw_path: Union[Path,str]
        A string or a Path object of the path to your raw (e.g., intensity) images that will be the input for segmentation
    raw_file_type: str
        The raw file type (e.g., ".tiff" or ".czi")
    seg_path: Union[Path, str]
        A string or a Path object of the path where the segmentation outputs will be saved 
    name_suffix: str
        An optional string to include before the segmentation suffix at the end of the output file. 
        For example, if the name_suffix was "20240105", the segmentation file output from the 1.1_masks workflow would include:
        "{base-file-name}-20240105-masks"
    {}_settings: Union[List, None]
        For each workflow that you wish to include in the batch processing, 
        fill out the information in the associated settings list. 
        The necessary settings for each function are included below.


    Output files are saved as .tiff files with the same base file name as the raw image, with the addition of the name_suffix (if included) and the segmentation suffix (e.g., "masks" or "golgi").

    """
    start = time.time()
    count = 0

    if isinstance(raw_path, str): raw_path = Path(raw_path)
    if isinstance(seg_path, str): seg_path = Path(seg_path)

    if not Path.exists(seg_path):
        Path.mkdir(seg_path)
        print(f"The specified 'seg_path' was not found. Creating {seg_path}.")
    
    if not name_suffix:
        name_suffix=""

    # reading list of files from the raw path
    img_file_list = list_image_files(raw_path, raw_file_type)

    for img in img_file_list:
        count = count + 1
        print(f"Beginning segmentation of: {img}")
        seg_list = []

        # read in raw file and metadata
        raw_file = BioImage(img)
        img_data = np.squeeze(raw_file.data) # np.squeeze removes single-dimensional entries from the shape of an array
        meta_dict = {"file_name": img}

        # run masks function
        if masks_settings:
            print("MASK NEEDS EDITING - CURRENTLY JUST A PLACEHOLDER HERE")
            # masks = infer_masks(img_data, *masks_settings)
            # export_inferred_organelle(masks, name_suffix+"masks", meta_dict, seg_path)
            # seg_list.append("masks")

        if golgi_settings:
            golgi_seg = infer_golgi(img_data, *golgi_settings)
            export_inferred_organelle(golgi_seg, name_suffix+"golgi", meta_dict, seg_path)  
            seg_list.append("golgi")

        end = time.time()
        print(f"Processing for {img} completed in {(end - start)/60} minutes.")

    return print(f"Batch processing complete: {count} images segmented in {(end-start)/60} minutes.")

#####################################
## FOR QUALITY CHECKING SEGMENTATIONS
#####################################
def filter_segmentation(suffix, filt, edited, raw, status="Fail"):
    """
    This function applies filters to the object segmentations to ensure they meet criteria to run the quantification.
    After the filter is performed, users may view the filtered image, and choose to keep it or edit the previously edited image and rerun the filter. 

    Parameters:
    ----------

    suffix : str
        The suffix to identify the specific segmentation being filtered.
    filt : Union[int, str, None]
        The method of filtering to apply in the QC_filter function. The value must equal either an integer, 'Largest', 'Brightest', or 'ER'.
    edited : np.ndarray
        The edited segmentation image.
    raw : np.ndarray
        The raw input image.
    status : str
        The current status of the segmentation. Defaults to "Fail".

    Returns:
    -------
    Tuple[np.ndarray, str]
        A tuple containing the filtered segmentation and the status ("Pass" or "Fail" or "N/A").

    """

    if not (filt is None):
        if len(np.unique(label(edited))) > 2:
            
            status = "Fail"
            settings = get_settings()
            settings.application.ipy_interactive = False
            viewer2 = napari.Viewer()
            print(f"\nYour {suffix} segmentation contains MORE THAN ONE {suffix} object. For quantification, you must only have ONE {suffix} object, attempting to correct this automatically...")
            filtered_obj_seg = QC_filter(edited, raw, method=filt)

            if len(np.unique(filtered_obj_seg)) == 2:
                print(f"The image has been processed to automatically remove any small objects using the {filt} method.")
                viewer2.add_image(raw, name=f'{suffix}_raw', blending='additive')
                viewer2.add_labels(edited.copy(), name=f'{suffix}_seg')
                viewer2.add_labels(filtered_obj_seg.copy(), name=f'{suffix}_seg_filtered')
                settings = get_settings()
                settings.application.ipy_interactive = False
                print(f"Head to the Napari window to see your filtered {suffix} segmentation output!")
                print(f"Note: if further edits are desired, please edit the {suffix}_seg layer instead of the {suffix}_seg_filtered layer.")
                print("Please close the Napari window to continue.")
                
                napari.run()
                settings.application.ipy_interactive = True
                seg_edited = viewer2.layers[f'{suffix}_seg'].data
                seg_filtered = viewer2.layers[f'{suffix}_seg_filtered'].data

                if not (filtered_obj_seg.copy() == seg_filtered).all():
                    print(f"You have erroneously eddited the {suffix}_seg_filtered layer, restarting from beginning of the filtering process...")
                    return filter_segmentation(suffix, filt, edited, raw, status)

                if not (edited.copy() == seg_edited).all():
                    print(f"You have edited the {suffix}_seg layer, now retrying the filtering process...")
                    return filter_segmentation(suffix, filt, seg_edited, raw, status)
                else:
                    print(f"You appear satsified with the {suffix} segmentation, saving...")     
                    status = "Pass"
                    return (filtered_obj_seg, status)
                    
            elif len(np.unique(filtered_obj_seg)) > 2:
                print("We tried to remove small objects, but there are still multiple cell mask objects in the image. Please try other 'filter_cell' values above or edit the segmentation manually in Napari again.")
                viewer2.add_image(raw, name=f'{suffix}_raw', blending='additive')
                viewer2.add_labels(edited, name=f'{suffix}_seg')
                viewer2.add_labels(filtered_obj_seg, name=f'{suffix}_seg_filtered')

                print(f"Head to the Napari window to see your filtered {suffix} segmentation output!")
                print(f"Note: please edit the {suffix}_seg layer instead of the {suffix}_seg_filtered layer, or change the filter type chosen for this segmentation and rerun the block when prompted later.")
                print("Please close the Napari window to continue.")

                napari.run()

                if not (filtered_obj_seg == viewer2.layers[f'{suffix}_seg_filtered'].data).all():
                    print(f"You have erroneously eddited the {suffix}_seg_filtered layer, restarting from beginning of the filtering process...")
                    return filter_segmentation(suffix, filt, edited, raw, status)

                if not (viewer2.layers[f'{suffix}_seg'].data == edited).all():
                    print(f"You have edited the {suffix}_seg layer, now retrying the filtering process...")
                    return filter_segmentation(suffix, filt, viewer2.layers[f'{suffix}_seg'].data, viewer2.layers[f'raw'].data, status)
                else:
                    print(f"As you have not chosen to edit the {suffix}_seg layer, we will now return the previous segmentation.")
                    return (edited, status)
            else:
                print("There are no objects in the segmentation... Please check your segmentation files, and/or obj_filter value. We will now return the prior segmentation.")
                return (edited, status)
        else:
            print(f"Your {suffix} segmentation looks good, no corrections needed!")
            status = "Pass"
            return (edited, status)
    else:
        print(f"You have chosen not to filter the {suffix} segmentation, returning original segmentation...")
        return (edited, "N/A")
    
def edit_segmentation(suffix, viewer, edit):
    """
    This function enables editing of segmentation masks in Napari based on chosen segmentations.

    Parameters:
    ---------- 

    suffix : str
        The suffix of the segmentation file that is also used to name the layers in the Napari viewer.
    viewer : napari.Viewer
        The Napari viewer instance used previously for displaying all segmentations for an image.
    edit : bool
        A True/False flag to indicate whether editing of the image is desired or not.

    Returns:
    -------
    np.ndarray
        The edited segmentation mask as a NumPy array.
    """
    if edit:
        settings = get_settings()
        settings.application.ipy_interactive = False
        viewer2 = napari.Viewer()
        print("\nA new Napari viewer has been opened.")
        print(f"You may now begin editing the {suffix} segmentation.")
        viewer2.add_image(viewer.layers['raw'].data, name=f'raw')
        try:
            viewer2.add_image(viewer.layers[f'{suffix}_raw'].data, name=f'{suffix}_raw', blending='additive')
        except (ValueError, KeyError):
            print(f"No raw image found for {suffix}.")
        viewer2.add_labels(viewer.layers[f'{suffix}_seg'].data, name=f'{suffix}_seg')
        print(f"Head to the Napari window to edit your {suffix} segmentation output in the {suffix}_seg layer.")
        print(f"When you close out of the viewer, the edited {suffix} segmentation will be saved automatically")
        napari.run()
        settings.application.ipy_interactive = True
        return viewer2.layers[f'{suffix}_seg'].data
    else:
        return viewer.layers[f'{suffix}_seg'].data