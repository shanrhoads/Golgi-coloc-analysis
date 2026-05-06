from pathlib import Path
import pandas as pd

from infer_subc.core.file_io import (read_czi_image, list_image_files)

from infer_subc.core.img import *

from typing import List, Union, Dict
from infer_subc.utils.batch import list_image_files, find_segmentation_tiff_files
from infer_subc.core.file_io import read_czi_image, read_tiff_image
import numpy as np
import time
from infer_subc.utils.stats import (get_contact_metrics_3D, get_org_morphology_3D, get_XY_distribution, 
                                    get_Z_distribution, get_region_morphology_3D)
import itertools 

from bioio import BioImage

##################################################################
### FOR PROCESSING INENSITY VALUES WHEN THERE ARE NOT CELL/NUCLEUS MASKS####
##################################################################
def batch_process_intensity_quant(out_file_name: str,
                                  seg_path: Union[Path,str],
                                  out_path: Union[Path, str], 
                                  raw_path: Union[Path,str], 
                                  raw_file_type: str,
                                  raw_chan_axis:int,
                                  raw_Z_axis:int,
                                  raw_X_axis:int,
                                  raw_Y_axis:int,
                                  organelle_names: List[str],
                                  channel_dict: Dict[int, str],
                                  seg_suffix:Union[str, None]=None) -> int:
    """  
    Batch-quantifies the total fluorescence intensity of each specified channel within one or more segmented organelle masks and across the whole image for all raw image files in a given directory, then calculates the fraction of each channel's signal that falls within each organelle mask.

    Parameters:
    ----------
    out_file_name: str
        the prefix to use when naming the output datatables
    seg_path: Union[Path,str]
        Path or str to the folder that contains the segmentation tiff files
    out_path: Union[Path, str]
        Path or str to the folder that the output datatables will be saved to
    raw_path: Union[Path,str]
        Path or str to the folder that contains the raw image files
    raw_file_type: str
        the file type of the raw data; ex - ".tiff", ".czi"
    organelle_names: List[str]
        a list of all organelle names that will be analyzed; the names should be the same as the suffix used to name each of the tiff segmentation files
        Note: the intensity measurements collect per region (from get_region_morphology_3D function) will only be from channels associated to these organelles 
    channel_dict: Dict[int, str]
        a dictionary mapping channel indices (keys) to their names (values) in the raw image. Only channels specified in this dictionary will be used for intensity measurements. 
        They will be identified as the associated keys in the output datatables.
    seg_suffix:Union[str, None]=None
        any additional text that is included in the segmentation tiff files between the file stem and the segmentation suffix; this was specified during batch 
        processing, if it was used. If not used, indicate None.


    Measurements collected:
    -----------------------
    One row per image. Columns are generated dynamically from channel_dict and organelle_names:

    - {ch_name}-ch_intensity-sum_in-whole-image : sum of all pixel intensities in the {ch_name} channel across the entire image
    - {ch_name}-ch_intensity-sum_in-{org_name}  : sum of all pixel intensities in the {ch_name} channel within voxels where the {org_name} segmentation mask is nonzero
    - portion-{ch_name}-ch-intensity_in-{org_name} : fraction of the total channel intensity that falls within the organelle mask ({org_name} sum / whole-image sum); the key colocalization metric


    Returns:
    ----------
    count: int
        the number of images processed
    
    The output datatables that are saved to the out_path directly.
    """
    start = time.time()
    count = 0

    # Checking the file inputs
    if isinstance(raw_path, str): raw_path = Path(raw_path)
    if isinstance(seg_path, str): seg_path = Path(seg_path)
    if isinstance(out_path, str): out_path = Path(out_path)
    
    if not Path.exists(out_path):
        Path.mkdir(out_path)
        print(f"making {out_path}")

    quant_csv_path = out_path / f"{out_file_name}-intensity_quantification.csv"
    if Path.exists(quant_csv_path):
        raise FileExistsError(f"{quant_csv_path} already exists! Please change the out_file_name or delete the existing file to avoid overwriting.")
    

    # reading list of files from the raw path
    img_file_list = list_image_files(raw_path, raw_file_type)

    # containers to collect data tabels
    quant_tabs = []

    # begin processing for each image in the list
    for img_f in img_file_list:
        print(f"Processing image {img_f.stem}...")
        count = count + 1
        
        # find the associated segmentation files for this image
        seg_suffix = "-" if seg_suffix is None else seg_suffix
        filez = find_segmentation_tiff_files(img_f, organelle_names, seg_path, seg_suffix)
        print(f"found files: {filez}")

        # read in raw file and metadata
        raw_file = BioImage(img_f)
        img_data = np.squeeze(raw_file.data)

        # transpose raw image to ZYX order if necessary
        if (raw_chan_axis, raw_Z_axis, raw_Y_axis, raw_X_axis) != (0, 1, 2, 3):
            img_data = np.transpose(img_data, (raw_chan_axis, raw_Z_axis, raw_Y_axis, raw_X_axis))

        # create intensities from raw file as list based on the channel order provided
        intensities = [img_data[ch] for ch, name in channel_dict.items()]

        # store organelle images as list
        organelles = [read_tiff_image(filez[org]) for org in organelle_names]
        

        # container to store per image metrics
        per_image_metrics = {"object": []}
    
        for ch_name in channel_dict.values():
            per_image_metrics[f"{ch_name}-ch_intensity-sum"] = []

        ########## MAIN MEASUREMENTS ##########
        per_image_metrics["object"].append("whole-image")
        for ch, ch_name in channel_dict.items():
            whole_img_intensity_sum = np.sum(intensities[ch])
            per_image_metrics[f"{ch_name}-ch_intensity-sum"].append(whole_img_intensity_sum)

        for org, org_name in zip(organelles, organelle_names):
            per_image_metrics["object"].append(org_name)
            for ch, ch_name in channel_dict.items():
                org_intensity_sum = np.sum(intensities[ch][org > 0])
                per_image_metrics[f"{ch_name}-ch_intensity-sum"].append(org_intensity_sum)

        per_img_tab = pd.DataFrame(per_image_metrics)
        per_img_tab.insert(0, "file_name", img_f.stem)
        quant_tabs.append(per_img_tab)

        end2 = time.time()
        print(f"Completed processing for {count} images in {(end2-start)/60} mins.")


    final_quant = pd.concat(quant_tabs, ignore_index=True)
    final_quant = final_quant.set_index(["file_name", "object"]).unstack(level=-1)
    final_quant.columns = [f"{col[0]}_in-{col[1]}" for col in final_quant.columns.values]

    # calculate ratio of organelle intensity to whole image intensity for each channel
    for ch_name in channel_dict.values():
        for org_name in organelle_names:
            final_quant[f"portion-{ch_name}-ch-intensity_in-{org_name}"] = final_quant[f"{ch_name}-ch_intensity-sum_in-{org_name}"] / final_quant[f"{ch_name}-ch_intensity-sum_in-whole-image"]

    final_quant.to_csv(quant_csv_path)

    end = time.time()
    print(f"Quantification for {count} files is COMPLETE! Files saved to '{out_path}'.")
    print(f"It took {(end - start)/60} minutes to quantify these files.")
    return count



##################################################################
### FOR PROCESSING ONLY IF THERE IS A CELL/REGION MASK TO USE ####
##################################################################
def make_all_metrics_tables(source_file: str,
                             list_obj_names: List[str],
                             list_obj_segs: List[np.ndarray],
                             list_intensity_img: List[np.ndarray],
                             list_region_names: List[str],
                             list_region_segs: List[np.ndarray],
                             mask: str,
                             dist_centering_obj:str, 
                             dist_num_bins: int,
                             dist_center_on: bool=False,
                             dist_keep_center_as_bin: bool=True,
                             dist_zernike_degrees: Union[int, None]=None,
                             scale: Union[tuple,None] = None,
                             include_contact_dist:bool=True):
    """
    Measure the composition, morphology, distribution, and contacts of multiple organelles in a cell

    Parameters:
    ----------
    source_file: str
        file path; this is used for recorder keeping of the file name in the output data tables
    list_obj_names: List[str]
        a list of object names (strings) that will be measured; this should match the order in list_obj_segs
    list_obj_segs: List[np.ndarray]
        a list of 3D (ZYX) segmentation np.ndarrays that will be measured per cell; the order should match the list_obj_names 
    list_intensity_img: List[np.ndarray]
        a list of 3D (ZYX) grayscale np.ndarrays that will be used to measure fluoresence intensity in each region and object
    list_region_names: List[str]
        a list of region names (strings); these should include the mask (entire region being measured - usually the cell) 
        and other sub-mask regions from which we can meausure the objects in (ex - nucleus, neurites, soma, etc.). It should 
        also include the centering object used when created the XY distribution bins.
        The order should match the list_region_segs
    list_region_segs: List[np.ndarray]
        a list of 3D (ZYX) binary np.ndarrays of the region masks; the order should match the list_region_names.
    mask: str
        a str of which region name (contained in the list_region_names list) should be used as the main mask (e.g., cell mask)
    dist_centering_obj:str
        a str of which region name (contained in the list_region_names list) should be used as the centering object in 
        get_XY_distribution()
    dist_num_bins: int
        the number of concentric rings to draw between the centering object and edge of the mask in get_XY_distribution()
    dist_center_on: bool=False,
        for get_XY_distribution:
        True = distribute the bins from the center of the centering object
        False = distribute the bins from the edge of the centering object
    dist_keep_center_as_bin: bool=True
        for get_XY_distribution:
        True = include the centering object area when creating the bins
        False = do not include the centering object area when creating the bins
    dist_zernike_degrees: Union[int, None]=None
        for get_XY_distribution:
        the number of zernike degrees to include for the zernike shape descriptors; if None, the zernike measurements will not 
        be included in the output
    scale: Union[tuple,None] = None
        a tuple that contains the real world dimensions for each dimension in the image (Z, Y, X)
    include_contact_dist:bool=True
        whether to include the distribution of contact sites in get_contact_metrics_3d(); True = include contact distribution

    Returns:
    ----------
    4 Dataframes of measurements of organelle morphology, region morphology, contact morphology, and organelle/contact distributions

    """
    start = time.time()
    count = 0

    # segmentation image for all masking steps below
    cell_mask_img = list_region_segs[list_region_names.index(mask)]

    region_tabs = []
    org_tabs = []
    dist_tabs = []
    XY_bins = []
    XY_wedges = []
    contact_tabs = []

    ### NEW: process images per cell ###
    cell_num = np.unique(cell_mask_img)
    for i in cell_num[cell_num!=0]:
        # create new mask with only a single cell
        mask = cell_mask_img==i

        # run the rest of the original code a cell column will be added to each tab to differentiate between cells in the same image
        ######################
        # measure cell regions
        ######################
        # create np.ndarray of intensity images
        raw_image = np.stack(list_intensity_img)
        
        if list_region_names:
            for r, r_name in enumerate(list_region_names):
                region = list_region_segs[r]
                region_metrics = get_region_morphology_3D(region_seg=region, 
                                                        region_name=r_name,
                                                        channel_names=list_obj_names,
                                                        intensity_img=raw_image, 
                                                        mask=mask,
                                                        scale=scale)
                region_metrics.insert(loc=0,column='cell',value=i) 
                region_tabs.append(region_metrics)
                

        ##############################################################
        # loop through all organelles to collect measurements for each
        ##############################################################
        # containers to collect per organelle information


        for j, target in enumerate(list_obj_names):
            # organelle intensity image
            org_img = list_intensity_img[j]

            # organelle segmentation
            if target == 'ER':
                # ensure ER is only one object
                org_obj = (list_obj_segs[j] > 0).astype(np.uint16)
            else:
                org_obj = list_obj_segs[j]


            ##########################################################
            # measure organelle morphology & number of objs contacting
            ##########################################################
            org_metrics = get_org_morphology_3D(segmentation_img=org_obj, 
                                                seg_name=target,
                                                intensity_img=org_img, 
                                                mask=mask,
                                                scale=scale)

            org_metrics.insert(loc=0,column='cell',value=i) 
            org_tabs.append(org_metrics)

            ################################
            # measure organelle distribution 
            ################################
            centering = list_region_segs[list_region_names.index(dist_centering_obj)]
            XY_org_distribution, XY_bin_masks, XY_wedge_masks = get_XY_distribution(mask=mask,
                                                                                    centering_obj=centering,
                                                                                    obj=org_obj,
                                                                                    obj_name=target,
                                                                                    scale=scale,
                                                                                    num_bins=dist_num_bins,
                                                                                    center_on=dist_center_on,
                                                                                    keep_center_as_bin=dist_keep_center_as_bin,
                                                                                    zernike_degrees=dist_zernike_degrees)
            Z_org_distribution = get_Z_distribution(mask=mask, 
                                                    obj=org_obj,
                                                    obj_name=target,
                                                    center_obj=centering,
                                                    scale=scale)
            
            org_distribution_metrics = pd.merge(XY_org_distribution, Z_org_distribution,on=["object", "scale"])
            org_distribution_metrics.insert(loc=0,column='cell',value=i) 

            dist_tabs.append(org_distribution_metrics)
            XY_bins.append(XY_bin_masks)
            XY_wedges.append(XY_wedge_masks)

        ### NEW: only run contacts if there is more than one organelle listed
        if len(list_obj_names) > 1:
            #######################################
            # collect non-redundant contact metrics 
            #######################################
            # list the non-redundant organelle pairs
            contact_combos = list(itertools.combinations(list_obj_names, 2))

            # loop through each pair and measure contacts
            for pair in contact_combos:
                # pair names
                a_name = pair[0]
                b_name = pair[1]

                # segmentations to measure
                if a_name == 'ER':
                    # ensure ER is only one object
                    a = (list_obj_segs[list_obj_names.index(a_name)] > 0).astype(np.uint16)
                else:
                    a = list_obj_segs[list_obj_names.index(a_name)]
                
                if b_name == 'ER':
                    # ensure ER is only one object
                    b = (list_obj_segs[list_obj_names.index(b_name)] > 0).astype(np.uint16)
                else:
                    b = list_obj_segs[list_obj_names.index(b_name)]
                

                if include_contact_dist == True:
                    contact_tab, contact_dist_tab = get_contact_metrics_3D(a, a_name, 
                                                                        b, b_name, 
                                                                        mask, 
                                                                        scale, 
                                                                        include_dist=include_contact_dist,
                                                                        dist_centering_obj=centering,
                                                                        dist_num_bins=dist_num_bins,
                                                                        dist_zernike_degrees=dist_zernike_degrees,
                                                                        dist_center_on=dist_center_on,
                                                                        dist_keep_center_as_bin=dist_keep_center_as_bin)
                    dist_tabs.append(contact_dist_tab)
                else:
                    contact_tab = get_contact_metrics_3D(a, a_name, 
                                                        b, b_name, 
                                                        mask, 
                                                        scale, 
                                                        include_dist=include_contact_dist)
                    
                contact_tab.insert(loc=0,column='cell',value=i) 
                contact_tabs.append(contact_tab)


    ###########################################
    # combine all tabs into one table per type:
    ###########################################
    if org_tabs:
        final_org_tab = pd.concat(org_tabs, ignore_index=True)
        final_org_tab.insert(loc=0,column='image_name',value=source_file.stem)
    else:
        final_org_tab = None

    if contact_tabs:
        final_contact_tab = pd.concat(contact_tabs, ignore_index=True)
        final_contact_tab.insert(loc=0,column='image_name',value=source_file.stem)
    else:
        final_contact_tab = None

    if dist_tabs:
        combined_dist_tab = pd.concat(dist_tabs, ignore_index=True)
        combined_dist_tab.insert(loc=0,column='image_name',value=source_file.stem)
    else:
        combined_dist_tab = None

    if region_tabs:
        final_region_tab = pd.concat(region_tabs, ignore_index=True)
        final_region_tab.insert(loc=0,column='image_name',value=source_file.stem)
    else:
        final_region_tab = None

    end = time.time()
    print(f"It took {(end-start)/60} minutes to quantify one image.")
    return final_org_tab, final_contact_tab, combined_dist_tab, final_region_tab

def batch_process_quantification(out_file_name: str,
                                  seg_path: Union[Path,str],
                                  out_path: Union[Path, str], 
                                  raw_path: Union[Path,str], 
                                  raw_file_type: str,
                                  raw_chan_axis:int,
                                  raw_Z_axis:int,
                                  raw_X_axis:int,
                                  raw_Y_axis:int,
                                  organelle_names: List[str],
                                  organelle_channels: List[int],
                                  region_names: List[str],
                                  masks_file_name: str,
                                  mask: str,
                                  dist_centering_obj:str, 
                                  dist_num_bins: int,
                                  dist_center_on: bool=False,
                                  dist_keep_center_as_bin: bool=True,
                                  dist_zernike_degrees: Union[int, None]=None,
                                  include_contact_dist: bool = True,
                                  scale:bool=True,
                                  seg_suffix:Union[str, None]=None) -> int :
    """  
    batch process segmentation quantification (morphology, distribution, contacts); this function is currently optimized to process images from one file folder per image type (e.g., raw, segmentation)
    the output csv files are saved to the indicated out_path folder

    Parameters:
    ----------
    out_file_name: str
        the prefix to use when naming the output datatables
    seg_path: Union[Path,str]
        Path or str to the folder that contains the segmentation tiff files
    out_path: Union[Path, str]
        Path or str to the folder that the output datatables will be saved to
    raw_path: Union[Path,str]
        Path or str to the folder that contains the raw image files
    raw_file_type: str
        the file type of the raw data; ex - ".tiff", ".czi"
    organelle_names: List[str]
        a list of all organelle names that will be analyzed; the names should be the same as the suffix used to name each of the tiff segmentation files
        Note: the intensity measurements collect per region (from get_region_morphology_3D function) will only be from channels associated to these organelles 
    organelle_channels: List[int]
        a list of channel indices associated to respective organelle staining in the raw image; the indices should listed in same order in which the respective segmentation name is listed in organelle_names
    region_names: List[str]
        a list of regions, or masks, to measure; the order should correlate to the order of the channels in the "masks" output segmentation file
    masks_file_name: str
        the suffix of the "masks" segmentation file; ex- "masks_B", "masks", etc.
        this function currently does not accept indivial region segmentations 
    mask: str
        the name of the region to use as the mask when measuring the organelles; this should be one of the names listed in regions list; usually this will be the "cell" mask
    dist_centering_obj:str
        the name of the region or object to use as the centering object in the get_XY_distribution function
    dist_num_bins: int
        the number of bins for the get_XY_distribution function
    dist_center_on: bool=False,
        for get_XY_distribution:
        True = distribute the bins from the center of the centering object
        False = distribute the bins from the edge of the centering object
    dist_keep_center_as_bin: bool=True
        for get_XY_distribution:
        True = include the centering object area when creating the bins
        False = do not include the centering object area when creating the bins
    dist_zernike_degrees: Union[int, None]=None
        for get_XY_distribution:
        the number of zernike degrees to include for the zernike shape descriptors; if None, the zernike measurements will not 
        be included in the output
    include_contact_dist:bool=True
        whether to include the distribution of contact sites in get_contact_metrics_3d(); True = include contact distribution
    scale:bool=True
        a tuple that contains the real world dimensions for each dimension in the image (Z, Y, X)
    seg_suffix:Union[str, None]=None
        any additional text that is included in the segmentation tiff files between the file stem and the segmentation suffix
    


    Returns:
    ----------
    count: int
        the number of images processed
        
    """
    start = time.time()
    count = 0

    if isinstance(raw_path, str): raw_path = Path(raw_path)
    if isinstance(seg_path, str): seg_path = Path(seg_path)
    if isinstance(out_path, str): out_path = Path(out_path)
    
    if not Path.exists(out_path):
        Path.mkdir(out_path)
        print(f"making {out_path}")
    
    # reading list of files from the raw path
    img_file_list = list_image_files(raw_path, raw_file_type)

    # list of segmentation files to collect
    segs_to_collect = organelle_names + masks_file_name

    # containers to collect data tabels
    org_tabs = []
    contact_tabs = []
    dist_tabs = []
    region_tabs = []
    for img_f in img_file_list:
        print(f"Processing image {img_f.stem}...")
        count = count + 1
        filez = find_segmentation_tiff_files(img_f, segs_to_collect, seg_path, seg_suffix)

        # read in raw file and metadata
        img_data, meta_dict = read_czi_image(filez["raw"])

        # transpose raw image to ZYX order if necessary
        if raw_chan_axis != 0:
            img_data = np.transpose(img_data, (raw_chan_axis, raw_Z_axis, raw_Y_axis, raw_X_axis))

        # create intensities from raw file as list based on the channel order provided
        intensities = [img_data[ch] for ch in organelle_channels]

        # define the scale
        if scale is True:
            scale_tup = meta_dict['scale']
        else:
            scale_tup = None

        # load regions as a list based on order in list (should match order in "masks" file)
        masks = [] 
        for m in masks_file_name:
            mfile = read_tiff_image(filez[m])
            masks.append(mfile)
        regions = [masks[r] for r, region in enumerate(region_names)]

        # store organelle images as list
        organelles = [read_tiff_image(filez[org]) for org in organelle_names]
        

        org_metrics, contact_metrics, dist_metrics, region_metrics = make_all_metrics_tables(source_file=img_f,
                                                                                             list_obj_names=organelle_names,
                                                                                             list_obj_segs=organelles,
                                                                                             list_intensity_img=intensities, 
                                                                                             list_region_names=region_names,
                                                                                             list_region_segs=regions, 
                                                                                             mask=mask,
                                                                                             dist_centering_obj=dist_centering_obj,
                                                                                             dist_num_bins=dist_num_bins,
                                                                                             dist_center_on=dist_center_on,
                                                                                             dist_keep_center_as_bin=dist_keep_center_as_bin,
                                                                                             dist_zernike_degrees=dist_zernike_degrees,
                                                                                             scale=scale_tup,
                                                                                             include_contact_dist=include_contact_dist)

        org_tabs.append(org_metrics)
        contact_tabs.append(contact_metrics)
        dist_tabs.append(dist_metrics)
        region_tabs.append(region_metrics)
        end2 = time.time()
        print(f"Completed processing for {count} images in {(end2-start)/60} mins.")

    if any(x is not None for x in org_tabs):
        final_org = pd.concat(org_tabs, ignore_index=True)
        org_csv_path = out_path / f"{out_file_name}_organelles.csv"
        final_org.to_csv(org_csv_path)  
    if any(x is not None for x in contact_tabs):
        final_contact = pd.concat(contact_tabs, ignore_index=True)
        contact_csv_path = out_path / f"{out_file_name}_contacts.csv"
        final_contact.to_csv(contact_csv_path)
    if any(x is not None for x in dist_tabs):
        final_dist = pd.concat(dist_tabs, ignore_index=True)
        dist_csv_path = out_path / f"{out_file_name}_distributions.csv"
        final_dist.to_csv(dist_csv_path)
    if any(x is not None for x in region_tabs):
        final_region = pd.concat(region_tabs, ignore_index=True)
        region_csv_path = out_path / f"{out_file_name}_regions.csv"
        final_region.to_csv(region_csv_path)


    end = time.time()
    print(f"Quantification for {count} files is COMPLETE! Files saved to '{out_path}'.")
    print(f"It took {(end - start)/60} minutes to quantify these files.")
    return count

def batch_summary_stats(csv_path_list: List[str],
                         out_path: str,
                         out_preffix: str):
    """" 
    csv_path_list: List[str],
        A list of path strings where .csv files to analyze are located.
    out_path: str,
        A path string where the summary data file will be output to
    out_preffix: str
        The prefix used to name the output file.    
    """
    ds_count = 0
    fl_count = 0
    ###################
    # Read in the csv files and combine them into one of each type
    ###################
    org_tabs = []
    contact_tabs = []
    dist_tabs = []
    region_tabs = []

    for loc in csv_path_list:
        ds_count = ds_count + 1
        loc=Path(loc)
        files_store = sorted(loc.glob("*.csv"))
        for file in files_store:
            fl_count = fl_count + 1
            stem = file.stem

            org = "organelles"
            contacts = "contacts"
            dist = "distributions"
            regions = "_regions"

            if org in stem:
                test_orgs = pd.read_csv(file, index_col=0)
                test_orgs.insert(0, "dataset", stem[:-11])
                org_tabs.append(test_orgs)
            if contacts in stem:
                test_contact = pd.read_csv(file, index_col=0)
                test_contact.insert(0, "dataset", stem[:-9])
                contact_tabs.append(test_contact)
            if dist in stem:
                test_dist = pd.read_csv(file, index_col=0)
                test_dist.insert(0, "dataset", stem[:-14])
                dist_tabs.append(test_dist)
            if regions in stem:
                test_regions = pd.read_csv(file, index_col=0)
                test_regions.insert(0, "dataset", stem[:-8])
                region_tabs.append(test_regions)
    
    org_df = pd.concat(org_tabs,axis=0, join='outer')
    # contacts_df = pd.concat(contact_tabs,axis=0, join='outer')
    dist_df = pd.concat(dist_tabs,axis=0, join='outer')
    regions_df = pd.concat(region_tabs,axis=0, join='outer')

    ###################
    # adding new metrics to the original sheets
    ###################
    # TODO: include these labels when creating the original sheets
    # contact_cnt = contacts_df[["dataset", "image_name", "object", "label", "volume"]]
    # contact_cnt[["orgA", "orgB"]] = contact_cnt["object"].str.split('X', expand=True)
    # contact_cnt[["A_ID", "B_ID"]] = contact_cnt["label"].str.split('_', expand=True)
    # contact_cnt["A"] = contact_cnt["orgA"] +"_" + contact_cnt["A_ID"].astype(str)
    # contact_cnt["B"] = contact_cnt["orgB"] +"_" + contact_cnt["B_ID"].astype(str)

    # contact_cnt_percell = contact_cnt[["dataset", "image_name", "orgA", "A_ID", "object", "volume"]].groupby(["dataset", "image_name", "orgA", "A_ID", "object"]).agg(["count", "sum"])
    # contact_cnt_percell.columns = ["_".join(col_name).rstrip('_') for col_name in contact_cnt_percell.columns.to_flat_index()]
    # unstacked = contact_cnt_percell.unstack(level='object')
    # unstacked.columns = ["_".join(col_name).rstrip('_') for col_name in unstacked.columns.to_flat_index()]
    # unstacked = unstacked.reset_index()
    # for col in unstacked.columns:
    #     if col.startswith("volume_count_"):
    #         newname = col.split("_")[-1] + "_count"
    #         unstacked.rename(columns={col:newname}, inplace=True)
    #     if col.startswith("volume_sum_"):
    #         newname = col.split("_")[-1] + "_volume"
    #         unstacked.rename(columns={col:newname}, inplace=True)
    # unstacked.rename(columns={"orgA":"object", "A_ID":"label"}, inplace=True)
    # unstacked.set_index(['dataset', 'image_name', 'object', 'label'])

    # contact_percellB = contact_cnt[["dataset", "image_name", "orgB", "B_ID", "object", "volume"]].groupby(["dataset", "image_name", "orgB", "B_ID", "object"]).agg(["count", "sum"])
    # contact_percellB.columns = ["_".join(col_name).rstrip('_') for col_name in contact_percellB.columns.to_flat_index()]
    # unstackedB = contact_percellB.unstack(level='object')
    # unstackedB.columns = ["_".join(col_name).rstrip('_') for col_name in unstackedB.columns.to_flat_index()]
    # unstackedB = unstackedB.reset_index()
    # for col in unstackedB.columns:
    #     if col.startswith("volume_count_"):
    #         newname = col.split("_")[-1] + "_count"
    #         unstackedB.rename(columns={col:newname}, inplace=True)
    #     if col.startswith("volume_sum_"):
    #         newname = col.split("_")[-1] + "_volume"
    #         unstackedB.rename(columns={col:newname}, inplace=True)
    # unstackedB.rename(columns={"orgB":"object", "B_ID":"label"}, inplace=True)
    # unstackedB.set_index(['dataset', 'image_name', 'object', 'label'])

    # contact_cnt = pd.concat([unstacked, unstackedB], axis=0).sort_index(axis=0)
    # contact_cnt = contact_cnt.groupby(['dataset', 'image_name', 'object', 'label']).sum().reset_index()
    # contact_cnt['label']=contact_cnt['label'].astype("Int64")

    # org_df = pd.merge(org_df, contact_cnt, how='left', on=['dataset', 'image_name', 'object', 'label'], sort=True)
    # org_df[contact_cnt.columns] = org_df[contact_cnt.columns].fillna(0)

    ###################
    # summary stat group
    ###################
    group_by = ['dataset', 'image_name', 'cell', 'object']
    sharedcolumns = ["SA_to_volume_ratio", "equivalent_diameter", "extent", "euler_number", "solidity", "axis_major_length"]
    ag_func_standard = ['mean', 'median', 'std']

    ###################
    # summarize shared measurements between org_df and contacts_df
    ###################
    org_cont_tabs = []
    for tab in [org_df]: #, contacts_df]:
        tab1 = tab[group_by + ['volume']].groupby(group_by).agg(['count', 'sum'] + ag_func_standard)
        tab2 = tab[group_by + ['surface_area']].groupby(group_by).agg(['sum'] + ag_func_standard)
        tab3 = tab[group_by + sharedcolumns].groupby(group_by).agg(ag_func_standard)
        shared_metrics = pd.merge(tab1, tab2, 'outer', on=group_by)
        shared_metrics = pd.merge(shared_metrics, tab3, 'outer', on=group_by)
        org_cont_tabs.append(shared_metrics)

    org_summary = org_cont_tabs[0]
    # contact_summary = org_cont_tabs[1]

    ###################
    # group metrics from regions_df similar to the above
    ###################
    regions_summary = regions_df[group_by + ['volume', 'surface_area'] + sharedcolumns].set_index(group_by)

    ###################
    # summarize extra metrics from org_df
    ###################
    # columns2 = [col for col in org_df.columns if col.endswith(("_count", "_volume"))]
    # contact_counts_summary = org_df[group_by + columns2].groupby(group_by).agg(['sum'] + ag_func_standard)
    # org_summary = pd.merge(org_summary, contact_counts_summary, 'outer', on=group_by)#left_on=group_by, right_on=True)

    ###################
    # summarize distribution measurements
    ###################
    # organelle distributions
    hist_dfs = []
    for ind in range(0,len(dist_df.index)):
        selection = dist_df.iloc[[ind]] #    selection = dist_df.loc[[ind]]
        bins_df = pd.DataFrame()
        wedges_df = pd.DataFrame()
        Z_df = pd.DataFrame()
        CV_df = pd.DataFrame()

        bins_df[['bins', 'masks', 'obj']] = selection[['XY_bins', 'XY_mask_vox_cnt_perbin', 'XY_obj_vox_cnt_perbin']]
        wedges_df[['bins', 'masks', 'obj']] = selection[['XY_wedges', 'XY_mask_vox_cnt_perwedge', 'XY_obj_vox_cnt_perwedge']]
        Z_df[['bins', 'masks', 'obj']] = selection[['Z_slices', 'Z_mask_vox_cnt', 'Z_obj_vox_cnt']]

        dfs = [selection[['dataset', 'image_name', 'object', 'cell']].reset_index()]
        for df, prefix in zip([bins_df, wedges_df, Z_df], ["XY_bins_", "XY_wedges_", "Z_slices_"]):
            single_df = pd.DataFrame(list(zip(df["bins"].values[0][1:-1].split(", "), 
                                            df["obj"].values[0][1:-1].split(", "), 
                                            df["masks"].values[0][1:-1].split(", "))), columns =['bins', 'obj', 'mask']).astype(int)
            
            if "Z_" in prefix:
                single_df =  single_df.drop(single_df[single_df['mask'] == 0].index)
                single_df['bins'] = (single_df["bins"]/max(single_df.bins)*9.99).apply(np.floor)+1
                single_df = single_df.groupby("bins").agg(['sum']).reset_index()
                single_df.columns = ['bins',"obj","mask"]
        
            single_df['mask_fract'] = single_df['mask']/single_df['mask'].max()
            # single_df['obj_normed_tocell'] = (single_df["obj"]*single_df["mask_fract"]).fillna(0)
            single_df['obj_perc_per_bin'] = (single_df["obj"] / single_df["obj"].sum())*100
            single_df['obj_portion_normed_tobin'] = (single_df["obj_perc_per_bin"]/single_df["mask_fract"]).fillna(0)

            sumstats_df = pd.DataFrame()

            s = single_df['bins'].repeat(single_df['obj_portion_normed_tobin']*100)

            sumstats_df['hist_mean']=[s.mean()]
            sumstats_df['hist_median']=[s.median()]
            if single_df['obj_portion_normed_tobin'].sum() != 0: sumstats_df['hist_mode']=[s.mode().iloc[0]]
            else: sumstats_df['hist_mode']=['NaN']
            sumstats_df['hist_min']=[s.min()]
            sumstats_df['hist_max']=[s.max()]
            sumstats_df['hist_range']=[s.max() - s.min()]
            sumstats_df['hist_stdev']=[s.std()]
            sumstats_df['hist_skew']=[s.skew()]
            sumstats_df['hist_kurtosis']=[s.kurtosis()]
            sumstats_df['hist_var']=[s.var()]
            sumstats_df.columns = [prefix+col for col in sumstats_df.columns]
            dfs.append(sumstats_df.reset_index())

        CV_df = pd.DataFrame(list(zip(selection["XY_obj_cv_perbin"].values[0][1:-1].split(", "))), columns =['CV']).astype(float)
        sumstats_CV_df = pd.DataFrame()
        sumstats_CV_df['XY_bin_CV_mean'] = CV_df.mean()
        sumstats_CV_df['XY_bin_CV_median'] = CV_df.median()
        sumstats_CV_df['XY_bin_CV_std'] = CV_df.std()
        dfs.append(sumstats_CV_df.reset_index().drop(['index'], axis=1))

        combined_df = pd.concat(dfs, axis=1).drop(columns="index")
        hist_dfs.append(combined_df)
    dist_org_summary = pd.concat(hist_dfs, ignore_index=True)


    # nucleus distribution
    nuc_dist_df = dist_df[["dataset", "image_name", "cell",
                        "XY_bins", "XY_center_vox_cnt_perbin", "XY_mask_vox_cnt_perbin",
                        "XY_wedges", "XY_center_vox_cnt_perwedge", "XY_mask_vox_cnt_perwedge",
                        "Z_slices", "Z_center_vox_cnt", "Z_mask_vox_cnt"]].set_index(["dataset", "image_name", "cell"])
    nuc_hist_dfs = []
    for idx in range(0,len(nuc_dist_df.index)):
        selection = nuc_dist_df.iloc[[idx]] #.iloc[[0]]
        bins_df = pd.DataFrame()
        wedges_df = pd.DataFrame()
        Z_df = pd.DataFrame()

        bins_df[['bins', 'center', 'masks']] = selection[['XY_bins', 'XY_center_vox_cnt_perbin', 'XY_mask_vox_cnt_perbin']]
        wedges_df[['bins', 'center', 'masks']] = selection[['XY_wedges', 'XY_center_vox_cnt_perwedge', 'XY_mask_vox_cnt_perwedge']]
        Z_df[['bins', 'center', 'masks']] = selection[['Z_slices', 'Z_center_vox_cnt', 'Z_mask_vox_cnt']]

        dfs = [selection.reset_index()[['dataset', 'image_name', 'cell']]]
        for df, prefix in zip([bins_df, wedges_df, Z_df], ["XY_bins_", "XY_wedges_", "Z_slices_"]):
            single_df = pd.DataFrame(list(zip(df["bins"].values[0][1:-1].split(", "), 
                                            df["masks"].values[0][1:-1].split(", "),
                                            df["center"].values[0][1:-1].split(", "))), columns =['bins', 'mask', 'obj']).astype(int)

            if "Z_" in prefix:
                single_df =  single_df.drop(single_df[single_df['mask'] == 0].index)
                single_df['bins'] = (single_df["bins"]/max(single_df.bins)*9.99).apply(np.floor)+1
                single_df = single_df.groupby("bins").agg(['sum']).reset_index()
                single_df.columns = ['bins',"mask","obj"]
        
            single_df['mask_fract'] = single_df['mask']/single_df['mask'].max()
            # single_df['obj_normed_tocell'] = (single_df["obj"]*single_df["mask_fract"]).fillna(0)
            single_df['obj_perc_per_bin'] = (single_df["obj"] / single_df["obj"].sum())*100
            single_df['obj_portion_normed_tobin'] = (single_df["obj_perc_per_bin"]/single_df["mask_fract"]).fillna(0)

            sumstats_df = pd.DataFrame()

            s = single_df['bins'].repeat(single_df['obj_portion_normed_tobin']*100)

            sumstats_df['hist_mean']=[s.mean()]
            sumstats_df['hist_median']=[s.median()]
            if single_df['obj_portion_normed_tobin'].sum() != 0: sumstats_df['hist_mode']=[s.mode().iloc[0]]
            else: sumstats_df['hist_mode']=['NaN']
            sumstats_df['hist_min']=[s.min()]
            sumstats_df['hist_max']=[s.max()]
            sumstats_df['hist_range']=[s.max() - s.min()]
            sumstats_df['hist_stdev']=[s.std()]
            sumstats_df['hist_skew']=[s.skew()]
            sumstats_df['hist_kurtosis']=[s.kurtosis()]
            sumstats_df['hist_var']=[s.var()]
            sumstats_df.columns = [prefix+col for col in sumstats_df.columns]
            dfs.append(sumstats_df.reset_index())
        combined_df = pd.concat(dfs, axis=1).drop(columns="index")
        nuc_hist_dfs.append(combined_df)
    dist_center_summary = pd.concat(nuc_hist_dfs, ignore_index=True)
    dist_center_summary.insert(2, column="object", value="nuc")

    dist_summary = pd.concat([dist_org_summary, dist_center_summary], axis=0).set_index(group_by).sort_index()


    ###################
    # add normalization
    ###################
    # organelle area fraction
    area_fractions = []
    for idx in org_summary.index.unique():
        org_vol = org_summary.loc[idx][('volume', 'sum')]
        cell_vol = regions_summary.loc[idx[:-1] + ('cell',)]["volume"]
        afrac = org_vol/cell_vol
        area_fractions.append(afrac)
    org_summary[('volume', 'fraction')] = area_fractions
    # TODO: add in line to reorder the level=0 columns here

    # contact sites volume normalized
    # norm_toA_list = []
    # norm_toB_list = []
    # for col in contact_summary.index:
    #     norm_toA_list.append(contact_summary.loc[col][('volume', 'sum')]/org_summary.loc[col[:-1]+(col[-1].split('X')[0],)][('volume', 'sum')])
    #     norm_toB_list.append(contact_summary.loc[col][('volume', 'sum')]/org_summary.loc[col[:-1]+(col[-1].split('X')[1],)][('volume', 'sum')])
    # contact_summary[('volume', 'norm_to_A')] = norm_toA_list
    # contact_summary[('volume', 'norm_to_B')] = norm_toB_list

    # number and area of individuals organelle involved in contact
    # cont_cnt = org_df[group_by]
    # cont_cnt[[col.split('_')[0] for col in org_df.columns if col.endswith(("_count"))]] = org_df[[col for col in org_df.columns if col.endswith(("_count"))]].astype(bool)
    # cont_cnt_perorg = cont_cnt.groupby(group_by).agg('sum')
    # cont_cnt_perorg.columns = pd.MultiIndex.from_product([cont_cnt_perorg.columns, ['count_in']])
    # for col in cont_cnt_perorg.columns:
    #     cont_cnt_perorg[(col[0], 'num_fraction_in')] = cont_cnt_perorg[col].values/org_summary[('volume', 'count')].values
    # cont_cnt_perorg.sort_index(axis=1, inplace=True)
    # org_summary = pd.merge(org_summary, cont_cnt_perorg, on=group_by, how='outer')


    ###################
    # flatten datasheets and combine
    # TODO: restructure this so that all of the datasheets and unstacked and then reorded based on shared level 0 columns before flattening
    ###################
    # org flattening
    org_final = org_summary.unstack(-1)
    for col in org_final.columns:
        if col[1] in ('count_in', 'num_fraction_in') or col[0].endswith(('_count', '_volume')):
            if col[2] not in col[0]:
                org_final.drop(col,axis=1, inplace=True)
    # new_col_order = ['dataset', 'image_name', 'object', 'volume', 'surface_area', 'SA_to_volume_ratio', 
    #              'equivalent_diameter', 'extent', 'euler_number', 'solidity', 'axis_major_length', 
    #              'ERXLD', 'ERXLD_count', 'ERXLD_volume', 'golgiXER', 'golgiXER_count', 'golgiXER_volume', 
    #              'golgiXLD', 'golgiXLD_count', 'golgiXLD_volume', 'golgiXperox', 'golgiXperox_count', 'golgiXperox_volume', 
    #              'lysoXER', 'lysoXER_count', 'lysoXER_volume', 'lysoXLD', 'lysoXLD_count', 'lysoXLD_volume', 
    #              'lysoXgolgi', 'lysoXgolgi_count', 'lysoXgolgi_volume', 'lysoXmito', 'lysoXmito_count', 'lysoXmito_volume', 
    #              'lysoXperox', 'lysoXperox_count', 'lysoXperox_volume', 'mitoXER', 'mitoXER_count', 'mitoXER_volume', 
    #              'mitoXLD', 'mitoXLD_count', 'mitoXLD_volume', 'mitoXgolgi', 'mitoXgolgi_count', 'mitoXgolgi_volume', 
    #              'mitoXperox', 'mitoXperox_count', 'mitoXperox_volume', 'peroxXER', 'peroxXER_count', 'peroxXER_volume', 
    #              'peroxXLD', 'peroxXLD_count', 'peroxXLD_volume']
    # new_cols = org_final.columns.reindex(new_col_order, level=0)
    # org_final = org_final.reindex(columns=new_cols[0])
    org_final.columns = ["_".join((col_name[-1], col_name[1], col_name[0])) for col_name in org_final.columns.to_flat_index()]

    #renaming, filling "NaN" with 0 when needed, and removing ER_std columns
    for col in org_final.columns:
        # if '_count_in_' or '_fraction_in_' in col:
        #     org_final[col] = org_final[col].fillna(0)
        # if col.endswith(("_count_volume","_sum_volume", "_mean_volume", "_median_volume")):
        #     org_final[col] = org_final[col].fillna(0)
        # if col.endswith("_count_volume"):
        #     org_final.rename(columns={col:col.split("_")[0]+"_count"}, inplace=True)
        if col.startswith("ER_std_"):
            org_final.drop(columns=[col], inplace=True)
    org_final = org_final.reset_index()

    # contacts flattened
    # contact_final = contact_summary.unstack(-1)
    # contact_final.columns = ["_".join((col_name[-1], col_name[1], col_name[0])) for col_name in contact_final.columns.to_flat_index()]

    # #renaming and filling "NaN" with 0 when needed
    # for col in contact_final.columns:
    #     if col.endswith(("_count_volume","_sum_volume", "_mean_volume", "_median_volume")):
    #         contact_final[col] = contact_final[col].fillna(0)
    #     if col.endswith("_count_volume"):
    #         contact_final.rename(columns={col:col.split("_")[0]+"_count"}, inplace=True)
    # contact_final = contact_final.reset_index()

    # distributions flattened
    dist_final = dist_summary.unstack(-1)
    dist_final.columns = ["_".join((col_name[1], col_name[0])) for col_name in dist_final.columns.to_flat_index()]
    dist_final = dist_final.reset_index()

    # regions flattened & normalization added
    regions_final = regions_summary.unstack(-1)
    regions_final.columns = ["_".join((col_name[1], col_name[0])) for col_name in regions_final.columns.to_flat_index()]
    regions_final['nuc_area_fraction'] = regions_final['nuc_volume'] / regions_final['cell_volume']
    regions_final = regions_final.reset_index()

    # combining them all
    # combined = pd.merge(org_final, contact_final, on=["dataset", "image_name"], how="outer")
    combined = pd.merge(org_final, dist_final, on=["dataset", "image_name", "cell"], how="outer")
    combined = pd.merge(combined, regions_final, on=["dataset", "image_name", "cell"], how="outer").set_index(["dataset", "image_name", "cell"])
    combined.columns = [col.replace('sum', 'total') for col in combined.columns]

    ###################
    # export summary sheets
    ###################
    org_summary.to_csv(out_path + f"/{out_preffix}per_org_summarystats.csv")
    # contact_summary.to_csv(out_path + f"/{out_preffix}per_contact_summarystats.csv")
    dist_summary.to_csv(out_path + f"/{out_preffix}distribution_summarystats.csv")
    regions_summary.to_csv(out_path + f"/{out_preffix}per_region_summarystats.csv")
    combined.to_csv(out_path + f"/{out_preffix}summarystats_combined.csv")

    print(f"Processing of {fl_count} files from {ds_count} dataset(s) is complete.")
    return f"{fl_count} files from {ds_count} dataset(s) were processed"