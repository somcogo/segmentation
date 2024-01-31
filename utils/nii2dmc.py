import SimpleITK as sitk
import os
import time
import pydicom
import nibabel as nib
import numpy as np
from glob import glob


def writeSlices(series_tag_values, new_img, i, out_dir, zooms=(1, 1, 1)):
    image_slice = new_img[:,:,i]
    # zooms = (zooms[0], zooms[1], 1)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time

    # Setting the type to CT preserves the slice location.
    # image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,int(i*zooms[2])))))) # Image Position (Patient)
    image_slice.SetMetaData("0020|0013", str(i)) # Instance Number

    image_slice.SetMetaData("0020|1041", str(i*zooms[2])) # Slice location


    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(out_dir,'slice' + str(i).zfill(4) + '.dcm'))

    writer.Execute(image_slice)


def nifti2dicom_1file(in_dir, out_dir):
    """
    This function is to convert only one nifti file into dicom series

    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    os.makedirs(out_dir, exist_ok=True)
    img_id = in_dir[-14:-7]
    f = nib.load(in_dir)
    zooms = f.header.get_zooms()

    new_img = sitk.ReadImage(in_dir)
    np_img = sitk.GetArrayFromImage(new_img)
    np_img = (np_img - np_img.min())/(np_img.max()-np_img.min()) * 255
    if 'seg' in in_dir:
        np_img = np_img[0]
    new_img = sitk.GetImageFromArray(np_img)
    new_img = sitk.Cast(new_img, sitk.sitkInt16)
    print(np_img.shape)
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    uid = "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time

    direction = new_img.GetDirection()
    
    series_tag_values = [("0008|0031",modification_time), # Series Time
                    ("0008|0021",modification_date), # Series Date
                    ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                    ("0020|000e", uid), # Series Instance UID 
                    ("0020|000d", uid), # Series Instance UID 
                    ("0008|0018", uid), # Series Instance UID 
                    ("0008|0030", modification_time), # Series Time
                    ("0028|0030", str(list(zooms[:2]))), # Pixel spacing
                    ("0020|0037", '\\'.join(map(str, ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])))),# Image Orientation (Patient)
                    ("0008|103e", "Created-Pycad"), # Series Description
                    ("0010|0020", str(img_id)),
                    ]

    # Write slices to output directory
    list(map(lambda i: writeSlices(series_tag_values, new_img, i, out_dir, zooms), range(new_img.GetDepth())))

def nifti2dicom_mfiles(nifti_dir, out_dir=''):
    """
    This function is to convert multiple nifti files into dicom files

    `nifti_dir`: You enter the global path to all of the nifti files here.
    `out_dir`: Put the path to where you want to save all the dicoms here.

    PS: Each nifti file's folders will be created automatically, so you do not need to create an empty folder for each patient.
    """

    images = glob(nifti_dir + '/*.nii.gz')

    for image in images:
        print(image)
        o_path = out_dir + '/' + os.path.basename(image)[:-7]
        os.makedirs(o_path, exist_ok=True)

        nifti2dicom_1file(image, o_path)

def convertNsave(arr,file_dir, index=0):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put 
    the name of each slice while using a for loop to convert all the slices
    """
    
    dicom_file = pydicom.dcmread('data/dcmimage.dcm')
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    dicom_file.save_as(os.path.join(file_dir, 'slice{}.dcm'.format(index)))

def nifti2dicom_1file2(nifti_dir, out_dir):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    nifti_file = nib.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    number_slices = nifti_array.shape[2]

    for slice_ in range(number_slices):
        convertNsave(nifti_array[:,:,slice_], out_dir, slice_)