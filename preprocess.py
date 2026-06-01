import os
import pandas as pd
import numpy as np
import pydicom
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def resample_image(image, is_label, target_spacing=(1.5, 1.5, 3.0)):
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [
        int(round(orig_size[0] * (orig_spacing[0] / target_spacing[0]))),
        int(round(orig_size[1] * (orig_spacing[1] / target_spacing[1]))),
        int(round(orig_size[2] * (orig_spacing[2] / target_spacing[2])))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resample.Execute(image)

def _process_patient(args):
    idx, row, preprocessed_dir = args
    pt_id = row['PatientID']
    out_dir = os.path.join(preprocessed_dir, pt_id)
    os.makedirs(out_dir, exist_ok=True)
    ct_out, mask_out = os.path.join(out_dir, "ct.npy"), os.path.join(out_dir, "mask.npy")
    if os.path.exists(ct_out) and os.path.exists(mask_out): return True
    try:
        reader = sitk.ImageSeriesReader()
        ct_files = reader.GetGDCMSeriesFileNames(row['CT_Dir'])
        reader.SetFileNames(ct_files)
        ct_img = reader.Execute()
        uid_to_z = {pydicom.dcmread(f, stop_before_pixels=True).SOPInstanceUID: i for i, f in enumerate(ct_files)}
        ct_size = ct_img.GetSize()
        combined_mask = np.zeros((ct_size[2], ct_size[1], ct_size[0]), dtype=np.uint8)
        seg_ds = pydicom.dcmread(row['SEG_File'])
        k_nums, t_nums = [], []
        for seg in getattr(seg_ds, 'SegmentSequence', []):
            lab = getattr(seg, 'SegmentLabel', '').lower()
            desc = getattr(seg, 'SegmentDescription', '').lower()
            if 'kidney' in lab or 'kidney' in desc: k_nums.append(seg.SegmentNumber)
            elif 'tumor' in lab or 'mass' in lab or 'tumor' in desc or 'mass' in desc: t_nums.append(seg.SegmentNumber)
        seg_pix = seg_ds.pixel_array
        for i, frame in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
            uid = frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            if uid in uid_to_z:
                z = uid_to_z[uid]
                s_num = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                if s_num in k_nums: combined_mask[z][seg_pix[i] > 0] = 1
                elif s_num in t_nums: combined_mask[z][seg_pix[i] > 0] = 2
        mask_img = sitk.GetImageFromArray(combined_mask)
        mask_img.CopyInformation(ct_img)
        ct_res, mask_res = resample_image(ct_img, False), resample_image(mask_img, True)
        ct_arr, mask_arr = sitk.GetArrayFromImage(ct_res), sitk.GetArrayFromImage(mask_res)
        def window(arr, vmin, vmax): return (np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin)
        ct_dual = np.stack([window(ct_arr, -150, 250), window(ct_arr, -30, 170)], axis=0)
        np.save(ct_out, ct_dual.astype(np.float32))
        np.save(mask_out, mask_arr.astype(np.uint8))
        return True
    except Exception as e: print(f"Error {pt_id}: {e}"); return False

def discover_dataset(base_dir="c4kc_kits"):
    data = []
    patients = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    patients.sort()

    def _get_patient_data(pt_id):
        pt_dir = os.path.join(base_dir, pt_id)
        studies = [d for d in os.listdir(pt_dir) if os.path.isdir(os.path.join(pt_dir, d))]
        if not studies: return None
        study_dir = os.path.join(pt_dir, studies[0])
        series = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
        
        potential_seg_files = []
        potential_ct_dirs = []
        for s in series:
            s_dir = os.path.join(study_dir, s)
            files = [f for f in os.listdir(s_dir) if f.endswith(".dcm")]
            if not files: continue
            ds = pydicom.dcmread(os.path.join(s_dir, files[0]), stop_before_pixels=True)
            modality = getattr(ds, "Modality", "")
            if modality == "SEG": potential_seg_files.append(os.path.join(s_dir, files[0]))
            elif modality == "CT": potential_ct_dirs.append((s_dir, len(files)))
        
        if potential_seg_files and potential_ct_dirs:
            potential_ct_dirs.sort(key=lambda x: x[1], reverse=True)
            return {"PatientID": pt_id, "CT_Dir": potential_ct_dirs[0][0], "SEG_File": potential_seg_files[0]}
        return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(_get_patient_data, patients))
    
    data = [r for r in results if r is not None]
    df = pd.DataFrame(data)
    
    # Add train/val/test split
    np.random.seed(42)
    p = np.random.permutation(len(df))
    train_end = int(0.7 * len(df))
    val_end = int(0.85 * len(df))
    
    df['split'] = 'train'
    df.loc[p[train_end:val_end], 'split'] = 'val'
    df.loc[p[val_end:], 'split'] = 'test'
    
    df.to_csv("dataset.csv", index=False)
    print(f"Generated dataset.csv with {len(df)} patients and train/val split.")
    return df

if __name__ == "__main__":
    if not os.path.exists("dataset.csv"):
        df = discover_dataset()
    else:
        df = pd.read_csv("dataset.csv")
    
    prep_dir = "preprocessed"
    os.makedirs(prep_dir, exist_ok=True)
    args = [(i, r, prep_dir) for i, r in df.iterrows()]
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as ex:
        list(ex.map(_process_patient, args))
    print("Preprocessing complete.")
