import streamlit as st
import os, tempfile, shutil, torch, time
import numpy as np
import SimpleITK as sitk
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import Resize
from torch.amp import autocast
from huggingface_hub import hf_hub_download
from scipy.ndimage import label

def get_fused_image(vol, mask, show_k=False):
    bg = (np.clip(vol, -150, 250) + 150) / 400.0
    fused = np.stack([bg]*3, axis=-1)
    if mask.any():
        if show_k: fused[mask == 1] = 0.5 * fused[mask == 1] + 0.5 * np.array([0, 1.0, 0])
        fused[mask == 2] = 0.4 * fused[mask == 2] + 0.6 * np.array([1.0, 0, 0])
    return bg, fused

def resample_itk(img, spacing=(1.5, 1.5, 3.0), is_lab=False):
    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(spacing)
    res.SetSize([int(np.round(img.GetSize()[i] * (img.GetSpacing()[i] / spacing[i]))) for i in range(3)])
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_lab else sitk.sitkLinear)
    return res.Execute(img)

def get_bbox(mask, margin=15):
    nz, ny, nx = np.any(mask, axis=(1, 2)), np.any(mask, axis=(0, 2)), np.any(mask, axis=(0, 1))
    if not any(nz): return None
    z, y, x = np.where(nz)[0], np.where(ny)[0], np.where(nx)[0]
    return [max(0, z[0]-margin), min(mask.shape[0], z[-1]+margin+1),
            max(0, y[0]-margin), min(mask.shape[1], y[-1]+margin+1),
            max(0, x[0]-margin), min(mask.shape[2], x[-1]+margin+1)]

@st.cache_resource
def load_models(repo, device):
    m1 = SegResNet(3, 16, 2, 1).to(device)
    m1.load_state_dict(torch.load(hf_hub_download(repo, "model_s1.pth"), map_location=device))
    m2 = SegResNet(3, 32, 3, 1, blocks_down=[1, 2, 2, 4]).to(device)
    m2.load_state_dict(torch.load(hf_hub_download(repo, "model_s2.pth"), map_location=device))
    return m1.eval(), m2.eval()

def main():
    st.set_page_config("Kidney & Tumor Segmentation", layout="wide")
    st.title("🏥 Two-Stage Kidney & Tumor Segmentation")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    repo = "kssrikar4/KidneySeg-UNet"
    with st.spinner("Loading models..."): m1, m2 = load_models(repo, dev)
    files = st.file_uploader("Select DICOM series or NIfTI", accept_multiple_files=True, type=["dcm", "nii", "nii.gz"])
    if files and st.button("🚀 Run Segmentation"):
        with st.spinner("Processing..."):
            tmp = tempfile.mkdtemp()
            try:
                for f in files:
                    with open(os.path.join(tmp, f.name), "wb") as b: b.write(f.getbuffer())
                if len(files) == 1 and files[0].name.endswith(('.nii', '.gz')): img = sitk.ReadImage(os.path.join(tmp, files[0].name))
                else:
                    r = sitk.ImageSeriesReader()
                    ids = r.GetGDCMSeriesIDs(tmp)
                    if not ids: return st.error("No DICOM found.")
                    r.SetFileNames(r.GetGDCMSeriesFileNames(tmp, ids[0]))
                    img = r.Execute()
                v_np = sitk.GetArrayFromImage(resample_itk(img))
                win = lambda a, lo, hi: (np.clip(a, lo, hi) - lo) / (hi - lo)
                dual = np.stack([win(v_np, -150, 250), win(v_np, -30, 170)], 0)
                with torch.no_grad(), autocast(dev):
                    m1_low = (m1(Resize((128,128,128))(torch.from_numpy(dual).float()).unsqueeze(0).to(dev)).sigmoid() > 0.5).float()
                    m1_full = Resize(v_np.shape, mode="nearest")(m1_low.squeeze(0)).squeeze(0).cpu().numpy()
                final, (lab_m, n) = (m1_full > 0).astype(np.uint8), label(m1_full)
                for i in range(1, n + 1):
                    box = get_bbox(lab_m == i)
                    if not box: continue
                    z0, z1, y0, y1, x0, x1 = box
                    inp2 = torch.from_numpy(np.concatenate([dual[:, z0:z1, y0:y1, x0:x1], m1_full[z0:z1, y0:y1, x0:x1][None]], 0)).unsqueeze(0).float().to(dev)
                    with torch.no_grad(), autocast(dev):
                        p2 = (sliding_window_inference(inp2, (128,128,128), 4, m2).sigmoid() > 0.5).float().squeeze().cpu().numpy()
                    final[z0:z1, y0:y1, x0:x1][p2 > 0.5] = 2
                st.session_state.update({'v': v_np, 'p': final})
            except Exception as e: st.error(f"Error: {e}")
            finally: shutil.rmtree(tmp)
    if 'v' in st.session_state:
        v, p = st.session_state['v'], st.session_state['p']
        c1, c2, c3 = st.columns([2, 1, 1.5])
        idx = c1.slider("Slice", 0, v.shape[0]-1, v.shape[0]//2)
        vid, sh_k = c2.checkbox("▶️ Play Video"), c3.checkbox("Show Kidney", False)
        c3.write(f"K: {np.sum(p==1)*6.75/1000:.1f}mL | T: {np.sum(p==2)*6.75/1000:.1f}mL")
        cols = st.columns(2)
        if vid:
            phs = [c.empty() for c in cols]
            while vid:
                for i in range(v.shape[0]):
                    bg, f = get_fused_image(v[i], p[i], sh_k)
                    phs[0].image(bg, use_container_width=True)
                    phs[1].image(f, use_container_width=True)
                    time.sleep(0.05)
        else:
            bg, f = get_fused_image(v[idx], p[idx], sh_k)
            cols[0].image(bg, use_container_width=True)
            cols[1].image(f, use_container_width=True)

if __name__ == "__main__": main()
