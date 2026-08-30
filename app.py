import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Enterprise HPLC/DAD Processor", layout="wide")

# --- Mathematical & Chemometric Models ---
def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def baseline_poly(y, degree=3):
    x = np.arange(len(y))
    coefs = np.polyfit(x, y, degree)
    return np.polyval(coefs, x)

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-((x - cen) ** 2) / wid)

def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        y += gaussian(x, params[i], params[i+1], params[i+2])
    return y

def generate_dad_chromatogram():
    time = np.linspace(0, 20, 2000)
    wavelengths = np.linspace(200, 400, 100)
    
    baseline = 0.5 * np.sin(time/3) + 0.1 * time 
    # API peak with a closely co-eluting impurity (hard to integrate manually)
    peak1 = 15 * np.exp(-((time - 5)**2) / 0.05)
    peak_coeluting = 4 * np.exp(-((time - 5.3)**2) / 0.04) 
    peak3 = 8 * np.exp(-((time - 14)**2) / 0.1)
    
    signal_1d = baseline + peak1 + peak_coeluting + peak3 + np.random.normal(0, 0.1, 2000)
    
    # 3D Matrix
    spec1 = np.exp(-((wavelengths - 254)**2) / 400)
    spec2 = np.exp(-((wavelengths - 260)**2) / 300)
    spec3 = np.exp(-((wavelengths - 230)**2) / 500)
    
    z_matrix = np.outer(peak1, spec1) + np.outer(peak_coeluting, spec2) + np.outer(peak3, spec3)
    z_matrix += np.random.normal(0, 0.02, (2000, 100)) 
    
    return time, wavelengths, signal_1d, z_matrix

# --- UI Sidebar Setup ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
st.sidebar.title("Data Pipeline")
uploaded_file = st.sidebar.file_uploader("Ingest Raw Data (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    time = df.iloc[:, 0].values
    signal = df.iloc[:, 1].values
else:
    st.sidebar.info("Demo Mode: Synthesizing API + Co-eluting Impurity.")
    time, wavelengths, signal, z_matrix = generate_dad_chromatogram()

st.sidebar.markdown("---")
st.sidebar.header("Chemometric Parameters")
baseline_model = st.sidebar.selectbox("Baseline Algorithm", ["Asymmetric Least Squares (ALS)", "Polynomial Fit"])
if "ALS" in baseline_model:
    lam = st.sidebar.number_input("ALS Smoothness (λ)", value=100000, step=10000)
    p = st.sidebar.number_input("ALS Asymmetry (p)", value=0.005, format="%.3f")
else:
    poly_deg = st.sidebar.slider("Polynomial Degree", 1, 7, 3)

apply_smoothing = st.sidebar.checkbox("Savitzky-Golay Smoothing", value=True)
prominence = st.sidebar.slider("Peak Prominence", 0.1, 20.0, 0.5, 0.1)
enable_deconv = st.sidebar.checkbox("Attempt Gaussian Deconvolution", value=True)

# --- Processing Pipeline ---
processed_signal = savgol_filter(signal, 15, 3) if apply_smoothing else signal

if "ALS" in baseline_model:
    baseline = baseline_als(processed_signal, lam, p)
else:
    baseline = baseline_poly(processed_signal, poly_deg)
    
corrected_signal = processed_signal - baseline

# Peak Detection
peaks, properties = find_peaks(corrected_signal, prominence=prominence)
widths_half, width_heights, left_ips, right_ips = peak_widths(corrected_signal, peaks, rel_height=0.5)

# Noise calculation for LOD/LOQ (using last 10% of the chromatogram)
noise_region = corrected_signal[-int(len(corrected_signal)*0.1):]
noise_std = np.std(noise_region) if np.std(noise_region) > 0 else 0.001

# Gaussian Deconvolution
deconv_signal = np.zeros_like(corrected_signal)
if enable_deconv and len(peaks) > 0:
    guess = []
    for i, p_idx in enumerate(peaks):
        guess.extend([corrected_signal[p_idx], time[p_idx], (widths_half[i] * (time[1]-time[0]))**2])
    try:
        popt, _ = curve_fit(multi_gaussian, time, corrected_signal, p0=guess, maxfev=2000)
        deconv_signal = multi_gaussian(time, *popt)
    except:
        st.sidebar.error("Deconvolution failed to converge.")

# --- Main Dashboard ---
st.title("🔬 Enterprise Chemometrics & Validation Suite")
st.markdown("Automated pipeline for peak deconvolution, baseline correction, and ICH Q2 Pharmacopeial validation.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Peaks Detected", len(peaks))
col2.metric("Baseline Drift Variance", f"{np.var(baseline):.2f}")
col3.metric("System Noise (σ)", f"{noise_std:.4f} mAU")
col4.metric("Deconvolution Status", "Active" if enable_deconv else "Inactive")

tab_2d, tab_3d, tab_metrics = st.tabs(["📈 Chromatogram & Deconvolution", "🌈 3D DAD Spectrogram", "📋 ICH Q2/SST Audit Report"])

with tab_2d:
    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='Raw Detector Signal', line=dict(color='lightgray', width=1)))
    fig_2d.add_trace(go.Scatter(x=time, y=baseline, mode='lines', name=f'{baseline_model}', line=dict(color='red', dash='dash', width=1)))
    fig_2d.add_trace(go.Scatter(x=time, y=corrected_signal, mode='lines', name='Corrected Profile', line=dict(color='blue', width=2)))
    
    if enable_deconv and np.any(deconv_signal):
        fig_2d.add_trace(go.Scatter(x=time, y=deconv_signal, mode='lines', name='Gaussian Mixture Fit', line=dict(color='green', dash='dot', width=2)))
        
    fig_2d.add_trace(go.Scatter(x=time[peaks], y=corrected_signal[peaks], mode='markers', name='Centroids', marker=dict(color='orange', size=8, symbol='x')))
    fig_2d.update_layout(xaxis_title="Retention Time (min)", yaxis_title="Absorbance (mAU)", margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig_2d, use_container_width=True)

with tab_3d:
    if not uploaded_file:
        ds = 5
        fig_3d = go.Figure(data=[go.Surface(z=z_matrix[::ds, :], x=wavelengths, y=time[::ds], colorscale='Plasma')])
        fig_3d.update_layout(scene=dict(xaxis_title='Wavelength (nm)', yaxis_title='Time (min)', zaxis_title='mAU'), margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("3D DAD Visualization requires multi-dimensional UV-Vis data. Uploaded CSV is 1D time-series.")

with tab_metrics:
    if len(peaks) > 0:
        peak_data = []
        for i, peak_idx in enumerate(peaks):
            tr = time[peak_idx]
            w50 = widths_half[i] * (time[1]-time[0])
            height = corrected_signal[peak_idx]
            area = simpson(y=corrected_signal[max(0, peak_idx-30):min(len(time)-1, peak_idx+30)], dx=time[1]-time[0])
            
            # ICH Q2 Validation Metrics
            sn_ratio = height / (2 * noise_std) 
            lod = 3.3 * (noise_std / height) * 100 if height > 0 else 0 # Approx LOD %
            loq = 10.0 * (noise_std / height) * 100 if height > 0 else 0 # Approx LOQ %
            theoretical_plates = 5.54 * (tr / w50)**2 if w50 > 0 else 0
            
            resolution = 1.18 * (tr - time[peaks[i-1]]) / (w50 + (widths_half[i-1]*(time[1]-time[0]))) if i > 0 else 0.0
            
            peak_data.append({
                "Peak": f"P{i+1}",
                "RT (min)": round(tr, 3),
                "Area": round(area, 2),
                "S/N Ratio": round(sn_ratio, 1),
                "LOD Est. (%)": round(lod, 4),
                "LOQ Est. (%)": round(loq, 4),
                "Plates (N)": int(theoretical_plates),
                "Resolution (Rs)": round(resolution, 2) if i > 0 else "-"
            })
            
        report_df = pd.DataFrame(peak_data)
        st.dataframe(report_df, use_container_width=True)
        
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Export Audit-Ready LIMS Report (CSV)", data=csv, file_name='ich_validation_report.csv', mime='text/csv')