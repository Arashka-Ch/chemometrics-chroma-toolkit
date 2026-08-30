# chemometrics-chroma-toolkit
Python dashboard for automated HPLC baseline correction, peak deconvolution, and ICH Q2 analytical method validation.
# 🔬 Enterprise Chemometrics & Analytical Validation Suite

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chemometrics-chroma-toolkit.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

An enterprise-grade, GMP-adjacent computational toolkit engineered for pharmaceutical and cosmeceutical R&D laboratories. This application automates the processing of complex Diode Array Detector (DAD) multi-wavelength data, mathematically resolves co-eluting impurities, and dynamically calculates critical validation parameters compliant with **ICH Q2** and **Pharmacopeial (USP/EP)** guidelines.

## 💡 Technical Architecture & Capabilities
*   **Gaussian Peak Deconvolution:** Employs advanced `scipy.optimize` non-linear curve fitting (Gaussian Mixture Models) to mathematically separate hidden, co-eluting impurity peaks that cannot be resolved via standard chromatographic methods.
*   **Algorithmic Model Selection:** Allows analytical chemists to toggle between Asymmetric Least Squares (ALS) sparse matrix regression or iterative polynomial fitting to detrend severely wandering baselines.
*   **ICH Q2 Analytical Validation:** Automatically calculates system noise ($\sigma$) to rigorously report **Signal-to-Noise ($S/N$)**, estimated **Limit of Detection (LOD)**, and **Limit of Quantitation (LOQ)**.
*   **System Suitability Testing (SST):** Real-time generation of USP Theoretical Plates ($N$), Peak Resolution ($R_s$), and Full Width at Half Maximum (FWHM).
*   **3D DAD Spectrogram Visualization:** Interactively renders multidimensional time-wavelength-absorbance matrices via Plotly to evaluate peak purity.

## 🚀 Deployment Instructions

This repository is optimized for deployment in isolated computational environments (such as Ubuntu WSL via Miniforge/Mamba).

**1. Clone the Repository**
```bash
git clone https://github.com/Arashka-Ch/chemometrics-chroma-toolkit.git
cd chemometrics-chroma-toolkit
```

**2. Provision the Environment**
```bash
mamba create -n chroma_env python=3.10 -y
mamba activate chroma_env
pip install -r requirements.txt
```

**3. Launch the Application**
```bash
streamlit run app.py
```
