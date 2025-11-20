import streamlit as st
import numpy as np
import pandas as pd

# Import your engine
from Vectorized_Tensored_Quantum_Econometrics import (
    PolicySpace,
    VectorizedPanelModel,
    QuantumPolicySimulator,
    generate_synthetic_irish_agri_panel,
    build_policy_config_qubo
)

st.set_page_config(page_title="Quantum Econometric Simulator", layout="wide")

st.title("🧠⚛️ Quantum–Tensored Econometric Simulator for Irish Agri Policy")
st.markdown("Developed by **Jit** — Powered by Vectorized Tensor Econometrics + QUBO Optimisation + Quantum Policy States.")

tabs = st.tabs(["Model", "Quantum Horizon", "QUBO", "Real Data"])

with tabs[0]:
    # -------------------------------------------------------
    # 1. Controls
    # -------------------------------------------------------

    st.sidebar.header("Simulation Controls")

    T = st.sidebar.slider("Time Periods (Synthetic)", 5, 30, 20)
    H = st.sidebar.slider("Policy Horizon (Steps)", 1, 10, 5)

    st.sidebar.markdown("---")
    st.sidebar.header("Policy Hyperparameters")

    fert_levels = st.sidebar.multiselect(
        "Fertiliser Subsidy Levels",
        [0.0, 0.5, 1.0],
        default=[0.0, 0.5, 1.0],
    )

    carbon_levels = st.sidebar.multiselect(
        "Carbon Tax Levels (€ / tonne)",
        [0.0, 50.0, 100.0],
        default=[0.0, 50.0, 100.0],
    )

    rnd_levels = st.sidebar.multiselect(
        "R&D Support",
        [0.0, 0.3, 0.6],
        default=[0.0, 0.3, 0.6],
    )

    # -------------------------------------------------------
    # 2. Build System from Controls
    # -------------------------------------------------------

    st.subheader("Model Initialization")

    levers = {
        "fertiliser_subsidy": fert_levels,
        "carbon_tax": carbon_levels,
        "R_and_D_support": rnd_levels
    }

    policy_space = PolicySpace(levers)


    # -------------------------------------------------------
    # Data Source Selection
    # -------------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.header("Data Source")

    data_source = st.sidebar.radio(
        "Select Data Source",
        ("Synthetic Panel", "Real GHG Panel (CSV Upload)"),
        index=0
    )

    uploaded_file = None
    if data_source == "Real GHG Panel (CSV Upload)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Irish Copernicus GHG CSV",
            type=["csv"]
        )

    # -------------------------------------------------------
    # 2A. Load Data (Synthetic or Real)
    # -------------------------------------------------------

    def load_real_ghg_panel(df: pd.DataFrame):
        # Ensure required column exists
        if "ghg_kgco2e_per_ha" not in df.columns:
            st.error("CSV must contain column: ghg_kgco2e_per_ha")
            st.stop()

        years = sorted(df["year"].unique())
        counties = sorted(df["county"].unique())

        T_real = len(years)
        R_real = len(counties)
        S = 1  # no sector

        year_to_idx = {y: i for i, y in enumerate(years)}
        county_to_idx = {c: i for i, c in enumerate(counties)}

        # Identify regressors
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols = ["year", "ghg_kgco2e_per_ha"]
        regressor_names = [c for c in num_cols if c not in drop_cols]

        K_real = len(regressor_names)

        y_real = np.zeros((T_real, R_real, S))
        X_real = np.zeros((T_real, R_real, S, K_real))

        for row in df.itertuples(index=False):
            ti = year_to_idx[getattr(row, "year")]
            ri = county_to_idx[getattr(row, "county")]
            y_real[ti, ri, 0] = getattr(row, "ghg_kgco2e_per_ha")
            for k, col in enumerate(regressor_names):
                X_real[ti, ri, 0, k] = getattr(row, col)

        meta_real = {
            "years": years,
            "counties": counties,
            "regressor_names": regressor_names,
            "y_name": "ghg_kgco2e_per_ha",
            "y_desc": "GHG emissions per ha"
        }
        return y_real, X_real, meta_real

    # Decide which dataset to load
    if data_source == "Synthetic Panel":
        y, X, meta = generate_synthetic_irish_agri_panel(T=T)
    else:
        if uploaded_file is None:
            st.warning("Upload a CSV file to proceed with real data.")
            st.stop()
        df_real = pd.read_csv(uploaded_file)
        y, X, meta = load_real_ghg_panel(df_real)

    panel_model = VectorizedPanelModel(use_fixed_effects=True)
    panel_model.fit(y, X)

    st.success("Panel Model Fitted Successfully")
    st.write("**Estimated Coefficients:**")
    st.json({k: float(v) if np.isscalar(v) else v.tolist() 
             for k, v in panel_model.summary().items()})

    # -------------------------------------------------------
    # 3. Policy Sensitivity Matrix
    # -------------------------------------------------------

    regressor_names = meta["regressor_names"]
    K = len(regressor_names)
    P = len(levers)

    def safe_index(name):
        return regressor_names.index(name) if name in regressor_names else None

    idx_lag_y = safe_index("lag_y")
    idx_carbon = safe_index("carbon_price")

    policy_sensitivity = np.zeros((P, K))
    if idx_carbon is not None:
        policy_sensitivity[0, idx_carbon] = -0.5  # fertiliser subsidy lowers carbon cost
        policy_sensitivity[1, idx_carbon] = +1.0  # carbon tax increases carbon_price regressor
    if idx_lag_y is not None:
        policy_sensitivity[2, idx_lag_y] = +0.3   # R&D raises productivity (lag_y)

with tabs[1]:
    # -------------------------------------------------------
    # 4. Simulator
    # -------------------------------------------------------

    simulator = QuantumPolicySimulator(policy_space, panel_model, X, policy_sensitivity)

    U_seq = []
    for t in range(H):
        U_seq.append(policy_space.random_unitary(seed=100 + t))

    results = simulator.simulate_horizon(U_seq)
    policy_hist = results["policy_features"]
    y_hat_hist = results["y_hat"]

    # -------------------------------------------------------
    # 5. Display Policy Features Over Horizon
    # -------------------------------------------------------

    st.subheader("Policy Feature Expectations Over Time")
    df_pol = pd.DataFrame(policy_hist, columns=policy_space.lever_names)
    st.dataframe(df_pol)

    st.line_chart(df_pol)

    # -------------------------------------------------------
    # 6. Show Ŷ Predictions
    # -------------------------------------------------------

    st.subheader("Predicted Outcomes (Example Slice: t=0, region=0)")
    st.write(pd.DataFrame(y_hat_hist[0, 0, :, :]))

    # -------------------------------------------------------
    # 7. Quantum State Vector Visualisation (|psi|^2)
    # -------------------------------------------------------

    if "psi" in results:
        st.subheader("Quantum State Vector Visualisation (|ψ|²)")

        # psi is expected to have shape (H, dim)
        psi_seq = results["psi"]
        psi0 = psi_seq[0]

        prob_vec = np.abs(psi0)**2
        df_probs = pd.DataFrame({
            "State Index": np.arange(len(prob_vec)),
            "Probability": prob_vec
        })

        st.bar_chart(df_probs.set_index("State Index"))

    # -------------------------------------------------------
    # 8. Full Tensor Visualisation (X Tensor)
    # -------------------------------------------------------

    st.subheader("Full Tensor Visualisation (X Tensor)")

    # X has shape (T, R, S, K)
    T_dim, R_dim, S_dim, K_dim = X.shape

    st.write(f"Tensor Shape: **T={T_dim}**, **R={R_dim}**, **S={S_dim}**, **K={K_dim}**")

    # Dropdowns to select tensor slices
    t_sel = st.number_input("Select Time Index (t)", min_value=0, max_value=T_dim - 1, value=0)
    r_sel = st.number_input("Select Region Index (r)", min_value=0, max_value=R_dim - 1, value=0)
    s_sel = st.number_input("Select Sector Index (s)", min_value=0, max_value=S_dim - 1, value=0)

    # Extract the vector X[t, r, s, :]
    tensor_slice = X[int(t_sel), int(r_sel), int(s_sel), :]

    df_tensor = pd.DataFrame({
        "Regressor": regressor_names,
        "Value": tensor_slice
    })

    st.write("Selected Tensor Slice (X[t, r, s, :]):")
    st.dataframe(df_tensor)

    st.bar_chart(df_tensor.set_index("Regressor"))

with tabs[2]:
    # -------------------------------------------------------
    # 7. QUBO Optimisation
    # -------------------------------------------------------

    st.subheader("QUBO Optimal Policy Configuration")

    aggregator = lambda y_hat: -float(np.mean(y_hat))

    qubo_result = simulator.optimise_policy_via_qubo(
        aggregator=aggregator,
        penalty_lambda=10.0,
        max_configs=None
    )

    st.write("**Chosen Config ID:**", qubo_result["chosen_config_id"])
    st.write("**Policy Feature Vector:**")
    st.json(dict(zip(policy_space.lever_names, qubo_result["chosen_policy_features"])))

with tabs[3]:
    st.subheader("Real GHG Data Preview")
    if data_source == "Real GHG Panel (CSV Upload)" and uploaded_file is not None:
        st.write("Uploaded CSV:")
        st.dataframe(df_real.head())
    else:
        st.info("Switch to 'Real GHG Panel' in the sidebar and upload a CSV to preview real data here.")
