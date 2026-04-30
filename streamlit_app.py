from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR: Path = Path(__file__).resolve().parent
DATA_PATH: Path = BASE_DIR / "Mall_Customers.csv"
MODEL_PATH: Path = BASE_DIR / "kmeans_mall_customers.joblib"
FEATURES: list[str] = ["Annual Income (k$)", "Spending Score (1-100)"]


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontro el archivo: {path.name}")
    return pd.read_csv(path)


def calculate_elbow_and_silhouette(x_scaled: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for k in range(k_min, k_max + 1):
        model: KMeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels: np.ndarray = model.fit_predict(x_scaled)
        inertia: float = float(model.inertia_)
        silhouette: float = float(silhouette_score(x_scaled, labels))
        records.append({"k": float(k), "inertia": inertia, "silhouette": silhouette})
    metrics_df: pd.DataFrame = pd.DataFrame(records)
    metrics_df["k"] = metrics_df["k"].astype(int)
    return metrics_df


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "No se encontro el modelo exportado. Ejecuta primero la celda final del notebook para crear el .joblib."
        )
    bundle: Any = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Formato de modelo invalido: se esperaba un diccionario con model y scaler.")
    required_keys: set[str] = {"model", "scaler", "features"}
    if not required_keys.issubset(bundle.keys()):
        raise KeyError("El .joblib no contiene las claves requeridas: model, scaler y features.")
    return bundle


def app() -> None:
    st.set_page_config(page_title="Segmentacion de Clientes - K-Means", layout="wide")
    st.title("Segmentacion de Clientes con K-Means")
    st.caption("Mall Customers Dataset - Metodo del codo, silueta y clasificacion de nuevo cliente.")

    try:
        df: pd.DataFrame = load_data(DATA_PATH)
    except Exception as exc:
        st.error(f"Error al cargar datos: {exc}")
        return

    x_raw: pd.DataFrame = df[FEATURES].copy()
    scaler_preview: StandardScaler = StandardScaler()
    x_scaled_preview: np.ndarray = scaler_preview.fit_transform(x_raw)
    metrics_df: pd.DataFrame = calculate_elbow_and_silhouette(x_scaled_preview)
    best_k_by_silhouette: int = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metodo del Codo")
        fig_elbow, ax_elbow = plt.subplots(figsize=(7, 4))
        ax_elbow.plot(metrics_df["k"], metrics_df["inertia"], marker="o")
        ax_elbow.set_xlabel("Numero de clusters (K)")
        ax_elbow.set_ylabel("Inercia (WCSS)")
        ax_elbow.set_title("Curva del codo")
        ax_elbow.grid(alpha=0.3)
        fig_elbow.tight_layout()
        st.pyplot(fig_elbow)
        plt.close(fig_elbow)

    with col2:
        st.subheader("Coeficiente de Silueta")
        fig_sil, ax_sil = plt.subplots(figsize=(7, 4))
        ax_sil.plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="#2ca02c")
        ax_sil.axvline(best_k_by_silhouette, color="red", linestyle="--", label=f"Mejor K = {best_k_by_silhouette}")
        ax_sil.set_xlabel("Numero de clusters (K)")
        ax_sil.set_ylabel("Silhouette Score")
        ax_sil.set_title("Silueta por numero de clusters")
        ax_sil.legend(loc="best")
        ax_sil.grid(alpha=0.3)
        fig_sil.tight_layout()
        st.pyplot(fig_sil)
        plt.close(fig_sil)

    st.markdown(f"**Sugerencia automatica de K (por silueta):** {best_k_by_silhouette}")
    selected_k: int = st.slider("Selecciona K para visualizar clusters", min_value=2, max_value=10, value=best_k_by_silhouette)

    scaler_fit: StandardScaler = StandardScaler()
    x_scaled_fit: np.ndarray = scaler_fit.fit_transform(x_raw)
    kmeans_fit: KMeans = KMeans(n_clusters=selected_k, random_state=42, n_init=20)
    clusters_fit: np.ndarray = kmeans_fit.fit_predict(x_scaled_fit)

    plot_df: pd.DataFrame = x_raw.copy()
    plot_df["Cluster"] = clusters_fit
    centroids_scaled: np.ndarray = kmeans_fit.cluster_centers_
    centroids_original: np.ndarray = scaler_fit.inverse_transform(centroids_scaled)

    st.subheader("Clusters generados")
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5))
    scatter = ax_cluster.scatter(
        plot_df["Annual Income (k$)"],
        plot_df["Spending Score (1-100)"],
        c=plot_df["Cluster"],
        cmap="tab10",
        alpha=0.7,
    )
    ax_cluster.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        c="black",
        marker="X",
        s=220,
        label="Centroides",
    )
    ax_cluster.set_xlabel("Annual Income (k$)")
    ax_cluster.set_ylabel("Spending Score (1-100)")
    ax_cluster.set_title(f"Visualizacion de clientes segmentados (K={selected_k})")
    ax_cluster.legend(loc="best")
    ax_cluster.grid(alpha=0.2)
    fig_cluster.colorbar(scatter, ax=ax_cluster, label="Cluster")
    fig_cluster.tight_layout()
    st.pyplot(fig_cluster)
    plt.close(fig_cluster)

    st.subheader("Clasificar nuevo cliente")
    with st.form("new_customer_form"):
        annual_income: float = st.number_input("Annual Income (k$)", min_value=0.0, value=60.0, step=1.0)
        spending_score: float = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        submit_predict: bool = st.form_submit_button("Clasificar cliente")

    if submit_predict:
        new_customer_df: pd.DataFrame = pd.DataFrame(
            [{"Annual Income (k$)": annual_income, "Spending Score (1-100)": spending_score}]
        )
        new_customer_scaled: np.ndarray = scaler_fit.transform(new_customer_df)
        predicted_cluster: int = int(kmeans_fit.predict(new_customer_scaled)[0])
        st.success(f"El nuevo cliente pertenece al cluster: {predicted_cluster}")

        st.subheader("Ubicacion de la prediccion en la grafica de clusters")
        fig_prediction, ax_prediction = plt.subplots(figsize=(10, 5))
        scatter_prediction = ax_prediction.scatter(
            plot_df["Annual Income (k$)"],
            plot_df["Spending Score (1-100)"],
            c=plot_df["Cluster"],
            cmap="tab10",
            alpha=0.55,
            label="Clientes del dataset",
        )
        ax_prediction.scatter(
            centroids_original[:, 0],
            centroids_original[:, 1],
            c="black",
            marker="X",
            s=220,
            label="Centroides",
        )
        ax_prediction.scatter(
            annual_income,
            spending_score,
            c="red",
            marker="*",
            s=380,
            edgecolors="white",
            linewidths=1.2,
            label=f"Nuevo cliente (Cluster {predicted_cluster})",
        )
        ax_prediction.set_xlabel("Annual Income (k$)")
        ax_prediction.set_ylabel("Spending Score (1-100)")
        ax_prediction.set_title(f"Cliente predicho sobre clusters (K={selected_k})")
        ax_prediction.grid(alpha=0.2)
        ax_prediction.legend(loc="best")
        fig_prediction.colorbar(scatter_prediction, ax=ax_prediction, label="Cluster")
        fig_prediction.tight_layout()
        st.pyplot(fig_prediction)
        plt.close(fig_prediction)

    st.subheader("Uso del modelo exportado (.joblib)")
    try:
        bundle: dict[str, Any] = load_bundle(MODEL_PATH)
        model: KMeans = bundle["model"]
        scaler: StandardScaler = bundle["scaler"]
        st.info(
            f"Modelo cargado desde `{MODEL_PATH.name}` con K={model.n_clusters}. "
            f"Features: {', '.join(bundle['features'])}"
        )

        demo_df: pd.DataFrame = pd.DataFrame(
            [{"Annual Income (k$)": 75.0, "Spending Score (1-100)": 82.0}]
        )
        demo_scaled: np.ndarray = scaler.transform(demo_df)
        demo_cluster: int = int(model.predict(demo_scaled)[0])
        st.write(f"Ejemplo con modelo exportado: cliente (75, 82) -> cluster {demo_cluster}")
    except Exception as exc:
        st.warning(f"Aun no se pudo cargar el .joblib exportado: {exc}")


if __name__ == "__main__":
    app()
