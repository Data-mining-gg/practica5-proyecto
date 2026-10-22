import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def columnas_numericas(df: pl.DataFrame) -> list[str]:
    return [
        col
        for col, dtype in df.schema.items()
        if dtype.is_numeric() and col != "datetime"
    ]


def porcion_entrenamiento(
    df: pl.DataFrame, prop_entrenamiento: float = 0.8
) -> pl.DataFrame:
    df = df.sort("datetime")
    corte_entrenamiento = int(df.height * prop_entrenamiento)
    return df.slice(0, corte_entrenamiento)


def verificar_estandarizacion(
    df: pl.DataFrame, prop_entrenamiento: float = 0.8
) -> pl.DataFrame:
    df_entrenamiento = porcion_entrenamiento(df, prop_entrenamiento)
    numericas = columnas_numericas(df_entrenamiento)

    return pl.DataFrame(
        {
            "variable": numericas,
            "media": [df_entrenamiento[col].mean() for col in numericas],
            "desviacion": [df_entrenamiento[col].std() for col in numericas],
        }
    )


def ajustar_pca(df: pl.DataFrame, prop_entrenamiento: float = 0.8) -> PCA:
    df_entrenamiento = porcion_entrenamiento(df, prop_entrenamiento)
    numericas = columnas_numericas(df_entrenamiento)

    return PCA().fit(df_entrenamiento.select(numericas).to_numpy())


def tabla_varianza(pca: PCA) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "componente": [f"PC{i + 1}" for i in range(pca.n_components_)],
            "valor_propio": pca.explained_variance_,
            "varianza_explicada": pca.explained_variance_ratio_,
            "varianza_acumulada": np.cumsum(pca.explained_variance_ratio_),
        }
    )


def seleccionar_componentes(pca: PCA, umbral: float = 0.8) -> int:
    acumulada = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(acumulada, umbral) + 1)


def graficar_scree(pca: PCA, umbral: float = 0.8):
    componentes = np.arange(1, pca.n_components_ + 1)
    acumulada = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(9, 5))
    plt.bar(
        componentes,
        pca.explained_variance_ratio_,
        alpha=0.7,
        label="Varianza explicada",
    )
    plt.plot(
        componentes, acumulada, marker="o", color="tab:red", label="Varianza acumulada"
    )
    plt.axhline(umbral, linestyle="--", color="gray", label=f"Umbral {umbral:.0%}")
    plt.xticks(componentes, [f"PC{i}" for i in componentes])
    plt.ylabel("Proporción de varianza")
    plt.title("Scree plot", fontsize=13, pad=15)
    plt.legend()
    plt.tight_layout()
    plt.show()


def tabla_loadings(pca: PCA, variables: list[str], n_componentes: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "variable": variables,
            **{f"PC{i + 1}": pca.components_[i] for i in range(n_componentes)},
        }
    )


def graficar_loadings(loadings: pl.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        loadings.drop("variable").to_numpy(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        xticklabels=loadings.drop("variable").columns,
        yticklabels=loadings["variable"].to_list(),
    )
    plt.title("Loadings de los componentes retenidos", fontsize=13, pad=15)
    plt.tight_layout()
    plt.show()


def proyectar(df: pl.DataFrame, pca: PCA, n_componentes: int) -> pl.DataFrame:
    numericas = columnas_numericas(df)
    proyeccion = pca.transform(df.select(numericas).to_numpy())[:, :n_componentes]

    return pl.DataFrame(proyeccion, schema=[f"pc{i + 1}" for i in range(n_componentes)])


def graficar_proyeccion_2d(
    df: pl.DataFrame, pca: PCA, n_muestra: int = 20000, semilla: int = 42
):
    muestra = df.sample(n=min(n_muestra, df.height), seed=semilla)
    proyeccion = proyectar(muestra, pca, 2)
    horas = muestra["datetime"].dt.hour().to_numpy()

    plt.figure(figsize=(9, 7))
    dispersion = plt.scatter(
        proyeccion["pc1"], proyeccion["pc2"], c=horas, cmap="twilight", s=4, alpha=0.5
    )
    plt.colorbar(dispersion, label="Hora del día")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("Proyección en 2D de los componentes principales", fontsize=13, pad=15)
    plt.tight_layout()
    plt.show()


def incorporar_componentes(
    df: pl.DataFrame, pca: PCA, n_componentes: int
) -> pl.DataFrame:
    df = df.sort("datetime")
    return pl.concat([df, proyectar(df, pca, n_componentes)], how="horizontal")
