"""Exporta el gráfico por año y una demo HTML que se puede abrir sin servidor."""

from base64 import b64encode
from html import escape
import json
from pathlib import Path
from string import Template

from matplotlib.figure import Figure
import pandas as pd


COLORS = ("#576c83", "#ae481a", "#076b55")


def write_year_plot(path: Path, summary: dict) -> None:
    years = list(summary["test_by_year"])
    figure = Figure(figsize=(8.6, 3.7), layout="constrained")
    axis = figure.subplots()
    for (model, color) in zip(summary["test"]["models"], COLORS):
        errors = [summary["test_by_year"][year]["models"][model]["mae_s"] for year in years]
        axis.plot(years, errors, label=model, color=color, marker="o", linewidth=2)
    axis.set_xticks(years, [f"{year}\n(n={summary['test_by_year'][year]['rows']})" for year in years])
    axis.set_ylabel("MAE (segundos)", fontsize=12)
    axis.tick_params(labelsize=11)
    axis.set_ylim(bottom=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="best", frameon=False, fontsize=10)
    figure.savefig(path, dpi=145)


def write_report(output: Path, summary: dict, test: pd.DataFrame) -> None:
    plot = output / "mae_by_year.png"
    write_year_plot(plot, summary)
    table_rows = []
    for model, metrics in summary["test"]["models"].items():
        mae = f"{metrics['mae_s']:.2f} s".replace(".", ",")
        coverage = f"{metrics['coverage']:.1%}".replace(".", ",") if "coverage" in metrics else "&mdash;"
        width = f"{metrics['mean_width_s']:.1f} s".replace(".", ",") if "mean_width_s" in metrics else "&mdash;"
        table_rows.append(
            f'<tr><th scope="row">{escape(model)}</th><td>{mae}</td>'
            f"<td>{coverage}</td><td>{width}</td></tr>"
        )
    columns = ["artist", "song", "year", "actual_s", "mlp_p10_s", "mlp_p50_s", "mlp_p90_s"]
    songs = test[columns].sort_values(["artist", "song", "year"], key=lambda item: item.astype(str).str.casefold())
    records = json.dumps(songs.to_dict(orient="records"), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Template(Path(__file__).with_name("report.html").read_text(encoding="utf-8"))
    (output / "report.html").write_text(template.substitute(
        test_first=summary["splits"]["test"]["years"][0],
        test_last=summary["splits"]["test"]["years"][1],
        test_rows=summary["test"]["rows"],
        table_rows="".join(table_rows),
        plot=b64encode(plot.read_bytes()).decode("ascii"),
        records=records,
    ), encoding="utf-8")
