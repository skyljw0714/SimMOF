import os
import math
from typing import Dict, Any
from config import working_dir


class ZeoppOutputAgent:

    @staticmethod
    def _read_res_file(mof: str, work_dir: str) -> dict:
        res_path = os.path.join(work_dir, f"{mof}.res")
        with open(res_path, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            return {
                "included_sphere": float(parts[1]),
                "free_sphere": float(parts[2]),
                "included_sphere_along_free_path": float(parts[3]),
            }

    @staticmethod
    def _read_vol_file(mof: str, work_dir: str) -> dict:
        vol_path = os.path.join(work_dir, f"{mof}.vol")
        with open(vol_path, "r") as f:
            for line in f:
                if line.startswith("@"):
                    values = line.strip().split()
                    return {
                        "AV_A3": float(values[7]),
                        "AV_Volume_fraction": float(values[9]),
                        "AV_cm3_g": float(values[11]),
                    }
        return {}

    @staticmethod
    def _read_sa_file(mof: str, work_dir: str) -> dict:
        sa_path = os.path.join(work_dir, f"{mof}.sa")
        with open(sa_path, "r") as f:
            for line in f:
                if line.startswith("@"):
                    values = line.strip().split()
                    return {
                        "ASA_A2": float(values[7]),
                        "ASA_m2_cm3": float(values[9]),
                        "ASA_m2_g": float(values[11]),
                    }
        return {}

    @staticmethod
    def _weighted_quantile(values, weights, quantile):
        total = float(sum(weights))
        if total <= 0.0:
            return None
        threshold = quantile * total
        cumulative = 0.0
        for value, weight in zip(values, weights):
            cumulative += float(weight)
            if cumulative >= threshold:
                return float(value)
        return float(values[-1]) if values else None

    @classmethod
    def _read_psd_file(cls, mof: str, work_dir: str) -> dict:
        exact_path = os.path.join(work_dir, f"{mof}.psd_histo")
        if os.path.exists(exact_path):
            psd_path = exact_path
        else:
            candidates = [
                os.path.join(work_dir, name)
                for name in os.listdir(work_dir)
                if name.endswith(".psd_histo")
            ]
            if not candidates:
                raise FileNotFoundError(f"No Zeo++ PSD histogram found in {work_dir}")
            psd_path = max(candidates, key=os.path.getmtime)

        metadata = {}
        pore_size_A = []
        counts = []
        cumulative = []
        derivative = []
        in_table = False

        metadata_fields = {
            "Bin size (A)": ("bin_size_A", float),
            "Number of bins": ("number_of_bins", int),
            "From": ("range_from_A", float),
            "To": ("range_to_A", float),
            "Total samples": ("total_samples", int),
            "Accessible samples": ("accessible_samples", int),
            "Fraction of sample points in node spheres": (
                "fraction_sample_points_in_node_spheres",
                float,
            ),
            "Fraction of sample points outside node spheres": (
                "fraction_sample_points_outside_node_spheres",
                float,
            ),
        }

        with open(psd_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.lower().startswith("bin count cumulative_dist derivative_dist"):
                    in_table = True
                    continue
                if not in_table:
                    for label, (key, converter) in metadata_fields.items():
                        prefix = f"{label}:"
                        if line.startswith(prefix):
                            value = line[len(prefix):].strip()
                            metadata[key] = converter(float(value)) if converter is int else converter(value)
                            break
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    size, count, cumulative_value, derivative_value = map(float, parts[:4])
                except ValueError:
                    continue
                pore_size_A.append(size)
                counts.append(count)
                cumulative.append(cumulative_value)
                derivative.append(derivative_value)

        if not pore_size_A:
            raise RuntimeError(f"No PSD histogram rows parsed from {psd_path}")

        total_count = float(sum(counts))
        nonzero_indices = [i for i, count in enumerate(counts) if count > 0.0]
        count_mode_index = max(range(len(counts)), key=lambda i: counts[i])
        derivative_mode_index = max(range(len(derivative)), key=lambda i: derivative[i])
        if total_count > 0.0:
            mean = sum(size * count for size, count in zip(pore_size_A, counts)) / total_count
            variance = (
                sum(count * (size - mean) ** 2 for size, count in zip(pore_size_A, counts))
                / total_count
            )
            std = math.sqrt(max(variance, 0.0))
        else:
            mean = None
            std = None

        total_samples = metadata.get("total_samples")
        accessible_samples = metadata.get("accessible_samples")
        accessible_fraction = (
            float(accessible_samples / total_samples)
            if total_samples and accessible_samples is not None
            else None
        )

        return {
            "file": psd_path,
            "metadata": metadata,
            "summary": {
                "n_histogram_rows": len(pore_size_A),
                "histogram_count_sum": total_count,
                "accessible_sample_fraction": accessible_fraction,
                "mode_by_count_A": float(pore_size_A[count_mode_index]),
                "mode_by_derivative_A": float(pore_size_A[derivative_mode_index]),
                "mean_by_count_A": float(mean) if mean is not None else None,
                "std_by_count_A": float(std) if std is not None else None,
                "p10_by_count_A": cls._weighted_quantile(pore_size_A, counts, 0.10),
                "p50_by_count_A": cls._weighted_quantile(pore_size_A, counts, 0.50),
                "p90_by_count_A": cls._weighted_quantile(pore_size_A, counts, 0.90),
                "min_nonzero_bin_A": (
                    float(pore_size_A[nonzero_indices[0]]) if nonzero_indices else None
                ),
                "max_nonzero_bin_A": (
                    float(pore_size_A[nonzero_indices[-1]]) if nonzero_indices else None
                ),
            },
            "histogram": {
                "pore_size_A": pore_size_A,
                "count": counts,
                "cumulative_distribution": cumulative,
                "derivative_distribution": derivative,
            },
            "note": (
                "Zeo++ sampled pore-size histogram. Peak locations depend on the probe/channel "
                "radius, Monte Carlo sample count, and Zeo++ PSD algorithm."
            ),
        }

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        work_dir   = context.get("work_dir", working_dir)
        zeopp_info = context.get("zeopp_info", {})
        results    = context.setdefault("results", {})

        

        if results.get("zeopp_status") != "ok":
            print("[ZeoppOutputAgent] zeopp_status != ok -> skipping parsing")
            return context

        mof     = zeopp_info.get("MOF")
        command = zeopp_info.get("command", "")

        if not mof:
            print("[ZeoppOutputAgent] ERROR: MOF name missing in zeopp_info.")
            results["zeopp_status"] = "output_missing_mof"
        elif "-psd" in command:
            parsed = self._read_psd_file(mof, work_dir)
            prop_type = "pore_size_distribution"
        elif "-res" in command:
            parsed = self._read_res_file(mof, work_dir)
            prop_type = "pore_diameter"
        elif "-vol" in command:
            parsed = self._read_vol_file(mof, work_dir)
            prop_type = "accessible_volume"
        elif "-sa" in command:
            parsed = self._read_sa_file(mof, work_dir)
            prop_type = "surface_area"
        else:
            print("[ZeoppOutputAgent] WARNING: unknown command type, no parser matched.")
            parsed = {}
            prop_type = "unknown"

        results["zeopp"] = {
            "type": prop_type,
            "mof": mof,
            "command": command,
            "raw": parsed,
        }

        return context
