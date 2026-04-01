
import re
import math
from pathlib import Path
from typing import Optional

def parse_trappe_resi_atoms(top_file: str, resi_name: str):
    text = Path(top_file).read_text()
    m = re.search(
        rf"RESI\s+{re.escape(resi_name)}\b.*?(?=\nRESI\s+|\Z)",
        text,
        flags=re.S,
    )
    if not m:
        raise ValueError(f"RESI {resi_name} not found in {top_file}")

    block = m.group(0)
    atoms = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("ATOM"):
            parts = line.split()
            if len(parts) >= 3:
                atoms.append((parts[1], parts[2]))  
    if not atoms:
        raise ValueError(f"No ATOM lines found in RESI {resi_name}")
    return atoms


def read_xyz_atoms(xyz_path: str):
    lines = Path(xyz_path).read_text().splitlines()
    n = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""
    atoms = []
    for i in range(2, 2 + n):
        parts = lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Bad XYZ atom line: {lines[i]}")
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((sym, x, y, z))
    return comment, atoms


def write_xyz(xyz_path: str, comment: str, rows):
    out = [str(len(rows)), comment]
    for sp, x, y, z in rows:
        out.append(f"{sp:4s} {x: .8f} {y: .8f} {z: .8f}")
    Path(xyz_path).write_text("\n".join(out) + "\n")


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def order_carbons_linear(carbons, cc_cutoff=1.85):
    n = len(carbons)
    if n == 0:
        raise ValueError("No carbons found")
    if n == 1:
        return [0]

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _dist(carbons[i], carbons[j]) < cc_cutoff:
                adj[i].append(j)
                adj[j].append(i)

    degrees = [len(a) for a in adj]
    ends = [i for i, d in enumerate(degrees) if d == 1]
    if len(ends) != 2:
        raise RuntimeError(f"Not a linear chain. degrees={degrees}, ends={ends}")

    start = ends[0]
    order = [start]
    prev = None
    cur = start
    while len(order) < n:
        nxts = [k for k in adj[cur] if k != prev]
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt

    if len(order) != n:
        raise RuntimeError("Failed to traverse full linear chain.")
    return order


def needs_ua_conversion(resi_name: str) -> bool:
    return re.fullmatch(r"C\d+A", resi_name) is not None


def convert_allatom_xyz_to_trappe_ua_xyz(
    original_xyz: str,
    top_file: str,
    resi_name: str,
    out_xyz: str,
    out_xyz_site: Optional[str] = None,
    cc_cutoff: float = 1.85,
):
    sites = parse_trappe_resi_atoms(top_file, resi_name)
    n_sites = len(sites)

    comment, atoms = read_xyz_atoms(original_xyz)
    carbons = [(x, y, z) for (sym, x, y, z) in atoms if sym.upper() == "C"]

    if len(carbons) != n_sites:
        raise RuntimeError(
            f"Carbon count in xyz={len(carbons)} != UA sites in RESI {resi_name}={n_sites}. "
            f"Converter assumes #sites == #carbons (linear alkanes)."
        )

    order = order_carbons_linear(carbons, cc_cutoff=cc_cutoff)

    rows_site = []
    rows_elem = []
    for idx_site, c_idx in enumerate(order):
        site_name, _ = sites[idx_site]
        x,y,z = carbons[c_idx]
        rows_site.append((site_name, x, y, z))   
        rows_elem.append(("C", x, y, z))         

    if out_xyz_site is not None:
        write_xyz(out_xyz_site, comment, rows_site)
    write_xyz(out_xyz, comment, rows_elem)
