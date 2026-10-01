#!/usr/bin/env python3
"""Protein Studio - live protein structure editor (Flask + Biopython + 3Dmol.js)."""

import copy
import io
import math
import os
import urllib.request

import numpy as np
from Bio.PDB import PDBIO, NeighborSearch, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.vectors import calc_dihedral
from flask import Flask, Response, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
S = {"st": None, "undo": [], "redo": [], "name": ""}
KEEP = {"N", "CA", "C", "O", "OXT", "CB"}


# ---------- structure helpers ----------
def mdl():
    return S["st"][0]


def residues(ch):
    return [r for r in mdl()[ch] if r.id[0] == " "]


def get_res(ch, rn):
    for r in residues(ch):
        if r.id[1] == rn:
            return r
    raise ValueError(f"Residue {ch}{rn} not found")


def linked(a, b):
    return "C" in a and "N" in b and np.linalg.norm(a["C"].coord - b["N"].coord) < 2.0


def dih(*at):
    return math.degrees(calc_dihedral(*[x.get_vector() for x in at]))


def phi_psi(rs, i):
    r, phi, psi = rs[i], None, None
    if not all(k in r for k in ("N", "CA", "C")):
        return None, None
    if i and linked(rs[i - 1], r):
        phi = dih(rs[i - 1]["C"], r["N"], r["CA"], r["C"])
    if i + 1 < len(rs) and linked(r, rs[i + 1]):
        psi = dih(r["N"], r["CA"], r["C"], rs[i + 1]["N"])
    return phi, psi


def rotate(atoms, origin, axis, deg):
    a, k = math.radians(deg), axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)  # Rodrigues
    for at in atoms:
        at.coord = (R @ (at.coord - origin) + origin).astype("f4")


# ---------- edit operations ----------
def set_angle(ch, rn, which, target):
    """Set backbone phi/psi by rigidly rotating the shorter side of the chain."""
    rs = residues(ch)
    i = next(k for k, r in enumerate(rs) if r.id[1] == rn)
    r = rs[i]
    cur = phi_psi(rs, i)[0 if which == "phi" else 1]
    if cur is None:
        raise ValueError(f"{which} is undefined for {ch}{rn} (chain end or gap)")
    if which == "phi":
        a, b = r["N"], r["CA"]
        head = [x for x in r if x.name not in ("N", "H", "HN")]
    else:
        a, b = r["CA"], r["C"]
        head = [x for x in r if x.name in ("O", "OXT")]
    moving = head + [x for q in rs[i + 1 :] for x in q]
    skip = {id(x) for x in moving} | {id(a), id(b)}
    other = [x for q in rs for x in q if id(x) not in skip]
    axis, origin, d = b.coord - a.coord, b.coord.copy(), target - cur
    if len(other) < len(moving):
        rotate(other, origin, axis, -d)
    else:
        rotate(moving, origin, axis, d)


def mutate(ch, rn, new):
    """Backbone-preserving mutation: keeps N/CA/C/O/CB, drops the old side chain."""
    r = get_res(ch, rn)
    for at in [x for x in r if x.name not in KEEP]:
        r.detach_child(at.id)
    if new == "GLY" and "CB" in r:
        r.detach_child("CB")
    elif new != "GLY" and "CB" not in r:  # place ideal Cb
        n, ca, c = (r[k].coord for k in ("N", "CA", "C"))
        b, cc = ca - n, c - ca
        a = np.cross(b, cc)
        xyz = -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca
        r.add(Atom("CB", xyz.astype("f4"), 0.0, 1.0, " ", " CB ", 0, "C"))
    r.resname = new


def delete(ch, rn):
    r = get_res(ch, rn)
    mdl()[ch].detach_child(r.id)


# ---------- analysis ----------
def mass(a):
    try:
        return a.mass
    except Exception:
        return 12.0


def stats():
    atoms = list(mdl().get_atoms())
    xyz = np.array([a.coord for a in atoms])
    m = np.array([mass(a) for a in atoms])
    com = (xyz * m[:, None]).sum(0) / m.sum()
    rg = math.sqrt((m * ((xyz - com) ** 2).sum(1)).sum() / m.sum())
    bad = set()
    for a, b in NeighborSearch(atoms).search_all(2.2):
        ra, rb = a.get_parent(), b.get_parent()
        same = ra.get_parent() is rb.get_parent()
        if ra is rb or (same and abs(ra.id[1] - rb.id[1]) < 2):
            continue
        if a.name == b.name == "SG" or ra.id[0] == "W" or rb.id[0] == "W":
            continue
        bad.add((ra.get_parent().id, ra.id[1]))
        bad.add((rb.get_parent().id, rb.id[1]))
    rama, nres = [], 0
    for ch in mdl():
        rs = residues(ch.id)
        nres += len(rs)
        for i, r in enumerate(rs):
            p, s = phi_psi(rs, i)
            if p is not None and s is not None:
                rama.append([ch.id, r.id[1], r.resname, round(p, 1), round(s, 1)])
    return {
        "residues": nres,
        "atoms": len(atoms),
        "mass": round(m.sum() / 1000, 2),
        "rg": round(rg, 2),
        "clashes": len(bad),
        "clash_res": sorted(bad)[:60],
        "rama": rama,
    }


def snapshot():
    S["undo"].append(copy.deepcopy(S["st"]))
    del S["undo"][:-20]
    S["redo"].clear()


def payload():
    out = io.StringIO()
    io_ = PDBIO()
    io_.set_structure(S["st"])
    io_.save(out)
    return jsonify(
        pdb=out.getvalue(),
        stats=stats(),
        name=S["name"],
        undo=len(S["undo"]),
        redo=len(S["redo"]),
    )


def fail(e):
    return jsonify(error=str(e)), 400


# ---------- routes ----------
@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.post("/api/load")
def load():
    b = request.get_json(force=True)
    try:
        if b.get("pdb_id"):
            pid = b["pdb_id"].strip().upper()
            url = f"https://files.rcsb.org/download/{pid}.pdb"
            text = urllib.request.urlopen(url, timeout=20).read().decode()
            S["name"] = pid
        else:
            text, S["name"] = b["text"], b.get("name", "upload")
        st = PDBParser(QUIET=True).get_structure("p", io.StringIO(text))
        for extra in list(st)[1:]:
            st.detach_child(extra.id)
        S.update(st=st, undo=[], redo=[])
        return payload()
    except Exception as e:
        return fail(f"Could not load structure: {e}")


@app.post("/api/edit")
def edit():
    b = request.get_json(force=True)
    op = b["op"]
    try:
        if op in ("undo", "redo"):
            src, dst = (
                (S["undo"], S["redo"]) if op == "undo" else (S["redo"], S["undo"])
            )
            if not src:
                raise ValueError(f"Nothing to {op}")
            dst.append(S["st"])
            S["st"] = src.pop()
        else:
            if S["st"] is None:
                raise ValueError("Load a structure first")
            if b.get("snap", True):
                snapshot()
            if op == "angle":
                for w in ("phi", "psi"):
                    if b.get(w) is not None:
                        set_angle(b["chain"], b["rn"], w, float(b[w]))
            elif op == "mutate":
                mutate(b["chain"], b["rn"], b["to"])
            elif op == "delete":
                delete(b["chain"], b["rn"])
            else:
                raise ValueError(f"Unknown operation {op}")
        return payload()
    except Exception as e:
        return fail(e)


@app.get("/api/export")
def export():
    out = io.StringIO()
    w = PDBIO()
    w.set_structure(S["st"])
    w.save(out)
    return Response(
        out.getvalue(),
        mimetype="chemical/x-pdb",
        headers={"Content-Disposition": f"attachment; filename={S['name']}_edited.pdb"},
    )


if __name__ == "__main__":
    print("Protein Studio running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=False)
