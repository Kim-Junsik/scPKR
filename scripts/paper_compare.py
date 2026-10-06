"""The published tables, transcribed, with this model's row added.

The numbers for the other models are TRANSCRIBED FROM THE PAPERS' OWN TABLES and are
not reproducible from this repository - nothing here can recompute them. They live in
source so that a table can be redrawn without the screenshots, which is how they were
lost once already.

scPKR's row is measured: scripts/paper_table.py over the runs in results/final, the
sample gate with the gamma realisation at 4,096 cells, so that the L2 and the five
cell-eval columns beside it come from one prediction rather than two. The mean the
model states - the soft gate, which is what an L2 between means asks for - is carried
separately in SOFT_L2 and is the lower of the two everywhere.
"""
import argparse

LOWER = {"L2", "MSE", "MAE"}
COLUMNS = ["L2", "MSE", "MAE", "DE-Spearman", "Pearson d", "DS",
           "Pearson dhat", "Pearson dhat20"]

NA = None  # the papers print N.A.; we print it the same way
MISSING = "miss"  # a column this repository has not measured

TABLES = {
"Table 1  Norman additive split": [
 ("Control",    [3.9937, 0.01839, 0.03953, NA,      NA,      0.5135, -0.1695, -0.1297]),
 ("Additive",   [1.9395, 0.00448, 0.02276, 0.5564,  0.9024,  0.9686,  0.8584,  0.9244]),
 ("scGPT",      [3.4112, 0.01349, 0.03796, 1.07e-5, 0.5304,  0.5404,  0.2165,  0.2414]),
 ("Geneformer", [1.9132, 0.00410, 0.02360, 0.3741,  0.7732,  0.8241, -0.0078,  0.2239]),
 ("GEARS",      [3.5531, 0.01387, 0.06624, 0.5624,  0.7421,  0.8601, -0.0089,  0.2032]),
 ("CPA",        [5.7629, 0.03435, 0.07894, 0.0713,  0.3845,  0.6021, -0.0039,  0.2254]),
 ("STATE",     [17.3330, 0.30059, 0.24705, 0.5288, -0.0108,  0.5135, -0.0069,  0.2515]),
 ("scDFM",      [1.7043, 0.00315, 0.02155, 0.5705,  0.8853,  0.9737,  0.8468,  0.9260]),
 ("CellFlow",   [1.7064, 0.00392, 0.02207, 0.5503,  0.8678,  0.9321,  0.8395,  0.8988]),
 ("scPKR (ours)",
                [1.6515, 0.00395, 0.02239, 0.5210,  0.8740,  0.9419, MISSING, MISSING]),
],
"Table 2  Norman holdout split - Single": [
 ("Control",    [2.6834, 0.0095, 0.0263, NA,      NA,     0.5217,  0.1618,  0.1982]),
 ("scGPT",      [2.5007, 0.0080, 0.0259, -0.1139, 0.4503, 0.5680,  0.0747,  0.0798]),
 ("GEARS",      [2.5641, 0.0105, 0.0466,  0.3569, 0.6646, 0.8271,  0.6356,  0.7914]),
 ("Geneformer", [1.6962, 0.0036, 0.0191,  0.3566, 0.6075, 0.8070,  0.5620,  0.6513]),
 ("CPA",        [5.8060, 0.0356, 0.0853,  0.1168, 0.2837, 0.5796, -0.0028,  0.0802]),
 ("STATE",     [18.2543, 0.3333, 0.2693,  0.6116, 0.0004, 0.5236,  0.0154,  0.2386]),
 ("scDFM",      [1.6186, 0.0030, 0.0190,  0.6957, 0.7127, 0.8914,  0.6659,  0.8116]),
 ("CellFlow",   [1.6758, 0.0035, 0.0191,  0.2860, 0.7109, 0.8072,  0.6138,  0.6753]),
 ("scPKR (ours)",
                [1.4610, 0.00308, 0.01663, 0.5190, 0.7325, 0.8984, MISSING, MISSING]),
],
"Table 2  Norman holdout split - Double": [
 ("Control",    [4.1882, 0.0207, 0.0423, NA,      NA,     0.5322, -0.1303, -0.0265]),
 ("scGPT",      [3.5171, 0.0153, 0.0362, -0.0665, 0.5693, 0.5578,  0.2814,  0.2652]),
 ("GEARS",      [3.7458, 0.0156, 0.0708,  0.2543, 0.7552, 0.8766,  0.6407,  0.8413]),
 ("Geneformer", [2.0819, 0.0050, 0.0237,  0.3468, 0.7361, 0.8067,  0.6245,  0.7261]),
 ("CPA",        [5.7891, 0.0357, 0.0796,  0.3652, 0.4176, 0.6311,  0.2432,  0.2870]),
 ("STATE",     [18.4458, 0.3404, 0.2733,  0.4071, 0.0061, 0.5289, -0.0023,  0.2580]),
 ("scDFM",      [2.0309, 0.0047, 0.0235,  0.5676, 0.8357, 0.9189,  0.7769,  0.8688]),
 ("CellFlow",   [2.1042, 0.0049, 0.0236,  0.5074, 0.8095, 0.8622,  0.6780,  0.7155]),
 ("scPKR (ours)",
                [2.0458, 0.00576, 0.02580, 0.5130, 0.8215, 0.8592, MISSING, MISSING]),
],
"Table 3  ComboSciPlex": [
 ("Control",    [5.3716, 0.0324, 0.0698, NA,      NA,     0.5714, MISSING, MISSING]),
 ("scGPT",      [1.6934, 0.0031, 0.0251, -0.1261, 0.8322, 0.8571, MISSING, MISSING]),
 ("CPA",        [1.6592, 0.0029, 0.0240,  0.7906, 0.8150, 0.8980, MISSING, MISSING]),
 ("scDFM",      [1.6567, 0.0028, 0.0220,  0.8289, 0.8933, 0.8776, MISSING, MISSING]),
 ("scPKR (ours)",
                [2.2804, 0.00530, 0.03303, 0.8021, 0.8368, 0.8571, MISSING, MISSING]),
],
}

# The soft gate: the mean the model states, computed exactly rather than sampled. An L2
# between population means is what this column asks for, and the realisation only adds
# variance to it - 0.047 to 0.091 on three blocks, and 0.697 on ComboSciPlex, which is
# out of line with the rest and not yet explained.
SOFT_L2 = {"Table 1  Norman additive split": 1.5608,
           "Table 2  Norman holdout split - Single": 1.4136,
           "Table 2  Norman holdout split - Double": 1.9761,
           "Table 3  ComboSciPlex": 1.5836}


def _rank(rows: list, j: int) -> tuple:
    """Best and runner-up in column j, over the rows that reported it."""
    seen = [(r[1][j], r[0]) for r in rows
            if r[1][j] is not NA and r[1][j] != MISSING]
    if not seen:
        return None, None
    order = sorted(seen, key=lambda t: t[0], reverse=COLUMNS[j] not in LOWER)
    best = order[0][0]
    second = next((v for v, _ in order if v != best), None)
    return best, second


def render(name: str, rows: list, markdown: bool, digits: int) -> str:
    out = [f"### {name}", ""]
    width = max(len(r[0]) for r in rows) + 2
    head = f"{'Model':{width}s}" + "".join(f"{c:>16s}" for c in COLUMNS)
    arrow = "".join(f"{'(lower)' if c in LOWER else '(higher)':>16s}" for c in COLUMNS)
    out += [head, f"{'':{width}s}" + arrow, "-" * len(head)]
    for label, values in rows:
        cells = []
        for j, v in enumerate(values):
            best, second = _rank(rows, j)
            if v is NA:
                cells.append(f"{'N.A.':>16s}")
            elif v == MISSING:
                cells.append(f"{'-':>16s}")
            else:
                text = f"{v:.{digits}f}" if abs(v) >= 1e-4 else f"{v:.2e}"
                if markdown and v == best:
                    text = f"**{text}**"
                elif markdown and second is not None and v == second:
                    text = f"_{text}_"
                cells.append(f"{text:>16s}")
        out.append(f"{label:{width}s}" + "".join(cells))
    soft = SOFT_L2.get(name)
    if soft is not None:
        out += ["",
                f"scPKR's L2 from the mean it states rather than from sampled cells: "
                f"{soft:.4f}."]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true",
                        help="mark the best value bold and the runner-up italic")
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args()
    for name, rows in TABLES.items():
        print(render(name, rows, args.markdown, args.digits))
        print()


if __name__ == "__main__":
    main()
