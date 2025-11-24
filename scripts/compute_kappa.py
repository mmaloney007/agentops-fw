import csv, sys
from collections import Counter

def cohen_kappa(labels_a, labels_b):
    assert len(labels_a) == len(labels_b) and len(labels_a)>0
    n = len(labels_a)
    agree = sum(1 for a,b in zip(labels_a, labels_b) if a==b)
    p0 = agree / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[x]/n)*(cb[x]/n) for x in set(ca)|set(cb))
    if pe == 1: return 1.0
    return (p0 - pe) / (1 - pe)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "annotations.csv"
    la, lb = [], []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            a = row.get("annotator1_label"); b = row.get("annotator2_label")
            if a and b: la.append(a); lb.append(b)
    if not la:
        print("No overlapping labels found."); sys.exit(1)
    print(f"Cohen's kappa: {cohen_kappa(la, lb):.3f}")
