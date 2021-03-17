
import numpy as np

def compute_all_aps(ap_dicts, classes):

    ap50 = 0.0
    ap75 = 0.0
    ap = 0.0
    final_ap = {50: 0, 55: 0, 60: 0, 65: 0, 70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 95: 0}
    for key in range(50, 100, 5):
        for k, v in ap_dicts[key].items():
            final_ap[key] += v
        final_ap[key] = final_ap[key] / classes

    ap = np.mean(list(final_ap.values()))
    ap50 = final_ap[50]
    ap75 = final_ap[75]
    print("-" * 100)
    print(final_ap)
    print("-" * 100)
    return ap, ap50, ap75
