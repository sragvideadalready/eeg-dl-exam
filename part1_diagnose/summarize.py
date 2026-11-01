import json, sys, numpy as np
r = [json.loads(l)["test_acc"] for l in open(sys.argv[1])]
print(np.round(r, 3), "mean %.3f" % np.mean(r))
