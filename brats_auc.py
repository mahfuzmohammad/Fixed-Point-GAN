import numpy as np
from glob import glob
from PIL import Image
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

imgs = glob("brats_syn_256_lambda0.1/results/*.jpg")
gts = [0]*975 + [1]*486

preds = []

for i in range(len(imgs)):
	im = np.array(Image.open("brats_syn_256_lambda0.1/results/{}-images.jpg".format(i+1)))
	rows = np.split(im, im.shape[0]//256, axis=0)

	for r in rows:
		cols = np.split(r, 5, axis=1)
		preds.append(np.max( np.mean(cols[1], axis=-1) ))

print(roc_auc_score(gts, preds))