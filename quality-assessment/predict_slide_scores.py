import csv
import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import pickle
import pandas
import matplotlib.pyplot as plt
import glob


parser = argparse.ArgumentParser(description='Estimate WSI subjective scores')
parser.add_argument('--quality_overlays_dir', type=str, default='quality-overlays')
parser.add_argument('--tumor_mask_dir', default='', help='add another mask, e.g. tumor mask on top of tissue mask')
parser.add_argument('--slide_scores_filename', default='slide_scores.csv', type=str)
args = parser.parse_args()


def main():
    pred_usblty=list()
    pred_focus = list()
    pred_stain = list()
    qc_filename_list = glob.glob(os.path.join(args.quality_overlays_dir, '*.npy'))
    comments= len(qc_filename_list)*['']
    dist_list_sum = np.zeros((1,7)).astype(np.float)
    focus_model = pickle.load(open('quality-assessment/model_focus_score.pkl', 'rb'))
    usblty_model = pickle.load(open('quality-assessment/model_usblty_score.pkl', 'rb'))
    stain_model = pickle.load(open('quality-assessment/model_stain_score.pkl', 'rb'))
    for i, npy_path in enumerate(qc_filename_list):
        slide_info_loaded = np.array(np.load(npy_path, allow_pickle=True))
        usblty = np.array(slide_info_loaded[()]['usblty'])
        normal = np.array(np.array(slide_info_loaded[()]['normal']))
        focus_artfcts = np.array(np.array(slide_info_loaded[()]['focus_artfcts']))
        stain_artfcts = np.array(np.array(slide_info_loaded[()]['stain_artfcts']))
        folding_artfcts = np.array(np.array(slide_info_loaded[()]['folding_artfcts']))
        other_artfcts = np.array(np.array(slide_info_loaded[()]['other_artfcts']))
        processed_region = np.array(np.array(slide_info_loaded[()]['processed_region']))

        if args.tumor_mask_dir:
            try:
                mask_path = os.path.join(args.tumor_mask_dir,
                                         Path(npy_path).stem.replace('_quality_overlays', '')+'.png')
                mask = cv2.imread(mask_path, -1)
                mask = cv2.resize(mask, (usblty.shape[1], usblty.shape[0]), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
                usblty[mask < 1] = 0
                normal[mask < 1] = 0
                focus_artfcts[mask < 1] = 0
                stain_artfcts[mask < 1] = 0
                folding_artfcts[mask < 1] = 0
                other_artfcts[mask < 1] = 0
                processed_region[mask < 1] = 0
            except:
                print('No tumor mask for', npy_path)
                pred_usblty.extend(['NA'])
                pred_stain.extend(['NA'])
                pred_focus.extend(['NA'])
                comments[i] = 'No tumor mask'
                continue

        if processed_region.sum() == 0:
            print('zero tissue mask:', npy_path)
            pred_usblty.extend([0.])
            pred_stain.extend(['NA'])
            pred_focus.extend(['NA'])
            comments[i] = 'No tissue'
            with open('no_tissue.csv', 'a', ) as file:
                wr = csv.writer(file)
                wr.writerow([os.path.basename(npy_path)])
            continue

        dist_list_s = np.array([usblty[processed_region > 0].sum(),
                                normal[processed_region > 0].sum(),
                                focus_artfcts[processed_region > 0].sum(),
                                stain_artfcts[processed_region > 0].sum(),
                                folding_artfcts[processed_region > 0].sum(),
                                other_artfcts[processed_region > 0].sum(),
                                processed_region[processed_region > 0].sum()])
        dist_list_sum = dist_list_sum + dist_list_s
        fa = focus_artfcts[processed_region > 0]
        sa = stain_artfcts[processed_region > 0]

        if len(fa) == 1:
            a = fa
            b = sa
            comments[i] = 'one tile only'
        else:
            a = np.mean(fa[fa >= np.percentile(fa, 80)])
            b = np.mean(sa[sa < np.percentile(sa, 90)])

        feats_usblty = [usblty[processed_region > 0].mean(), usblty[processed_region > 0].std(),
                        stain_artfcts[processed_region > 0].mean(), stain_artfcts[processed_region > 0].std(),
                        focus_artfcts[processed_region > 0].mean(), focus_artfcts[processed_region > 0].std()]

        feats_focus = [a, focus_artfcts[processed_region > 0].mean(), focus_artfcts[processed_region > 0].std()]
        feats_stain = [b, stain_artfcts[processed_region > 0].mean(), stain_artfcts[processed_region > 0].std()]

        pred_usblty.extend(np.clip(np.round(usblty_model.predict([feats_usblty]), 1), 0, 1))
        pred_focus.extend(np.clip(np.round(focus_model.predict([feats_focus])), 0, 10))
        pred_stain.extend(np.clip(np.round(stain_model.predict([feats_stain])), 0, 10))

    print('average tile scores:', dist_list_sum/dist_list_sum[0, -1])
    with open(args.slide_scores_filename, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(['slide_id', 'usability_score', 'focus_score', 'staining_score', 'comments'])
        for i, npy_file in enumerate(qc_filename_list):
            slide_id = (os.path.basename(npy_file)).replace('_quality_overlays.npy', '')
            wr.writerow([slide_id, pred_usblty[i], pred_focus[i], pred_stain[i], comments[i]])

        bin_pred_usabilty = [item for item in pred_usblty if not item == 'NA']
        bin_pred_usabilty = [0 if item < .5 else 1 for item in bin_pred_usabilty]
        print('Percentage of usable slides (cut off at 0.5):', 100*np.mean(bin_pred_usabilty))


if __name__ == "__main__":
    main()
    print('done!')
