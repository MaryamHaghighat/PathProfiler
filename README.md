PathProfiler: Quality Assessment of Histopathology Whole-Slide Image Cohorts
===================================================

In this study, a quality assessment pipeline is  proposed in which possible multiple artefacts are predicted in a same region along with diagnostic usability of the image. 

**How does it work?** A  multi-task deep neural network is trained to predict if an image tile is usable for diagnosis/research and the  kind of artefacts present in the image tile. Quality overlays are then generated from image tile predictions. Quality overlays are further mapped to a standard  scoring system to predict the usability,  focus and staining quality of the whole slide images.

===================================================
### Tissue Segmentation
<img src="imgs/tissue_segmentation.jpg" align="center" />

A UNET segmentation model ([download](https://drive.google.com/file/d/1otWor5WnaJ4W9ynTOF1XS755CsxEa4qj/view?usp=sharing)) is trained on multiple tissue types including prostate and colon tissue to separate tissue from background. It can be deployed by running "**tissue_segmentation/run.py**" with the following arguments: 

* `--slide_dir`:  path to slide directory
* `--slide_id`:  slide filename (or "*" for all slides)
* `--save_folder`:  path to save results
* `--mask_magnification`:  magnification power of generated tissue masks. It is recommended to use 1.25 or 2.5.
* `--mpp_level_0`:  manually enter mpp at level 0 if not available in slide properties as "slide.mpp['MPP']"

===================================================
###  Tile extraction (parallel processing on CPU)
"**tile-extract/tiling.py**" extracts tiles from WSIs passing the following arguments:

* `--slide_dir`:  path to slide directory
* `--slide_id`:  slide filename (or "*" for all slides)
* `--save_folder`:  path to save results
* `--tile_magnification`:  magnification at which tiles are extracted
* `--mask_magnification`:  magnification power of tissue masks
* `--tile_size`:  pixel size of the tiles; the default is 256
* `--stride`:  stride; the default is 256
* `--mask_dir`:  path to save tissue masks 
* `--mask_ratio`:  the minimum acceptable masked area (available tissue) to extract tile
* `--mpp_level_0`:  manually enter mpp at level 0 if not available in slide properties as "slide.mpp['MPP']"

===================================================
### Quality assessment (parallel processing on CPU)
<img src="imgs/pipeline.jpg" align="center" />

A multi-label ResNet18 model ([download](https://drive.google.com/file/d/13egPkDufR6W4aTBUAAf8uV6zQxwdBx6r/view?usp=sharing)) with 6 outputs of linear activation function is trained on image tiles from ProMPT prostate cancer cohort.  
Annotated tiles are  256x256 in size and have been exctracted at 5X magnification. Tiles are further downsampled to 224x224 to accomodate for the model.
The model outputs are:

* `output1`:  predicts the usability of an image, where 1 indicates the image is of appropriate quality for diagnosis.
* `output2`:  predicts if an image looks normal, where 1 indicates no artefact seen.
* `output3`:  predicts if an image is out of focus, where 1 indicates severe, 0.5 slight, and 0 no focus issues.
* `output4`:  predicts if an image has staining issues, where 1 indicates severe, 0.5 slight, and 0 no staining issues.
* `output5`:  predicts if an image has tissue folding, where 1 indicates that tissue folding is present.
* `output6`:  predicts if an any other artefacts such as dirt, glue, ink, cover slip edge, diathermy, bubbles, calcification and tissue scoring is present, where
1 indicates that other artefacts are present.

**Notes**: ProMPT dataset has very limited areas of folded tissue and hence our dataset does not include a various forms of tissue folding in histology.

<img src="imgs/overlays.png" align="center" />


To start quality assessment tool, run:
**python quality-assessment/run.py** by passing the following arguments:
* `--slide_dir`:  path to slides
* `--slide_id`:  slide filename or "*" for going through all slides.
* `--mpp_level_0`: manually enter mpp at level 0 if not available in slide properties as "slide.mpp['MPP']"
* `--mask_dir`: path to tissue mask folder (the output folder of tissue segmentation step)
* `--mask_magnification`: the magnification power of tissue masks 
* `--overlay_magnification`: the magnification power of generated quality overlays (heatmaps) 
* `--mask_ratio`: the minimum ratio of masked area (tissue) in an image tile to proceed tile processing
* `--save_folder`: folder path to save results. 
Quality overlays are collected in a dictionary and saved as **slide_name.npy** with key values as below:
``` shell 
{'usblty': quality overlay from output1, 'normal': quality overlay from output2, 
'stain_artfcts': quality overlay from output3, 'focus_artfcts': quality overlay from output4, 
'folding_artfcts': quality overlay from output5, 'other_artfcts': quality overlay from output6 
'processed_region': regions that have been processed during quality assessment} 
```
    
**Notes:**
- the pixel size of quality overlays saved in "slide_name.npy" is (slide_size_at_5X) / 256.
- each pixel value in the quality overlay represents quality prediction value for a tile of  256*256. 
- quality overlays can be easily regenerated from "slide_name.npy" at magnification "X" by: 

`overlay at magnification X = overlay.repeat(X*256/5, axis=0).repeat(X*256/5, axis=1)`



### Mapping quality overlays to standard whole-slide quality scores  
Three separate linear regression models are used to predict WSI usability, focus and staining scores. 
To map the quality overlays to standard slide-level scores, run:
**python predict_slide_scores.py** by passing the following arguments:
* `--quality_overlays_dir`:  path to the quality overlays folder
* `--tumor_mask_dir`:  add another mask (e.g. tumor mask) on top of tissue mask; default is None.
* `--slide_scores_filename`:  csv filename to save standard quality scores for each slide

Our model has been only exposed to artefacts in ProMPT, a local cohort of 4732 histology slides of prostate cancer collated between 2001-2018 as part of a UK-based observational study. While the model has not been trained or validated on external cohorts, we estimated quality overlays for WSIs in TCGA-prostate and FOCUS datasets  ([here](https://drive.google.com/drive/folders/1D9fIt67dBxaOqWcOXYpZWiKaYGhAXwZu?usp=sharing)) for further community investigation. Estimated  standard WSI quality scores for TCGA ([here](https://drive.google.com/file/d/1Kuz1TOQ_HHFKdeuV1bNpTAl58EDLXXM1/view?usp=sharing)) and tumor regions of FOCUS slides ([here](https://drive.google.com/file/d/1emvYNkmCuCjdDvUQOH6W4zdgOUEoLNVG/view?usp=sharing)) are also provided . With the help of community to  collect various artefacts in different tissue types, we believe the model performance will improve for external cohorts and hence this work can be extended to a comprehensive and clinically relevant quality assessment tool. 

### Examples 
####run the pipeline on TCGA slides

1- Tissue segmentation
``` shell
python tissue-segmentation/run.py --save_folder 'tissue-masks/TCGA' --slide_dir 'TCGA_slides_Directory' --mask_magnification 1.25 --model 'checkpoint_147800.pth'
```
2- Generate quality overlays:
``` shell
python quality-assessment/run.py --slide_dir 'TCGA_slides_Directory' --mask_dir 'tissue-masks/TCGA' --mask_magnification 1.25 --save_folder 'quality-assessment/quality-overlays/TCGA' --model 'checkpoint_106.pth' --slide_id '*.svs'
```
3- Predict slide-level scores
``` shell
python quality-assessment/predict_slide_scores.py --quality_overlays_dir quality-assessment/quality-overlays/TCGA --slide_scores_filename TCGA_slide_scores.csv
```
#### Extract tiles from TCGA slides
``` shell
python tile-extract/tiling.py --save_folder 'tiles' --slide_dir 'TCGA_slides_Directory' --mask_magnification 1.25  --mask_dir 'tissue-masks/TCGA'  --slide_id '*.svs'
```

## To do
Current model is trained and tested on the ProMPT cohort produced and scanned in the Cellular Pathology Department at Oxford University Hospitals NHS Foundation Trust (OUHFT). To improve the model performance on other cohorts (e.g TCGA) we intend to incorporate images from different cohorts to the training process using semi-supervised techniques. However, a community effort is required to build a comprehensive quality assessment tool for histology slides and  validate results on other cohorts.

## Contact authors
maryam.haghighat@eng.ox.ac.uk

stefano.malacrino@nds.ox.ac.uk

korsuk.sirinukunwattana@eng.ox.ac.uk

## License
This code is made available under the GPLv3 License.

## Funding
This work is supported by the PathLAKE Centre of Excellence for digital pathology and AI which is funded, managed and delivered by Innovate UK on behalf of UK Research and Innovation (UKRI). 

## Reference
Please refer to our pre-print (https://www.medrxiv.org/content/10.1101/2021.09.24.21263762v1) for detailed results.
If you find our work useful in your research or if you use parts of this code please consider citing our paper:
```
@article {Haghighat_PathProfiler,
	author = {Haghighat, Maryam and Browning, Lisa and Sirinukunwattana, Korsuk and Malacrino, Stefano and Alham, Nasullah Khalid and Colling, Richard and Cui, Ying and Rakha, Emad and Hamdy, Freddie and Verrill, Clare and Rittscher, Jens},
	title = {PathProfiler: Automated Quality Assessment of Retrospective Histopathology Whole-Slide Image Cohorts by Artificial Intelligence, A Case Study for Prostate Cancer Research},
	elocation-id = {2021.09.24.21263762},
	year = {2021},
	doi = {10.1101/2021.09.24.21263762},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2021/09/27/2021.09.24.21263762},
	eprint = {https://www.medrxiv.org/content/early/2021/09/27/2021.09.24.21263762.full.pdf},
	journal = {medRxiv}
}
```
