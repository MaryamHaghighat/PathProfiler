# Training

In case you are interested to train your own segmentation model.

### 1. Data

The data folder should be organised in the following structure

```
data
-- train
	-- imgs
		-- img1.jpg
	-- labels
		-- img1.jpg
-- val
	-- imgs
		-- img2.jpg
	-- labels
		-- imgs.jpg
```

 If you have data organised in other formats, it might be convenient to adjust the [SegDataset](https://github.com/MaryamHaghighat/PathProfiler/blob/1288073ce7dcac046c073532a89919d96a74b8e7/tissue-segmentation/datasets.py#L113) class.

A label image should be a binary image where positive pixels take the value 255 and negative pixels 0.

### 2. Visualisation

We use [visdom](https://github.com/fossasia/visdom) to visualise the training progression. To start a visdom server

````
python -m visdom.server
````



## Author

Korsuk Sirinukunwattana (korsuk.sirinukunwattana@eng.ox.ac.uk)



###  

