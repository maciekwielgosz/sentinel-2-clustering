# Sentinel-2-clustering 

This is just a simple clustering of the data in order to see how it separates the categories.

Steps to follow:
1. Provide the data in HR folder
2. Run `prepare_data.py`
3. Run `compute_svd.py`
4. Run `cluster.py`


# Super-Resolution of Satellite Imagery With Deep Learning

In this research, we used **SRCNN** and **SwinIR Transformer** networks to investigate super-resolution techniques on various types of land cover using the Sentinel-2 dataset. We enhanced the resolution of these images from 20m/pixel to 10m/pixel, aiming to understand the relationship between land cover types and super-resolution processing.

An accuracy assessment was conducted using several metrics and different frequency domains. Our results showed a significant variation in metrics depending on the land cover types. The PSNR (peak signal to noise ratio) indicated 'forest' as the best-performing class and 'urban' as the least successful for both SRCNN and SwinIR.

The experiment highlights that super-resolution results differ for each land cover type, suggesting a need for consideration when applying super-resolution to specific land cover types.


