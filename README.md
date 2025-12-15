# ECE-253-Fundamentals-of-Digital-Image-Processing-Final-Project
ECE 253 DIP final project, Old Photo Restoration and Enhancement

To clone this repository:
```bash
git clone https://github.com/Nachiket-Laul/ECE-253-Fundamentals-of-Digital-Image-Processing-Final-Project.git
```

For best results, use high-resolution scanned images.
## 🔧 Installation

### Install Required Libraries
Open the VS Code terminal using **Ctrl + `** and run:

```bash
pip install opencv-python numpy 
```

## ▶️ Running the Code
As per the requirement run the Traditional_Pipeline_for_restoration_and_enhancement.py for a single image restoration and enhancement.

```bash
python Traditional_Pipeline_for_restoration_and_enhancement.py
```
If you wish to run and compare different methodologies proposed, run:

```bash
python Comparison_between_methodologies_for_traditional_pipeline.py
```
## 📝 Note

- Ensure the input image name and path is correctly specified in the script before execution for the Traditional_Pipeline_for_restoration_and_enhancement.py.
- Ensure the old_photos folder path name and output folder name is correctly specified in the script Comparison_between_methodologies_for_traditional_pipeline.py.
- The restoration quality depends on the level of degradation in the input image.
- Parameter values may need slight tuning for different photographs.
- For best results, use high-resolution scanned images.


## ⭐ Special Credits
The GFPGAN is implemented using the the code in https://github.com/TencentARC/GFPGAN.git which is also forked in our profile.

You can try implementing the GFPGAN first and then the traditional pipeline for better results.






