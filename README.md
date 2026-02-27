# Visual-COT: Video Understanding via Reasoning Chains
### USYD IT Capstone Project (Master of Computer Science)

## Project Overview
This repository contains the implementation of our Capstone project at the **University of Sydney**. We are building upon the [Video-R1](https://github.com/tulerfeng/Video-R1) framework to implement and evaluate Chain-of-Thought (COT) reasoning for complex video understanding.

## Team Collaboration (Important!)
To our team members: Please follow these guidelines to ensure smooth collaboration:
- **Issue Tracking**: If you encounter any problems (e.g., environment setup, platform-specific errors, or code bugs), please **submit an Issue** in this repository. 
  - Whether you are using **AutoDL**, local servers, or other cloud platforms, feel free to report your findings or blockers there.
  - This helps us keep track of all technical challenges in one place.
- **Git Usage**: If you are not familiar with Git commands, you can still contribute by uploading results or feedback through the GitHub web interface or the Issue section.

## Current Progress
- [x] Base repository initialized and structured.
- [x] Model weights (`Qwen2.5-VL-7B-COT-SFT`) verified and download scripts prepared.
- [x] Basic environment requirements documented.

## Project Structure
- `src/`: Core source code and experimental scripts.
- `README_ORIGIN.md`: Original documentation and setup instructions from Video-R1.
- `download_model.py`: Utility script to fetch the required model weights.

## Getting Started
1. **Environment**: This project is platform-agnostic. While we currently use AutoDL, you can deploy it on any platform with sufficient GPU memory.
2. **Model**: Run `python download_model.py` to prepare the weights locally (do not upload weights to GitHub).
3. **Troubleshooting**: Check the `Issues` tab for existing solutions or to post a new question.

##  Acknowledgement
This project is a derivative work of **Video-R1**. We acknowledge the original authors for their foundational contributions.
