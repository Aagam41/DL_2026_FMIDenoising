
<p align="center">
  <a href="https://ai4life.eurobioimaging.eu/open-calls/">
    <img src="https://public.grand-challenge-user-content.org/b/833/banner.x15.jpeg" width="100%">
  </a>
</p>


# Calcium Imaging Denoising Challenge AI4Life-CIDC25


## Example submission container 

Welcome to the AI4Life-CIDC25 denoising challenge! 
Check [The Challenge Page](https://ai4life-cidc25.grand-challenge.org/) for all the details about the challenge. 

On this page, you can find an example submission for the Grand Challenge platform.

### Submission checklist
- [ ] Check that all the requirements are contained in the `requirements.txt` and `Dockerfile`.
- [ ] Test your container locally with `do_test_run.sh`.
- [ ] Create a gzip archive from your image, run `do_save.sh`. This may take a while!
- [ ] Go to [this page](https://ai4life-cidc25.grand-challenge.org/how-to-submit-your-algorithm/) for further instructions on how to submit to the Grand Challenge.


### How to use this code
1. Example inputs are stored in the `test/input/` folder. 
2. Look through the contents of the [inference.py](inference.py) script.
3. Run [do_test_run.sh](do_test_run.sh) to build and test the container execution.
4. The resulting image should appear in the `test/output/` folder.

### What is in the example? 

Here, the example submission contains just dummy predictions. You can use this as a template to create your own submission.

### Useful links

Make sure to check Grand Challenge [documentation](https://grand-challenge.org/documentation/participate-in-a-challenge/) with any questions you may have.  

For **any** other questions or issues, create a topic on the [challenge forum](https://grand-challenge.org/forums/forum/ai4life-calcium-imaging-denoising-challenge-2025-798/) or drop us an email through the *Email organizers* button on the challenge page.


### Thank you for participating, and we are looking forward to receiving your submission!
