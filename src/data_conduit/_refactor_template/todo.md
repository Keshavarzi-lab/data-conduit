To Do list for Refactor:

1. Complete the `datastructure` module creation, ie deprecation of unnecessary components of sessiongroups in favour of datastructure.py approach.

2. Add in a `trials` module. 

3. Add in any functionality/components that were present in `qc` module within data-conduit before that are not present within the `_refactor` subfolder. 

4. Determine how to fix DLC/pose object not following datasource format. 

5. Decide on how best to reorganise submodules within core. 

6. Set up `integrations` correctly. 

7. Ensure that all paths are correct. Ensure that `API_docs` works correctly too. 

8. Decide what examples should be present in `examples` folder. These will eventually be moved outside of src/data-conduit, but shall be used within here for now. 
    - Current ideas are separate demos for:
        i. Creating a new reader and adding it to the available readers. 
        ii. Create and apply monosource object. 
        iii. Create and apply multisource object. Extension: demonstrate how virtual array selectors work.
        iv. Demonstrate how to use level selectors. 
        v. Demonstrate how to set up and use datastructures
        vi. Demonstrate how to set up and use trials. 
        vii. Demonstrate all other functionality that was demonstrated in the demos notebook before within the `data_conduit_demos` subfolder. 
        viii... Any further demos that are appropriate. 
        (final demo). end to end example based on simplified case from one or two sessions from lab.

9. Outline (and potentially implement if it can be automated) how to incorporate test coverage for library.

10. Tidy up documentation, add full readmes and comments, both docstrings and more in-line comments. 

11. Create helpful visuals and other features for GitHub pages.

12. Fix any additional component for package like dependencies list, ruff, configs, etc. 

13. Page deployment stuff?

14. Other?