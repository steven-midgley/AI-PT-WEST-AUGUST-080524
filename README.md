# ASU-VIRT-AI-PT-11-2023-U-LOLC

## ASU-AI-Bootcamp

### Extra Helpfull Instructor Recordings:
* [Python Practice Codes for the Break](https://pythonprogramming.net/)
* [More Python Practice(200 Mini-Projects)](https://thecleverprogrammer.com/2021/01/14/python-projects-with-source-code/)
* [Set Up SSH Key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

* [GitLab](https://zoom.us/rec/share/jVyJIva85n9JEn_ay0sbl9Yoa3RYwuWpY5LiYU4loXLCf7x54DpEY0hgrOAfGU8B.YwMtILEWTyxSDSmB?startTime=1664669201000)

* [ATM Activity](https://zoom.us/rec/share/jClmBy6CWbeNSu6R9ZRwKWMMb5-3WutbuYwcmJBmu94PZ5MX4Jt7THzg6HdmT4o.MPrZh98_XpyqfP-J?startTime=1664667713000)

* [GitHub, Challenges, Readme](https://zoom.us/rec/share/vCbsxgKalwlcHw5MkfWTAk5usZlo9-1lsy8IIdJc-i8niyNEN3R7n6iCTSI4EVVm.Tv3hmeKWncn7M68r?startTime=1664669495000)

* [Pandas Demo](https://zoom.us/rec/share/cNNhE83OMwS0NbboHybJ_qnn9IPjm-M_s3dyoEqkxCkgKqoIO_l1udLiMDV4QHtZ.YzyEaHZT-rInHADw?startTime=1665372022000)

* [Github Branching](https://zoom.us/rec/share/DN_KItdVPFezV6GLiPAcA0uPcooMUjSOAZWbGZxgqvsQVqsZUWEHRwDkH92Chrci.lxoBBcanh1QKibS7)

```conda
ACTIVATE COND ENVIROMENT: conda activate env_name
CREATE NEW CONDA ENV: conda create -n env_name python=3.xx (xx resembles what python version you want)
DELETE AN ENV INSIDE ANACONDA: conda env remove --name ENVIRONMENT
INSTALL PACKAGES IN YOUR CONDA ENV FROM txt file: pip install -r requirements.txt
CREATE ENV FROM YAML: conda env create --name environment_name -f environment.yml
EXPORT CONDA TO YAML: conda env export > environment_droplet.yml

```

### Jupyter Lab Flicks

* Please follow the instructions in the article: [The article](https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874)

A summary of the article basically is:
In every enviroment you activate you need to run the following two commands:
* `conda install ipykernel`
* `ipython kernel install --user --name=new-env` where `new-env` is you enviroment name (i.e. dev or whatever you called it)